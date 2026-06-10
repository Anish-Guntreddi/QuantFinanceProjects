"""EventDrivenBacktester — deterministic main loop with T+1 pending-order buffer.

Design invariants (from RESEARCH.md Pattern 1 and locked decisions):

  1. T+1 fill guarantee: An order generated from bar T's signal fills at bar T+1's
     open.  The loop flushes _pending_orders BEFORE advancing the data cursor.

  2. Deterministic: Two runs with identical config and seed produce byte-identical
     equity curves.

  3. No reset() method — fresh instances only.

  4. EOD cancellation: Unfilled pending orders at end-of-data are logged at WARNING,
     stored in results.cancelled_orders with zero cost, and excluded from trade stats.

  5. Risk seam: A signal that violates risk limits never produces a fill (blocked at
     generate_orders, which calls risk_manager.validate_order).

Loop order per iteration:
  (1) Flush _pending_orders using _peek_next_bar → fill_at_open → portfolio.on_fill
      APPLIED IMMEDIATELY, so bar T+1's signals/sizing see post-fill state
      (codex review finding 1: routing fills through the queue made them process
      after T+1 market/signal events due to FILL's low same-timestamp priority)
  (2) data_handler.update_bars() → enqueue MarketEvents
  (3) Drain EventQueue:
        MARKET  → strategy.calculate_signals
        SIGNAL  → portfolio.generate_orders (enqueue OrderEvents)
        ORDER   → append to _pending_orders (NEVER fill same bar)
  (4) portfolio.mark_to_market ONCE per bar (codex review finding 3: per-event
      MTM produced N equity points per bar for N symbols)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from qbacktest.events import (
    EventQueue,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from qbacktest.execution.handler import SimulatedExecutionHandler
from qbacktest.execution.commission import ZeroCommission
from qbacktest.execution.slippage import ZeroSlippage
from qbacktest.metrics.performance import MetricsReport, compute_metrics
from qbacktest.portfolio.portfolio import Portfolio
from qbacktest.risk.manager import RiskManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """Configuration for EventDrivenBacktester.

    All fields have sensible defaults for quick prototyping.
    """
    initial_capital: float = 100_000.0
    position_size: float = 0.1
    max_position_weight: float = 0.2
    max_gross_exposure: float = 1.0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None


# ---------------------------------------------------------------------------
# BacktestResults
# ---------------------------------------------------------------------------


@dataclass
class BacktestResults:
    """Results of a completed backtest run.

    Attributes
    ----------
    equity_curve:
        DatetimeIndex → net equity values.
    gross_returns:
        Per-bar returns before transaction costs.
    net_returns:
        Per-bar returns after transaction costs.
    metrics:
        Full MetricsReport (Sharpe, Sortino, drawdown, etc.)
    trades:
        List of all FillEvents (filled orders).
    cancelled_orders:
        List of OrderEvents that were buffered but never filled (EOD cancellations).
    gross_sharpe:
        Convenience passthrough from metrics.gross_sharpe.
    net_sharpe:
        Convenience passthrough from metrics.net_sharpe.
    cost_bps:
        Convenience passthrough from metrics.cost_bps.
    """
    equity_curve: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    metrics: MetricsReport
    trades: list[FillEvent]
    cancelled_orders: list[OrderEvent]
    gross_sharpe: float
    net_sharpe: float
    cost_bps: float


# ---------------------------------------------------------------------------
# EventDrivenBacktester
# ---------------------------------------------------------------------------


class EventDrivenBacktester:
    """Deterministic event-driven backtester.

    Parameters
    ----------
    data_handler:
        DataHandler providing ``update_bars()``, ``get_latest_bars()``, and the
        engine-internal ``_peek_next_bar()`` (not part of the strategy-facing API).
    strategy:
        Strategy subclass implementing ``calculate_signals()``.
    portfolio:
        Portfolio instance.  If None, one is built from ``config``.
    execution_handler:
        SimulatedExecutionHandler (or custom).  If None, one with Zero models is built.
    config:
        BacktestConfig controlling capital, sizing, and risk limits.
        If None, defaults are used.

    Notes
    -----
    No ``reset()`` method — instantiate a fresh object for each run.
    """

    def __init__(
        self,
        data_handler,
        strategy,
        portfolio: Portfolio | None = None,
        execution_handler: SimulatedExecutionHandler | None = None,
        config: BacktestConfig | None = None,
    ) -> None:
        self.data_handler = data_handler
        self.strategy = strategy
        self.config: BacktestConfig = config if config is not None else BacktestConfig()

        # Build defaults from config if not supplied
        if portfolio is None:
            risk_manager = RiskManager(
                max_position_weight=self.config.max_position_weight,
                max_gross_exposure=self.config.max_gross_exposure,
            )
            self.portfolio = Portfolio(
                initial_capital=self.config.initial_capital,
                position_size=self.config.position_size,
                risk_manager=risk_manager,
            )
        else:
            self.portfolio = portfolio

        if execution_handler is None:
            self.execution_handler = SimulatedExecutionHandler(
                slippage_model=ZeroSlippage(),
                commission_model=ZeroCommission(),
            )
        else:
            self.execution_handler = execution_handler

        # T+1 pending-order buffer (orders wait here until next bar's open)
        self._pending_orders: list[OrderEvent] = []

        # Accumulate fills and cancelled orders
        self._fills: list[FillEvent] = []
        self._cancelled_orders: list[OrderEvent] = []

        # Gross equity tracking: parallel equity that ignores commission cost
        # We track cost-free equity by adding back commission on each fill.
        # We maintain the same equity curve shape but reconstruct gross as:
        #   gross_equity[t] = net_equity[t] + cumulative_commission[t]
        self._cumulative_commission_at_bar: list[float] = []

        # EventQueue for the current iteration
        self._queue = EventQueue()

        # Wire strategy to data handler
        strategy.initialize(data_handler)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> BacktestResults:
        """Execute the full backtest and return BacktestResults.

        Loop order per bar:
          1. Flush _pending_orders using _peek_next_bar + fill_at_open
          2. Advance data via update_bars → enqueue MarketEvents
          3. Drain queue: MARKET → signals + mtm; SIGNAL → orders; ORDER → buffer; FILL → accounting

        After the loop: remaining _pending_orders are cancelled (logged WARNING).
        """
        while True:
            # --- Step 1: Flush pending orders at T+1 open ----------------------
            # Fills are applied to the portfolio IMMEDIATELY (not queued), so
            # this bar's signals and sizing observe post-fill cash/positions.
            self._flush_pending_orders()

            # --- Step 2: Advance data ------------------------------------------
            market_events = self.data_handler.update_bars()
            if not market_events:
                # No more bars — loop complete
                break

            for me in market_events:
                self._queue.put(me)

            # --- Step 3: Drain event queue -------------------------------------
            self._drain_queue()

            # --- Step 4: Mark to market ONCE per bar ----------------------------
            bar_timestamp = market_events[0].timestamp
            prices = self._latest_close_prices()
            self.portfolio.mark_to_market(bar_timestamp, prices)

            # --- Track cumulative commission snapshot after each bar -----------
            self._cumulative_commission_at_bar.append(
                self.portfolio.cumulative_costs
            )

        # --- EOD cancellation -------------------------------------------------
        for order in self._pending_orders:
            logger.warning(
                "EOD cancellation: order %s for %s (no T+1 bar available)",
                order.order_id,
                order.symbol,
            )
            self._cancelled_orders.append(order)
        self._pending_orders.clear()

        # --- Build results ----------------------------------------------------
        return self._build_results()

    # ------------------------------------------------------------------
    # Pending-order flush (T+1 fill)
    # ------------------------------------------------------------------

    def _flush_pending_orders(self) -> None:
        """Fill every pending order at the next bar's open price.

        For each order, call the engine-internal _peek_next_bar without advancing
        the cursor. Fills are applied to the portfolio IMMEDIATELY — never queued —
        so the upcoming bar's signals and sizing observe post-fill state.
        If no next bar exists, the order stays in the buffer (will be cancelled
        at EOD by the main loop).
        """
        if not self._pending_orders:
            return

        filled_indices = []
        for i, order in enumerate(self._pending_orders):
            next_bar = self.data_handler._peek_next_bar(order.symbol)
            if next_bar is None:
                # No T+1 bar — will be cancelled at EOD
                continue
            fill = self.execution_handler.fill_at_open(order, next_bar)
            if fill is not None:
                self._handle_fill_event(fill)
                filled_indices.append(i)

        # Remove filled orders from buffer (reverse order to preserve indices)
        for i in reversed(filled_indices):
            self._pending_orders.pop(i)

    # ------------------------------------------------------------------
    # Queue drain
    # ------------------------------------------------------------------

    def _drain_queue(self) -> None:
        """Process all events currently in the queue."""
        while not self._queue.empty():
            event = self._queue.get()
            if event.event_type == EventType.MARKET:
                self._handle_market_event(event)
            elif event.event_type == EventType.SIGNAL:
                self._handle_signal_event(event)
            elif event.event_type == EventType.ORDER:
                self._handle_order_event(event)
            elif event.event_type == EventType.FILL:
                self._handle_fill_event(event)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_market_event(self, event: MarketEvent) -> None:
        """Generate signals from the strategy (MTM happens once per bar in run())."""
        signals = self.strategy.calculate_signals(event)
        for signal in signals:
            self._queue.put(signal)

    def _latest_close_prices(self) -> dict[str, float]:
        """Latest available close per symbol, for the once-per-bar MTM."""
        prices: dict[str, float] = {}
        for sym in self.data_handler._symbols:
            latest = self.data_handler.get_latest_bars(sym, 1)
            if not latest.empty:
                prices[sym] = float(latest["close"].iloc[-1])
        return prices

    def _handle_signal_event(self, signal: SignalEvent) -> None:
        """Convert signal to orders using portfolio's position sizing."""
        # Get current price for this symbol
        latest = self.data_handler.get_latest_bars(signal.symbol, 1)
        if latest.empty:
            logger.warning("No bars available for signal on %s", signal.symbol)
            return
        price = float(latest["close"].iloc[-1])

        orders = self.portfolio.generate_orders(signal, price)
        for order in orders:
            self._queue.put(order)

    def _handle_order_event(self, order: OrderEvent) -> None:
        """Buffer order — NEVER fill on the same bar (T+1 invariant)."""
        self._pending_orders.append(order)

    def _handle_fill_event(self, fill: FillEvent) -> None:
        """Delegate all accounting to portfolio.on_fill."""
        self.portfolio.on_fill(fill)
        self._fills.append(fill)

    # ------------------------------------------------------------------
    # Results construction
    # ------------------------------------------------------------------

    def _build_results(self) -> BacktestResults:
        """Build BacktestResults from portfolio state and fills."""
        equity_data = self.portfolio.equity_curve
        if not equity_data:
            # Empty run (no bars)
            empty_series = pd.Series([], dtype=float, name="equity")
            empty_returns = pd.Series([], dtype=float)
            empty_metrics = compute_metrics(
                empty_series, empty_returns, empty_returns, [], 0.0, 0.0
            )
            return BacktestResults(
                equity_curve=empty_series,
                gross_returns=empty_returns,
                net_returns=empty_returns,
                metrics=empty_metrics,
                trades=list(self._fills),
                cancelled_orders=list(self._cancelled_orders),
                gross_sharpe=0.0,
                net_sharpe=0.0,
                cost_bps=0.0,
            )

        # Build net equity Series
        timestamps = [t for t, _ in equity_data]
        net_equities = [e for _, e in equity_data]
        net_equity_series = pd.Series(net_equities, index=timestamps, name="equity")

        # Build gross equity Series: add back cumulative commission at each bar.
        # Both lists are appended exactly once per bar in run() — any drift is a
        # structural bug, so fail loudly instead of silently padding.
        n = len(net_equities)
        cum_commissions = self._cumulative_commission_at_bar
        if len(cum_commissions) != n:
            raise RuntimeError(
                f"equity/commission snapshot misalignment: {n} equity points vs "
                f"{len(cum_commissions)} commission snapshots"
            )
        gross_equities = [
            net_equities[i] + cum_commissions[i]
            for i in range(n)
        ]
        gross_equity_series = pd.Series(gross_equities, index=timestamps, name="gross_equity")

        # Compute returns from equity series
        net_returns = net_equity_series.pct_change().dropna()
        gross_returns = gross_equity_series.pct_change().dropna()

        # Total costs = cumulative commissions
        total_costs = self.portfolio.cumulative_costs

        # Compute metrics
        metrics = compute_metrics(
            equity_curve=net_equity_series,
            gross_returns=gross_returns,
            net_returns=net_returns,
            trade_pnls=list(self.portfolio.trade_pnls),
            total_traded_value=self.portfolio.total_traded_value,
            total_costs=total_costs,
        )

        return BacktestResults(
            equity_curve=net_equity_series,
            gross_returns=gross_returns,
            net_returns=net_returns,
            metrics=metrics,
            trades=list(self._fills),
            cancelled_orders=list(self._cancelled_orders),
            gross_sharpe=metrics.gross_sharpe,
            net_sharpe=metrics.net_sharpe,
            cost_bps=metrics.cost_bps,
        )
