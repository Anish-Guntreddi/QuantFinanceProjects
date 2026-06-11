---
phase: 5
slug: defiregimenet
status: planned
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-11
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (quant venv, Python 3.11) |
| **Config file** | portfolio_projects/defiregimenet/pyproject.toml — created in 05-01 Task 1 |
| **Quick run command** | `python -m pytest tests/ -q -x` (from portfolio_projects/defiregimenet/) |
| **Full suite command** | `python -m pytest tests/ -q` (from portfolio_projects/defiregimenet/) |
| **Estimated runtime** | ~60-120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -q -x`
- **After every plan wave:** Run `python -m pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| Skeleton + editable install | 05-01 T1 | 1 | DFR-01 | smoke | `python -c "import defiregimenet"` (from /tmp) | n/a | ⬜ pending |
| CryptoGenerator + data quality + ccxt | 05-01 T2 | 1 | DFR-01 | unit | `python -m pytest tests/test_synthetic.py -q` | created in task | ⬜ pending |
| conftest + Wave-0 stubs + live AST guard | 05-01 T3 | 1 | DFR-01, DFR-02 | unit | `python -m pytest tests/ -q` | created in task | ⬜ pending |
| Quarantined labels + forward tests | 05-02 T1 | 2 | DFR-02 | unit | `python -m pytest tests/test_labels.py -q` | ✅ W0 stub (05-01) | ⬜ pending |
| Causal features + perturbation oracle | 05-02 T2 | 2 | DFR-02 | unit | `python -m pytest tests/test_features.py tests/test_labels.py -q` | ✅ W0 stub (05-01) | ⬜ pending |
| Per-token causal detector + oracle | 05-03 T1 | 2 | DFR-03 | unit | `python -m pytest tests/test_regime.py -q` | ✅ W0 stub (05-01) | ⬜ pending |
| Diagnostics + k-sensitivity reuse | 05-03 T2 | 2 | DFR-03, DFR-06 | unit | `python -m pytest tests/test_diagnostics.py -q` | ✅ W0 stub (05-01) | ⬜ pending |
| Classifier wrappers (LR, XGB) | 05-04 T1 | 2 | DFR-04 | unit | `python -m pytest tests/test_classifiers.py -q` | ✅ W0 stub (05-01) | ⬜ pending |
| RegimeCVEvaluator + embargo invariant | 05-04 T2 | 2 | DFR-04 | unit | `python -m pytest tests/test_cv_evaluator.py -q` | ✅ W0 stub (05-01) | ⬜ pending |
| Per-token forecast comparison | 05-05 T1 | 2 | DFR-05 | unit | `python -m pytest tests/test_forecast.py -q` | ✅ W0 stub (05-01) | ⬜ pending |
| StudentsT GARCH robustness variant | 05-05 T2 | 2 | DFR-05 | unit | `python -m pytest tests/test_forecast.py -q` | ✅ W0 stub (05-01) | ⬜ pending |
| Cramér's V + cross-token matrix | 05-06 T1 | 2 | DFR-06 | unit | `python -m pytest tests/test_cross_token.py -q` | ✅ W0 stub (05-01) | ⬜ pending |
| DGP shared-regime association | 05-06 T2 | 2 | DFR-06, DFR-01 | integration | `python -m pytest tests/test_cross_token.py -q` | ✅ W0 stub (05-01) | ⬜ pending |
| Pipeline assembly + frozen results | 05-07 T1 | 3 | DFR-07, DFR-04 | integration | `python -m pytest tests/test_pipeline.py tests/test_labels.py -q` | ✅ W0 stub (05-01) | ⬜ pending |
| ReportBuilder figures + summary.md | 05-07 T2 | 3 | DFR-07, DFR-06 | integration | `python -m pytest tests/test_pipeline.py -q` | ✅ W0 stub (05-01) | ⬜ pending |
| Runner + integration tests | 05-08 T1 | 4 | DFR-07 | integration | `python -m pytest tests/test_report.py -q` | ✅ W0 stub (05-01) | ⬜ pending |
| Publication-style README | 05-08 T2 | 4 | DFR-07 | smoke | inline python check: sections + >= 6 figures | n/a | ⬜ pending |
| API freeze + lazy imports | 05-09 T1 | 5 | QUAL-01 | unit | `python -m pytest tests/test_api.py -q` | created in task | ⬜ pending |
| Strict suite x2 + codex audit | 05-09 T2 | 5 | QUAL-01, QUAL-04 | gate | `python -m pytest tests/ -q --tb=short` (twice) + `codex exec --sandbox read-only` | n/a | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/conftest.py` — seeded synthetic crypto panel fixtures (24/7 calendar, fat tails, vol clustering, shared 4-state latent regime) — 05-01 T3
- [ ] Test stubs per module with module-level skip + owning plan number — 05-01 T3
- [ ] AST-walk label-quarantine enforcement test LIVE from Wave 0 (passes trivially, guards all wave-2 executors) — 05-01 T3
- [ ] Package skeleton + editable install into quant venv — 05-01 T1

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Figure visual quality (regime timelines, cross-token heatmap) | DFR-06 | Plot aesthetics not assertable | Open reports/figures/*.png after runner |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (all stub test files created in 05-01 T3)
- [x] No watch-mode flags
- [x] Feedback latency < 120s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** planned — pending execution
