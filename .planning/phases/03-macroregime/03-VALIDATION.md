---
phase: 3
slug: macroregime
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-10
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (quant/ venv) |
| **Config file** | `portfolio_projects/macroregime/pyproject.toml` — Wave 0 creates with `filterwarnings = ["error::FutureWarning"]` |
| **Quick run command** | `cd portfolio_projects/macroregime && python3 -m pytest tests/ -x -q` |
| **Full suite command** | `cd portfolio_projects/macroregime && python3 -m pytest tests/ -v -W error::FutureWarning` |
| **Estimated runtime** | ~90 seconds (rolling HMM re-fits are the heavy item — keep test windows small) |

---

## Sampling Rate

- **After every task commit:** quick command
- **After every plan wave:** full suite
- **Before verification:** full suite green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| (populated by planner) | | | MCR-01..08 | unit/oracle/integration | pytest node ids | ❌ W0 | ⬜ pending |

Key oracle/property tests required by success criteria:
- `tests/test_regimes.py::test_causality_future_data_does_not_change_past_regimes` — causal sequence at t identical when computed on data[:t] vs data[:T>t] (MCR-04, THE phase-defining oracle)
- `tests/test_macro_data.py::test_point_in_time_mask` — series value for month m invisible until m + release_lag; strategy view as-of date d contains only published rows (MCR-02)
- `tests/test_regimes.py::test_label_alignment_stable_across_refits` — state ordering by economic observable stable across rolling re-fits (MCR-05)
- `tests/test_benchmarks.py::test_identical_costs_across_strategies` — regime strategy and all 3 benchmarks run through the same engine config/costs (MCR-07)
- `tests/test_synthetic_macro.py::test_hmm_recovers_planted_regimes` — HMM on full sample recovers planted Markov regimes above chance (MCR-01)
- end-to-end: `python3 run_macroregime.py --quick` exits 0, writes report + figures (MCR-08)

---

## Wave 0 Requirements

- [ ] `portfolio_projects/macroregime/pyproject.toml` — skeleton, strict pytest config
- [ ] `tests/conftest.py` — seeded fixtures, small synthetic macro panel fixture
- [ ] Stub test files with the node ids above

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Report figure legibility | MCR-08 | Rendering | Open reports/figures/*.png |
| Codex audit | QUAL-04 | External gate | codex read-only review: point-in-time masking + causal HMM + allocation cost parity |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity maintained
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
