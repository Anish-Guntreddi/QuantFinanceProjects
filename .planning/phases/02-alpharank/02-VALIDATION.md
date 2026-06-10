---
phase: 2
slug: alpharank
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-10
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (quant/ venv) |
| **Config file** | `portfolio_projects/alpharank/pyproject.toml` — Wave 0 creates with `filterwarnings = ["error::FutureWarning"]` |
| **Quick run command** | `cd portfolio_projects/alpharank && python3 -m pytest tests/ -x -q` |
| **Full suite command** | `cd portfolio_projects/alpharank && python3 -m pytest tests/ -v -W error::FutureWarning` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick command
- **After every plan wave:** Run full suite command
- **Before verification:** Full suite must be green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| (populated by planner) | | | ALR-01..09 | unit/property/integration | pytest node ids | ❌ W0 | ⬜ pending |

Key oracle/property tests required by success criteria:
- `tests/test_features.py::test_no_feature_uses_future_data` — leakage assertion: features at t computed from data ≤ t only (ALR-02)
- `tests/test_labels.py::test_forward_rank_labels_hand_computed` — labels match hand-computed forward-return ranks (ALR-03)
- `tests/test_validation.py::test_purged_cv_no_train_test_overlap` — purge/embargo property: no train index within purge window of any test index (ALR-05)
- `tests/test_validation.py::test_no_standard_kfold_anywhere` — grep-style guard: sklearn KFold not imported anywhere in src (ALR-05)
- `tests/test_analytics.py::test_ic_hand_computed` — IC/rank-IC math vs hand-computed values (ALR-06)
- `tests/test_synthetic.py::test_planted_alpha_recoverable` — synthetic generator's planted IC is recovered within tolerance (ALR-01)
- `tests/test_portfolio_construction.py::test_decile_long_short_weights` — top/bottom decile weights sum to +1/-1 (ALR-07)
- end-to-end: `python3 run_pipeline.py --quick` exits 0, writes report figures (ALR-09)

---

## Wave 0 Requirements

- [ ] `portfolio_projects/alpharank/pyproject.toml` — package skeleton with pytest strict config, qbacktest path dependency
- [ ] `portfolio_projects/alpharank/tests/conftest.py` — seeded fixtures, small synthetic panel fixture
- [ ] Stub test files with exact node ids above

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Report figure legibility | ALR-09 | Rendering aesthetics | Open reports/figures/*.png after run_pipeline.py |
| Codex leakage audit | QUAL-04 | External validator gate | codex exec read-only review focused on label/feature leakage and CV correctness |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
