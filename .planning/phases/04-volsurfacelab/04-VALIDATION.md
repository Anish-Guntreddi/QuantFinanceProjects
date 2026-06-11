---
phase: 4
slug: volsurfacelab
status: planned
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-11
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (quant venv, Python 3.11) |
| **Config file** | portfolio_projects/volsurfacelab/pyproject.toml — Wave 0 installs |
| **Quick run command** | `python -m pytest tests/ -q -x` (from portfolio_projects/volsurfacelab/) |
| **Full suite command** | `python -m pytest tests/ -q` (from portfolio_projects/volsurfacelab/) |
| **Estimated runtime** | ~30-60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -q -x`
- **After every plan wave:** Run `python -m pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

All commands run from `portfolio_projects/volsurfacelab/` with the quant venv interpreter.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01 T1 | 04-01 | 1 | VSL-01 | smoke | `python -c "import volsurfacelab"` + `python -m pytest tests/ -q` (stubs skip) | creates them | ⬜ pending |
| 04-01 T2 | 04-01 | 1 | VSL-01 | unit | `python -m pytest tests/test_chain.py -q` | created in T1/T2 | ⬜ pending |
| 04-02 T1 (RED) | 04-02 | 2 | VSL-02 | unit (failing) | `python -m pytest tests/test_iv_solver.py -q` (expect nonzero exit) | ✅ W0 stub | ⬜ pending |
| 04-02 T2 (GREEN) | 04-02 | 2 | VSL-02 | unit | `python -m pytest tests/test_iv_solver.py -q` | ✅ | ⬜ pending |
| 04-03 T1 (RED) | 04-03 | 2 | VSL-03 | unit (failing) | `python -m pytest tests/test_svi.py -q` (expect nonzero exit) | ✅ W0 stub | ⬜ pending |
| 04-03 T2 (GREEN) | 04-03 | 2 | VSL-03 | unit + negative | `python -m pytest tests/test_svi.py -q` | ✅ | ⬜ pending |
| 04-04 T1 (RED) | 04-04 | 2 | VSL-05 | unit (failing) | `python -m pytest tests/test_forecast.py -q` (expect nonzero exit) | ✅ W0 stub | ⬜ pending |
| 04-04 T2 (GREEN) | 04-04 | 2 | VSL-05 | unit | `python -m pytest tests/test_forecast.py -q` | ✅ | ⬜ pending |
| 04-05 T1 (RED) | 04-05 | 2 | VSL-06, VSL-07 | unit (failing) | `python -m pytest tests/test_strategy.py -q` (expect nonzero exit) | ✅ W0 stub | ⬜ pending |
| 04-05 T2 (GREEN) | 04-05 | 2 | VSL-06, VSL-07 | unit | `python -m pytest tests/test_strategy.py -q` + no-qbacktest grep | ✅ | ⬜ pending |
| 04-06 T1 | 04-06 | 3 | VSL-03, VSL-08 (wiring) | integration | `python -m pytest tests/test_pipeline.py -q` | created in task | ⬜ pending |
| 04-06 T2 | 04-06 | 3 | VSL-04 | integration | `python -m pytest tests/test_report.py -q` | created in task | ⬜ pending |
| 04-07 T1 | 04-07 | 4 | VSL-08 | integration | `python -m pytest tests/test_integration.py -q` + CLI smoke run | ✅ W0 stub | ⬜ pending |
| 04-07 T2 | 04-07 | 4 | VSL-08 (QUAL-02) | doc + suite | `grep Limitations README.md` + `python -m pytest tests/ -q` | created in task | ⬜ pending |
| 04-08 T1 | 04-08 | 5 | QUAL-01 | unit | `python -m pytest tests/test_api.py -q` + lazy-import subprocess check | created in task | ⬜ pending |
| 04-08 T2 | 04-08 | 5 | QUAL-01, QUAL-04 | full suite ×2 + codex | `python -m pytest tests/ -q` (twice) + `codex exec --sandbox read-only ...` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Covered by plan 04-01 (wave 1):

- [ ] `tests/conftest.py` — fix_seeds autouse + session-scope `chain` and `underlying_returns` fixtures
- [ ] Test stubs: `tests/test_iv_solver.py` (VSL-02), `tests/test_svi.py` (VSL-03), `tests/test_forecast.py` (VSL-05), `tests/test_strategy.py` (VSL-06/07), `tests/test_integration.py` (VSL-08) — each one skip-marked placeholder
- [ ] `tests/test_chain.py` — implemented fully in plan 04-01 (VSL-01)
- [ ] Package skeleton + editable install into quant venv; pyproject pytest config with `filterwarnings = ["error::FutureWarning"]`

`tests/test_pipeline.py`, `tests/test_report.py`, `tests/test_api.py` are created by the plans that implement them (04-06, 04-08) — acceptable because each is created test-first within its own plan, so no plan ships code without same-plan automated verification.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Surface/smile figure visual quality | VSL-04 | Plot aesthetics not assertable | Open reports/figures/*.png after `python run_pipeline.py --quick`; check smile shape (downward skew, rho=-0.3), 3D surface coverage over k∈[-1.5,1.5], T∈[0.25,1.0] |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 90s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved by planner 2026-06-11
