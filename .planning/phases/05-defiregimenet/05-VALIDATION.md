---
phase: 5
slug: defiregimenet
status: draft
nyquist_compliant: false
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
| **Config file** | portfolio_projects/defiregimenet/pyproject.toml — Wave 0 installs |
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
| (filled by planner per plan/task breakdown) | | | DFR-01..DFR-07 | unit/integration | `python -m pytest tests/ -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/conftest.py` — seeded synthetic crypto panel fixture (24/7 calendar, fat tails, vol clustering, 4 latent regimes)
- [ ] Test stubs per module incl. the AST-walk label-quarantine enforcement test
- [ ] Package skeleton + editable install into quant venv

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Figure visual quality (regime timelines, cross-token heatmap) | DFR-06 | Plot aesthetics not assertable | Open reports/figures/*.png after runner |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
