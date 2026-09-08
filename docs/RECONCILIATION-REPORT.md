# Reconciliation execution report

Date: 2026-09-08  
Canonical repository: `dmitry-goryunov/Storage`  
Baseline: `main` at `2e1b5ddacd9649b211e7ce45cedd3bbb944a7f1a`  
Working branch: `review/reconcile-model-fixes`

## Outcome

The September review was ported by hand onto a fresh clone of canonical `main`.
The stale Drive working tree was used only as review evidence. No source file,
virtual environment, cache or Git metadata was copied from it.

The reconciled branch fixes findings 1, 6, 9 and 18, and completes finding 4
with a generic post-build reachability check. Finding 10 is resolved in both
apps. Existing fixes on `main` were preserved.

## Verified changes

| Finding | Change | Evidence |
|---|---|---|
| 18 | Removed three-decimal rounding from expected exercise and delta aggregation | Contract value reprices from daily delta at relative error below `1e-9` |
| 1 | Swing per-MWh metrics now divide by expected physical exercise, not hedge delta | Call-swing stochastic metric changed from 20.0093 to 20.0845 EUR/MWh on the current workbook |
| 6 | Tree spacing uses peak term volatility; inputs and every live probability are validated | Variable-volatility test is valid; unstable configurations raise; flat-vol output is bit-identical |
| 4 | Added terminal permitted-state probability check after the forward pass | Ratchet-constrained but calendar-feasible case now raises instead of returning the sentinel |
| 9 | Delta includes the discount factor and is defined as discounted forward-price sensitivity | Discounted repricing identity passes below `1e-9` |
| 10 | Added convexity and local-sensitivity explanation to both apps | Both Streamlit entrypoints pass smoke tests |

## Product and portfolio impact

The plan expected six products, but the current `products.xlsx` contains two.
Both were checked.

| Product | Quantity | Baseline | Reconciled | Change |
|---|---:|---:|---:|---:|
| `call_swing_2010` | Raw value, EUR | 1,205,068.7769 | 1,205,068.7769 | 0 |
| `call_swing_2010` | Profiled metric, EUR/MWh | 20.0268 | 20.0268 | 0 |
| `call_swing_2010` | Stochastic metric, EUR/MWh | 20.0093 | 20.0845 | +0.0752 (+0.376%) |
| `storage_2010` | Raw value, EUR | 549,074.1441 | 549,074.1441 | 0 |
| `storage_2010` | Total, EUR/MWh of capacity | 9.1512 | 9.1512 | 0 |

Storage remains normalised by working capacity, not net exercise. Its optimal
injections and withdrawals net to approximately zero, so value per net exercised
MWh is deliberately undefined. The portfolio MtM regression remains
approximately -56,901 EUR and passes unchanged.

## Verification gates

- Original baseline: 9 tests passed in 20.64 seconds in the first isolated run.
- Reconciled suite: 27 tests passed in 7.12 seconds with warmed Numba caches.
- A second clean virtual environment installed from `requirements-lock.txt` and
  passed all tests: 27 passed in 6.94 seconds.
- Tests include the greedy deterministic solver anchor, portfolio MtM anchor,
  monthly finite differences, exact mandatory volume, multi-clip ratchets,
  terminal reachability, curve and tree repricing, app startup and notebook JSON.
- Continuous integration now installs the lock on Python 3.12 and runs the suite
  on pushes and pull requests.

These are software verification results. They are not evidence of market model
calibration or independent model validation.

## Curve-method decision

The existing additive monthly correction was compared with an experimental
least-squares knot solve. Both repriced monthly contracts to numerical precision.

| Curve | Maximum month-boundary daily step, additive / knot (EUR/MWh) | Call value change | Storage value change |
|---|---:|---:|---:|
| Repository `curve.csv` | 0.476 / 0.150 | +0.017% | +0.749% |
| Moderate 22/32 zigzag | 4.340 / 0.819 | +5.454% | +14.918% |
| One-month 45 spike on 25 | 6.807 / 1.448 | +4.969% | +31.925% |
| 45 to 18 slope | 0.033 / 0.023 | +0.0001% | -0.039% |

Decision: retain the additive correction in this reconciliation. The knot solve
reduces month-boundary steps, but it changes exercise and valuation materially in
stress cases and can increase spline overshoot. Replacing it is a model change,
not a safe defect port. It requires explicit curve-shape acceptance criteria and
calibration evidence before implementation.

## Source inventory decision

Canonical source remains GitHub. Drive-only `code_review.md`, `finding.md`,
`review_checks.py` and `test_model.py` remain historical evidence. Their live
assertions were adapted into `tests/test_reconciliation.py`. Stale Drive source,
debug scripts, `.git`, `.venv` and cache directories are not release inputs.

## Deferred work

The remaining low-level items are recorded in `.planning/ROADMAP.md`. They were
not mixed into this correctness port because tunnels, the backstop grid, tie
tolerance and the inventory-state API require separate behavioural decisions.
