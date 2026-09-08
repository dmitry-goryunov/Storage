# Reconciliation plan — Drive review branch vs GitHub `main`

Date: 2026-09-08
Trigger: [PROCESS-AND-PROGRESS-REVIEW.md](PROCESS-AND-PROGRESS-REVIEW.md), whose central
claim — that the September review was built on a stale baseline — is confirmed below.
Supersedes the sequencing in [.planning/ROADMAP.md](../.planning/ROADMAP.md), which was
derived from that stale baseline.

---

## 1. The divergence, measured

```
$ git rev-list --left-right --count origin/main...review/model-fixes
18      6

$ git merge-base origin/main review/model-fixes
4a1ca46  "new"
```

18 commits on `origin/main` that the review branch does not have; 6 the other way. The
earlier statement that the branch was "5 commits ahead of `main` and fast-forwardable"
was measured against the **local** `main` ref, which is itself 18 commits behind
`origin/main`. That statement is withdrawn — **do not fast-forward or merge `main` from
this branch.**

The deeper problem: the merge base `4a1ca46` contains only `Storage_and_Swing.ipynb` and
`curve.csv`. `storage_model.py` was never committed at the fork point. It reached
`origin/main` through 18 subsequent commits and reached the Drive copy as an untracked
working file. **The two lines therefore share no file history for the library at all** —
this is not a merge, it is a port.

Remotes: `origin` = github.com/dmitrygoryunov2000/Storage, `publish` =
github.com/dmitry-goryunov/Storage. Both were at `2e1b5dd` after `git fetch --all`. The
branch `review/model-fixes` exists only in the Drive clone and has never been pushed.

## 2. What `origin/main` gained in those 18 commits

Verified by reading the blobs, not by inference:

- `storage_kernels.py` — `_tree_core`, `run_model`, `probabilities`, `get_exercise`,
  `get_delta` split out of `storage_model.py`
- **`strat` semantics changed**: it now holds the *signed clip count moved per state*,
  not `±1`. `compute_all_metrics` is consequently three lines
  (`action = strat[:n_t] * v_step`), and `probabilities` reads `dk = int(round(strat[...]))`
- Asymmetric injection/withdrawal capacity: `apply_ratchets(fullness, inj_mult, wdr_mult)`,
  `ratchet_arrays`, `load_ratchets`, `ratchets.xlsx`; new `clips_per_day` parameter
- Portfolio valuation: `portfolio.ipynb`, `portfolio_app.py`, `products.xlsx`,
  `list_products`, `load_product_params`, `params_for_run_valuation`
- `tests/test_regression.py` — 251 lines, including a portfolio MtM anchor
  (`total_mtm ≈ -56,901`) and a greedy-solver cross-check at `n_p = 0`
- Docstrings throughout, a `.gitignore`, `quotes_2.csv`, `quotes.xlsx`

## 3. Finding-by-finding status against `origin/main`

Every row was checked against the blob at `origin/main`, with the location where it holds.

### Already fixed there — do not port

| # | Finding | How `main` fixed it |
|---|---|---|
| 2 | `flat()`/`profiled()` read price state 0 | **Better than my fix.** `flat()` is now the window-mean forward — a genuinely different quantity, not an alias; `profiled()` reads `v[0, n_p, ·]`. `tests/test_regression.py` even asserts `v[0,0,·] == 0.0` as "the old (buggy) boundary read" |
| 3 | Three ratchet conventions | **Fixed by the refactor.** With `strat` holding the signed clip count, DP, forward pass and metrics all read one number. My whole-clip validation is no longer the right shape |
| 4 | `−1e9` sentinel returned as a price | `assert_cycle_feasible` — a static capacity pre-check (clips needed vs active days × clips/day), cheaper than my post-build probability-mass check |
| 5 | Curve does not reprice input contracts | Implemented — but with the **flat additive per-month shift**, the approach [SPEC R1](SPEC-remaining-work.md) rejected for leaving month-boundary steps. See §6 |
| 7 | `n_p_full` default 10 | Now 30, with a convergence help string |
| 8 | `wdr_days` dead input | Removed |
| 15 | Curve coverage error message | `price_curve.isna()` guard with a count |

### Still live there — port these

| # | Finding | Location on `main` | Severity |
|---|---|---|---|
| **1** | `stochastic_metric = full_eur / np.sum(s.delta)` — value ÷ price-weighted volume instead of MWh exercised | `storage_model.py:502`, `:561`, and **`profiled()` at `:245`** | **High — and worse on `main` than it was here.** See §4 |
| **6** | `dx = vol_curve[0] * sqrt(3*dt)` — a vol term structure yields negative probabilities and silent NaNs | `storage_model.py:672` | High (silent) |
| 9 | `delta` ignores `d_curve`; the repricing identity breaks under discounting | `compute_all_metrics`, `:697-707` | Medium, latent |
| 10 | Monthly delta table has no convexity caption | `streamlit_app.py` (no mention of convexity/re-hedging) | Low |
| 11 | `get_exercise`, `get_delta` (`storage_kernels.py:215,225`), `valuation` (`storage_model.py:689`), `check_curve` (`:93`) unused | as listed | Low |
| 12 | `probabilities` takes `mintunnel`/`max_tunnel` and never reads them | `storage_kernels.py:182-184` | Low |
| 13 | Tunnels cannot bind; `1000.0 * v_step` penalty scale is arbitrary | `storage_kernels.py:117` | Low, but see §7 |
| 14 | DP grid runs a month past the exercise window (24 % waste on a 3-month deal) | `storage_model.py:128,130` | Low, perf |
| 17 | `strat` tie tolerance is absolute (`1e-6`), not relative to deal scale | `storage_kernels.py:171` | Low |
| **18** | `np.round(..., 3)` in `compute_all_metrics` masks the repricing identity at 1e-7 | `storage_model.py:702,705` | Low alone, **blocks the master test** |
| 19 | `n_op_start` means two things | Now *documented* (`:196`) but not fixed | Low |

Not applicable: 16 (partly addressed — `sVol` is now documented as annualised, though the
default is still 0.9), 20 (no `debug_delta*.py` on `main`).

## 4. The finding that matters most

`main`'s fix to finding 2 made finding 1 *more* consequential, not less.

`profiled()` now deliberately reads the central node so it works at `n_p > 0` — and still
divides by `np.sum(self.delta)`:

```python
def profiled(self):
    ACQ = np.sum(self.delta)
    return self.v[0, self.n_p, self.n_op_start] / ACQ      # storage_model.py:245
```

At `n_p = 0` that was harmless: `delta ≡ exp_ex` because S = F at every node. At `n_p = 30`
the denominator is the price-weighted forward-equivalent volume, which differs from MWh
exercised by 8.4 % in the measured base case — and moves the wrong way as optionality is
added. `tests/test_regression.py::test_profiled_central_node_nonzero_at_np30` exercises
exactly that path, and `test_flat_is_window_mean_and_below_profiled` asserts
`profiled() >= flat()` on a distorted quantity.

This is the single highest-value item to port, and it should carry its own before/after
table on `main`'s six products plus the portfolio MtM anchor.

## 5. Port mechanics

`git cherry-pick 18d1fb0` will not work and should not be attempted: no shared file
history, and the fix touches functions that `main` has since moved to
`storage_kernels.py` or rewritten (`strat` semantics, `compute_all_metrics`).

Instead:

1. **Fresh clone of `origin/main`, outside Google Drive.** A `.git` directory inside a
   synced folder risks index corruption; `.venv` and `__pycache__` should never sync
   either. (The Drive copy currently contains all three.)
2. Re-apply findings 1, 6, 18 by hand against `main`'s current code — three small,
   independent changes. Each is a few lines; the analysis behind them is already written.
3. Port the *reasoning*, not the diff: `code_review.md`, the peer review in `finding.md`
   and `review_checks.py` describe what to check and why, and none of it exists on `main`.
4. Leave findings 9–14, 17, 19 for a second pass, sequenced by severity.

## 6. What happens to Phase 2 (exact curve repricing)

It shrinks from "implement the correction" to "decide whether to replace a working one".

`main` already reprices the input contracts via the flat additive per-month shift. The
knot-solve argument in SPEC R1 was that the additive shift leaves ~0.1 EUR discontinuities
at month boundaries, and the model compares adjacent days to decide exercise. That
argument still stands, but it is now a **refinement of a working correction**, not a fix
for a missing one — a much smaller and more debatable piece of work.

Required before deciding: measure the actual month-boundary steps `main`'s correction
produces on `curve.csv`, and whether they change any exercise decision. If they do not,
Phase 2 should be dropped rather than done.

## 7. Test-suite reconciliation

`test_model.py`'s 13 tests do not drop into `main` unchanged. Specifically:

- `test_price_methods_work_on_a_full_tree` asserts `flat() == v/volume`. On `main`,
  `flat()` means the window-mean forward. **Rewrite against `profiled()` only.**
- `test_delta_reprices_the_contract` and `test_mandatory_quota_is_met_exactly` need
  finding 18 ported first, or they must relax to ~1e-5.
- The tree tests import `build_tree` from `storage_model` (still correct) but the
  probability internals now live in `storage_kernels`.
- `test_fractional_ratchets_are_rejected` is obsolete — `main`'s `strat`-as-clip-count
  removes the three-way inconsistency. Replace with a test that `run_model`,
  `probabilities` and `compute_all_metrics` agree on clips moved under a non-unit
  `clips_per_day` and an asymmetric ratchet table.
- Everything else ports with only import changes.

Merge target is `tests/`, preserving `main`'s portfolio MtM anchor and greedy cross-check.
Adopting the external review's additions, in priority order: finite-difference validation
of monthly deltas (already written — `review_checks.py` section 4, needs promoting to an
assertion), zero-volatility and deterministic analytical cases, single-thread vs parallel
comparison, and sign/unit tests for put, call and storage deltas.

## 8. Sequence

| Step | Work | Done when |
|---|---|---|
| 1 | Fresh clone of `origin/main` outside Drive; run `tests/test_regression.py` | Green baseline recorded, with timings |
| 2 | Port finding 18 (rounding) | Repricing identity measurable at 1e-9 on `main` |
| 3 | Port finding 1 (metric denominator), incl. `profiled()` | Before/after on six products + portfolio MtM; `test_regression.py` still green |
| 4 | Port finding 6 (`dx`, tree validation) | Vol-term-structure test passes; flat-vol results bit-identical |
| 5 | Merge the suites under `tests/` per §7 | One suite, no skips reported as passes |
| 6 | Re-measure `main`'s curve correction (§6) | Decision recorded: refine or drop |
| 7 | Regenerate the roadmap from the reconciled baseline | `.planning/` matches reality |
| 8 | Second pass: findings 9–14, 17, 19 | — |

Steps 2–4 are small and independent; each should be its own commit on a branch off
`origin/main`, pushed, so this never happens again.

## 9. Adopted from the process review

Accepted without qualification: progress reporting must show both denominators (findings
fixed vs new-roadmap requirements — currently 0 % in `STATE.md` against 7 of 20 findings
fixed); the valuation-impact classification is too categorical (a changed default that
moves reported extrinsic by 15 % matters to users even if the DP is untouched); "the DP
and tree are correct" should read "no defect was found in the tested domain"; the
repricing identity is necessary but not sufficient, since a shared error across DP,
forward pass and metrics would preserve it; software verification is not model validation
and no calibration evidence is in scope anywhere; the planning documents have begun to
duplicate each other; dependencies are unpinned with no clean-environment test.

Two clarifications on the record: the independent finite-difference validation the review
asks for **already exists** (`review_checks.py` section 4 — every 2027 monthly bucket
against a ±5 bp DP bump, matching to ±0.4 %); it needs promoting into the suite, not
writing. And the review lists only the app-level overlaps with `main`; the actual overlap
is larger — findings 2, 3, 4, 5 and 15 are also already fixed there — while finding 1, the
highest-severity item, is still live and unmentioned.

## 10. Risks

- **Porting by hand loses the red-green evidence.** Mitigate by writing each test against
  `main` *before* its fix, exactly as in the September pass.
- **The portfolio MtM anchor (`≈ -56,901`) may move** when finding 1 is fixed, if any
  portfolio metric flows through `profiled()` or `stochastic_metric`. Check before
  changing the anchor — a moved anchor is a result, not a nuisance.
- **The Drive copy will keep drifting** while it stays a second working tree. Once the
  port is merged, delete it or make it a read-only reference.
