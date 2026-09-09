# Project status

**As of 2026-09-08.** Canonical `main` is `49d5411`, with every open PR merged: nothing is
in flight. Update this file when that changes — a status document that lags is worse than
none.

| | |
|---|---|
| Repository | [dmitry-goryunov/Storage](https://github.com/dmitry-goryunov/Storage) — the single writable source |
| Working copy | `H:\My Drive\Github\dmitry-goryunov\Storage`, tracking `main` |
| Tests | `python -m pytest -q` → **57 passed** |
| CI | `.github/workflows/test.yml`, pinned from `requirements-lock.txt`, on every push and PR |
| Environment | System Python 3.12. There is deliberately no venv in the Drive folder — build one outside it |

---

## What this is

A valuation library for gas storage and swing contracts on TTF: a trinomial price tree with
Ornstein–Uhlenbeck mean reversion, a dynamic program over (time × price × inventory), and
Numba-compiled kernels. Around it sit two Streamlit apps, four notebooks and a product
workbook.

## Where it stands in one line

**Verified, not validated.** The code now computes what it claims, and the claims are
pinned by tests. Nothing yet establishes that the claims are the right ones for a traded
price — no calibration, no parameter provenance. Treat every number as exploratory.

## How it got here

A model review in September 2026 found 10 defects plus 10 minor ones. It was developed
against a stale copy of the repository that had diverged from `main` (18 commits versus 6,
with no shared file history for the library), which an external process review caught before
anything was merged. Every finding was then re-checked against `main`'s actual code and only
the live ones ported.

| PR | | |
|---|---|---|
| [#1](https://github.com/dmitry-goryunov/Storage/pull/1) | Port the September review onto canonical `main` | merged |
| [#2](https://github.com/dmitry-goryunov/Storage/pull/2) | `delta` becomes an undiscounted hedge volume; conventions documented; dead code removed | merged |
| [#3](https://github.com/dmitry-goryunov/Storage/pull/3) | Separate inventory grid size from initial state | merged |
| [#4](https://github.com/dmitry-goryunov/Storage/pull/4) | `Products.ipynb` | merged |
| [#5](https://github.com/dmitry-goryunov/Storage/pull/5) | Intrinsic split benchmarked net of the strike | merged |

The full review, its evidence and the reconciliation record are archived under
[`docs/reconciliation-history/2026-09-08/`](reconciliation-history/2026-09-08/) — 27 files,
including the original code review, the peer-reviewed put-delta investigation, the process
review that caught the stale baseline, and the migration log.

## What was wrong, and what it cost

| | Effect |
|---|---|
| Per-MWh metrics divided value by `sum(delta)` instead of MWh exercised | 8.4 % error, and it moved the **wrong way** as optionality was added |
| `flat()`/`profiled()` read price state 0 | Silently returned `0.0` on any `n_p > 0` tree |
| `dx` sized from `vol_curve[0]` | A vol term structure produced negative probabilities and silent NaNs |
| `np.round(..., 3)` in the metrics | Masked the repricing identity at 1e-7 |
| Coverage guard ran after `smoothen_curve` | Curve gaps surfaced as SciPy's `y must contain only finite values` |
| `wdr_days` dropped by `run_valuation` | 30, 45, 90 and 365 all priced a storage deal identically |
| Intrinsic benchmarked against a raw forward average | Shifted by the strike: −27.7 EUR/MWh on a K=28 call, +20.2 on a K=20 put |
| `value_storage` lost its starting-inventory assignment in the P3.1 refactor | Reported a value for an empty store beside profiles for a full one — 6,041 MWh in against 66,041 out. Introduced 2026-09-08, found and fixed the next day |
| A curve could start before the quote it was built from | Look-ahead, and the gap was back-filled with the quote's day-ahead price — two months of 52.00 stamped across Jan–Feb 2026 for a March quote. The Streamlit app shipped with exactly this default (valDate 2026-01-01 against FDDate 2026-01-05) |
| `Products.ipynb` read `AS_OF` only on the `quotes` branch | With `CURVE_SOURCE = "csv"` the as-of date did nothing: 6 March 2026 still priced off curve.csv's stored 28.00. Same trap class as `wdr_days` |
| Intrinsic on a flat curve looked like a bug | It is not: the optimiser prices `DF·F`, and at 10 % a flat 40 slopes 36.836 → 33.342 across 2027. `intrinsic_components` now splits it into day selection and financing, exactly |

Notebook outputs were also materially stale — `Swing_new.ipynb` showed a swing worth 2.11
EUR/MWh where the model now gives 3.22.

## What holds

These are asserted in `tests/`, not claimed here:

- **The repricing identity.** `sum_i DF_i·delta_i·fwd_i == V0` for a zero-cost deal, and
  with the cost leg subtracted (`inj_cost·injected + wdr_cost·withdrawn`) it closes to
  **1e-15 for every product type**, struck and asymmetric included.
- **Volumes.** A mandatory quota is met exactly; `days == window` prices at the flat forward.
- **The tree.** Probabilities in [0,1], forward fitting exact to 1e-15, terminal log-std
  matching the OU stationary value, unstable configurations rejected rather than NaN.
- **Delta as a derivative.** Every monthly bucket matches a finite-difference bump of the DP
  to ±0.4 % at 5 bp.
- **The portfolio anchor.** Total MtM −56,901 EUR, end to end from the workbook.

## What does not hold

**Calibration.** `sVol` defaults to 0.9 and `sMR` to 1.0, neither with a recorded estimation
window, method or as-of date. There is no convention tying a valuation to a quote date in
its output, and no parameter-sensitivity standard. This is the largest remaining risk and
the one item that cannot be closed without market data and a judgement about what
"calibrated" means here. See the calibration section of
[MODEL-CONVENTIONS.md](MODEL-CONVENTIONS.md).

**Bucketed hedges are local.** The monthly deltas are correct derivatives but strongly
convex — at a 1 % move in one month, July is out by +8.6 % and November by +43.9 %, while
quota-forced December holds at −1.3 %. Re-hedge; do not carry them across a large move.

**Tree width is a real parameter.** On the most option-like deal, `n_p = 5` misprices by
0.74 EUR/MWh and `n_p = 10` by 0.19; it is converged by 20. The app default is 30.

## Open work

From [`.planning/ROADMAP.md`](../.planning/ROADMAP.md). Everything here needs a decision
first — the work that did not is finished.

| | Item | Why it is open |
|---|---|---|
| P1.1 | Tunnel semantics — hard vs soft, before/after action | The tunnels cannot bind as built, and the `1000·v_step` penalty scale is arbitrary |
| P1.2 | Curve-shape acceptance criteria | `main` already reprices its input contracts; the knot solve is now a refinement, worth doing only against stated criteria |
| P1.3 | Production discount-curve source and settlement timing | `discount_rate` now prices time value and the optimiser prefers early withdrawal; where a *real* curve comes from is still open, and settlement is assumed on the exercise day |
| P1.4 | Withdrawal capacity, remaining half | Rates are whole clips per day, so anything slower than one clip/day is inexpressible. Also `inj_days` still means two things |
| P2.1 | Shorten the terminal backstop | 24 % of the grid on a three-month deal, but it moves indices near the terminal condition |
| P2.2 | Scale-aware exercise tie threshold | `1e-6` is absolute, on values that scale with deal size |
| P3.4 | Justify or change the 0.9 default vol | Part of calibration |

Also open and needing a data source rather than a decision: **`ttf q.xlsx` row 4172** (quote
date 2010-03-12) differs in 52 of 57 columns between the two former copies of the repo. It
is never the row selected for pricing, but it is wrong for any backtest.

## Running it

```bash
python -m pytest -q                  # 34 tests, ~14 s
streamlit run streamlit_app.py       # single-deal valuation
streamlit run portfolio_app.py       # portfolio mark-to-market
jupyter lab                          # notebooks below
```

| Notebook | |
|---|---|
| `Products.ipynb` | One deal family at four sizes — put swing 2027 at 10/30/90/180 days, with hedge and convergence checks |
| `Swing_new.ipynb` | Swing valuation and the intrinsic/extrinsic decomposition |
| `forward.ipynb` | Valuation off a chosen historical curve date |
| `pricing.ipynb` | Single product through `run_valuation` |
| `portfolio.ipynb` | Portfolio MtM |

All four execute cleanly against the current library and their committed outputs are current.

## Where things live

| | |
|---|---|
| [`docs/MODEL-CONVENTIONS.md`](MODEL-CONVENTIONS.md) | What the inputs and outputs mean — signs, units, discounting, the invariant, and what is not calibrated. **Read this before using a number.** |
| [`docs/DELTA-CONVENTION.md`](DELTA-CONVENTION.md) | Why `delta` is an undiscounted hedge volume, with the measured cost of the alternative |
| [`docs/RECONCILIATION-REPORT.md`](RECONCILIATION-REPORT.md) | What was ported onto `main` and what it moved |
| [`docs/reconciliation-history/2026-09-08/`](reconciliation-history/2026-09-08/) | The September review and its evidence, read-only |
| [`.planning/ROADMAP.md`](../.planning/ROADMAP.md) | Open work, prioritised |
| `storage_model.py`, `storage_kernels.py` | The library; kernels are split out so edits here do not trigger a Numba recompile |
| `tests/` | `test_regression.py` (product and portfolio anchors) and `test_reconciliation.py` (the September findings) |

Branch `review/model-fixes` holds the September review's own history and exists nowhere
else; keep it.
