# Project status

**As of 2026-09-10 (evening).** An independent review of the storage day landed, overturned
two results recorded as established — the ratchet cap and the case for a second factor — and
was itself replied to with an implementation plan. **The inventory-bound repair is done, and the ratchet
discretisation is now measured, gated and reported** — though the shipped notebook still
chooses a grid 0.53 % short of the gate, deliberately and in print. Read
[REVIEW-RESPONSE-2026-09-10.md](REVIEW-RESPONSE-2026-09-10.md) and
[REVIEW-REPLY-AND-ACTION-PLAN-2026-09-10.md](REVIEW-REPLY-AND-ACTION-PLAN-2026-09-10.md)
together. Update this file when that changes — a status document that lags is worse than none.

| | |
|---|---|
| Repository | [dmitry-goryunov/Storage](https://github.com/dmitry-goryunov/Storage) — the single writable source. `origin` points here directly as of 2026-09-10; it had been on the pre-rename `dmitrygoryunov2000` URL and reaching this one through a GitHub redirect |
| Working copy | `H:\My Drive\Github\dmitry-goryunov\Storage`, tracking `main`. **Stays on Drive by decision, 2026-09-09** — see the note below |
| Tests | `python -m pytest -q` → **115 passed** |
| CI | `.github/workflows/test.yml`, pinned from `requirements-lock.txt`, on every push and PR |
| Environment | System Python 3.12. There is deliberately no venv in the Drive folder — build one outside it |

---

## The working copy lives on Google Drive

Decided 2026-09-09, knowing the cost. `H:` is Google Drive File Stream, which presents as
**FAT32** behind a sync daemon rather than as a disk — no real file locking, no symlinks,
case-insensitive. Measured against a local clone of the same repository:

| | Drive | local | |
|---|---:|---:|---:|
| `git status` | 0.181 s | 0.036 s | 5× |
| write 200 small files | 4.071 s | 0.132 s | 31× |
| read them back | 1.218 s | 0.034 s | 36× |

**The failure you will actually hit** is a stale lock file in `.git`: a zero-byte file left
when the sync daemon touches `.git` mid-operation. It blocks the operation with *"Another
git process seems to be running"* when none is. Twice as `packed-refs.lock` on 2026-09-09
(blocking `git checkout -b` and `git push`), once as **`AUTO_MERGE.lock`** on 2026-09-10
during a fast-forward merge. In every case so far **the operation itself succeeded** and
only the lock was left behind, so check the result before assuming it failed. Then, having
confirmed no git process is really running:

```bash
rm -f .git/*.lock
git fsck --no-progress --no-dangling      # clean every time so far
```

The operation that tripped it still succeeds — only ref-packing is blocked. `git config
gc.auto 0` would stop git attempting it at all, at the cost of never auto-packing loose
objects; not set today.

Two incidents trace to this filesystem beyond the lock: the September stale-baseline
divergence (18 commits against 6, no shared history), and a mid-session edit to
`ttf q.xlsx` that silently changed every valuation reading it. GitHub, not Drive, is the
backup — everything committed is on `origin`, and Drive only covers work that is not.

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

The library now prices the time value of money. `discount_rate` discounts each day's
cash flow, the benchmark is present-valued on the same footing, and the optimiser
reschedules: a seller pulls exercise earlier to be paid sooner, a buyer pushes it later to
hold the cash. A flat forward curve can therefore show settlement-timing value because the
optimiser prices `DF·F`. That is a result under the model's exercise-day settlement and rate
conventions, not evidence that a flat observed gas curve is economically inconsistent. See
the discounting sections of [MODEL-CONVENTIONS.md](MODEL-CONVENTIONS.md).

Rate configuration is now validated through one path used by the library, product workbook,
`Products.ipynb` and the single-deal app. An explicit `discount_rate=0.0` counts as a
selected market/valuation rate and cannot silently coexist with treasury scenario fields.
The app and notebook expose the two supported modes; the latter remains one selected
treasury scenario for the whole deal, not asymmetric cash-balance funding.

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
| [#6](https://github.com/dmitry-goryunov/Storage/pull/6) | Time value of money: `discount_rate`, the day-selection/financing split, `delta_pv`, borrow vs invest rates, `STRIKE`, and a total P&L bridge | merged |
| `review/correct-rate-economics` | External economic review of #6: explicit funding direction, three-way intrinsic attribution, one rate-validation path, and the P0 verification gates | merged, no PR |

A day's findings and corrections from the time-value work are in
[FINDINGS-2026-09-09.md](FINDINGS-2026-09-09.md), including four claims this documentation
had wrong.

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
| `price_per_mwh` took an absolute value | A strike above the curve flips a put swing's value positive, so a deal that pays you was reported as a 12.905 EUR/MWh cost, and `vs flat` was out by twice the moneyness. Caught by the section 4 cross-check the day it was added |
| Intrinsic on a flat curve looked like a bug | It is not under exercise-day settlement: the optimiser prices `DF·F`, and at 10 % a flat 40 slopes 36.836 → 33.342 across 2027. On a shaped curve the price/timing split also has an interaction |

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
- **An independent closed form.** A call swing collapsed to one exercise day with the quota waived is a European call, and reproduces Black-76 to under 1 % across five strikes and two maturities — at the Clewlow & Strickland (1999a) variance for the one-factor Schwartz model, `sVol²(1−e^(−2·sMR·T))/(2·sMR)`, not `sVol²T`. The tree carries that variance across the term structure to within 0.25 % from three months to five years. This is the only check here against something outside the model; everything above it is internal consistency.
- **An independent deterministic oracle.** Exhaustive enumeration of every feasible
  four-day schedule reproduces the DP for put swing, call swing and storage.
- **Rate plumbing.** Workbook and app treasury inputs reach the model without an inferred
  direction; `Products.ipynb` executes every section in a clean process under the invest
  scenario and rejects committed stale outputs.

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
first — the work that did not is finished. The order below predates the review;
[REVIEW-RESPONSE-2026-09-10.md](REVIEW-RESPONSE-2026-09-10.md) proposes a different one and is
the document to argue with.

| | Item | Why it is open |
|---|---|---|
| ~~P1.1~~ | ~~Tunnel semantics — hard vs soft, before/after action~~ | **Done 2026-09-10 (evening).** Hard, on the opening balance. Three defects repaired: nearest rounding relaxed both sides, the balance was rebuilt without its opening inventory, and the check compared an expectation. The terminal condition was hardened with it — a `-1e9` penalty is purchasable too |
| P1.2 | Curve-shape acceptance criteria | `main` already reprices its input contracts; the knot solve is now a refinement, worth doing only against stated criteria |
| P1.3 | Production discount-curve source, purpose and settlement timing | `discount_rate` now prices time value; where a *real* market curve comes from, whether a rate is market/funding/hurdle, and the contractual settlement dates remain open |
| ~~P1.4~~ | ~~Withdrawal capacity, remaining half~~ | **Done 2026-09-10 (evening).** The loss is a **sawtooth** — zero where `rate × multiplier` lands on an integer, 33 % just below one, even on the *mild* shipped profile. `describe_ratchet_rates()` reports it per inventory level; `value_storage` now refuses a loss over **10 %** by default; the notebook runs at 1,920 clips and prints its 0.19 % residual; `forward.ipynb` reports the profile and says when `ratchets.xlsx` is inert. `inj_days` no longer means two things at once. Left open by choice: paying ~8 s a valuation for the 0.5 % value-convergence grid |
| P2.1 | Shorten the terminal backstop | 24 % of the grid on a three-month deal, but it moves indices near the terminal condition |
| ~~P2.2~~ | ~~Scale-aware exercise tie threshold~~ | **Considered and declined 2026-09-10.** It cannot misprice a deal — nominal cash is identical to nine decimals whether the store turns once or twice — but it can double the reported hedge, and only at a rate near 1e-9 with a hundredfold size contrast. Left as is |
| P3.4 | Justify or change the 0.9 default vol | Part of calibration |
| P4.1 | A second factor — **it is a calibration question, not an architecture one** | The correlations stand (c1/c6 **0.801**, c6/c12 **0.771**, c1/c24 **0.633**) but the shortfall is **3.1×**, not nine. At a *fixed* short factor a common long factor is worth exactly nothing to a homogeneous contract — true, and misleading on its own. A **calibrated** second factor takes volatility out of `sigma_chi`, and then a store **loses** up to **8 %**, because a store monetises short-horizon variance and that is what moved. For a struck swing even the SIGN depends on the calibration anchor (**+0.58 %** holding spot variance fixed, **−10.23 %** holding terminal variance fixed, same contract and same `sigma_xi`). No scalar anchor settles it. [`two_factor_probe.py`](../two_factor_probe.py) is the experiment; the panel fit comes first |
| ~~P4.2~~ | ~~Volumetric fuel loss~~ | **Done 2026-09-10.** `fuel_loss` charges the injection price leg; 1.5 % retention costs 8.9 % of value. Forced decision D-O3 — `delta` becomes the traded volume, which is what keeps the repricing identity closing |
| ~~P4.4~~ | ~~Report hedge stability~~ | **Done 2026-09-10.** A 1 % bump per month, measured against the book's largest position. Oct–Dec move 0.1 %; Apr–Aug move 44–51 %, because months priced alike leave the optimiser flipping between them |
| **P4.3** | **Calibrate at the sensitivity that matters** | A spot-fitted vol is the wrong target for a spread product; the answer swings tenfold across plausible `sMR` |

Also open and needing a data source rather than a decision: **`ttf q.xlsx` row 4172** (quote
date 2010-03-12) differs in 52 of 57 columns between the two former copies of the repo. It
is never the row selected for pricing, but it is wrong for any backtest.

## Running it

```bash
python -m pytest -q                  # 115 tests, ~90 s
streamlit run streamlit_app.py       # single-deal valuation
streamlit run portfolio_app.py       # portfolio mark-to-market
jupyter lab                          # notebooks below
```

| Notebook | |
|---|---|
| `Products.ipynb` | One put/call swing family at four sizes, 10/30/90/180 days, with rate mode, hedge and convergence checks |
| `Storage_30_60.ipynb` | The simple store: 30 days to fill, 60 to empty, valued and hedged at 0 % and at 10 % funding |
| `Storage_30_65.ipynb` | Asymmetric-rate storage — 30 days to fill, 65 to empty. Sizes the inventory grid from the rates and refuses to price a deal it cannot express |
| `SwingVsOption.ipynb` | A swing priced beside a vanilla call — the vol convention, the Black-76 anchor, and the ladder from a call to a mandatory swing |
| `Swing_new.ipynb` | Swing valuation and the intrinsic/extrinsic decomposition |
| `forward.ipynb` | Valuation off a chosen historical curve date |
| `pricing.ipynb` | Single product through `run_valuation` |
| `portfolio.ipynb` | Portfolio MtM |

All notebook code cells compile. The test suite executes every `Products.ipynb` section in a
fresh process under a reduced-width treasury-scenario smoke configuration and requires all
stored outputs to be empty. Run it in Jupyter to regenerate the full displayed tables and
charts. The other committed outputs are unchanged by this review.

## Where things live

| | |
|---|---|
| [`docs/FINDINGS-2026-09-10-evening.md`](FINDINGS-2026-09-10-evening.md) | **Start here for the evening.** What was done, what was discovered doing it, and what is left — including three claims of mine that were wrong |
| [`docs/REVIEW-REPLY-AND-ACTION-PLAN-2026-09-10.md`](REVIEW-REPLY-AND-ACTION-PLAN-2026-09-10.md) | The reply to the response: three qualifications, and the delivery order the repairs followed |
| [`docs/REVIEW-RESPONSE-2026-09-10.md`](REVIEW-RESPONSE-2026-09-10.md) | What survived the independent review, what did not, and the outstanding work ranked |
| [`docs/INDEPENDENT-REVIEW-2026-09-10.md`](INDEPENDENT-REVIEW-2026-09-10.md) | The review itself, with its evidence archive beside it. Every file it inspected hashes identical to this working copy |
| [`docs/FINDINGS-2026-09-10.md`](FINDINGS-2026-09-10.md) | The storage day — dated inventory bounds, ratchets, fuel loss, the delta split and hedge stability; five defects, three claims corrected, four decisions. **Three claims in it are withdrawn** — see the response |
| [`docs/DESIGN-P4.1-two-factor.md`](DESIGN-P4.1-two-factor.md) | Plan for the second factor: the measured case, the lattice-vs-LSMC fork, and step-by-step |
| [`docs/FINDINGS-2026-09-09.md`](FINDINGS-2026-09-09.md) | What the time-value work found and corrected — nine defects, four wrong claims, the behaviour now pinned by tests, and three process traps |
| [`docs/CODEX-HANDOVER-2026-09-09.md`](CODEX-HANDOVER-2026-09-09.md) | Handover for continuing the corrected project in the Codex extension for Visual Studio Code |
| [`docs/MODEL-CONVENTIONS.md`](MODEL-CONVENTIONS.md) | What the inputs and outputs mean — signs, units, discounting, the invariant, and what is not calibrated. **Read this before using a number.** |
| [`docs/DELTA-CONVENTION.md`](DELTA-CONVENTION.md) | Why `delta` is an undiscounted hedge volume, with the measured cost of the alternative |
| [`docs/RECONCILIATION-REPORT.md`](RECONCILIATION-REPORT.md) | What was ported onto `main` and what it moved |
| [`docs/reconciliation-history/2026-09-08/`](reconciliation-history/2026-09-08/) | The September review and its evidence, read-only |
| [`.planning/ROADMAP.md`](../.planning/ROADMAP.md) | Open work, prioritised |
| `storage_model.py`, `storage_kernels.py` | The library; kernels are split out so edits here do not trigger a Numba recompile |
| `tests/` | `test_regression.py` (product and portfolio anchors) and `test_reconciliation.py` (the September findings) |

Branch `review/model-fixes` holds the September review's own history and exists nowhere
else; keep it.
