# Project status

**As of 2026-09-11 (later still).** S1–S5 and S8 (documentation reconciliation) are done and
pushed. **S6 is scoped but not started in code** — no file has been edited for it yet. S7 has
not been scoped at all. See the day-by-day account below, and the note directly under this one
for exactly where S6 stands.

**S6 plan, recorded before writing any code.** Checklist items 19–20 (data manifest and the
delivery-averaged observation function). `storage_model.py` already has the canonical
month/front-month convention (`month_start`, `month_end`, `front_month_start`,
`monthly_curve_from_quote` at lines 355–384) — `forward.ipynb` cell 4 independently
reimplements the identical logic inline (`_month_start`/`_month_end`/`_front_month_start`),
a pre-existing duplication this pass did not create and is not in scope to fix. The plan is to
build on the library functions, not add a third copy:

- `quote_data.py`: `build_delivery_panel(quotes, source_hash, columns=None)` — long-format
  panel (`quote_date, contract_identifier, delivery_start, delivery_end, price_eur_mwh,
  source_column, source_hash, validity_flag`) with `contract_identifier` keyed on the
  *delivery month* (via `storage_model.front_month_start`), not the column name, so a
  continuous-rank roll (TTFc2 on one date, TTFc1 on the next, same delivery month) is
  recognised as the same contract and a rank that rolls *off* (front month entering delivery)
  correctly stops rather than being bridged to the new front month. Plus `build_returns(panel)`
  (aligned log-returns per contract, with actual elapsed calendar days recorded, non-positive
  prices excluded and counted, no silent bridging across a missing quote) and
  `build_data_manifest(quotes, provenance)` (source fingerprint, quote-date range, column
  definitions, cleaning version, duplicates, missing-observation count, filters).
- New `delivery_model.py`: `flat_forward_delivery_loading(kappa, t, A, B)` — the closed form
  from guide §9.3, with the `kappa -> 0` limit (`= 1`) handled explicitly rather than dividing
  by zero — and `delivery_averaged_loading(kappa, t, dates, forward_prices)`, the discrete
  price-weighted sum for a non-flat curve. Test against the guide's own analytic pair
  (month-end point 0.11932561 vs flat-forward delivery average 0.12443854, sigma 0.5, kappa 1,
  equal one-month periods ending 0.5y/1y) and against a synthetic panel with a month-end roll,
  a weekend, one missing quote and two delivery identifiers, expected returns worked by hand.

Not yet decided: whether to also point `forward.ipynb` at the new panel builder while touching
it. Revisit after the above is implemented and tested — do not let that decision block S6's
own completion bar (guide §9: traceable source/delivery identity, matched-interval returns,
delivery weighting reproduced on synthetic examples).

**As of 2026-09-11 (later).** [IMPLEMENTATION-GUIDE-2026-09-11.md](IMPLEMENTATION-GUIDE-2026-09-11.md)'s S1–S5 are all done — every one of its acceptance pack's 31 checks now passes, up from 9 when the guide landed. What remains is S6/S7, the calibration and model-comparison work proper, blocked on TTF data past the workbook's 6 March 2026 endpoint regardless of further engineering. See the day-by-day account below.

**As of 2026-09-10 (evening).** An independent review of the storage day landed, overturned
two results recorded as established — the ratchet cap and the case for a second factor — and
was itself replied to with an implementation plan. **The inventory-bound repair is done, and the ratchet
discretisation is now measured, gated and reported** — though the shipped notebook still
chooses a grid 0.19 % short of the 3,840-clip gate, deliberately and in print. Read
[REVIEW-RESPONSE-2026-09-10.md](REVIEW-RESPONSE-2026-09-10.md) and
[REVIEW-REPLY-AND-ACTION-PLAN-2026-09-10.md](REVIEW-REPLY-AND-ACTION-PLAN-2026-09-10.md)
together.

**2026-09-11.** A second independent review of the evening's work and an [implementation guide](IMPLEMENTATION-GUIDE-2026-09-11.md) built on it landed. Every claim in both was verified before acting (guide §14): **five P1 items are open**, three of them in code written the previous evening — the workbook cache, the convergence gate and the notebook's hard-coded convergence status — and one, the days-to-rate conversion, that silently prices a 30/90 store as 30/30 through the workbook and API routes. Its acceptance pack scores **9 of 31** on this tree.

**S1 done (checklist items 1–3).** `normalise_storage_contract()` preserves a storage contract's requested capacity, MWh/day rates and opening/terminal MWh exactly, or refuses the request — replacing `max(1, round(n_states / days))`, which had no failure mode and always returned *something*: 30/90 at N=30 silently priced 30/30 at **+EUR 131,369 (+5.51 %)** above the requested contract, and 5,000 MWh of opening inventory rounded to **zero** clips, both with no error. `resolve_grid()` gained the matching check for an explicit capacity/v_step/n_states combination that does not agree. All 11 counterexamples that were `xfail`-pinned went to `XPASS(strict)` the moment the fix landed — proving it — and the marks are now removed; the one old test that asserted the wrong contract as correct is corrected alongside. The guide's own untouched acceptance pack confirms it independently: **20 of 31 then pass**, up from 9.

**S2 mostly done.** Both `describe_ratchet_rates()`/`worst_ratchet_rate_loss()` **and** `assert_ratchets_expressible()` read `curve.max()` — the single largest active-day rate across the whole span — so a contract with more than one active rate only ever had its *fastest* day checked. An alternating 1-/2-clip-day schedule at a 0.5 multiplier reported **zero loss** on both, while the 1-clip days floored to zero and could not move at all: the guard built specifically to catch that let it straight through. Both now check every distinct active-day rate, and the diagnostic separately caps by physical headroom *before* flooring, so a boundary state one clip from full correctly reports zero loss (exact) rather than either a spurious percentage or a null. The reference store — one active rate per side — reproduces its previously-recorded 32.2 %/33.1 % exactly, confirming no regression. `benchmarks.storage_params()` had the identical independent-rounding defect and now routes through the same normaliser; every `n_states` this module actually uses is a multiple of `lcm(30,60)=60`, so nothing changed for any existing case, only a genuinely incompatible grid now refuses.

`Storage_30_65.ipynb`'s `rates_for()` and `forward.ipynb`'s storage branch both reimplemented the same silent `max(1, round(...))`, independently of the library — both now delegate to `sm.normalise_storage_contract()`. Verified byte-identical to the pre-migration output for every case checked (`Storage_30_65.ipynb`'s own test suite gate; `forward.ipynb`'s default swing *and* a storage override, via a git-stash A/B comparison, since nothing executes that notebook in CI). `Storage_30_65.ipynb`'s deliberately-incompatible grid comparison (§13) now shows `REFUSED` for 4 of 5 candidate grids instead of silently pricing them wrong. **Checked and found not to need migration**: `Storage_30_60.ipynb` constructs its grid bottom-up with its own hard `assert` and never hits the defect (its `_RESIDUAL` dict is a separate, S4-scoped issue); `pricing.ipynb`/`Swing_new.ipynb` pass `inj_days`/`wdr_days` as fixed equal values, never through a days-to-rate conversion; `Products.ipynb` doesn't touch storage sizing at all; `streamlit_app.py` never exposes asymmetric days-based storage rates in the first place — it's explicitly symmetric-only, satisfying the guide's second acceptable option for that file.

**S3 done.** [`quote_data.py`](../quote_data.py) is a new shared module (no Streamlit import, so `benchmarks.py` can use it too): one cleaning rule, and a cache addressed by a **SHA-256 of the source's actual bytes**, not a path string or a modification time. Until 2026-09-11 `benchmarks.load_quote_matrix` checked one fixed parquet path regardless of what was requested — an explicit request for a synthetic one-row workbook (`TTFc1=999`), given an older modification time than the repository's own cache, returned the repository's 4,171 rows (`TTFc1=10`) instead, and a same-path edit with its mtime restored stayed silently stale. Both are now impossible by construction: two different sources can never collide on one cache entry, and identical bytes always find their own regardless of path or timestamp. `portfolio_app.py`'s `load_quote_matrix_local` had a second, independent instance of the same defect one layer up — `@st.cache_data` hashes its call *arguments*, and the old signature was `(xlsx_path, parquet_path)`, two path *strings* that don't change when a file's bytes do. It now takes the fingerprint as an explicit (non-underscore) argument, computed outside the cached function, so Streamlit's own in-memory cache also observes a content change at an unchanged path — verified directly against `st.cache_data` (not `AppTest`, which never reaches this code; it's gated behind a submit button the smoke test doesn't click), with a **negative control** confirming the pre-fix single-argument signature is genuinely stale on the same scenario. Acceptance pack: **23 of 31** (the pack's own D-group checks reach into `benchmarks._PARQUET_CACHE`, which no longer exists now caching moved into `quote_data`; adapted the *local verification copy* only, per the pack's own README guidance for exactly this case — the canonical `docs/ACCEPTANCE-PACK-2026-09-11.zip` is untouched, dated evidence).

**S4 done.** `convergence_verdict()` returned a bare bool and had **seven** independently-confirmed ways to fail open, matching the pack's own oracle exactly: NaN/infinite values compare `False` against any tolerance, so three non-finite rows in a row silently *passed*; `n_states` was never checked for being sorted or actually doubling — duplicate, descending or arbitrary-step grids all "converged" by accident; a refused row was filtered *out* and the steps either side of the gap it left were treated as adjacent (`240 → REFUSED → 960 → 1920` silently became `240 → 960 → 1920`); and an absolute-EUR fallback was an unconditional OR on the *size of the move* rather than gated on the *values themselves* being near zero — `10,000 → 9,500 → 9,000` (5 % steps, plainly material) passed because each 500 EUR move happened to be under a 1,000 EUR floor. It now returns `(status, steps, message)` — `within_declared_tolerance` / `outside_tolerance` / `insufficient` / `invalid` — distinguishing "diverged" from "never enough evidence to judge," and requires a *genuine doubling* between adjacent, unrefused rows before a step counts at all. All nine of the pack's own `gate()` cases (seven defects plus two controls that must keep working) reproduce exactly, independently, before touching the pack itself.

`Storage_30_60.ipynb`'s `_RESIDUAL` lookup — a table measured once on 2026-09-10 for one specific curve, keyed on `N_STATES` alone, evaluated *before* the curve, ratchets or bounds even existed — is deleted outright. A new `VERIFY_CONVERGENCE` flag (default `False`, since it costs three more full valuations) runs a **live** ladder against whatever curve/ratchets/bounds are currently in the notebook, through the same `benchmarks.convergence_verdict()` the pack checks; the default path prints an honest "not checked" rather than reusing a stale number. Verified both ways: the default path still executes clean and fast (the existing notebook gate, unchanged), and the live path reproduces the previously-recorded 1,920/3,840-clip values (3,026,417.92 / 3,032,172.02) to the last printed digit while genuinely recomputing them.

`price_grid_ladder` is relabelled rather than rebuilt: it refines the price tree's *boundary width* (`n_p`, at a fixed one-day time step) and was previously documented as testing "price discretisation" more broadly — building genuine daily-substep refinement is a materially larger feature, out of scope here, so its docstring and the report label now say precisely what it does and does not test instead.

Acceptance pack: **30 of 31** — every check passes except S5's `P-terminal_anchor`.

**S5 done — the guide's implementation checklist is now complete, 31 of 31 on the acceptance pack.** `two_factor_probe.py`'s `_ou_lattice` built its transition *probabilities* from an Euler discretisation (`mean = (1-kappa*DT)*chi`, `var = sigma^2*DT`) instead of the OU process's own exact one-step moments (`mean = exp(-kappa*DT)*chi`, `var = sigma^2*(1-exp(-2*kappa*DT))/(2*kappa)`). At kappa 4, `DT = 1/12` (`kappa*DT = 1/3`, not small), Euler's mean-reversion factor is 0.6667 against the exact 0.7165 — a 7 % per-step error compounding over every one of the 23 steps, not just the last. Separately, `matched_sig_chi`'s "terminal" anchor defaulted its horizon to `N_T*DT` (2.0y) — one step *past* the actual last decision at `(N_T-1)*DT` (1.9167y) — so it matched a variance nothing in the probe is ever valued at. Together these produced the reviewed 8.95 % mismatch between the anchor and what the lattice actually propagated; both are now exact, and the mismatch is zero to machine precision (verified directly against the pack's own `P-terminal_anchor` formula, and independently against the closed-form OU variance at six kappa values including 0 and near-zero).

Fixing the transition law surfaced a second, previously-latent defect: with the exact (smaller) per-step variance, the lattice's *spacing* — fixed at `sigma*sqrt(3*DT)`, sized for the larger Euler variance — violated the trinomial's own stability condition at high kappa, producing probabilities as negative as −0.33 at the kappa=0 boundary node. `dx` is now derived from the same exact variance the transition uses (`dx = sqrt(3*var)`), which the guide's own §8.5 sequence explicitly asks be validated at kappa 0 and near zero — a case the pack itself does not test, and one the previous implementation would have failed had anyone exercised it.

The qualitative findings are unchanged — a calibrated second factor still costs a store value at every kappa above 0.2 (now 4.97 % at kappa 4, spot anchor, down from a previously reported ~8 %), and the struck swing's answer still flips sign between anchors (+0.81 % spot vs −7.76 % terminal, down from +0.58 %/−10.23 %) — only the specific percentages moved, uniformly smaller, since Euler had been overstating how sharply mean reversion departs from a random walk at this step size. Update this file when that changes — a status document that lags is worse than none.

**S8 done — documentation reconciliation, against [`docs/IMPLEMENTATION-GUIDE-2026-09-11.md`](IMPLEMENTATION-GUIDE-2026-09-11.md) §11's own table, not just this file's own paraphrase of it.** Every current document that still described the dated-bound penalty, the "nine times too little" spread shortfall or the "no gap over three days" DA-observation claim as live now carries the fix or a correction link: [`MODEL-CONVENTIONS.md`](MODEL-CONVENTIONS.md) (the penalty paragraph, the kernel-constants table, and a new section on `normalise_storage_contract`'s requested-versus-effective distinction), [`.planning/ROADMAP.md`](../.planning/ROADMAP.md) (the same penalty language in P1.1, the DA-gap and ninefold claims in P4.1, and a flag on that item's still-unreconciled extrinsic-by-kappa table), and `Storage_30_60.ipynb` (cells 3 and 10). [`REVIEW-RESPONSE-2026-09-10.md`](REVIEW-RESPONSE-2026-09-10.md)'s provenance section now dates its 34-file/101-test claim to the exact pre-repair commit (`1a01848`, verified against the archive's own recorded `storage_model.py` hash) rather than leaving it looking current, and links [`PROJECT-REVIEW-2026-09-10-evening.md`](PROJECT-REVIEW-2026-09-10-evening.md) — the second, 37-file review this guide itself was built from. `benchmarks.py`'s delivery-averaging comment no longer claims point maturities are a volatility *upper bound*; that direction was disproved by a counterexample (0.12444 delivery-averaged against 0.11933 point, same kappa and sigma), so the comment now says so, and [`FINDINGS-2026-09-10-evening.md`](FINDINGS-2026-09-10-evening.md) carries a correction link where it repeated the same assumption. `two_factor_probe.py`'s narrative and printed "Reading" section now say plainly that its section 2 is a variance-*allocation* scenario at fixed mean reversion and independent factors, not a fit to any market observation, and that correlation could move `sigma_chi` the other way.

One row of the guide's table is deliberately not fully closed: `DESIGN-P4.1-two-factor.md` already carries its 2026-09-10 correction notice, but the "link a new current calibration specification" half of that row has nothing to link to yet — that specification is S7's own deliverable, not something S8 can produce ahead of it. Revisit once S7 lands. All 154 tests still pass; this slice changed no valuation code, so the acceptance pack (last confirmed 31/31 after S5) was not rerun.

| | |
|---|---|
| Repository | [dmitry-goryunov/Storage](https://github.com/dmitry-goryunov/Storage) — the single writable source. `origin` points here directly as of 2026-09-10; it had been on the pre-rename `dmitrygoryunov2000` URL and reaching this one through a GitHub redirect |
| Working copy | `H:\My Drive\Github\dmitry-goryunov\Storage`, tracking `main`. **Stays on Drive by decision, 2026-09-09** — see the note below |
| Tests | `python -m pytest -q` → **154 passed** as of `362f526` (S5, the last of this pass's commits). Rerun for the current count; a passed-count claim does not outlive the next commit |
| CI | `.github/workflows/test.yml`, pinned from `requirements-lock.txt`, on every push and PR |
| Environment | System Python 3.12. There is deliberately no venv in the Drive folder — build one outside it. **It is not the pinned environment**: the working machine runs NumPy 2.4.3 / pandas 2.3.3 / SciPy 1.17.1 / Numba 0.65.1 against `requirements-lock.txt`'s 2.5.3 / 3.0.5 / 1.18.1 / 0.67.0, so a green local run is evidence about this machine, not about CI |

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

**Verified, not validated.** Two independent reviews (2026-09-10 and its evening follow-up)
each found real defects in how physical inputs were converted to grid quantities, in the
inventory-bound checker, in the workbook cache, and in the convergence gate — not
hypothetical ones, reproduced against `main`. [IMPLEMENTATION-GUIDE-2026-09-11.md](IMPLEMENTATION-GUIDE-2026-09-11.md)'s
S1–S5 closed all of them: physical capacity/rate/boundary-inventory conversion now refuses an
incompatible request rather than silently rounding it (S1), the ratchet diagnostic checks
every distinct rate rather than only the fastest (S2), the quote cache is content-addressed
so two different sources can never collide (S3), and the convergence gate can actually fail
(S4). What remains unresolved is calibration, not code: `sVol`/`sMR` have no estimation
window or provenance, and no market data past the workbook's 6 March 2026 endpoint has been
brought in to fit them (S6/S7). Treat every valuation as exploratory until that lands.

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
python -m pytest -q                  # 154 tests as of 362f526, ~90 s
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
| [`docs/IMPLEMENTATION-GUIDE-2026-09-11.md`](IMPLEMENTATION-GUIDE-2026-09-11.md) | **The current plan.** Eight delivery slices from strict physical conversion to a calibration decision, with an acceptance pack; §14 is the project's verification of it |
| [`docs/PROJECT-REVIEW-2026-09-10-evening.md`](PROJECT-REVIEW-2026-09-10-evening.md) | Second independent review, of the evening's repairs — credits the bound work on 48 exhaustive cases and finds five P1 items, all reproduced |
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
