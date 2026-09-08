# Gas Storage & Swing Pricing Model

## What This Is

A quantitative library for valuing natural gas storage and swing contracts on the TTF market.
`storage_model.py` builds a trinomial price tree with Ornstein-Uhlenbeck mean reversion and
solves a dynamic program over a joint (time x price x volume) state space, JIT-compiled and
parallelised with Numba. `streamlit_app.py` is a local front end and `Swing_new.ipynb` the
driver notebook, pricing six TTF swing products plus a storage case against `curve.csv`
(48 monthly contracts, Jan 2026 - Dec 2029) and `quotes.csv`. One author, one implementer,
no packaging or deployment target.

## Core Value

Every reported number is reproducible and defensible: the model prices exactly what it claims
to price, and no valuation moves without a documented before/after.

## Requirements

### Validated

<!-- Shipped on branch review/model-fixes and locked by test_model.py (13/13). -->

- ✓ `stochastic_metric` divides by exercised volume, not by `sum(delta)` (code_review #1) — Batch 1
- ✓ One `price_per_mwh()` with the correct price state and denominator; `flat()`/`profiled()` no longer return 0 on a full tree (code_review #2) — Batch 1
- ✓ One whole-clip ratchet convention; fractional ratchets rejected at assignment (code_review #3) — Batch 2
- ✓ `build()` checks terminal feasibility and raises instead of returning the `-1e9` sentinel as a price (code_review #4) — Batch 2
- ✓ `dx` sized from `max(sVol)` plus a tree stability check; vol term structures no longer produce silent NaNs (code_review #6) — Batch 2
- ✓ Curve-coverage failure names the real cause, and `load_direct_curve` checks uploads (code_review #15) — Batch 2
- ✓ Cosmetic `round(..., 3)` removed; it was masking the repricing identity at 1e-7 (code_review #18) — Batch 2
- ✓ Regression suite exists: `test_model.py`, 3/13 against the baseline, 13/13 after the fixes — Batch 0

### Active

<!-- Remaining work, per docs/SPEC-remaining-work.md. IDs defined in REQUIREMENTS.md. -->

- [ ] **DELTA-01..03** (SPEC R6 / code_review #9) — document the delta convention, restate invariant I1, test it under a non-flat `d_curve`
- [ ] **CURVE-01..05** (SPEC R1 / code_review #5) — exact monthly curve repricing via the PCHIP knot solve; the only remaining valuation-changing item
- [ ] **APP-01..03** (SPEC R2, R3, R4 / code_review #7, #8, #10) — converged `n_p_full` default, remove the dead `wdr_days` input, caption the monthly delta table
- [ ] **CLEAN-01..04** (SPEC R5.1, R5.2 / code_review #11, #12, #13, #19) — remove dead code, fix the tunnels and the penalty scale, disentangle `n_op_start`
- [ ] **DOCS-01..03** (SPEC R5.3, R5.4, R5.5 / code_review #14, #16, #20) — README accuracy, fold the debug scripts into `review_checks.py`, settle DP-grid truncation

### Out of Scope

- Rewriting the DP solver or the trinomial tree — both were verified correct: probabilities in [0,1], forward fitting exact to 1e-15, value converged in `n_p`, race-free parallel passes (SPEC "Out of scope", D-1)
- The mandatory-quota artefacts in finding.md — the January step-down and the December delta amplification are confirmed correct model behaviour, independently validated (D-2)
- A graduated soft terminal penalty — the multiplier needed to replicate the hard constraint is ~4x and depends on `n_p` and `sVol`, i.e. on discretisation rather than contract economics (D-3)
- Bucketed gamma on the monthly delta table — a compute-bearing feature (~24 builds, ~5 s), not warranted as housekeeping (D-O4); available as a future feature
- Asymmetric injection/withdrawal capacity — a legitimate future feature, to be specified on its own rather than smuggled in through a dead `wdr_days` input (D-O3)
- Relative tolerance in `strat`'s no-exercise test (code_review #17) — immaterial while the repricing identity holds to 0.1 EUR; not in the SPEC
- "Correcting" README's `1e10` claim — it is true: `bigdummy = 1e10` in `run_model` is the per-transition penalty, a different quantity from the `-1e9` terminal sentinel

## Context

- **Review provenance.** A model review dated 2026-09-08 produced 10 findings plus 10 minor
  ones, executed rather than inferred and reproducible with `python review_checks.py <section>`.
  Batches 0-2 fixed findings 1, 2, 3, 4, 6, 15 and 18 and took the suite from 3/13 to 13/13.
  No valuation changed in those batches: `intrinsic`, `extrinsic`, `total` and `v0` are identical
  before and after for all six quoted products, the finding.md put swing and the storage case;
  only `stochastic_metric` moved (+1.2 % to +5.5 % for the call swings, -7.6 % for the put swing).
- **Branch state.** Work sits on `review/model-fixes`: `9492d90` baseline, `d999c1f` review,
  `0021a19` regression suite, `18d1fb0` fixes. `origin/main` predates all four.
- **Pipeline.** `map_curve_to_dates` -> `smoothen_curve` -> `build_tree` -> `run_model` (DP)
  -> `probabilities` (forward pass) -> `compute_all_metrics` (`exp_ex`, `delta`).
- **Measured state of the remaining work.** The curve misses its input contracts by up to
  0.1023 EUR/MWh (mean 0.0218) against a 1.27 EUR/MWh intrinsic value; the Streamlit
  `n_p_full = 10` default discards ~15 % of extrinsic value (1.0054 vs 1.1789 EUR/MWh);
  with a 3 % discount curve the undiscounted delta profile overstates the hedge by 4.92 %
  (`V0 = -652,146.1` vs `sum(delta*F) = -684,200.5`).
- **Evidence base.** `finding.md` carries the put-swing delta investigation and an independent
  peer review: bucketed deltas match finite differences to within +/-0.4 % at a 5 bp bump but
  are strongly convex at 1 % (Jul +8.6 %, Sep +18.7 %, Nov +43.9 %, quota-forced Dec -1.3 %).
- **Dead-code targets, confirmed present.** `check_curve` (`storage_model.py:45`),
  `valuation` (`:718`), `get_exercise` (`:766`), `get_delta` (`:780`); tunnel penalty
  `1000.0 * v_step` (`:668`); Streamlit `n_p_full` value 10 (`:101`) and `wdr_days` (`:107`, `:145`).

## Constraints

- **Invariant (I1)**: `sum_i d_curve[i] * delta[i] * fwd[i] == v[0, n_p, n_op_start]`, relative error < 1e-9 — the suite's master test; it ties the DP, the forward pass and the reported metrics together. Reduces to `sum_i delta[i] * fwd[i] == V0` while `d_curve` is all ones.
- **Invariant (I2)**: `python test_model.py` passes — 13/13 today, and green at every commit.
- **Invariant (I3)**: no change to a reported valuation lands without a documented before/after across the six products in `quotes.csv`, the finding.md put swing and the storage case.
- **Numerical**: smoothed monthly means must reproduce input contracts to < 1e-9, with no month-boundary discontinuity beyond the natural curve slope.
- **Numerical**: reported extrinsic value must be checked for convergence in `n_p` (`n_p = 20` converges to 0.001; a full `n_p = 60` build costs 0.22 s).
- **API contract (must not regress)**: one whole-clip ratchet convention; infeasibility raises rather than being priced; the curve must cover one month past `storageEnd` and say so when it does not; tree probabilities in [0,1] with `q.sum(1) == 1`.
- **Tech stack**: Python 3.12 with numpy, numba (JIT + parallel), scipy, pandas, matplotlib, streamlit, openpyxl — Windows dev machine, no packaging or deployment target.
- **Scale**: no constant may be arbitrary relative to deal size — the `1000.0 * v_step` penalty is 1e6 per unit, comparable to the whole contract value.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| D-O1: exact curve repricing is default-on, with `exact=False` to reproduce old behaviour | A curve that does not reprice its own input contracts is a defect, not a preference | — Pending (lands in Phase 2) |
| D-O2: `delta` stays an undiscounted physical hedge volume; `d_curve` is not applied to it | It is the number of forward MWh to trade — the DF cancels in `h = E[S_i*Q_i]/F_i`, so a discounted delta would not be tradeable | — Pending (lands in Phase 1) |
| D-O3: remove `wdr_days` from the app rather than wiring it through | It implies a control the model does not have; withdrawal capacity is expressed through `w_ratch` | — Pending (lands in Phase 3) |
| D-O4: caption the monthly delta table; no bucketed gamma | The convexity is measured and documented; a compute-bearing feature is not warranted as housekeeping | — Pending (lands in Phase 3) |
| D-S1: R1 uses the M x M knot solve, not a flat additive per-month shift | The additive shift hits the constraint but leaves ~0.1 EUR steps at month boundaries, and the model compares adjacent days to decide exercise | — Pending (lands in Phase 2) |
| D-S2: repricing tolerance is 1e-9, not 1e-6 | Stricter bound already implemented at `test_model.py:72` | ✓ Good (in force) |
| D-1: no rewrite of the DP or the tree | Both verified correct by executed checks | ✓ Good |
| D-2: the mandatory-quota artefacts are correct behaviour | Independently reproduced; the January step tracks `days` and is invariant to `n_p` | ✓ Good |
| D-3: keep the hard terminal constraint; do not revisit the soft penalty | The required multiplier (~4x) depends on `n_p` and `sVol`, i.e. on discretisation | ✓ Good |
| Ingest: accept the `code_review.md <-> finding.md` cross-reference cycle downgrade | The edges are one-line mutual citations, not transitive inclusions; both documents are self-contained and agree | ✓ Good |

**No decision here is ADR-locked.** Every entry is revisable; raise an ADR if one should become immutable.

---
*Last updated: 2026-09-08 after ingest of code_review.md, finding.md, docs/SPEC-remaining-work.md and README.md*
