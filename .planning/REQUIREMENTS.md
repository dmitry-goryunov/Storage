# Requirements: Gas Storage & Swing Pricing Model

**Defined:** 2026-09-08
**Core Value:** Every reported number is reproducible and defensible: the model prices exactly what it claims to price, and no valuation moves without a documented before/after.

Every requirement below traces to a requirement ID in `docs/SPEC-remaining-work.md` and/or a
numbered finding in `code_review.md`. Nothing is invented here; see `## Source Traceability`.

## v1 Requirements

### Delta convention and the master invariant (SPEC R6, code_review #9)

- [ ] **DELTA-01**: `delta` is documented at its definition site as an undiscounted *physical hedge volume* — the number of forward MWh to trade — with the reasoning that the discount factor cancels in `h = E[S_i*Q_i]/F_i`. `d_curve` is not applied to `delta`.
- [ ] **DELTA-02**: Invariant I1 is restated everywhere it appears as `sum_i d_curve[i] * delta[i] * fwd[i] == v[0, n_p, n_op_start]` (relative error < 1e-9), which reduces to today's `sum_i delta[i] * fwd[i] == V0` while `d_curve` is all ones.
- [ ] **DELTA-03**: A regression test builds a contract with a non-flat `d_curve` (3 % continuous) and asserts the restated I1 to < 1e-9; it fails if the discount weights are dropped from the identity.

### Exact monthly curve repricing (SPEC R1, code_review #5)

- [ ] **CURVE-01**: `smoothen_curve` solves the M x M knot system so the smoothed daily curve's monthly means reproduce every input contract to < 1e-9, iterating solve -> recompute PCHIP slopes -> re-solve and asserting exactness. Not the flat additive per-month shift (D-S1).
- [ ] **CURVE-02**: Partial first and last months are handled by building the matrix from the days actually present, rather than from a knot at the calendar midpoint against an average over available days.
- [ ] **CURVE-03**: `smoothen_curve(..., exact=False)` reproduces the pre-R1 curve exactly; exact repricing is the default (D-O1).
- [ ] **CURVE-04**: The corrected curve has no month-boundary discontinuity beyond the natural curve slope, verified by a test — the model compares adjacent days to decide exercise.
- [ ] **CURVE-05**: A before/after valuation table covering the six `quotes.csv` products, the `finding.md` put swing and the storage case is published in `code_review.md` (invariant I3).

### Streamlit front end (SPEC R2, R3, R4; code_review #7, #8, #10)

- [ ] **APP-01**: The sidebar `n_p_full` default is 30, with a note that the value must be checked for convergence (today's default of 10 discards ~15 % of extrinsic value: 1.0054 vs 1.1789 EUR/MWh).
- [ ] **APP-02**: The `wdr_days` input is removed from the sidebar and from `params`; the storage inputs state that injection and withdrawal capacity are symmetric and that withdrawal capacity is expressed through `w_ratch` (D-O3).
- [ ] **APP-03**: The "Monthly Native Deltas" table is captioned as a local ratio requiring re-hedging, naming the measured convexity — summer buckets decay within a single day's market move (July +8.6 %, September +18.7 %, November +43.9 % at a 1 % bump) while quota-forced December is stable at -1.3 % (D-O4).

### Dead code and API hygiene (SPEC R5.1, R5.2; code_review #11, #12, #13, #19)

- [ ] **CLEAN-01**: `get_exercise`, `get_delta`, `valuation` and `check_curve` are removed; nothing in the library, the app, the notebooks, `test_model.py` or `review_checks.py` still calls them.
- [ ] **CLEAN-02**: `probabilities` and `run_model` no longer take arguments they never read; the tunnels either bind (documented and tested) or `mintunnel`/`max_tunnel` are removed entirely — as constructed, `mintunnel` is all zeros and `max_tunnel = n_op` while valid states are `0..n_op-1`.
- [ ] **CLEAN-03**: The `1000.0 * v_step` tunnel penalty (1e6 per unit — comparable to the whole contract value, so it under-penalises large deals and over-penalises small ones) is replaced by a scale justified in a comment, with valuations unchanged or an I3 before/after table if any moves.
- [ ] **CLEAN-04**: `Storage.n_op_start` is disentangled: the volume-state count and the starting inventory have separate names, callers set the starting inventory explicitly, and a test asserts that calling `set_volume_states()` twice does not move the start state.

### Documentation and reproduction scripts (SPEC R5.3, R5.4, R5.5; code_review #14, #16, #20)

- [ ] **DOCS-01**: README matches the code — the curve-correction claim (true only after CURVE-01), the `sVol` default (0.9 at `storage_model.py:67`, documented as 0.6), the `flat()`/`profiled()` description and the "Intrinsic = profiled - flat" decomposition that their aliasing invalidates, the usage example that prints minus the flat price as "Extrinsic", the smoother described as "a natural cubic spline" where the code uses `PchipInterpolator` plus `CubicHermiteSpline`, and the API entries for functions CLEAN-01 deletes. README's `1e10` infeasibility claim is left intact, with a note recording that it is a different quantity from the `-1e9` terminal sentinel.
- [ ] **DOCS-02**: The five repo-root `debug_delta*.py` scripts are folded into `review_checks.py`; their unique checks are reachable via `python review_checks.py <section>` and no script re-implements `monthly_curve_from_quote` or `curve_df_for_storage`.
- [ ] **DOCS-03**: The DP-grid truncation question is settled in writing: either the grid ends at the exercise window (with an I3 before/after table) or the decision to keep `backStop` is recorded with its rationale and the measured cost — 30 idle steps of 760 for a one-year deal, 29 of 119 (24 %) for a three-month deal. The SPEC words this as "consider", so a documented decision is acceptance.

## v2 Requirements

Deferred. Tracked, not in the current roadmap.

### Hedging analytics

- **GAMMA-01**: Report a bucketed gamma alongside the monthly delta table (+/-1 % rerun, ~24 extra builds, ~5 s) — deferred by D-O4 in favour of the caption.

### Contract features

- **CAPY-01**: Asymmetric injection/withdrawal capacity — `n_op` from injection capacity, `w_ratch`/`max_vol` from withdrawal capacity — specified on its own rather than through the dead `wdr_days` input (D-O3).

### Numerical hygiene

- **TOL-01**: Make `strat`'s no-exercise test relative rather than the absolute `< 1e-6` (code_review #17) — immaterial today, worth revisiting if `v_step` or the price scale grows.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Rewriting the DP solver or the trinomial tree | Both verified correct: probabilities in [0,1], forward fitting exact to 1e-15, value converged in `n_p`, race-free parallel passes (SPEC "Out of scope", D-1) |
| "Fixing" the January step-down at Jan 31 | Confirmed correct model behaviour; the step tracks `days` and is invariant to `n_p` (20/30/45 identical to within 0.4 MWh/day) (D-2) |
| "Fixing" the December delta amplification (ratio 1.71) | Confirmed correct; it is the endpoint of a smooth trend in `E[S|exercise]/F`, validated against finite differences to +/-0.4 % (D-2) |
| Graduated soft terminal penalty | Reverted; the multiplier needed to replicate the hard constraint is ~4x and depends on `n_p` and `sVol`, i.e. on discretisation rather than contract economics (D-3) |
| "Correcting" README's `1e10` infeasibility claim | It is true — `bigdummy = 1e10` in `run_model` is the per-transition penalty, a different quantity from the `-1e9` terminal `t_p_curve` sentinel |
| Packaging, deployment, CI | Single-developer quant library plus a local Streamlit front end; no distribution target |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DELTA-01 | Phase 1 | Pending |
| DELTA-02 | Phase 1 | Pending |
| DELTA-03 | Phase 1 | Pending |
| CURVE-01 | Phase 2 | Pending |
| CURVE-02 | Phase 2 | Pending |
| CURVE-03 | Phase 2 | Pending |
| CURVE-04 | Phase 2 | Pending |
| CURVE-05 | Phase 2 | Pending |
| APP-01 | Phase 3 | Pending |
| APP-02 | Phase 3 | Pending |
| APP-03 | Phase 3 | Pending |
| CLEAN-01 | Phase 4 | Pending |
| CLEAN-02 | Phase 4 | Pending |
| CLEAN-03 | Phase 4 | Pending |
| CLEAN-04 | Phase 4 | Pending |
| DOCS-01 | Phase 5 | Pending |
| DOCS-02 | Phase 5 | Pending |
| DOCS-03 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0 ✓

## Source Traceability

Every v1 requirement back to its source. `SPEC` = `docs/SPEC-remaining-work.md`;
`#N` = numbered finding in `code_review.md`.

| Requirement | SPEC | code_review | Location |
|-------------|------|-------------|----------|
| DELTA-01 | R6 (decision O2) | #9 | `storage_model.py:793-811` (`compute_all_metrics`), `:97` (`d_curve`) |
| DELTA-02 | R6, I1 | #9 | SPEC §Invariants; `test_model.py:61-77` |
| DELTA-03 | R6 ("add a test with a non-flat `d_curve`") | #9 | `test_model.py`; `review_checks.py:287` (`discounting`) |
| CURVE-01 | R1 | #5 | `storage_model.py:20-42` (`smoothen_curve`) |
| CURVE-02 | R1 ("Also in scope") | #5 | `storage_model.py:29-31` (midpoint knots) |
| CURVE-03 | R1 (O1, decided default-on) | #5 | `storage_model.py:20` signature |
| CURVE-04 | R1 Acceptance | #5, D-S1 | `storage_model.py:36-40` |
| CURVE-05 | R1 Acceptance, I3 | #5 | `code_review.md`; `quotes.csv`; `finding.md` put swing |
| APP-01 | R2 | #7 | `streamlit_app.py:101` |
| APP-02 | R3 (decided: remove) | #8 | `streamlit_app.py:107`, `:145`; `value_storage` `storage_model.py:386-400` |
| APP-03 | R4 (decided: caption) | #10 | `streamlit_app.py:249` ("Monthly Native Deltas") |
| CLEAN-01 | R5.1 | #11 | `storage_model.py:45`, `:718`, `:766`, `:780` |
| CLEAN-02 | R5.2 | #12, #13 | `storage_model.py:111`, `:122`, `:626-628`, `:727-729` |
| CLEAN-03 | R5.2 | #13 | `storage_model.py:668-670` |
| CLEAN-04 | R5 (finding 19 bullet) | #19 | `storage_model.py:114`, `:126-129` (`set_volume_states`) |
| DOCS-01 | R5.3 | #16, #2, #5 | `README.md`; `storage_model.py:67`, `:213-214`, `:5`, `:36-37` |
| DOCS-02 | R5.4 | #20 | `debug_delta.py`, `debug_delta2..5.py`; `review_checks.py` |
| DOCS-03 | R5.5 ("consider") | #14 | `storage_model.py:75-77` (`backStop`) |

**Findings not routed as requirements:** 1, 2, 3, 4, 6, 15, 18 are fixed (see PROJECT.md
"Validated"); 17 is deliberately out of scope. No requirement derives from `finding.md`,
which is evidence, or from `README.md`, which describes existing rather than target behaviour.

---
*Requirements defined: 2026-09-08*
*Last updated: 2026-09-08 after roadmap creation*
