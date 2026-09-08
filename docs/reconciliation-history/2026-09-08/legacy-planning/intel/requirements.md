# Requirements — synthesized intel

All requirements below derive from **docs/SPEC-remaining-work.md** (precedence 0), which
is the agreed remaining-work plan. code_review.md (precedence 1) supplies measured
evidence and location references for the same items; finding.md and README.md yield no
requirements (see context.md).

Status vocabulary: `open` = outstanding work; `variant-unresolved` = requirement is real
but its acceptance has two competing forms that synthesis must not merge.

---

## REQ-exact-curve-repricing (SPEC R1 / review finding 5)
- source: docs/SPEC-remaining-work.md §R1
- evidence: code_review.md §5; storage_model.py:20-42 (`smoothen_curve`), :45 (`check_curve`)
- status: open — **the only remaining item that changes valuations** (SPEC §R1, Risk)
- scope: `smoothen_curve`, PCHIP knot solve, partial first/last month handling

**Description.** `smoothen_curve` fits PCHIP through monthly midpoints and stops, so the
daily curve misses each input contract by up to 0.1023 EUR/MWh (mean 0.0218, max 38 bp;
`alpha = 1.0` is no better at 0.1074; 2027 monthly errors run -0.102 to +0.076 EUR).
Intrinsic value is 1.27 EUR/MWh and is made of the month shape these errors distort, and
the monthly deltas the app reports are against contracts the model does not reprice.

**Approach (mandated, not optional).** The monthly average of the Hermite spline is a
linear function of the knot values: build the M x M matrix (M = contract months) and
solve for the knots that reproduce every contract exactly. PCHIP slopes are a nonlinear
function of the knots, so iterate solve -> recompute slopes -> re-solve (2-3 iterations)
and assert exactness. Explicitly NOT the flat additive per-month shift.

**Also in scope.** Partial first/last months: the knot sits at the calendar midpoint
while the average is taken over available days only, biasing the front of the curve.
Build the matrix from the days actually present.

**Acceptance criteria.**
- Smoothed monthly means reproduce input contracts to < 1e-9
- No month-boundary discontinuity beyond the natural curve slope
- Before/after valuation table per I3, published in code_review.md
- Tests added for both properties

**Blocked on decision D-O1** (default-on vs behind a flag) — see decisions.md.

---

## REQ-converged-np-default (SPEC R2 / review finding 7)
- source: docs/SPEC-remaining-work.md §R2
- evidence: code_review.md §7; streamlit_app.py:101 (`value=10`) — confirmed present
- status: open
- scope: streamlit_app.py sidebar default and caption

**Description.** The Streamlit default `n_p_full = 10` discards ~15 % of extrinsic value
(1.0054 vs a converged 1.1789 EUR/MWh). `n_p = 20` is converged to 0.001. The full
`n_p = 60` build costs 0.22 s, so there is no performance argument.

Convergence data (extrinsic, EUR/MWh): n_p 5 -> 0.537 (-54 %), 10 -> 1.005 (-15 %),
15 -> 1.166 (-1.3 %), 20 -> 1.180, 30 -> 1.179, 60 -> 1.179.

**Acceptance criteria.**
- Default raised to 30
- Sidebar notes that the value should be checked for convergence

---

## REQ-wdr-days-input (SPEC R3 / review finding 8)
- source: docs/SPEC-remaining-work.md §R3
- evidence: code_review.md §8; streamlit_app.py:107 collects `wdr_days`, :145 passes it,
  `value_storage` (:386-400) reads only `params["inj_days"]` — confirmed present
- status: **variant-unresolved** — do not merge; see INGEST-CONFLICTS.md WARNING
- scope: streamlit_app.py sidebar, `value_storage`, `set_volume_states`, `max_vol`

**Description.** `wdr_days` is dead input. Withdrawal capacity is silently forced equal
to injection capacity and the per-MWh normalisation uses injection days, so a user who
sets asymmetric capacities gets a number that does not answer their question.

**Acceptance variant (a) — wire it through.** `n_op` from injection capacity;
`w_ratch` / `max_vol` from withdrawal capacity. Asymmetric capacities produce a value
that reflects the withdrawal constraint.

**Acceptance variant (b) — remove the input.** The sidebar no longer collects
`wdr_days`; symmetric capacity is explicit rather than silent.

Both variants are stated as acceptable by the SPEC ("Either wire it through ... or
remove the input"). Synthesis does not pick.

---

## REQ-delta-convexity-caveat (SPEC R4 / review finding 10)
- source: docs/SPEC-remaining-work.md §R4
- evidence: code_review.md §10; finding.md peer review [bucket_convexity], [delta_fd]
- status: **variant-unresolved** — do not merge; see INGEST-CONFLICTS.md WARNING
- scope: streamlit_app.py "Monthly Native Deltas" table

**Description.** Bucketed deltas are correct local derivatives (+/-0.4 % at a 5 bp bump,
validated month by month against finite differences) but strongly convex. At a 1 % bump:
July +8.6 %, September +18.7 %, November +43.9 %, while quota-forced December is stable
at -1.3 %. TTF monthlies move 1-3 % on an ordinary day. Bumping each month by 1 %
separately and summing gives -721,107 vs -678,683 for a parallel bump — a 6 % gap that
is pure gamma.

**Acceptance variant (a) — report a bucketed gamma.** Rerun at +/-1 % (~24 extra builds,
~5 s) and surface gamma alongside delta.

**Acceptance variant (b) — caption only.** Caption the table as a local ratio requiring
re-hedging, noting that summer buckets decay within a single day's market move while
winter (quota-forced) buckets do not.

Stated by the SPEC as "Either ... or at minimum ...". Synthesis does not pick.

---

## REQ-dead-code-and-docs (SPEC R5 / review findings 11-14, 16, 19, 20)
- source: docs/SPEC-remaining-work.md §R5
- evidence: code_review.md §Minor 11-14, 16, 19, 20
- status: open; **scope gap on finding 19** — see INGEST-CONFLICTS.md WARNING
- scope: storage_model.py, README.md, repo-root debug scripts

**Sub-items, each with its own acceptance.**

- **R5.1 Remove dead code** — `get_exercise` (storage_model.py:766), `get_delta` (:780),
  both superseded by `compute_all_metrics`; `valuation` (:718, a one-element loop equal
  to `v[0, n_p, n_op_start]`); `check_curve` (:45, never called). All four confirmed
  still present.
- **R5.2 Tunnels and penalty scale** — drop the unused `mintunnel` / `max_tunnel`
  arguments to `probabilities`; either make the tunnels bind or remove them; replace the
  arbitrary `1000.0 * v_step` penalty scale (1e6 per unit — comparable to the whole
  contract value, so it under-penalises large deals and over-penalises small ones).
- **R5.3 README accuracy** — four named claims: (i) the curve-correction claim (becomes
  true only via R1), (ii) the `sVol` default (0.9 in code at storage_model.py:67 vs 0.6
  documented), (iii) the `flat()` / `profiled()` description (they are aliases of one
  `price_per_mwh`, storage_model.py:213-214, so the documented
  "Intrinsic = profiled - flat" decomposition does not hold), (iv) the usage example that
  prints `s_full.flat() - s_flat.flat()` as "Extrinsic". Two further inaccuracies were
  detected during synthesis — see INGEST-CONFLICTS.md INFO.
- **R5.4 Consolidate debug scripts** — fold the five repo-root `debug_delta*.py` scripts
  into `review_checks.py`; they duplicate helpers that now live in `storage_model`
  (`monthly_curve_from_quote`, `curve_df_for_storage`).
- **R5.5 DP grid truncation (consider, not mandate)** — consider ending the DP grid at
  the exercise window rather than `backStop`: 30 idle steps of 760 for a one-year deal,
  but 29 of 119 (24 %) for a three-month deal. Worded as "Consider", so acceptance is a
  decision plus rationale, not necessarily a code change.

**Not in R5:** review finding 17 (`strat` uses an absolute tolerance `< 1e-6` while `v`
keeps the maximised value). code_review.md calls it "currently immaterial"; the SPEC does
not list it. Recorded in context.md, not as a requirement.

---

## Historical — fixed, not outstanding work

code_review.md findings **1, 2, 3, 4, 6, 15 and 18** are marked fixed in its own status
table and are excluded by the SPEC ("Findings 1, 2, 3, 4, 6, 15 and 18 are fixed and
covered by test_model.py"). The two sources agree. Where code_review.md still states a
"Fix:" recommendation for these, it is historical context and must NOT be routed as work.
Detail in context.md.
