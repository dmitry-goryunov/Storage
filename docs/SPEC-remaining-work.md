# SPEC — Remaining work on the storage/swing pricing model

Status: active
Date: 2026-09-08
Supersedes: the batch plan agreed in review; derived from [code_review.md](../code_review.md) and [finding.md](../finding.md)

## Context

`storage_model.py` values gas storage and swing contracts on a trinomial tree with a
DP solver; `streamlit_app.py` is the front end. A model review (2026-09-08) produced
10 findings plus 10 minor ones. Findings 1, 2, 3, 4, 6, 15 and 18 are fixed and
covered by `test_model.py` (13 tests). This spec covers what remains.

## Invariants that must hold at every commit

- **I1** — The delta profile reprices the contract:
  `sum_i d_curve[i] * delta[i] * fwd[i] == v[0, n_p, n_op_start]`, relative error < 1e-9.
  This ties the DP, the forward pass and the reported metrics together; it is the suite's
  master test. The `d_curve` weights follow from O2 below (delta is an undiscounted hedge
  volume); with `d_curve` all ones — every configuration in use today — this reduces to
  `sum_i delta[i] * fwd[i] == V0`, which is what test_model.py asserts.
- **I2** — `python test_model.py` passes 13/13.
- **I3** — No change to a reported valuation lands without a documented before/after
  across the six products in `quotes.csv`, the finding.md put swing and the storage case.

## R1 — Exact monthly curve repricing (finding 5)

`smoothen_curve` fits PCHIP through monthly midpoints and stops, so the daily curve
misses each input contract by up to 0.1023 EUR/MWh (mean 0.0218, max 38 bp). The
README already claims a correction that does not exist. Intrinsic value is 1.27
EUR/MWh and is made of the month shape these errors distort; the monthly deltas the
app reports are against contracts the model does not reprice.

**Approach.** The monthly average of the Hermite spline is a linear function of the
knot values, so build the M×M matrix (M = contract months) and solve for the knots
that reproduce every contract exactly. PCHIP slopes are a nonlinear function of the
knots, so iterate solve → recompute slopes → re-solve (2–3 iterations) and assert
exactness. Not the flat additive per-month shift: it hits the constraint but leaves
~0.1 EUR steps at month boundaries, and the model compares adjacent days to decide
exercise.

**Also in scope.** Partial first/last months: the knot sits at the calendar midpoint
while the average is taken over available days only, biasing the front of the curve.
Building the matrix from the days actually present fixes it.

**Acceptance.**
- Smoothed monthly means reproduce input contracts to < 1e-9.
- No month-boundary discontinuity beyond the natural curve slope.
- Before/after valuation table per I3, published in code_review.md.
- Tests added for both properties.

**O1 — DECIDED 2026-09-08: default-on.** Exact repricing is the default, with
`exact=False` available to reproduce the old behaviour for comparison, on the grounds
that a curve which does not reprice its own inputs is a defect rather than a preference.
Accepted consequence: notebook and app results shift slightly on the next run.

**Risk.** This is the only remaining item that changes valuations.

## R2 — Converged default for `n_p_full` (finding 7)

The Streamlit default of 10 discards ~15 % of extrinsic value (1.0054 vs a converged
1.1789 EUR/MWh); n_p = 20 is converged to 0.001. The full n_p = 60 build costs 0.22 s,
so there is no performance argument. Raise the default to 30 and note convergence in
the sidebar.

## R3 — `wdr_days` is dead input (finding 8)

The sidebar collects `wdr_days` and `value_storage` never reads it — withdrawal
capacity is silently forced equal to injection capacity, and the per-MWh
normalisation uses injection days.

**DECIDED 2026-09-08: remove the input.** It implies a control the model does not have;
withdrawal capacity is actually expressed through `w_ratch`. Asymmetric
injection/withdrawal capacity stays a legitimate future feature, to be specified on its
own rather than smuggled in through a dead input.

## R4 — Delta convexity caveat on the monthly table (finding 10)

Bucketed deltas are correct local derivatives (±0.4 % at 5 bp) but strongly convex:
at a 1 % bump July is out by +8.6 %, September +18.7 %, November +43.9 %, while
quota-forced December is stable at −1.3 %. TTF monthlies move 1–3 % on an ordinary
day.

**DECIDED 2026-09-08: caption only.** Caption the table as a local ratio requiring
re-hedging. A bucketed gamma (±1 % rerun, ~24 builds, ~5 s) remains available as a
future feature, but is not warranted as housekeeping.

## R5 — Dead code and documentation (findings 11–14, 16, 19, 20)

- Remove `get_exercise`, `get_delta` (superseded by `compute_all_metrics`),
  `valuation` (a one-element loop), `check_curve` (never called).
- Drop the unused `mintunnel`/`max_tunnel` arguments to `probabilities`; either make
  the tunnels bind or remove them, and replace the arbitrary `1000.0 * v_step`
  penalty scale.
- README: the curve-correction claim (fixed by R1), the `sVol` default (0.9 in code
  vs 0.6 documented), the `flat()`/`profiled()` description, and the usage example
  that prints minus the flat price as "Extrinsic".
- Fold the five `debug_delta*.py` scripts into `review_checks.py`; they duplicate
  helpers that now live in `storage_model`.
- Consider ending the DP grid at the exercise window rather than `backStop`: 30 idle
  steps of 760 for a one-year deal, but 29 of 119 (24 %) for a three-month deal.
- Disentangle `Storage.n_op_start` (finding 19): it means "number of volume states" to
  `_init_volume_arrays` and "starting inventory" to every caller, which reassigns it
  after `set_volume_states()` has set it — so a later `set_volume_states()` call silently
  resets the start state. Acceptance: the two meanings have separate names, callers set
  the starting inventory explicitly, and a test asserts that calling
  `set_volume_states()` twice does not move the start state.

Two further README inaccuracies were found during ingest and belong to the README
bullet above: it describes the smoother as "a natural cubic spline" where the code uses
`PchipInterpolator` plus `CubicHermiteSpline`, and it documents `get_exercise` and
`valuation`, which this requirement deletes.

**Do not "correct" the `1e10` claim.** README's "Infeasible transitions are penalised
with a large dummy value (1e10)" is true: `bigdummy = 1e10` in `run_model` is the
per-transition penalty, a different quantity from the `-1e9` terminal `t_p_curve`
sentinel that findings 3 and 4 concern.

## R6 — Document the delta convention and restate I1 (finding 9, decision O2)

`compute_all_metrics` never applies `d_curve`, so the reported delta is an
undiscounted physical hedge volume while the value is discounted. With a 3 % curve
the profile overstates the hedge by 4.92 % and I1 breaks by construction. Two
defensible conventions:

- **(a) Physical hedge volume** — leave undiscounted, document it, and state I1 in
  undiscounted terms.
- **(b) Value sensitivity** — apply `d_curve` to the delta, keeping I1 exact.

**DECIDED 2026-09-08: (a), undiscounted physical hedge volume.** `delta` is the number
of forward MWh to trade: hedging day i with h forwards gives PV = h·DF_i·F_i·ε against
dV/dε = DF_i·E[S_i·Q_i], so the discount factor cancels and h = E[S_i·Q_i]/F_i. A
discounted delta would not be a tradeable quantity. `d_curve` is therefore NOT applied to
`delta`; instead invariant I1 carries the DF weights, which is exact both under a real
discount curve and today (where `d_curve` is all ones and I1 reduces to its current
form). Work: document the convention, restate I1, and add a test with a non-flat
`d_curve`.

## Out of scope

- Rewriting the DP or the tree: both were verified correct (probabilities in [0,1],
  forward fitting exact to 1e-15, value converged in n_p, race-free parallel passes).
- The mandatory-quota artefacts in finding.md (the January step, the December delta
  amplification): both confirmed correct model behaviour and independently validated.
