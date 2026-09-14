# Corrected design and implementation plan: monthly-reset swing

**Date:** 13 September 2026 (design); implementation and a running status log follow in
sec.13, dated as each slice landed  
**Original status, as first written:** design only; no valuation code was implemented by
this document at that time. **No longer current** -- sections 1-12 remain the product and
numerical specification this feature was built against, but Release 1A and Release 1B (a
restricted subset of this section's own definition -- see sec.14.1 and R-06's own finding,
2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING) are now implemented, tested and in
production use in `MonthlyResetSwing.ipynb`. **For current implementation status, read
[`docs/STATUS.md`](STATUS.md) first** -- it is the concise, dated, currently-true summary;
sec.13 below is the full development log the status entries are extracted from, kept for
traceability rather than as the first thing to read. (R-11's own finding: this document mixes
a static specification with a historical diary in one file, making "what is current" harder
to see than it should be. A full split into three separate documents, as the review's own
required correction describes, has not been done -- STATUS.md already serves as the "concise
current status" piece; splitting the sec.1-12 specification from the sec.13 diary into
separate files remains open.)  
**Target repository:** `dmitry-goryunov/Storage`

## 1. Executive decision

A swing whose monthly strike is fixed from market observations made in the preceding month
cannot be priced by broadcasting a deterministic strike into the existing daily cost arrays.
The strike is random before fixing and correlated with both future delivery prices and the
exercise policy.

For a single isolated reset, one additional path state can be sufficient. For recurring
monthly resets, one additional state is not sufficient. During delivery month `M`, the model
must retain the already-fixed strike `K_M` while simultaneously accumulating the observations
that will determine `K_(M+1)`. The active-month valuation state is therefore

```text
(time, price state, cumulative exercise volume, current strike, next reset accumulator)
```

or an economically equivalent reduced representation.

A dense implementation of this state space must first be built as a small exact benchmark.
For a qualified mandatory-volume structure, the current-strike dimension may be eliminated
analytically and exact dynamic programming may also be practical at production scale. For
other realistic contracts, regression Monte Carlo remains the likely production route. The
choice must be made from measured state sizes, convergence, memory and runtime rather than
assumed in advance.

This produces an internally consistent value under the chosen risk-neutral price dynamics.
It does not by itself establish that those dynamics or parameters are calibrated to market
prices. The repository's current Q volatility and mean reversion remain illustrative.

## 2. Contract definition

Let:

- `M` be a delivery month;
- `D_M` be its eligible delivery or exercise days;
- `O_M` be the fixing observations used to set the strike for `M`, normally dates in `M-1`;
- `Q_t^M` be the month-ahead market quote observed at time `t` for delivery in `M`;
- `a_t` be the contractual weight of fixing observation `t`;
- `K_M` be the strike applied to exercise in `M`.

The general reset formula is

```text
raw_K_M = sum(t in O_M, a_t * Q_t^M) / sum(t in O_M, a_t)
K_M     = rounding(cap(floor(alpha * raw_K_M + beta)))
```

`alpha`, `beta`, floors, caps and rounding are optional contractual terms. The simple case is
`alpha = 1`, `beta = 0`, with no floor, cap or rounding.

For a call swing, where the holder sells gas, the exercise cashflow before operational costs
is

```text
volume_i * (S_i - K_M)
```

For a put swing, where the holder buys gas, it is

```text
volume_i * (K_M - S_i)
```

This repository's put swing is a purchase obligation when the terminal volume is mandatory;
it is not a vanilla put with `max(K-S, 0)`. Optional total volume must remain an explicit
contract choice.

### 2.1 Commercial inputs that must not be inferred

The implementation must require or explicitly default all of the following:

1. Call or put direction.
2. Delivery dates and eligible exercise dates.
3. Daily maximum volume.
4. Minimum and maximum total volume, and whether these apply globally or separately by month.
5. Exact index name and source.
6. Whether `Q_t^M` is a futures settlement, forward assessment, published index or another
   observation.
7. Fixing dates, business-day calendar, time zone and publication time.
8. Equal, calendar-day, business-day or other contractual fixing weights.
9. Treatment of missing, late or corrected publications.
10. Multiplier, additive spread, floor, cap and rounding.
11. Exercise decision time relative to publication of that day's fixing.
12. Cash settlement date or invoicing convention.
13. Initial historical fixings when valuation occurs during or after a fixing window.

"Month-ahead" is not enough to determine these terms. In the repository,
`front_month_start(quote_date)` means the next calendar month; this is a project convention,
not evidence that a real contract uses the same published index or roll convention.

## 3. Information chronology

Reset observations and exercise dates are separate calendars.

For delivery month `M`:

1. Observations in `O_M` accumulate before delivery.
2. Once the final observation is published, `K_M` is fixed.
3. Exercise in `D_M` uses that fixed value.
4. While `K_M` is being used in month `M`, observations in `O_(M+1)` normally accumulate to
   set the next strike.

The fourth point creates the overlap that requires two reset variables during an active
month.

### 3.1 Valuation-date cases

For every delivery month, the fixing status as of the valuation timestamp must be one of:

- **fully fixed:** all observations are historical and `K_M` is a known input;
- **partially fixed:** the realised weighted sum and realised weight are inputs, while the
  remaining observations are stochastic;
- **wholly future:** no fixing has occurred and the complete window is stochastic.

The first delivery month's fixing window may be before the exercise period and still be
inside the valuation simulation. There is no general requirement to delay delivery by one
month. A supplied initial strike is required only when the relevant historical fixings are
already known but unavailable to the model, or when the term sheet explicitly fixes it.

No exercise decision may use an observation published later that day. The model must order
publication, exercise and settlement events explicitly to prevent look-ahead.

## 4. Current-engine boundary

The present engine has the following relevant properties:

- the dynamic-programming state is `(time i, price node j, volume state l)`;
- `value_call_swing` and `value_put_swing` accept one scalar strike;
- that strike is broadcast into the per-day scalar arrays `w_cost[i]` or `i_cost[i]`;
- the kernel's immediate exercise cashflow therefore cannot depend on a path-specific strike;
- `q[i,j]` is the unconditional probability of reaching node `j` from the valuation root;
- `p_u`, `p_m` and `p_d` contain the conditional one-step transitions;
- `exp(x[i,j])` represents the model's price on date `i`, not automatically the market quote
  for a future delivery month.

Consequently, the current fixed-strike wrapper cannot value a stochastic reset without a new
valuation layer. A pre-declared deterministic monthly strike schedule is different: it can be
implemented by filling the existing daily cost arrays non-uniformly and needs no reset state.

## 5. Model-consistent month-ahead quote

The reset must use a model value for the quoted delivery-month instrument, not the spot or
daily price at the observation node.

Let `P_i(j,k)` be the current lattice transition probability from node `j` at date `i` to node
`k` at date `i+1`, and let

```text
S_u(k) = exp(x[u,k]).
```

For each future delivery date `u`, define the conditional expected delivery price

```text
H_u(u,k) = S_u(k)
H_u(i,j) = sum(k, P_i(j,k) * H_u(i+1,k)),  i < u.
```

For a uniformly settled monthly futures-style quote,

```text
Q_i^M(j) = sum(u in D_M, b_u * H_u(i,j)) / sum(u in D_M, b_u),
```

where `b_u` are delivery weights. If the contractual quote is a forward whose delivery cash
flows require deterministic discount weighting, the numerator and denominator must carry
the corresponding discount factors. Futures and forwards must not be treated as identical
when their settlement conventions make that distinction material.

This discrete conditional-expectation calculation is preferred to inserting a continuous-time
OU projection formula into the lattice. It automatically respects:

- the transition law actually implemented;
- the per-date forward-distortion shift;
- finite tree boundaries;
- time-varying volatility and mean reversion;
- non-flat daily delivery weights.

`q` must not be used as a conditional transition. It is suitable for unconditional averaging
from the root, but it loses the information needed to continue from an individual node.

### 5.1 Projection acceptance tests

For every tested delivery month and observation date:

1. **One-step recursion**

   ```text
   Q_i^M(j) = sum(k, P_i(j,k) * Q_(i+1)^M(k))
   ```

   whenever no delivery settlement occurs between the two observations.

2. **Tower identity**

   ```text
   sum(j, q[i,j] * Q_i^M(j)) = Q_0^M.
   ```

3. **Initial-curve fit:** the root value reproduces the delivery-weighted input forward for
   `M` to the declared numerical tolerance.
4. **Limits:** repeat at zero volatility, zero mean reversion, near-zero mean reversion and
   several positive mean-reversion values.
5. **Boundary check:** measure occupation of the outer price nodes and repeat with a wider
   tree. A tower identity can pass on an inadequately narrow tree.

These checks replace the earlier proposal to derive a special closed form as the first task.
A continuous-time formula can remain an independent diagnostic, but the exact discrete
recursion is authoritative for the implemented lattice.

## 6. Sufficient state representation

### 6.1 Single isolated reset

If there is one fixing window followed by one delivery block and no simultaneous accumulation
for a later strike, an exact benchmark can use

```text
V_i(j, l, r),
```

where `r` is the running weighted sum or average. Once fixed, the same coordinate may be
frozen and interpreted as the delivery strike.

### 6.2 Recurring monthly resets

For the intended repeating structure, the active-month state is

```text
V_i(j, l, k, r),
```

where:

- `j` is the price node;
- `l` is cumulative exercised volume or remaining volume entitlement;
- `k` is the fixed strike applicable to the current delivery month;
- `r` is the running statistic for the next delivery month's strike.

The elapsed fixing weight is deterministic and does not require another state. It is safer to
carry a weighted sum and cumulative deterministic weight than to repeatedly average an
average:

```text
R_new = R_old + a_i * Q_i^(M+1)(j)
W_new = W_old + a_i,
A_new = R_new / W_new.
```

`W_new` is determined by the calendar. For a partially fixed month, initialise `R_old` and
`W_old` from supplied historical observations.

At the reset boundary:

```text
K_(M+1) = reset_formula(R_final / W_final)
R_next  = known historical seed or zero for the following fixing window.
```

The old `K_M` can then be discarded. Between boundaries, `K_M` remains fixed while `R_next`
evolves.

### 6.3 Exercise recursion

At a call-swing state, for an allowed exercise quantity `d`:

```text
exercise_value = DF_i * d * v_step * (S_i(j) - K_M - variable_cost_i)
                 + conditional continuation after moving l by d.
```

For a put swing:

```text
exercise_value = DF_i * d * v_step * (K_M - S_i(j) - variable_cost_i)
                 + conditional continuation after moving l by d.
```

Idle and all permitted partial daily quantities remain alternatives. Global and monthly
volume constraints must be enforced according to the term sheet. The decision uses only
information available at that timestamp.

### 6.4 Convexity and the conditional current-strike reduction

For fixed `(i, j, l, r)` within delivery month `M`, the value is convex in the already-fixed
current strike `k`, provided that:

- the feasible exercise set does not depend on `k`;
- the current strike enters the immediate cashflow only as a linear amount per exercised
  unit; and
- the value at the `M` to `M+1` rollover is independent of the discarded strike `K_M`.

For a call, suppressing the fixed accumulator coordinate from the notation, the recursion is

```text
V_i(j,l;k) = max over feasible d of {
    DF_i * d * v_step * (S_i(j) - k - variable_cost_i)
    + E[V_(i+1)(j', l-d; k) | j]
}.
```

The rollover value is constant in `k`. Backward induction then establishes convexity because
each action value is an affine immediate payoff plus an expectation of convex continuation
values, and the maximum of convex functions is convex. The put result is symmetric.

Convexity does not by itself make the value affine and does not establish how many strike
knots are required.

The stronger affine result holds only when every feasible policy has the same discounted
current-month exercise volume:

```text
discounted_volume(policy) = sum(i in D_M, DF_i * volume_i(policy)).
```

If this amount is the same constant `E_M` for all feasible policies, then

```text
call: V(k) = V(k0) - E_M * (k - k0)
put:  V(k) = V(k0) + E_M * (k - k0).
```

Examples satisfying the condition include zero interest rates, a common monthly cash
settlement discount factor with a fixed monthly physical volume, or another settlement
convention that makes discounted volume schedule-independent. A mandatory physical monthly
volume alone is not sufficient when exercise-day discount factors differ. In that case,
different exercise schedules can have different slopes in `k`; the value remains convex but
may be piecewise affine, and the optimal policy can change with the strike.

The originally reported one-month numerical run was at the engine's default zero rate only,
and was not evidence for the general positive-rate case. Committed as four fixtures in
`tests/test_reconciliation.py` (`test_zero_rate_mandatory_strike_affinity`,
`test_common_settlement_strike_affinity`, `test_daily_discounting_can_break_affinity`,
`test_optional_volume_strike_convexity`) against the existing call-swing engine directly --
no reset-swing code exists yet to test, but the current-strike k-slice this section is about
is exactly what a fixed scalar strike over one window already computes. The zero-rate and a
patched common-settlement (flat, non-unit discount factor) case both reproduce the exact
affine slope `-DF * total_volume` to the declared tolerance; the same contract under the
engine's actual day-varying discount convention shows genuine curvature (slope spread > 100
across the tested strikes, still convex, no chord violation) -- confirming the qualification
above is not just a theoretical caveat but a reproducible property of this engine's own
discounting convention.

Consequently, `n_K` may be collapsed to one reference evaluation plus the analytic slope only
after the schedule-independent discounted-volume condition has been checked. Otherwise the
implementation must retain a strike grid, use an exact upper-envelope representation, or use
another approximation whose error is demonstrated by convergence tests. Convexity can inform
adaptive knot placement, but it does not justify a particular knot count or location.

## 7. Numerical architecture

### 7.1 Exact benchmark solver

Build a deliberately small reference dynamic programme with explicit grids for `K_M` and the
next reset accumulator. Its purpose is correctness, not production scale.

Requirements:

- use two adjacent time slices for value wherever possible;
- store strategy compactly only when policy reconstruction is required;
- construct reachable, date-dependent strike and accumulator domains;
- interpolate continuation value between adjacent reset-grid points;
- define boundary behaviour explicitly and never extrapolate silently;
- measure convergence separately in the price, volume and accumulator grids, and in the
  strike grid whenever the analytic reduction does not apply;
- report memory and runtime from actual dimensions.

The state count during an active month grows approximately as

```text
n_price * n_volume * n_K * n_R.
```

The previous design counted only `n_R`. With 40 strike buckets and 40 accumulator buckets,
the naive multiplier is about 1,600, before allowing for time or policy storage. Section 6.4
permits `n_K` to be removed only when discounted current-month exercise volume is
schedule-independent. In that qualified case, the incremental reset multiplier is `n_R`
alone and a dense exact solver may be practical. Otherwise `n_K` remains a numerical
dimension and must be sized from a separate convergence ladder. Whether the exact solver is
usable for a realistic annual swing is therefore a measured result, not a design assumption.

### 7.2 Conditional production solver: regression Monte Carlo

When the exact solver exceeds the declared production resource budget, the recommended
realistic-size approximation is least-squares Monte Carlo or another regression
dynamic-programming method:

1. Simulate price paths under the same stated risk-neutral dynamics used for valuation.
2. Calculate `Q_t^M` on each path consistently with the delivery-month instrument.
3. Build historical, partial and future reset strikes pathwise using only observations
   available by each decision time.
4. Work backwards over exercise dates for every discrete cumulative-volume state.
5. Regress continuation value on the current Markov information, including price factor(s),
   current strike and next reset accumulator. Time and elapsed fixing weight are deterministic.
6. Optimise over idle and every permitted daily exercise quantity.
7. Evaluate the resulting policy on fresh out-of-sample paths.
8. Report standard errors and repeat across path count, regression basis and random seeds.

In-sample regression value is not the acceptance price. The reported lower-bound policy value
must come from independent paths. A perfect-foresight pathwise optimum may be reported only as
an upper bound and must never be labelled Monte Carlo value.

### 7.3 Relationship between solvers

The exact small solver is the independent benchmark for the production method. The production
method is accepted only when its out-of-sample confidence interval agrees with exact values on
small contracts across more than one curve, volatility, mean reversion and reset structure.

## 8. Point-reset prototype

A useful intermediate product replaces the monthly fixing average with one observation:

```text
K_M = Q_tau(M)^M,
```

where `tau(M)` is a specified fixing timestamp before delivery month `M`.

This removes the running-average state but not the current-strike state. For recurring resets,
the active state is still

```text
V_i(j, l, k).
```

At a reset boundary, the next strike becomes a deterministic function of the reset-date price
node and is carried through the next delivery month.

It is not generally valid to:

1. price an independent fixed-strike monthly swing at every reset node;
2. weight those values by unconditional `q[i,j]`; and
3. add the monthly results.

That procedure loses conditional continuation and fails when total-volume constraints couple
months. It can work only for a single isolated delivery month with no exercise before reset,
after constructing the correct conditional forward curve and discounting from each reset node.

The point-reset prototype is still recommended as Phase 1 because it tests state-dependent
strike handling without the second continuous reset state.

## 9. Valuation outputs and economic definitions

### 9.1 Primary result

The primary model output is total present value in EUR. All other metrics are derived and must
reconcile to that PV.

Also report:

- expected exercised volume by day and month;
- distribution of monthly reset strikes;
- expected strike conditional on exercise;
- exercise-price and strike correlation diagnostics;
- minimum/maximum-volume utilisation;
- numerical convergence status;
- Monte Carlo standard error when applicable.

### 9.2 Deterministic, intrinsic and extrinsic values

Use the following definitions:

- **Full PV:** stochastic prices and stochastic resets, with optimal adapted exercise.
- **Deterministic PV:** zero stochastic volatility, with reset strikes generated from the
  deterministic forward projection and the same optimisation constraints.
- **Extrinsic PV:** `Full PV - Deterministic PV`.
- **Flat PV:** value of a separately specified non-optimised exercise schedule. For mandatory
  volume this may be an even feasible schedule. For optional total volume there is no unique
  flat benchmark unless the reporting convention specifies one.
- **Profile/intrinsic uplift:** `Deterministic PV - Flat PV`, only when Flat PV is defined.

The current scalar-strike `daily_arithmetic_flat_metric` cannot be reused unchanged. A random
monthly strike is correlated with delivery prices, discount factors and potentially exercised
volume.

For optional-volume products, value per expected exercised MWh can become unstable or
misleading when expected volume is small. Report total EUR and expected volume separately;
show a per-MWh number only with its denominator and convention.

### 9.3 Hedge sensitivities

The existing expected-exercise `delta` captures the physical price leg but not the sensitivity
of the stochastic reset strike. The model must provide at least:

1. **Total quoted-curve delta:** bump each supplied market-curve vertex, rebuild daily forwards,
   month-ahead projections and reset strikes, then re-optimise.
2. **Physical-leg diagnostic:** bump delivery prices while freezing reset strikes.
3. **Index-leg diagnostic:** bump the index projection entering the strikes while freezing the
   delivery-price leg.

The total sensitivity is authoritative. The two diagnostics are an attribution and depend on
the chosen bump construction.

The existing identity

```text
sum(DF_i * delta_i * F_i) = PV
```

must not be assumed universally. An Euler scaling identity can hold when the entire contract,
including the reset formula, is homogeneous in the input curve and has no fixed cash terms.
Additive strike spreads, fixed EUR/MWh costs, floors, caps and rounding generally introduce a
residual. Finite-difference agreement is the primary Greek test.

## 10. Verification and acceptance plan

### 10.1 Contract and chronology tests

- fixing and delivery calendars are generated exactly from a miniature stated term sheet;
- a fully fixed strike is reproduced without stochastic reset state;
- partial fixings combine the historical sum with future observations correctly;
- a fixing published after exercise cannot influence that exercise decision;
- global and monthly volume constraints are distinguished and tested separately;
- missing index observations follow the declared disruption rule or raise.

### 10.2 Degenerate and hand-computable tests

- zero volatility gives the hand-computable deterministic reset and PV;
- a deterministic pre-declared monthly schedule reproduces the existing fixed-strike engine
  when all monthly strikes are equal;
- a one-observation reset equals the fixed-strike value only conditional on the realised reset
  node, not unconditionally before fixing;
- zero daily exercise capacity produces zero optional exercise value;
- mandatory full exercise reproduces the corresponding indexed linear cashflow;
- a one-day, two-price-node example is enumerated by hand;
- at zero rates or with a common settlement discount factor, a fixed monthly volume produces
  the analytic constant strike slope from Section 6.4;
- with daily discounting, a two-date mandatory-volume counterexample remains convex but can
  be non-affine and can change exercise timing as the strike changes;
- optional volume is convex in the current strike, with no chord violations on the tested
  strike grid.

### 10.3 Independent exact tests

- exhaustive non-anticipating enumeration of every policy on a two-to-four-date scenario tree;
- comparison of the new exact solver with that enumeration;
- comparison of the out-of-sample regression-Monte-Carlo policy with the exact solver on small
  contracts;
- a perfect-foresight value retained only as a labelled upper bound.

### 10.4 Numerical tests

- price-tree boundary widening;
- cumulative-volume grid convergence;
- strike-grid convergence;
- running-accumulator-grid convergence;
- interpolation-domain coverage and boundary occupation;
- path-count, seed and regression-basis convergence;
- finite and non-negative transition probabilities;
- no silent acceptance of NaN or infinite values.

### 10.5 Hedge and reconciliation tests

- central finite-difference deltas match reported total deltas;
- total delta includes both physical and reset-index channels;
- physical plus index attribution reconciles under the declared frozen-leg convention;
- global scaling identity is tested only for homogeneous cases;
- fixed spread/cost cases reproduce the predicted non-homogeneous residual;
- discounted cashflows reconstructed from the simulated or exact policy equal reported PV.

### 10.6 Regression tests

- every existing fixed-strike swing and storage test remains unchanged and passes;
- the existing public `run_valuation` API continues to reject unknown product types;
- monthly-reset products use a separate entry point until their output conventions are stable;
- deterministic monthly strike schedules do not enter the stochastic-reset solver.

## 11. Implementation stages

### Phase 0: freeze the product specification

Deliverables:

- `ResetSwingTerms` data structure with calendars, index, formula, volume constraints and
  historical-fixing state;
- strict validation and audit serialisation;
- tests for all chronology cases;
- explicit statement whether the first implementation is call, put or both.

Exit condition: a term sheet maps to one unambiguous event schedule and reset formula.

### Phase 1: node-specific forward projection and point reset

Suggested files:

```text
reset_forward.py
reset_swing_exact.py
tests/test_reset_forward.py
tests/test_reset_swing_point.py
```

Deliverables:

- exact lattice-conditional month-ahead quote table;
- one-step, tower and root-curve tests;
- state-dependent current-strike recursion for the point-reset product;
- exact miniature scenario-tree oracle;
- cash-PV output only, plus essential diagnostics.

Exit condition: the point-reset solver matches exhaustive enumeration and all projection
identities.

### Phase 2: averaged-reset exact benchmark

Deliverables:

- simultaneous current-strike and next-accumulator states;
- partial historical fixing support;
- reset-boundary rotation;
- interpolation and boundary diagnostics;
- a convergence ladder for `n_R`, plus either proof and testing of the analytic `n_K`
  reduction or a separate `n_K` convergence ladder;
- measured memory and runtime report.

Exit condition: all hand-computable and exact-tree cases pass, with convergence inside a
predeclared tolerance.

### Phase 3: realistic-size production solver

Suggested files:

```text
reset_swing_lsmc.py
tests/test_reset_swing_lsmc.py
```

Deliverables:

- risk-neutral path generation consistent with the chosen model;
- backward regression for every discrete volume state;
- out-of-sample policy valuation;
- confidence intervals and convergence report;
- exact-benchmark reconciliation suite.

Exit condition: independent-path values agree with exact small-case results within both the
declared numerical tolerance and Monte Carlo confidence interval.

### Phase 4: Greeks, reporting and application integration

Deliverables:

- total quote-bucket bump deltas;
- physical/index diagnostic attribution;
- revised PV, deterministic, extrinsic and optional flat metrics;
- audit export containing reset schedule, historical fixings, model parameters, convergence
  evidence and solver version;
- Streamlit inputs and validation;
- performance budget and clear refusal when a requested run exceeds it.

Exit condition: the application exposes no metric whose economic definition is absent from
the audit output and tests.

## 12. Minimum release gate

The monthly-reset swing must not be labelled implemented or reliable until all of the following
are true:

1. The contract terms and event ordering are explicit.
2. Node-specific month-ahead quotes pass conditional and unconditional projection identities.
3. Recurring resets retain both current strike and next accumulator.
4. Exercise is non-anticipating.
5. A small exact implementation matches exhaustive enumeration.
6. If the production solver is regression Monte Carlo (sec.1's own fork): the policy is
   valued out of sample and reconciles to the exact solver. If exact DP is itself the
   selected production solver (sec.1: practical "from measured state sizes, convergence,
   memory and runtime," not assumed) this item is satisfied by construction -- there is no
   separate out-of-sample policy to reconcile against, since the exact solver IS the
   benchmark. (2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-11 found the original,
   unconditional wording of this item inconsistent with sec.1's own explicit DP-or-Monte-Carlo
   fork.)
7. All numerical dimensions have measured convergence.
8. Reported total deltas pass bump-and-revalue checks and include the reset leg.
9. Existing fixed-strike and storage regressions remain green.
10. Outputs state plainly that the Q dynamics are illustrative until market calibration is
    established.

## 13. Implementation and status log

**This section's original title was "Recommended immediate next action," and its own opening
paragraph (below, kept verbatim) recommended exactly the sequence the entries under it then
carried out -- fulfilled long ago, and stale as a "next action" by the time of the 2026-09-14
INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-11 finding that named it directly.** What follows is
the running development and status log for everything built against sections 1-12's
specification, each entry dated as it landed; read [`docs/STATUS.md`](STATUS.md) first for a
concise summary of what is currently true.

Original recommendation (fulfilled, first entry below): implement Phase 0 and the conditional
month-ahead projection from Phase 1 before modifying `storage_kernels.py`. The projection can
be tested independently against the current tree and is required by every later solver. After
it passes, build the point-reset exact benchmark. Only then choose the averaged-reset
production implementation on measured state sizes and runtime, rather than assuming the dense
lattice will be practical.

**Done, 2026-09-14, no real term sheet -- an explicit prototype convention per sec.2.1/sec.14
instead (call-only, month-end point reset, a model-internal index, global not per-month
volume, no historical fixings):**

- `reset_forward.py` -- the conditional month-ahead projection, exactly as this section
  specifies (backward induction through the lattice's own `p_u`/`p_m`/`p_d`, no derived
  closed form). `storage_kernels.py` untouched; `storage_model.py::Storage.build()` gained
  three lines storing `p_u`/`p_m`/`p_d` on the instance (previously local to `build_tree`'s
  caller) so this module reads the exact transition law an actual valuation used, not a
  second copy of it. `tests/test_reset_forward.py`: one-step recursion and the tower identity
  at kappa in {0, 1e-6, 0.3, 1.0, 4.0}, root value against the input forward, a kappa=0
  sanity check, and the boundary-occupancy check sec.5.1 explicitly asks for (a tower
  identity passing on a tree too narrow to trust, demonstrated directly, not just avoided).
- `reset_terms.py` -- `ResetSwingTerms` and `build_reset_schedule`. Found and closed one
  real gap while building this: a partial FINAL month would need a strike fixed from a date
  after its own exercise had already stopped, which is refused outright (a partial FIRST
  month is fine -- that month's settlement reference is unaffected by when exercise starts).
  `tests/test_reset_terms.py` covers both, plus the fixing-strictly-precedes-exercise
  property and refusing a fixing that would fall before val_date (no historical-fixing
  input in this release).
- `reset_swing_exact.py::value_point_reset_call_swing` -- the point-reset exact benchmark
  itself (sec.7.1, sec.8), pure Python/NumPy. Uses sec.6.4's finding that a point reset's
  `k` is an exact function of a single lattice node -- not a continuous quantity needing
  its own interpolated grid -- so the state stays `(i, j, l, j_fix)` with `j_fix` an
  index, not a separate `n_K`-sized discretisation. `tests/test_reset_swing_point.py`
  pins it at near-zero volatility, where the model is deterministic and the answer is
  hand-computable: a single-month mandatory case, a two-month case whose global (not
  per-month) quota must be deferred across the month rotation for the right answer,
  and an optional-volume case that must touch zero-margin days not at all. All three
  matched their hand-computed values to the stated tolerance on the first correct run
  of the algorithm; the two bugs found before that (curve silently hardcoded instead of
  taken from the caller; a dead conditional left over from an earlier draft of the
  within-month loop) were caught by review before any test ran, not by a failing test.

**Done, same day, follow-through on what was flagged above as not yet done:**

- **Real volatility, not just the deterministic limit.** Every test above used near-zero
  volatility because that is what makes the answer hand-computable -- none of it exercised
  the genuinely stochastic core. `tests/test_reset_swing_stochastic.py` closes that:
  value rises with volatility for optional exercise (0.1 -> 1.2 moves PV from 2,628 to
  31,593 on a one-month case -- monotonic and not marginal, the standard option-pricing
  sanity check) and, more importantly, an independently-coded reference implementation
  (plain nested Python loops over every `(j,l,d)`, no NumPy shift/broadcast shared with
  `reset_swing_exact.py`) matches the vectorised solver exactly (`abs` tolerance 1e-6) across
  four real-volatility, real-mean-reversion, mandatory/optional/partial-volume
  configurations -- sec.10.3's independent-exact-test requirement, done at parameters that
  actually exercise the stochastic recursion rather than the sec.10.2 degenerate case.
- **A real bug, found by the convergence ladder, not by inspection.** Widening `n_p` from 6
  to 28 converges cleanly (0.023% -> 0.0000% successive relative moves). Refining `v_step`
  from 1000 to 125 MWh does not move the price at all once the daily rate is already exact
  on the coarser grid -- a genuine finding (the optimal policy here is bang-bang, so finer
  granularity adds no new achievable choice), not a weak test. But the FIRST attempt at
  that same ladder crashed on `v_step=2000`: `int(round(1000/2000))` rounds the 0.5 clips/day
  down to **zero** under Python's banker's rounding, silently pricing a swing that could
  never exercise anything as if that were the requested contract -- the identical defect
  class `normalise_storage_contract` exists to catch in the fixed-strike engine, now present
  in code barely a day old. Fixed the same way: `ResetSwingTerms.__post_init__` now calls
  `storage_model._grid_representable` on `daily_max_mwh` and on both global volume bounds,
  refusing construction rather than silently rounding. `tests/test_reset_terms.py` pins
  both the original defect and the fix.
- **Runtime at deal scale**, machine-indicative only per this repository's own standing
  caveat about timing claims: a full 12-month deal at `n_p=15`, `n_l=41` (a clip fine
  enough that halving it again does not move the answer, per the finding above) runs in
  under a second; `n_p=20` or `n_l=81` individually, under three. The deliberately
  unoptimised pure-Python/NumPy choice (sec.7.1: "small reference, not production scale")
  is comfortably practical at this scale without needing a Numba rewrite yet.

**Done, same day, closing the gap named directly above:**

- **sec.10.3's brute-force enumeration, in the strict sense this time.**
  `tests/test_reset_swing_exhaustive.py` builds a tiny hand-constructed scenario (three price
  nodes, one fixing, two exercise days, one mandatory clip) where waiting on day one forces
  day two, reducing "every non-anticipating policy" to one binary stopping choice per
  `(j_fix, j_day1)` pair -- 3x3=9 pairs, `2**9`=512 policies, each one's exact expected
  discounted value computed by literal path-probability summation, independent of (not
  reusing) `reset_swing_exact.py`'s own transition helpers. The DP matches the best of all
  512 to `1e-9`. This is the stronger check the previous entry explicitly said was still
  missing, not the differently-coded-reference one that already existed.
- **Hedge sensitivities.** `reset_swing_exact.compute_deltas` implements sec.9.3's three
  central-finite-difference measures. `value_point_reset_call_swing` gained an optional
  `quotes=` override precisely so the physical- and index-leg diagnostics can freeze one
  lattice's strikes while bumping the other's spot, or vice versa. On a six-month
  illustrative deal the two legs are large and nearly cancel (+48,961 physical, -43,738
  index, net total 5,224 EUR per EUR/MWh) -- the point of an indexed strike at all, now a
  pinned property (`tests/test_reset_swing_deltas.py`) rather than an assumption. `total`
  is authoritative; the legs are not asserted to sum to it exactly, matching sec.9.3's own
  caveat that the split is only additive to first order.
- **Somewhere a person would actually use it.** `MonthlyResetSwing.ipynb`, in the same
  style as `Storage_30_60.ipynb` (staleness guard, no stored output committed, a fresh-
  process execution test -- `test_monthly_reset_swing_notebook_executes_clean` in
  `tests/test_reconciliation.py`): term sheet, curve, PV, and the three-way delta print-out
  with the same "small total does not mean small risk" caveat as above, stated inline
  rather than left for a reader to discover.

**Release 1B: the averaged-reset exact benchmark, and a debugging story worth recording.**

- `reset_swing_averaged.py::value_averaged_reset_call_swing` -- sec.7.1's genuinely continuous
  running-average state, `(i, j, l, r)`, `r` on a discretised, linearly-interpolated grid
  (`accumulate_step`). Month-chaining collapses each month's own fresh outgoing accumulator
  to a representative slice before handing it to the preceding month (`_collapse_fresh_axis`)
  rather than carrying an ever-growing stack of axes -- see the module's own docstring for
  why that collapse is safe (the same "value at a not-yet-accumulated axis cannot depend on
  the bucket" boundary property Release 1A already uses for the pre-deal window).
- **Five real bugs, found and fixed before anything matched.** In order: (1) a diagonal
  collapse (`results[k][:,:,k]`) that wrongly conflated a month's own incoming strike bucket
  with its unrelated outgoing accumulator bucket; (2) a missing pre-deal accumulation step
  (month 1's strike was read off an arbitrary bucket instead of actually accumulated from
  `val_date`); (3) a 2D-to-3D broadcast that should have been an unconditional copy once the
  terminal was already stacked; (4) and (5) both in the sanity check meant to catch exactly
  this class of bug -- `_collapse_fresh_axis`'s spread computed the global max/min across every
  `(j,l,r)` state instead of each `(j,l)`'s own spread across `r`, so it was comparing
  unrelated economic states and always reported a huge, meaningless number, briefly rejecting
  an already-correct computation. Diagnosed throughout by the same method: hypothesise the
  expected property (a vol-continuity check -- averaged-reset PV should shrink toward
  point-reset's as vol -> 0 -- caught (1) and (2) directly), construct a minimal isolatable
  case, fix, re-verify.
- **A sixth apparent bug that was not one.** After all five fixes, a 2-month shaped-curve
  vol-continuity check still would not converge to point-reset's near-zero answer at low vol
  -- it plateaued around 5,600 EUR instead. Three successive brute-force cross-checks were
  built to isolate it, and each first *disagreed* with the DP before the actual cause was
  found to be the brute-force script's own scenario setup, not the DP: non-consecutive
  accumulation dates (the single-step-per-date propagation both `_run_month`'s accumulator
  loop and `_run_accumulation_only` use assumes consecutive calendar days), a pre-deal window
  not starting at `val_date + 1` (so the backward propagation stopped one or more steps short
  of `val_date`), and a hand-built month whose `fixing_date` did not equal the literal last
  day of the accumulation window feeding it. Once a scenario respected all three (exactly what
  `build_reset_schedule` already guarantees end to end), DP and brute force matched to ~1e-12.
  The real explanation for the plateau: the value-vs-K surface has a genuine kink at the
  exercise boundary, and linear interpolation across a kink converges only at first order,
  O(1/n_r) -- confirmed by a 4x grid refinement cutting the residual by very close to 4x, both
  in this investigation and as a pinned regression
  (`test_finer_r_grid_moves_averaged_reset_toward_point_reset_at_low_vol`). A coarse or wide
  `r_grid` is the single biggest source of apparent disagreement with point-reset; it is not a
  sign of a logic error, but it does mean `n_r` needs to be chosen generously (hundreds to a
  few thousand, bracketing the curve tightly) rather than assumed adequate at a small default.
- **`tests/test_reset_swing_averaged.py`** pins all of this: the literal brute-force
  enumeration (two accumulation days, two exercise days, matching to `abs=1e-6` on top of a
  fine local `r_grid`), the accumulator's zero-weight boundary property and its running-average
  arithmetic in isolation, the single-observation-window degenerate match to point-reset
  (`rel=1e-4`, using the real `ResetSwingTerms`/`build_reset_schedule` pipeline end to end),
  and the O(1/n_r) convergence direction and rate on the case that originally looked broken.

**Done, same day, closing all three gaps named directly above.**

- **A genuine two-month brute-force cross-check**, extending the single-month one: one
  pre-deal day fixes month A's strike, month A's own single exercise day is ALSO the sole day
  accumulating into month B's strike, one mandatory clip across both months. This is the first
  literal enumeration to exercise `_run_month(accumulate=True)` and `_collapse_fresh_axis` --
  the exact code path responsible for three of the five real bugs found earlier -- and it
  surfaced a genuine, previously-unknown risk before confirming correctness: the first attempt
  at this test bracketed `r_grid` only around month B's own projected-quote range, and month
  A's (materially probable, not tail) higher range silently CLAMPED rather than raised,
  producing a confidently wrong number with no exception at all. Once `r_grid` brackets both
  months' ranges, DP and brute force agree to ~1e-14.
- **`value_averaged_reset_call_swing` now checks that risk itself** rather than leaving it to
  the caller to discover: before running anything, it verifies `r_lo`/`r_hi` bracket every
  delivery month's own projected-quote range, weighted by `lattice["q"]` (the unconditional
  probability of each date/node pair) rather than a plain min/max over every lattice node --
  an unweighted check turned out to be useless in practice, since a trinomial lattice
  truncated to `n_p` steps genuinely piles up non-negligible probability at its own edge nodes
  once enough days have elapsed relative to `n_p` (an n_p=8, ~4-month lattice showed 3-4%
  probability sitting AT the edge, nowhere near a negligible tail), which an r_grid could
  never be wide enough to fully absorb without destroying the interpolation accuracy a tight
  grid exists to provide. Raises with the observed range and a clear explanation instead of
  silently mispricing.
- **Hedge sensitivities for the averaged reset.** `reset_swing_averaged.compute_deltas`, same
  three central-finite-difference measures and the same `quotes=` freeze-one-leg pattern as
  Release 1A's version (added to `value_averaged_reset_call_swing` for exactly this). On a
  two-month illustrative deal with a real curve step between the months, the legs are large
  and mostly offsetting (physical +9,911.63, index -9,895.16, `n_r=60`, recomputed 2026-09-14
  against the corrected fixing window -- the pre-fix figure quoted here originally, +9,885.12 /
  -9,848.41, used the wrong window and is superseded; the exact split varies with `n_r` since
  interpolation itself is not delta-neutral), same qualitative behaviour as Release 1A.
  `tests/test_reset_swing_averaged_deltas.py` pins the sign, the
  small-net-total-relative-to-the-legs property, and the mandatory-vs-optional physical-leg
  ordering, with tolerances loosened relative to Release 1A's own delta tests to reflect the
  added grid-interpolation noise (checked against leg size, not against `total` itself, which
  is a near-cancellation and so amplifies relative noise).
- **Wired into `MonthlyResetSwing.ipynb`.** A new section 5, using a smaller, separate term
  sheet (2 delivery months, `n_p=8`) purely so the notebook keeps running in a few seconds --
  the same 6-month, `n_p=15` term sheet the point-reset sections use would make the averaged
  reset's `O(n_r)`-per-incoming-bucket cost impractical for routine execution. Prices both
  reset conventions on the same smaller deal side by side, and reports the averaged reset's
  own three-way hedge attribution with the same "small total is not small risk" caveat as the
  point-reset section.  `tests/test_reconciliation.py::test_monthly_reset_swing_notebook_executes_clean`
  now also asserts the averaged-reset PV is finite and its deltas carry the expected signs.

**Still not done:** everything in sec.14.7's fixture list beyond what the tests above cover.
255 tests pass in the full repository suite (249 before this + 6: 2 in
`test_reset_swing_averaged.py`, 4 in `test_reset_swing_averaged_deltas.py`).

**Release 2: measured, then a Numba kernel, "stay in DP" per sec.14.1's own decision rule.**
Measured first: on the notebook's own 6-month, `n_p=15` term sheet, `n_r=20` -- already far
too coarse to trust, per the O(1/n_r) finding above -- took ~29s; extrapolating the measured
scaling to an `n_r` worth trusting reached on the order of an hour. Two numpy-vectorised
rewrites of `accumulate_step`'s interpolation (uniform-grid index arithmetic in place of
`np.interp`, one via `np.take_along_axis`, one via a width-loop with flat-indexed gathers)
were tried and reverted: both verified numerically identical to the original, and both gave
a genuine 2-5x speedup for large-`n_l` deals but were 1.3-5x WORSE for small-`n_l`/large-`n_r`
deals -- numpy per-call dispatch overhead dominating one regime, `np.interp`'s own tight C
loop winning the other, no shape-independent win. `reset_swing_kernels.py` (new module,
mirroring `storage_kernels.py`'s own established separate-file-for-Numba-caching pattern)
adds `run_month_accumulate_core`: the WHOLE per-k month recursion as one Numba-compiled
unit -- explicit nested loops, matching `storage_kernels.py`'s own style, not vectorised numpy
-- removing per-call dispatch overhead categorically rather than trading one deal shape's
speed for another's. Verified two ways before being wired into `_run_month`: a direct
comparison against `_run_month_accumulate_reference` (the old Python loop, kept as a named,
independent reference specifically so this comparison is not circular once `_run_month`
itself calls the kernel) across six randomised scenarios in
`tests/test_reset_swing_kernels.py`, matching to ~1e-10; then the full existing brute-force
suite in `tests/test_reset_swing_averaged.py`, unchanged, still passing with `_run_month` now
kernel-backed. One real trap found and closed structurally (not just patched in the one test
that hit it): a JIT "warm-up" call using sliced arrays is non-contiguous and so compiles a
DIFFERENT Numba specialisation than the real, contiguous call needs, silently leaving genuine
compile cost inside an apparently-warm timed block -- `_run_month` now calls
`np.ascontiguousarray` on every array it hands the kernel, closing this off for every caller,
not only the test that first tripped over it. Measured impact: the same 6-month deal's `n_r=30`
(54s before) now takes 1.5s; `n_r=300` (impractical before) reaches a residual gap under 0.3%
of PV to `n_r=600` in about 3 minutes; `n_r=600` itself takes 13.4 minutes, not the hour-plus
extrapolated for the pure-Python path. 262 tests pass (255 before this + 7 in
`tests/test_reset_swing_kernels.py`).

*PV figures above recomputed 2026-09-14 against the corrected fixing window (the P0 fix
entry immediately below): `n_r=30` PV=176,731.74 (1.3s), `n_r=300` PV=165,615.76 (170s),
`n_r=600` PV=165,226.49 (717s, ~12.0 minutes). The pre-fix PV figures originally quoted here
(182,057.25 / 170,806.60 / 170,393.52) used the wrong window and are superseded; they moved
because the corrected window is genuinely narrower (a real calendar month, not "every day
since val_date"), not because anything about the DP or the Numba kernel changed. The
qualitative claims (impractical before, ~1.5s/~3min/~12min after, residual gap under 0.3% of
PV between `n_r=300` and `n_r=600` -- 389/165,226 = 0.24% on the recomputed figures) all still
hold; only the absolute PVs moved.*

**P0 correctness fix: the averaged reset's fixing window was wrong (2026-09-14
INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-01/R-02).** An independent review of the branch,
requested and read in full, found -- and this project's own re-derivation from sec.3's "reset
observations and exercise dates are separate calendars" independently confirmed before any
fix was written -- that Release 1B's strike averaging window was constructed wrong in two
related ways, both silently: (R-01) the deal's first delivery month averaged its strike over
EVERY calendar day since `val_date`, not just the one calendar month sec.1 itself specifies
("fixed from market observations made in the preceding month"); (R-02) a non-first month's own
accumulation toward the NEXT month's strike used that month's `exercise_dates`, which only
equals the true fixing-observation window when the month is complete -- a partial delivery
month (mid-month `storage_start`) silently dropped the days before its own exercise window
began, even though sec.3 requires them. Both bugs are invisible to "does the code compute the
right average of whatever window it's given" verification, which is exactly what every
brute-force test up to this point checked; they needed checking the window ITSELF against the
contract, which none of them did.

**The fix, by layer:**
- `reset_terms.DeliveryMonth` gains `fixing_observation_dates`: every calendar day of the ONE
  month immediately before this one, computed independently of `exercise_dates` in
  `build_reset_schedule`. `ResetSwingTerms`'s own validation now requires `val_date` to precede
  the START of every month's fixing window, not merely its END (the old check) -- R-01 point 4's
  "refuse the valuation" option, since a val_date landing inside a window has no historical-fixing
  input to use instead.
- `reset_swing_kernels.run_month_accumulate_core` and `reset_swing_averaged.
  _run_month_accumulate_reference` now walk the full `fixing_observation_dates` window
  (accumulating on every day) with exercise applied ONLY on the trailing days that are also
  `exercise_dates` -- eliminating the conflation R-02 found, and (as a side effect) eliminating
  the separate gap-closing propagation loop the accumulating branch used to need, since the
  extended window already reaches the right starting point by construction.
- `value_averaged_reset_call_swing`'s pre-deal handling now accumulates over
  `months[0].fixing_observation_dates` only, then propagates (with NO further accumulation)
  through any remaining gap back to `val_date` -- replacing the old
  `date_range(val_date, fixing_date)[1:]` construction R-01 named directly.

**Verification, the same discipline as every other fix in this design:** two new,
DISCRIMINATIVE brute-force tests (not just re-checking the arithmetic, which was never wrong) --
one proving the pre-deal window ignores two "gap" days before the true fixing window (optional,
not mandatory, exercise: a mandatory single clip's expected payoff is zero regardless of which
days get averaged, by the tower property, so it could not have caught this; the correct 2-day
window and the old, wrong 4-day window differ by ~125%, not a rounding-level gap), the other
proving a fixing-only day (no exercise decision) still folds into the next month's strike. Both
match the code to ~1e-12/1e-14. `tests/test_reset_terms.py` separately pins
`fixing_observation_dates`'s own construction (5 new tests) and the stricter val_date
validation. `tests/test_reset_swing_kernels.py`'s randomised comparisons now include scenarios
with fixing-only days ahead of the exercise window, not only the degenerate case where the two
calendars coincide. 270 tests pass (262 before this + 8: 5 in `test_reset_terms.py`, 2 in
`test_reset_swing_averaged.py`, 1 in `test_reset_swing_kernels.py`'s existing parametrisation).

**Recomputed and republished, 2026-09-14, same day:** every PV, delta and timing figure
reported earlier in this document was computed under the WRONG window; the ones this document
itself quotes as specific numbers -- the Release 2 Numba-kernel benchmark immediately above
(now annotated in place with both the pre-fix and recomputed PVs) and the averaged-reset delta
example a few entries up (physical/index legs, now recomputed to +9,911.63/-9,895.16) -- have
now been recomputed against the corrected window and republished in place, rather than left
stale. `MonthlyResetSwing.ipynb` itself needed no republishing: it never stores output (the
staleness guard `test_monthly_reset_swing_notebook_executes_clean` enforces this), so it shows
correct figures automatically on its next run -- confirmed by executing it fresh:
averaged-reset PV 12,075.30 EUR against point-reset's 10,876.83 on its own smaller term sheet
(an 11.0% difference, itself a real, illustrative consequence of the fix -- the pre-fix
averaged PV on that same small scenario would have been computed over a wider, wrong window).
Point-reset figures throughout this document are unaffected by any of this: R-01/R-02 are
defects in `reset_swing_averaged.py` only, and point-reset never calls it. What is NOT yet
recomputed: any number in this document's own earlier historical/debugging narrative sections
(the "five bugs, then a sixth that wasn't" account, and similar) -- those describe what was
actually observed at the time as a diagnostic record, not a current-state claim, so they do
not need correcting. R-03 through R-11 from the same review (same-day information-ordering
assumption, production memory not yet reduced, "exact" terminology, Release 1B's
still-restricted scope, missing simultaneous global/monthly state, missing result-object
outputs, remaining validation gaps, missing sec.14.7 fixtures, and the document-structure
critique this section's own running-log format is an instance of) are not addressed by this
entry and remain open.

**R-09 (validation gaps) and R-04 (production memory), same day, closed together.**

R-09: `ResetSwingTerms.__post_init__` now rejects non-finite `daily_max_mwh`, `v_step_mwh`,
`global_min_mwh`, `global_max_mwh`, `vol`, `sMR`, `discount_rate` and `n_p` explicitly (the
existing range checks already rejected NaN as a side effect of NaN comparisons always being
False, but +inf passed every one of them, and NaN's own error message named the wrong
problem -- "must be strictly positive" -- rather than what actually failed), and a
non-integer `n_p`. `build_reset_schedule` now refuses a `global_min_mwh` that exceeds what
the deal could ever deliver (`daily_max_mwh` times the actual exercise-day count) -- checked
there rather than in `__post_init__` since it needs the schedule's own exercise-day count,
which `ResetSwingTerms` alone does not have. `value_averaged_reset_call_swing` now validates
`n_r >= 2` (an `n_r` of 1 would divide by zero inside `accumulate_step`/the kernel's own
`dr = (r_hi - r_lo) / (n_r - 1)`), finite `r_lo`/`r_hi`, and `r_lo < r_hi`, all before
building a lattice or doing any other work. Both `value_point_reset_call_swing` and
`value_averaged_reset_call_swing` now reject a non-finite computed PV explicitly, with the
inputs that produced it still in scope, rather than letting a NaN/inf surface several steps
downstream (inside a delta's own central difference, say) with no context left about which
valuation produced it. 28 new tests in `tests/test_reset_terms.py` and
`tests/test_reset_swing_averaged.py` (a mandatory-minimum-exceeds-deliverable scenario, and a
parametrised sweep of +inf/-inf/NaN across every numeric field).

R-04: `reset_swing_kernels.run_month_accumulate_core` used to return the complete,
un-collapsed `(n_r, width, n_l, n_r)` array for the caller to reduce with
`_collapse_fresh_axis` -- 10.1 GiB at the notebook's own `n_r=600`, an allocation this large
before even considering the working arrays inside a single call. The kernel now performs that
collapse itself, per incoming bucket `k`, before writing anything out -- exactly
`_collapse_fresh_axis`'s own reduction (per-(j,l) max-minus-min across the fresh axis, then
the worst one; overall max |value| floored at 1.0), just computed once per k inside the
compiled loop rather than by the caller on an array that no longer needs to exist. Returns
`(n_r, width, n_l)` plus `(n_r,)` spread and scale arrays; `_run_month` does the identical
raise-with-message the caller always did, reading from these instead of recomputing them from
a full array. This is safe because EVERY call to the kernel is immediately followed by
exactly this collapse in production -- confirmed by checking there is no other consumer of
the kernel's raw output. `_collapse_fresh_axis` itself still exists, now used only by
`_run_month_accumulate_reference` (the deliberately un-collapsed Python reference the kernel
is checked against) and by the kernel-comparison test, which applies the identical reduction
to the reference's own output before comparing, so both sides stay apples to apples -- and
separately confirms the kernel's own spread/scale agree with what `_collapse_fresh_axis`
computes from the full array, not just the final collapsed values.

Measured, not just computed from the returned array's own shape (the review's own explicit
ask: "Record peak memory as well as runtime"): process RSS on the notebook's 6-month term
sheet, `n_r=300` (2.70 GiB estimated for the pre-fix, un-collapsed array alone) stayed flat,
189.3 -> 189.5 MB, a +0.2 MB delta across the whole call -- the per-thread working arrays
Numba's own `prange` parallelism allocates evidently do not accumulate the way a naive
per-thread estimate would suggest. PVs are bit-for-bit unchanged from the pre-fix figures
(confirmed directly: `n_r=30` and `n_r=300` both reproduce this document's own Release 2
recomputation exactly) -- this is a pure memory/architecture change, not a numerical one.
298 tests pass (270 before this + 28).

**R-03 (same-day information ordering): named, not fixed -- it cannot be, without a real term
sheet to specify it.** On a day that is both a fixing-observation day for M+1 and an exercise
day for M, `_run_month_accumulate_reference` and `run_month_accumulate_core` both fold today's
quote into the running average FIRST, then make today's exercise decision against the ALREADY-
FIXED strike -- "fixing-before-exercise", correct only if settlement genuinely publishes before
the nomination deadline. The review's finding was not that this order is wrong (unknowable
without the missing term sheet) but that it was chosen silently. Now named explicitly in
`reset_swing_averaged.py`'s own module docstring, with a one-line cross-reference at the
kernel's own same-day code. The review also asked whether the choice is even consequential --
confirmed directly, not just argued: `accumulate_step` (linear interpolation) and
`_exercise_step_3d` (a pointwise max over exercise quantities) do not commute in general, so
"fixing-after-exercise" is a genuinely different computation, not an equivalent reformulation
reached a different way -- a synthetic same-day scenario in
`tests/test_reset_swing_averaged.py::test_same_day_ordering_is_fixing_before_exercise_and_the_choice_is_consequential`
shows the two orders diverging by over 400 EUR on values of order a few thousand, using only
the already brute-force-verified `accumulate_step`/`_exercise_step_3d` primitives directly (no
new arithmetic to re-verify -- the question is resequencing, not correctness of either step in
isolation). No production code path for "fixing-after-exercise" was added: with no term sheet
to justify implementing an alternative, the review's other acceptable option (recording
publication/exercise timestamps and letting the order follow from them) is deferred rather than
built speculatively. 299 tests pass (298 before this + 1).

**R-06 (scope overclaim) and R-11 (document structure), same day, scoped correction rather
than the review's full required one.** R-06: this document's own header (top of file) now
names the averaged reset as a restricted prototype of sec.14's own broader Release 1B
definition, not the full thing; `docs/STATUS.md` gained a permanent scope note above its
dated log doing the same, rather than editing every historical "Release 1B" mention
individually (the running-log entries are a record of what was believed true when written,
same reasoning as leaving old debugging-narrative numbers alone). `MonthlyResetSwing.ipynb`'s
own introduction, which specifically said "Release 1B ... is design work, not built" -- true
when written, false since section 5 was added -- is corrected in place, since notebook prose
(unlike a stored PV) has no staleness guard to catch this automatically.

R-11: the review's required correction is a full split into three separate documents (current
spec, concise status, dated diary). Not done -- `docs/STATUS.md` already serves the "concise
current status" role reasonably well, and splitting sections 1-12 (specification) from
section 13 (this running log) into separate files is a larger undertaking than the other
findings closed today, deferred rather than attempted partially. What WAS fixed: the specific
inconsistencies the review named directly -- this document's own header (previously "design
only; no valuation code is implemented," contradicted by sections 13/14 describing built,
tested code), section 13's title and opening paragraph (previously still phrased as a
"recommended immediate next action" that had been fulfilled for days), section 12 item 6
(previously required out-of-sample Monte Carlo validation unconditionally, inconsistent with
sec.1's own exact-DP-or-Monte-Carlo fork), and `docs/STATUS.md`'s own "the feature is now
complete" sentence (annotated in place, not rewritten, since R-06's restricted-prototype
finding is what made it wrong). The branch-tracking and test-count inconsistencies R-11 also
named were already fixed when R-11 itself was first read (see that entry, earlier in this
log). Still open: the actual three-way document split, and reconciling section 14.7's
fixture-completeness language against what tests currently exist (R-10's own scope).

**R-08 (result object), same day, MINIMAL scope: closed, not the full sec.14.8 specification.**
`reset_terms.ResetSwingResult` -- a frozen dataclass, `pv`/`reset_strikes`/`deltas` -- and two
new wrapper functions, `value_point_reset_call_swing_detailed` and
`value_averaged_reset_call_swing_detailed`, that call the existing, completely unmodified
`value_*_call_swing`/`compute_deltas` and package their outputs. Additive: every existing caller
of the plain functions is untouched and still gets a bare float back. `reset_strikes` is a
`{month.label: float}` point estimate -- the model's own root-date, centre-node projected
strike (point-reset's single fixing-date observation; averaged-reset's equal-weighted mean of
that same projection over the relevant fixing-observation window) -- not sec.14.8's full strike
distribution, and `deltas` is exactly `compute_deltas`'s existing dict, populated only when
`with_deltas=True`. Explicitly excluded, named as gaps in `ResetSwingResult`'s own docstring
rather than silently omitted: deterministic/extrinsic PV split, per-quote-vertex or
monthly-bucket deltas, convergence status, audit metadata -- all real new numerical work.
"Expected exercised volume" was in this task's originally approved scope but turned out to need
tracking the optimal exercise policy during backward induction, which neither DP currently
does -- dropped once that became clear, rather than approximated or silently skipped.

Two things worth recording from getting this wrong first, both caught before either shipped: (1)
an early draft of the averaged-reset wrapper derived each month's fixing window from the
PRECEDING month's own `exercise_dates` instead of reading that month's own
`fixing_observation_dates` field directly -- the exact R-01/R-02 conflation this session's
earlier fix exists to prevent, reintroduced by hand in new code, caught by re-deriving the
field's documented semantics against `reset_terms.py` itself before trusting the code, not by a
failing test (every fixture exercised so far happens to have a complete preceding month, where
the two windows coincide -- a gap for R-10's future fixtures to cover, not fixed here). (2) a
test written on the assumption that `H[i, n_p]` means "the curve's own value on date i" failed
where the assumption, not the code, was wrong: `H[i, n_p]` is the model's conditional
expectation, AS OF date i, of the price AT month-end, which at near-zero vol converges to the
DELIVERY month's own curve value regardless of which day within the fixing window you read it
from -- confirmed by cross-checking the two wrappers' `reset_strikes` against each other (agree
to 9 significant figures on a shared fixture) rather than against a hand guess. 304 tests pass
(299 before this + 5).

## 14. Implementation-readiness specification

This section defines the work required to turn the preceding design into an executable
specification. It does not declare the model implemented. The commercial terms listed in
Section 2.1 remain unknown until a real term sheet or an explicit prototype convention is
selected.

### 14.1 Staged release scope

The implementation should be divided into separately accepted releases.

#### Release 1A: point-reset exact benchmark

- call swing only;
- one fixing observation for each delivery month;
- the existing one-factor lattice and deterministic discount curve;
- discrete partial daily exercise quantities;
- explicit global and monthly volume constraints;
- cash PV and essential diagnostics only;
- a separate public entry point from `run_valuation`.

This restricted release is a recommended development convention, not an inferred commercial
contract. Its purpose is to validate the conditional quote and state-dependent strike before
introducing the averaging accumulator.

#### Release 1B: averaged-reset exact benchmark

- multiple weighted fixing observations;
- fully fixed, partially fixed and wholly future months;
- simultaneous current-strike and next-accumulator states;
- reset-boundary rotation;
- interpolation, boundary and convergence diagnostics.

#### Release 2: realistic-size production valuation

- retain exact dynamic programming when the analytic reduction and measured resource use make
  it practical;
- otherwise implement regression Monte Carlo with independent out-of-sample policy valuation;
- add Greeks and application integration only after the cash PV reconciles to the exact
  benchmark.

Put direction, floors, caps, rounding and detailed market-disruption provisions should be
added only when their terms are specified. They are not silently included in Release 1A.

### 14.2 Contract data model

The implementation must define an immutable `ResetSwingTerms` structure containing at least:

```python
@dataclass(frozen=True)
class ResetSwingTerms:
    direction: Literal["call", "put"]
    valuation_timestamp: pd.Timestamp
    exercise_dates: tuple[pd.Timestamp, ...]
    delivery_month_by_date: Mapping[pd.Timestamp, pd.Period]
    daily_max_mwh: float
    volume_step_mwh: float
    global_min_mwh: float
    global_max_mwh: float
    monthly_limits: Mapping[pd.Period, VolumeLimits] | None
    fixing_dates: Mapping[pd.Period, tuple[pd.Timestamp, ...]]
    fixing_weights: Mapping[pd.Period, tuple[float, ...]]
    publication_timestamps: Mapping[pd.Timestamp, pd.Timestamp]
    index_definition: IndexDefinition
    reset_formula: ResetFormula
    settlement_dates: Mapping[pd.Timestamp, pd.Timestamp]
    historical_fixings: Mapping[pd.Timestamp, float]
    disruption_rule: Literal["raise", "previous", "specified_fallback"]
```

The final Python types may differ, but the specification must state:

- the units of every field, including EUR, EUR/MWh and MWh;
- whether bounds are inclusive;
- whether global and monthly constraints apply separately or simultaneously;
- whether daily capacity must be an integer multiple of `volume_step_mwh`;
- which fields may be absent and the exact meaning of every default;
- how infeasible volume schedules are rejected;
- how historical observations and corrections are recorded in the audit output.

No calendar, index, publication or settlement convention may be inferred merely from the
description "month-ahead".

### 14.3 Event construction and ordering

`build_reset_schedule(terms)` must produce an auditable ordered event schedule. For each model
date, the candidate operations are:

1. establish the observable price-tree state;
2. publish an index observation scheduled before the exercise decision;
3. update the applicable reset accumulator;
4. finalise a strike when its fixing window is complete;
5. make the exercise decision using only the information then available;
6. attach the resulting cashflow to its contractual settlement date;
7. rotate the next strike and initialise the following accumulator at the month boundary.

This is not a universal same-day order. If nomination precedes publication, exercise must be
processed before that observation becomes available. The supplied timestamps decide the
order, and equal or contradictory timestamps must either follow an explicit priority rule or
raise a validation error.

### 14.4 Module interfaces and array contracts

The initial interfaces should be specified before implementation:

```python
build_reset_schedule(terms) -> ResetSchedule

project_month_ahead_quotes(
    tree,
    delivery_periods,
    delivery_weights,
) -> ConditionalQuoteTable

value_point_reset_exact(
    tree,
    terms,
    quote_table,
    controls,
) -> ResetSwingResult

value_average_reset_exact(
    tree,
    terms,
    quote_table,
    controls,
) -> ResetSwingResult
```

The corresponding conceptual dimensions are:

```text
conditional_quote[observation_time, delivery_month, price_node]
value[price_node, volume_state, strike_state, accumulator_state]
policy[time, price_node, volume_state, strike_state, accumulator_state]
```

Omitted dimensions may be removed by a proved reduction, but the returned metadata must say
which reduction was applied. Every array contract must define dtype, units, date alignment,
reachable indices and boundary values. The document must also choose one value convention:
cashflows discounted directly to the valuation date, or values expressed at each recursion
date. These conventions must not be mixed.

### 14.5 Required Bellman transitions

The detailed implementation section must supply separate pseudocode or equations for:

- an ordinary exercise date;
- a fixing-only date;
- a date containing both fixing and exercise events;
- finalisation of a monthly strike;
- month-boundary state rotation;
- terminal global and monthly volume enforcement;
- fully and partially fixed initialisation.

For every transition it must specify whether the accumulator is updated before or after the
exercise decision, how discounting is applied, how unreachable states are excluded and which
state variables survive the boundary. Interpolation must name the method, valid domain and
failure behaviour. Extrapolation must never occur silently.

### 14.6 Solver-selection rules

Use the following decision sequence:

1. Test whether discounted current-month volume is schedule-independent. If it is, apply the
   analytic strike reduction and verify its slope numerically.
2. If it is not, retain `n_K` or an exact upper-envelope equivalent and run an explicit strike
   convergence ladder.
3. Measure peak memory and runtime for the intended contract dimensions.
4. Use exact dynamic programming while it remains inside the declared application resource
   budget.
5. Introduce regression Monte Carlo only when the exact method exceeds that budget or fails
   another documented production requirement.

The local and Streamlit resource budgets are currently unknown. They must be measured and
recorded before a production-solver decision is final.

### 14.7 Reproducible acceptance fixtures

The following tests are required as named, committed fixtures:

```text
test_fixed_strike_equivalence
test_zero_volatility_reset
test_point_reset_by_exhaustive_enumeration
test_two_observation_average_reset
test_partially_fixed_month
test_fixing_after_exercise_not_visible
test_monthly_and_global_volume_limits
test_zero_rate_mandatory_strike_affinity
test_common_settlement_strike_affinity
test_daily_discounting_can_break_affinity
test_optional_volume_strike_convexity
test_reset_accumulator_grid_convergence
test_strike_grid_convergence
test_tree_boundary_convergence
```

Every fixture must contain complete inputs, an independently calculated expected result and
declared absolute and relative tolerances. Algebraic identities, benchmark PV comparisons and
Monte Carlo confidence checks require separate tolerances. Numerical values for those
tolerances are not fixed here because no reference runs have yet established defensible
levels.

### 14.8 Stable result object

`ResetSwingResult` must define at least:

```text
pv_eur
deterministic_pv_eur
extrinsic_pv_eur
expected_volume_mwh_by_date
expected_volume_mwh_by_month
reset_strike_distribution
expected_strike_conditional_on_exercise
total_curve_delta
physical_leg_delta
index_leg_delta
solver_name
numerical_controls
convergence_results
standard_error
warnings
audit_inputs
```

An inapplicable field must be explicitly `None` or carry a documented status; it must not be
silently omitted. The solver name, model parameters, reduction flags, grid sizes, random seed
where applicable and convergence evidence must accompany every reported PV.

### 14.9 Repository mapping and traceability

| Requirement | Suggested implementation | Primary test file |
|---|---|---|
| Contract validation and audit schedule | `reset_terms.py` | `tests/test_reset_terms.py` |
| Conditional month-ahead projection | `reset_forward.py` | `tests/test_reset_forward.py` |
| Point-reset exact recursion | `reset_swing_exact.py` | `tests/test_reset_swing_point.py` |
| Averaged fixing accumulator | `reset_swing_exact.py` | `tests/test_reset_swing_average.py` |
| Regression Monte Carlo policy | `reset_swing_lsmc.py` | `tests/test_reset_swing_lsmc.py` |
| Reporting and Greeks | `reset_swing_reporting.py` | `tests/test_reset_swing_reporting.py` |

`storage_kernels.py` should remain unchanged until the contract schedule and conditional quote
modules pass independently. The new product should use a separate public entry point until
its result conventions and regression suite are stable.

### 14.10 Definition of implementation-ready

The selected release is implementation-ready only when:

1. its supported commercial terms and exclusions are explicit;
2. every input has a type, unit, validation rule and default policy;
3. one term object maps to one deterministic ordered event schedule;
4. every state and Bellman transition is defined without relying on programmer judgement;
5. discounting and settlement conventions are consistent with the strike-dimension treatment;
6. every acceptance claim maps to a committed, reproducible test fixture;
7. numerical tolerances and failure behaviour are declared;
8. the output schema and audit metadata are stable;
9. existing fixed-strike swing and storage tests remain part of the release gate;
10. unresolved commercial or calibration choices are labelled unknown rather than defaulted
    silently.

The immediate document task is to complete this definition for Release 1A. After its tests
pass, extend the same specification to Release 1B using measured strike and accumulator state
sizes.
