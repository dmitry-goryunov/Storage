# Corrected design and implementation plan: monthly-reset swing

**Date:** 13 September 2026  
**Status:** design only; no valuation code is implemented by this document  
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
6. The production policy is valued out of sample and reconciles to the exact solver.
7. All numerical dimensions have measured convergence.
8. Reported total deltas pass bump-and-revalue checks and include the reset leg.
9. Existing fixed-strike and storage regressions remain green.
10. Outputs state plainly that the Q dynamics are illustrative until market calibration is
    established.

## 13. Recommended immediate next action

Implement Phase 0 and the conditional month-ahead projection from Phase 1 before modifying
`storage_kernels.py`. The projection can be tested independently against the current tree and
is required by every later solver. After it passes, build the point-reset exact benchmark.
Only then choose the averaged-reset production implementation on measured state sizes and
runtime, rather than assuming the dense lattice will be practical.

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
  and mostly offsetting (physical +9,878, index -9,840 on one configuration; the exact split
  varies with `n_r` since interpolation itself is not delta-neutral), same qualitative
  behaviour as Release 1A. `tests/test_reset_swing_averaged_deltas.py` pins the sign, the
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
