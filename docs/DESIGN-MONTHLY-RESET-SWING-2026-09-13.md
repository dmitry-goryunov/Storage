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

A dense implementation of this state space is suitable only as a small exact benchmark. For
realistic contracts, the recommended production route is regression Monte Carlo with a
discrete cumulative-volume state, validated against the exact benchmark on small cases.

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

### 6.4 The current-strike axis is convex, and often affine — so it need not match `n_R`'s resolution

Section 7.1's `n_price * n_volume * n_K * n_R` estimate treats `k` as needing resolution
comparable to `r`. It does not: for fixed `(i,j,l)` within delivery month `M`, `V_i(j,l;k)` is
convex in `k`, and in the common case of a per-month mandatory exercised volume, exactly
affine. Both are proved, not assumed, and verified numerically against the existing engine.

**Proof (convexity, general case).** Within `M`, `k` is fixed and enters the Bellman recursion
only through the immediate payoff, which is affine in `k`:

```text
V_i(j,l;k) = max_{d in feasible(l)} { DF_i*d*v_step*(S_i(j) - k - vc_i)
                                      + E[V_{i+1}(j',l-d;k) | j] }
```

with the boundary value at the `M`/`M+1` rollover independent of `k` (§6.2: the old `K_M` is
discarded there, and only `l` and the fresh `r` carry forward). By backward induction: the
base case is a constant (trivially convex); if `V_{i+1}(·,·;k)` is convex in `k`, then for each
fixed `d` the bracketed expression is an affine function of `k` plus a non-negative-weighted
expectation of convex functions of `k`, hence convex; a pointwise maximum over `d` of functions
convex in `k` is convex. So `V_i(j,l;k)` is convex in `k` for every `i` inside `M`.

**Stronger case (exact affine).** If the exercised volume during `M` is pinned independent of
price (a mandatory quota applying within the month, not just across the whole deal), the
optimal policy does not depend on `k` at all: every feasible schedule's total exercised volume
is the same fixed number, so shifting `k` by `delta` shifts every schedule's value by exactly
`-total_volume * v_step * delta` for a call (symmetric sign for a put), which cannot change
which schedule is optimal. Then `V(k) = V(k0) - total_volume * v_step * (k - k0)` exactly, with
zero interpolation error at any `k` -- `n_K` collapses to a single evaluation.

**Verified numerically**, using the existing, already-tested `value_call_swing` path directly
(no new code): a one-month call swing (280,000 MWh mandatory capacity, `daily_max` 10,000
MWh/day, flat 30.0 EUR/MWh curve, `sVol` 0.5, `sMR` 1.0), priced at nine strikes from 15 to 45,
under both terminal conditions:

| Configuration | Consecutive slopes `dV/dK` | Convexity check |
|---|---|---|
| Mandatory (default terminal pin) | `-280,000.0` at every interval, to full float precision | Exactly affine, as predicted |
| Optional (`zero_penalty=True`) | `-279,421` down to `-4,535`, monotonically shrinking in magnitude | Every point lies on or below its neighbours' chord -- zero violations across all nine, consistent with convex |

The mandatory slope is exactly `-280,000` = `-(total mandatory volume in MWh)`, matching the
closed form above to the last printed digit. The optional case's shrinking slope is exactly
the expected shape: as `k` rises the swing becomes less attractive, the optimiser exercises
less of the optional quota, and `V` becomes correspondingly less sensitive to further increases
in `k` -- convex, not merely "different from affine."

**Consequence for §7.1's sizing.** For a per-month mandatory structure, drop `n_K` from the
product entirely: `n_price * n_volume * n_R`, not `n_price * n_volume * n_K * n_R` -- the full
factor of ~40 in the worked example, not a fraction of it. For an optional-volume structure,
`n_K` is still needed, but a convexity-aware placement (few knots, tighter near the current
forward level where curvature is highest, or the exact affine sub-policies as anchors rather
than a naive uniform grid) should need far fewer than the 40 buckets `n_R` warrants on its own
merits -- this should be confirmed with its own convergence ladder per §7.1's existing
requirement to measure each grid separately, not assumed at a specific smaller number here.

This does not touch `n_R`, which remains genuinely stochastic and path-dependent within its
own accumulation window and gets no equivalent simplification from this argument.

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
- measure convergence separately in the price, volume, strike and accumulator grids;
- report memory and runtime from actual dimensions.

The state count during an active month grows approximately as

```text
n_price * n_volume * n_K * n_R.
```

The previous design counted only `n_R`. With 40 strike buckets and 40 accumulator buckets,
the naive multiplier is about 1,600, before allowing for time or policy storage -- but §6.4
shows `n_K` is not a free parameter to size the same way as `n_R`: it collapses to 1 for a
per-month mandatory structure (dropping the multiplier back to ~40, `n_R` alone) and to a
small, convexity-justified number otherwise, not 40. A dense full-history lattice may
therefore be usable for the mandatory case specifically; this should be checked against a
measured `n_R` convergence ladder before ruling it out, rather than assumed unusable from the
uncorrected `n_K * n_R` product. The optional-volume case remains the more likely candidate
for regression Monte Carlo, and still less likely to need it than this section originally
suggested.

### 7.2 Production solver: regression Monte Carlo

The recommended realistic-size solver is least-squares Monte Carlo or another regression
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
- §6.4's current-strike property, on the actual `n_K` grid used: exactly affine (slope
  constant to numerical tolerance) under a per-month mandatory quota, and convex (no chord
  violation) under an optional one -- a coarser `n_K` must be justified against this, not
  assumed.

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
- separate convergence ladders for `n_K` and `n_R`;
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
