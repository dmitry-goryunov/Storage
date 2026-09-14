# Independent review: monthly-reset swing

**Review date:** 14 September 2026  
**Reviewed branch:** `feature/unified-pricing-app`  
**Reviewed commit:** `abb6fff4254e5b68bf949b30db94a813974c25f1`  
**Review type:** design, implementation and test-evidence inspection  
**Outcome:** material correction required before Release 1B or Release 2 can be accepted

## 1. Executive conclusion

The fundamental modelling direction is correct. A strike that resets from future market
observations is stochastic before fixing and cannot be represented by broadcasting a
deterministic monthly strike through the existing fixed-strike swing engine. The conditional
month-ahead projection, the retention of reset information in the dynamic-programming state,
and the separate physical/index/total sensitivity channels are appropriate.

The point-reset implementation is a credible exploratory prototype for its explicitly limited
scope: call direction, one model-internal month-end observation per delivery month, global
volume limits, no historical fixings and exercise-day discounting.

The averaged-reset implementation is not yet correct for the convention stated in the design
and notebook. The first fixing average is built from the day after the valuation date through
the fixing date, rather than from the contractual preceding calendar month. A related error
uses exercise dates as fixing-observation dates when the first delivery month is partial. These
errors can change the reset strike and PV materially. The size and direction of the valuation
effect are unknown until corrected runs are performed.

Release 2 should not currently be described as production-ready. Numba materially improves
runtime, but the implementation retains quadratic accumulator-grid memory, lacks a declared
production resource budget, and has not demonstrated convergence to a predeclared valuation
tolerance.

## 2. Material reviewed

### 2.1 Documents

The following documents were read in full:

- `docs/STATUS.md`;
- `docs/DESIGN-MONTHLY-RESET-SWING-2026-09-13.md`.

The review compared the design, implementation history, release definitions, acceptance gates
and current-status claims for internal consistency.

### 2.2 Implementation

The following current branch files were inspected:

- `reset_terms.py`;
- `reset_forward.py`;
- `reset_swing_exact.py`;
- `reset_swing_averaged.py`;
- `reset_swing_kernels.py`;
- the monthly-reset additions exposing lattice transition arrays from `storage_model.py`;
- `MonthlyResetSwing.ipynb`.

The inspection covered contract validation, schedule construction, conditional projection,
Bellman recursion, month chaining, accumulator interpolation, strike handling, discounting,
finite-difference sensitivities, boundary checks, Numba/Python equivalence and memory
dimensions.

### 2.3 Tests

The following reset-specific test files were inspected:

- `tests/test_reset_terms.py`;
- `tests/test_reset_forward.py`;
- `tests/test_reset_swing_point.py`;
- `tests/test_reset_swing_stochastic.py`;
- `tests/test_reset_swing_exhaustive.py`;
- `tests/test_reset_swing_deltas.py`;
- `tests/test_reset_swing_averaged.py`;
- `tests/test_reset_swing_averaged_deltas.py`;
- `tests/test_reset_swing_kernels.py`;
- relevant monthly-reset sections in `tests/test_reconciliation.py`.

The latest GitHub Actions run for reviewed commit `abb6fff` completed successfully:
<https://github.com/dmitry-goryunov/Storage/actions/runs/34850650737>.

### 2.4 Repository state

At the review timestamp, `feature/unified-pricing-app` was 13 commits ahead of and 2 commits
behind `main`. The branches had diverged. This was checked through GitHub's commit comparison,
not inferred from the documents.

## 3. Review limitations

This was a substantive static code and evidence review, not a complete independent execution
audit.

The following were not independently performed:

- rerunning the complete test suite on a separate machine;
- reproducing the reported 1.5 second, 3 minute and 13.4 minute timings;
- profiling actual peak resident memory;
- independently recalculating every reported PV and delta;
- executing every notebook cell and comparing regenerated output;
- auditing every function in the wider 96 KB `storage_model.py`;
- constructing every missing Section 14 acceptance fixture.

The exact claim that 262 tests pass is reported by the project and is consistent with a green
CI run, but the numerical count was not independently reproduced in this review.

## 4. Findings

### R-01: the first averaged fixing window is constructed incorrectly

**Severity:** P0  
**Status:** observed directly in code  
**Affected file:** `reset_swing_averaged.py`

The first fixing-window dates are constructed as:

```python
pre_deal_dates = list(
    pd.date_range(terms.val_date, months[0].fixing_date, freq="D")
)[1:]
```

This means the strike average begins on the day after valuation. It does not begin on the first
contractual observation date.

For the averaged example in `MonthlyResetSwing.ipynb`:

- valuation date: 1 January 2026;
- first delivery month: May 2026;
- final fixing observation: 30 April 2026;
- stated convention: every calendar day in the preceding month, April 2026.

The implementation uses 2 January through 30 April, 119 observations. The stated convention
requires 1 April through 30 April, 30 observations.

If valuation occurs during the fixing month, the opposite error occurs. Already-realised
observations before valuation are silently omitted because the term object contains no
historical weighted sum or weight.

**Consequence:** the reset-strike distribution and PV do not represent the contract described
in the document or notebook. The direction and magnitude of the PV error depend on curve shape,
volatility, mean reversion and exercise policy.

**Required correction:**

1. Add explicit fixing-observation dates for each delivery month to `ResetSchedule`.
2. Accumulate only on those dates.
3. Propagate normally between valuation and the first future observation.
4. If valuation falls inside a fixing window, require the realised weighted sum and realised
   weight, or refuse the valuation.
5. Add an integration test proving that May's strike uses exactly 1-30 April when valuation is
   earlier than April.

### R-02: fixing observations are incorrectly coupled to exercise eligibility

**Severity:** P0  
**Status:** observed directly in code  
**Affected file:** `reset_swing_averaged.py`

For each non-final delivery month, `_run_month` accumulates the next month's reset observation
only over `month.exercise_dates`.

That happens to cover the full preceding calendar month when the current delivery month is
complete and every calendar day is exercisable. It fails for a partial first delivery month.

For a contract beginning 15 March, April's stated fixing window is 1-31 March. The
implementation accumulates only 15-31 March because those are the first month's exercise dates.

**Consequence:** the strike index is changed by the exercise-window boundary even though the
index calendar should be an independent contractual calendar.

**Required correction:** separate these schedules explicitly:

```text
exercise_dates
fixing_observation_dates[delivery_month]
publication_timestamps
settlement_dates
```

The Bellman loop must process fixing and exercise events from that event schedule rather than
assuming that every exercise date is also an index-observation date.

### R-03: same-day information ordering remains an unstated assumption

**Severity:** P1  
**Status:** conditional finding; commercial chronology is unknown  
**Affected files:** `reset_swing_averaged.py`, `reset_terms.py`

The averaged recursion folds the current day's quote into the accumulator before performing
the current day's exercise optimisation.

This is correct only if the fixing observation is available before the exercise or nomination
decision. If the published settlement occurs after nomination, the recursion gives the
decision access to information not contractually available at that time.

No real term sheet exists, so the correct order is unknown. The problem is not that the chosen
order is disproven; it is that the implementation selects it without recording the selection
in the term object or audit output.

**Required correction:** select an explicit prototype priority rule, or add publication and
exercise timestamps. Add both "fixing before exercise" and "fixing after exercise" miniature
tests.

### R-04: production-scale practicality is not established

**Severity:** P1  
**Status:** observed allocation plus resulting inference  
**Affected file:** `reset_swing_kernels.py`

The Numba kernel allocates:

```python
results = np.empty((n_r, width, n_l, n_r), dtype=np.float64)
```

For the reported six-month notebook-scale contract:

```text
n_r   = 600
width = 2 * n_p + 1 = 31
n_l   = global_max / v_step + 1 = 121
```

The `results` array alone therefore requires:

```text
600 * 31 * 121 * 600 * 8 bytes
= 10.80 GB decimal
= approximately 10.1 GiB
```

At `n_r=300`, the same array requires approximately 2.70 GB decimal before working arrays,
Numba parallel-thread temporaries and the rest of the valuation state.

The runtime improvement is material, but the evidence currently shows:

- approximately three minutes for `n_r=300`;
- approximately 13.4 minutes for `n_r=600`;
- a reported `300 -> 600` PV difference below 0.3%;
- no predeclared PV tolerance;
- no `600 -> 1,200` check;
- no measured peak-memory result;
- no declared local or Streamlit resource budget.

A successful run on the development machine does not establish suitability for a laptop,
Streamlit Cloud, production batch valuation or multi-bucket Greek calculation.

**Required correction:**

1. Collapse the mathematically fresh outgoing accumulator axis inside the kernel.
2. Return one representative slice per incoming strike bucket plus the maximum discarded-axis
   spread, rather than materialising and returning the complete four-dimensional array.
3. Record peak memory as well as runtime.
4. Declare the acceptable PV convergence tolerance before selecting `n_r`.
5. Test the full intended contract horizon and the complete bump-and-revalue workload.
6. Decide between exact DP and regression Monte Carlo only after those measurements.

### R-05: "exact" is being used too broadly

**Severity:** P1  
**Status:** documentation qualification

The averaged solver is a deterministic dynamic programme on a price lattice, but it interpolates
a continuous accumulator over a finite `r_grid`. It is therefore an approximation to the
continuous-accumulator problem.

In addition, `np.interp` clamps outside the grid. The wrapper checks projected quote values
only at cells whose unconditional probability is at least `1e-6`. Lower-probability cells can
still clamp silently. Their aggregate value effect is not measured.

The brute-force tests demonstrate correctness on selected small scenarios, including a
two-month chaining path. They do not prove that an arbitrary production `r_grid` has adequate
accuracy.

**Required wording:** "exact Bellman optimisation on the chosen price, volume and accumulator
grids" or "grid-DP benchmark", accompanied by an explicit convergence status. Do not describe
an unconverged finite-`n_r` result as the exact contract value.

### R-06: Release 1B's stated scope is not implemented

**Severity:** P1  
**Status:** direct design-to-code comparison

Section 14 defines Release 1B as including:

- multiple weighted fixing observations;
- fully fixed, partially fixed and wholly future months;
- simultaneous current-strike and next-accumulator state;
- reset-boundary rotation;
- interpolation, boundary and convergence diagnostics.

The current implementation supports:

- call direction only;
- equal calendar-day weights only;
- identity reset formula only;
- wholly future fixings only;
- no historical fixing input;
- deal-wide global volume limits only;
- a model-internal month-end price projection;
- exercise-day cashflow discounting;
- a scalar PV return rather than the specified result object.

The simultaneous reset states and month rotation are present. The other Release 1B items are
not.

**Outcome:** the implementation may be described as a restricted averaged-reset prototype,
but not as the Release 1B defined in Section 14.

### R-07: the general state specification is insufficient for simultaneous global and monthly limits

**Severity:** P1 design defect  
**Status:** logical state-sufficiency finding

The design claims that global and monthly volume limits can be enforced while using one
cumulative-volume state `l`.

If both constraint types apply, two histories can reach the same date and the same global
cumulative volume while having exercised different volumes since the current month began.
Those states have different remaining monthly entitlements but are indistinguishable under
`(i, j, l, k, r)`.

The restricted implementation avoids this defect because it supports global limits only.

**Required correction:** either:

- add a current-month cumulative-volume state alongside global cumulative volume; or
- formulate each month as a transition operator indexed by both global opening and closing
  volume states, with monthly feasibility enforced inside that operator.

### R-08: output and Greek requirements remain incomplete

**Severity:** P1  
**Status:** direct design-to-code comparison

The design requires a stable `ResetSwingResult` containing PV, deterministic and extrinsic
values, expected volume profiles, reset-strike distributions, conditional strike diagnostics,
deltas, controls, convergence evidence, warnings and audit inputs.

Both valuation entry points currently return a scalar float.

The implemented deltas are parallel-curve central differences. They are useful diagnostics,
but they are not the design's required quote-vertex or monthly-bucket deltas.

Missing or incomplete outputs include:

- expected exercised volume by date and month;
- reset-strike distribution;
- expected strike conditional on exercise;
- deterministic and extrinsic PV;
- numerical convergence status attached to the PV;
- peak memory and runtime;
- bucketed total, physical and index sensitivities;
- audit-ready contract, event and solver metadata.

### R-09: validation remains incomplete

**Severity:** P2, with potential P1 outcomes for malformed inputs  
**Status:** observed from validation paths

The grid-representability checks for daily and total volume are appropriate and close a real
silent-rounding defect. Additional validation is required:

- `n_r >= 2` must be enforced; the Numba interpolation spacing divides by `n_r - 1`;
- require finite `r_lo`, `r_hi` and `r_lo < r_hi`;
- require integer-valued `n_p`;
- reject non-finite `sMR` and `discount_rate`;
- either support exact zero volatility or change the zero-volatility acceptance requirement to
  an explicitly tested limiting-zero convention;
- reject a mandatory global minimum exceeding the maximum deliverable volume;
- validate supplied quote-table keys, shapes, date alignment and finiteness;
- reject non-finite PVs explicitly rather than relying on later comparisons.

### R-10: required acceptance fixtures are incomplete

**Severity:** P1 release governance  
**Status:** confirmed by test-file inspection and by the design's own implementation log

The following Section 14 fixtures or equivalent coverage remain absent or incomplete:

- fixed-strike equivalence;
- partially fixed month;
- fixing published after exercise is not visible;
- simultaneous monthly and global volume limits;
- exact zero-volatility reset through the public term/solver path;
- separate strike-grid convergence where applicable;
- production resource and convergence acceptance;
- stable result-object reconciliation.

The existing tests provide useful evidence, particularly the exhaustive point-reset stopping
test, the two-month averaged chain, the real-volatility reference comparison and the
Numba/Python comparison. They do not satisfy the complete release gate.

### R-11: status and design documents are internally inconsistent

**Severity:** P2 documentation and process risk  
**Affected files:** `docs/STATUS.md`,
`docs/DESIGN-MONTHLY-RESET-SWING-2026-09-13.md`, `MonthlyResetSwing.ipynb`

Observed inconsistencies include:

- the design header says no valuation code is implemented, while later sections describe
  Releases 1A, 1B and 2 as built;
- the design retains a "recommended immediate next action" that has already been performed;
- the minimum release gate always requires out-of-sample policy valuation, even though
  Section 14 allows exact DP to be the selected production solver;
- the design records that several fixtures are still missing while `STATUS.md` calls the
  feature complete;
- `STATUS.md` reports 262 tests at the top and 244 in the repository summary table;
- `STATUS.md` says the working copy tracks `main`, while the reviewed work is on
  `feature/unified-pricing-app`;
- the branch is 13 commits ahead and 2 behind `main`;
- the notebook introduction says Release 1B is not built, while later cells import and run it;
- the historical "later still" entries obscure which statements are current.

**Required correction:** separate the material into:

1. a current product and numerical specification;
2. a concise current implementation status;
3. a dated development log or archived findings document.

Only the first two should be used to decide whether a release is accepted.

## 5. Elements that are technically sound

The following conclusions survived review:

1. A stochastic reset cannot be represented as a deterministic strike schedule before fixing.
2. The conditional quote should be derived through the implemented transition law rather than
   by treating unconditional node probabilities as conditional transitions.
3. `reset_forward.py` correctly uses `p_u`, `p_m` and `p_d` for backward conditional
   expectation.
4. The point-reset fixing-node state `j_fix` is a valid reduction for the restricted
   single-observation convention.
5. Recurring averaged resets require simultaneous retention of the current strike and the next
   accumulator, or an equivalent month-operator representation.
6. The point-reset exhaustive stopping-policy test is meaningful.
7. The two-month averaged test exercises the month-chaining and fresh-axis collapse path.
8. The Numba kernel is checked against a retained Python reference across randomised shapes.
9. Total, physical-leg and index-leg sensitivities are economically distinct and the total
   sensitivity is the authoritative parallel-shift measure.
10. A small net parallel delta does not imply small basis or gross-leg risk.
11. The documents correctly state that current risk-neutral volatility and mean reversion are
    illustrative rather than market-calibrated.

## 6. Minimum corrective work before acceptance

### P0 correction package

1. Introduce an explicit fixing-observation schedule independent of exercise dates.
2. Correct first-month wholly future, partially fixed and fully fixed initialisation.
3. Add regression tests for:
   - a valuation date several months before the first fixing window;
   - valuation inside the fixing window;
   - a partial first exercise month;
   - a shaped curve for which the incorrect and correct fixing windows produce visibly
     different strikes and PVs.
4. Recalculate every averaged-reset notebook result after the fix.

### P1 numerical and release package

1. Reduce the Numba kernel's output memory by collapsing the fresh axis internally.
2. Measure runtime and peak memory on the full intended contract.
3. Predeclare the PV convergence and application resource limits.
4. Run price, volume and accumulator-grid ladders against those limits.
5. Complete the missing chronology and constraint tests.
6. Return a stable result object carrying convergence and audit metadata.
7. Add bucketed deltas if Release 2 is intended to include production hedging.

### Repository and documentation package

1. Reconcile `feature/unified-pricing-app` with the two commits by which it trails `main`.
2. Run the full pinned CI suite on the reconciled branch.
3. Replace contradictory current-status counts and branch statements.
4. Move the implementation diary out of the live design specification.
5. Change the release description to "restricted prototype" until its own acceptance matrix is
   complete.

## 7. Release assessment

| Component | Assessment |
|---|---|
| Conditional month-end projection | Implemented and supported by appropriate recursion tests |
| Point-reset call swing | Credible restricted exploratory prototype |
| Averaged-reset call swing | Not acceptable under the stated preceding-month convention until R-01 and R-02 are fixed |
| Numba kernel arithmetic | Reconciles to the retained Python reference on tested cases |
| Accumulator convergence | Direction investigated, but no accepted production tolerance |
| Production memory and runtime | Not established; current quadratic allocation is material |
| Contract chronology | Incomplete without independent fixing dates and same-day priority |
| Historical and partial fixings | Not implemented |
| Monthly volume constraints | Not implemented; general state design requires correction |
| Hedge reporting | Parallel sensitivities implemented; production bucketed hedge output incomplete |
| Calibration | Explicitly unresolved |
| Overall release | Verified prototype components, not a validated or production-ready monthly-reset product |

## 8. Final outcome

The review does not reject the underlying dynamic-programming approach. It identifies a
material schedule-construction error and several gaps between the restricted implementation
and the broader release specification.

The immediate priority is correctness of the fixing calendar. Further performance work or
application integration should wait until the averaged reset uses the contractual observation
window and the new counterexamples are pinned in tests.
