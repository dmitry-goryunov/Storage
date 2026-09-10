# Independent investigation of the storage findings and P4.1 design

10 September 2026. Source: the current `Storage` project, particularly `docs/FINDINGS-2026-09-10.md`, `docs/DESIGN-P4.1-two-factor.md`, `storage_model.py`, `storage_kernels.py`, the two test modules and `ttf q.xlsx`.

## Assessment

**Fix the inventory-bound checks and quantify ratchet discretisation error before implementing P4.1. Investigate additional curve factors, but do not adopt the present design unchanged.**

The empirical evidence for imperfect forward correlation is reproducible. The fuel implementation and much of the reported hedge behaviour also reproduce. However, an important physical conclusion in the findings reverses when the grid is refined, the new bound checker has three confirmed defects, and the proposed two-factor design rests on several incorrect mathematical arguments.

The existing suite passed independently: **101 passed in 48.14 seconds**. This is evidence about those tests, not evidence that all contractual constraints or the model specification are correct. No production source files were changed during this investigation.

| Priority | Finding | Evidence status |
|---|---|---|
| P0 | Dated bounds use expected inventory, omit initial inventory after day zero, and round inequalities in the wrong direction | Reproduced code defects |
| P0 | The reported 52.5% ratchet limit and infeasible 70% floor are consequences of the chosen integer grid | Reproduced reversal on the same physical inputs |
| P1 | The market correlation table is correct, but it does not establish that the proposed Schwartz–Smith specification fits seasonal spread risk | Statistics reproduced; exploratory fit remains materially deficient |
| P1 | The argument that LSMC loses cashflow reconciliation is wrong | Algebraic identity and independent Monte Carlo counterexample |
| P1 | The long factor cancels from the log forward ratio; under specified homogeneous cashflows it can also leave storage value unchanged | Algebraic derivation and independent two-factor DP counterexample |
| P1 | Mean-reversion sign reversal is not a valid acceptance gate | Analytic and numerical counterexamples |
| P1 | The claimed ninefold volatility shortfall and 4.2% baseline lack a sufficiently specified calculation | Not independently reproduced as stated |
| P2 | Fuel, net hedge, data gaps, grid cost and calibration statements need narrower wording | Mixed confirmed results and corrected generalisations |

## 1. Scope, provenance and execution

The files were retrieved from the [Storage folder](https://drive.google.com/drive/folders/1E8yDc92FS-WvsXblX__Qw9PQkX58HxCK), including its [docs folder](https://drive.google.com/drive/folders/1_VDRHCOZlCfvTfKMUtcD49DgNRfic4Te). The inspected model and kernel were last modified on 10 September at 18:22 UTC, and the main test file at 18:24 UTC. The findings and design were modified at 19:22 and 19:23 UTC respectively. SHA-256 hashes are in the evidence archive.

The model suite and model probes used Python 3.12.14, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1 and Numba 0.67.0. The numerical package versions match the relevant entries in `requirements-lock.txt`. Two Numba threads were used. Workbook statistics and the independent small DP oracle were computed separately from the production valuation implementation.

The review includes source inspection, the complete existing test suite, fresh contract examples, a Monte Carlo accounting check, an independent two-factor DP, historical return calculations and exploratory covariance fitting. It does not include a full Kalman calibration, a production two-factor engine, historical trading backtests or verification of the workbook against an independent market-data provider. Historical process claims about commits, synchronisation incidents and prior execution times were not independently verified.

## 2. Dated inventory bounds have three defects

### 2.1 The check tests an expectation instead of pathwise compliance

`assert_inventory_bounds`, at `storage_model.py:712`, reconstructs an expected balance from probability-weighted actions. An expectation above a floor does not establish that the floor holds in every reachable inventory state. The production kernel uses a finite penalty, so some states can deliberately breach a contractual bound.

An independently constructed stochastic contract demonstrates this on an actual optimiser-generated policy:

| Input or result | Value |
|---|---:|
| Working inventory | 10 clips of 1 MWh |
| Valuation date | 1 January 2026 |
| Injection opportunity | 1 January 2027, at a forward price of EUR 1/MWh |
| Withdrawal opportunities | 1 June and 1 December 2027 |
| Other forward prices | EUR 300/MWh |
| Volatility / mean reversion / price half-width | 0.8 / 1.0 / 30 |
| Required opening inventory on 2 June | 2 clips |
| Expected opening inventory | 7.424736 clips |
| Probability of opening with zero inventory | **0.014398%** |
| Current checker | **Accepts** |

This is a stress example, not a claim about the current notebook's breach frequency. Raising the price level to EUR 1,000 produces a **1.038170%** breach probability while the same checker accepts. The problem exists at any material probability that the contract requires to be zero.

**Remedy:** impose inventory bounds as hard admissibility conditions in the DP. Independently verify the probability mass below each floor and above each ceiling directly from `model.prob[index, :, :]`. A contractual bound needs a numerical probability tolerance, not an expected-inventory comparison. Validate both the intrinsic and stochastic passes.

### 2.2 Opening inventory is omitted after the first date

The code builds `closing = cumsum(net_actions)` and then prepends initial inventory only to the first opening balance. It should add initial inventory to the entire cumulative series. Directly calculating the balance from the state distribution is safer.

A 100 MWh store starting full and correctly holding all 100 MWh on 2 January is rejected against a 100% floor: the message reports **0 clips against 10**, although the state distribution holds **10 clips**. This is a false rejection of a feasible, compliant policy.

**Remedy:** use `sum_l l * sum_k prob[t,k,l]` for expected balance diagnostics, and the full distribution for contractual verification. Correct the cumulative reconstruction wherever it is used with non-zero initial inventory.

### 2.3 Nearest rounding changes the contract

`apply_inventory_bounds`, at `storage_model.py:659`, uses `round(fraction * n_states)` for both floors and ceilings.

On a ten-clip grid:

| Requested bound | Accepted model balance | Contractual error |
|---|---:|---|
| At least 71% | 70% | Below the requested minimum |
| At most 29% | 30% | Above the requested maximum |

Both cases were reproduced through `run_valuation`. The post-check validates the rounded bound, so it does not detect the changed contract.

**Remedy:** use ceiling for minimum inventory and floor for maximum inventory, with a documented floating-point tolerance around exact integers. Report the effective grid bound. Reject a resulting empty admissible interval, or require a finer grid. A non-grid-aligned equality requires explicit treatment.

## 3. Ratchet resolution changes both value and feasibility

The zero-rate guard is useful: it prevents a positive ratchet from silently becoming a complete shutdown. But avoiding zero clips does not make the remaining rate accurate. At four withdrawal clips per day, a 0.30 multiplier permits 1.2 clips physically and the kernel permits only one. The rate is already understated by 16.7% at that point; errors at other inventory levels can be larger.

I reproduced the stated 52.5% result, then refined only the inventory clip while preserving capacity, base daily MWh rates, ratchet functions, dates, forward curve and funding rate. This is the deterministic 600,000 MWh 30/60 store, 10% discount rate, March price 24.9, other summer months 25, winter months 30, and withdrawal ratchet `[0.3, 0.7, 1.0, 1.0]` at fullness `[0, 0.5, 0.8, 1]`.

| Inventory clips | Clip size, MWh | Peak opening inventory | Value, EUR | 70% floor on 1 October |
|---:|---:|---:|---:|---|
| 240 | 2,500.00 | **52.50%** | **1,279,390** | Rejected |
| 480 | 1,250.00 | **72.08%** | **1,751,509** | Accepted |
| 960 | 625.00 | 83.54% | 2,019,763 | Accepted |
| 1,920 | 312.50 | 88.54% | 2,134,480 | Accepted |
| 3,840 | 156.25 | 90.68% | 2,182,561 | Accepted |

The 480-clip value is **36.90% above** the 240-clip value. The 3,840-clip value is **70.59% above** it. Even the final refinement changes value by about 2.25%, so this is not a completed convergence demonstration.

An independent fractional-volume backward reachability calculation, using the same start-of-day ratchet convention and 92 withdrawal days from October through December, permits opening inventory of **92.7565%** on 1 October before requiring an empty terminal store. This is a physical reachability calculation, not a separate continuous-volume valuation.

**Conclusion:** withdrawal deliverability does constrain useful inventory, but the reported 52.5% cap is not an established property of the physical contract. The statement that the 70% floor is physically infeasible is disproven for these inputs. Six additional imposed floors on the same coarse grid do not resolve that distinction.

**Remedy:** replace the zero-only resolution guard with an explicit discretisation error policy. Report effective MWh/day rates by inventory level, establish value and feasibility convergence on fixed physical inputs, and consider continuous-action DP with inventory interpolation or an appropriate deterministic LP/MILP benchmark. Retain the zero-rate guard as an early check. Do not label refinement universally free or answer-neutral.

The test `test_ratchets_cap_the_store_through_the_exit_not_the_entry` currently asserts that the peak lies between 40% and 70% and that the 70% floor fails at 240 clips. It passes while encoding a discretisation result as a physical conclusion. It should be replaced or relabelled, with an independent refinement/feasibility test added.

## 4. Fuel and hedge results mostly reproduce, with qualifications

The injection multiplier `1 / (1 - fuel_loss)` is correct when fuel is a fraction of purchased gas and the injection rate is defined in inventory-added MWh. At 1.5% fuel, the multiplier is **1.0152284264**.

On the unratcheted, 60-clip flat-month reference at 10% funding and price half-width 15:

| Fuel loss | Model value, EUR |
|---:|---:|
| 0% | 3,212,220.14 |
| 1.5% | 2,925,765.10 |

The reduction is **8.9177%**, reproducing the reported 8.9%. This is a result for that contract and curve, not a general storage rule. Contracts whose injection capacity or cash tariff is defined on purchased rather than retained gas need a different mapping of those inputs.

The zero-net-intrinsic-hedge statement only holds for a closed inventory cycle with zero fuel loss. In a deterministic 100 MWh cycle with 1.5% fuel, net physical inventory movement is zero but net signed market hedge is **-1.522843 MWh**. Consequently, net hedge is not necessarily all extrinsic.

`delta = E[S_i Q_i] / F_i` is a price-weighted hedge sensitivity under the model's fixed-parameter, multiplicative curve-bump convention. In a stochastic run it is generally not simply expected traded volume `E[Q_i]`. It equals signed traded volume in the deterministic fuel test. The fuel adjustment is correct; the prose should retain this distinction.

The current 30/60 notebook's hedge instability is broadly reproduced. A +/-1% own-month bump moves April–August hedge positions by approximately **43.7% to 51.2% of the largest base position**. October, November and December move **0.104%, 0.109% and 0.191%**. A small October price bump also agrees with reported PV delta to relative error about **2.5e-11**. April is near policy switches; a centred small bump differs by approximately 0.105%, so a single smooth derivative should not be assumed there.

Those sensitivities are scenario results, not sufficient grounds for an operational instruction to put a hedge on and leave it indefinitely. Curve moves, model recalibration, constraints and time passage remain relevant.

The existing asymmetric-rate tests pass. For integer 30/65 days, `lcm(30,65) = 390` is the smallest clip count expressing both rates exactly; the grid actually contains 391 inventory levels including zero. The core conversion still rounds other user-supplied grid choices. The notebook solves its stated example; it does not eliminate the representation issue for every caller.

## 5. Historical statistics: reproduced, but the calibration problem is broader

The workbook contains **4,171 dated rows**, **55 forward columns**, and **4,117 numeric DA observations**, covering **12 March 2010 to 6 March 2026**. It also contains one blank row and some non-numeric/missing entries. The since-2015 raw return sample has 2,917 observations for the cited pairs.

I sorted the data chronologically, coerced non-numeric cells to missing, used positive prices, calculated daily log differences, and annualised standard deviations by `sqrt(252)`.

| Pair | Raw correlation | Raw annualised log-ratio volatility | Correlation excluding roll dates | Log-ratio volatility excluding roll dates |
|---|---:|---:|---:|---:|
| c1/c3 | 0.917095 | 0.282342 | 0.914376 | 0.283169 |
| c1/c6 | 0.801426 | 0.423435 | 0.794928 | 0.424831 |
| c6/c12 | **0.770636** | **0.373101** | **0.763070** | **0.377761** |
| c1/c12 | 0.742070 | 0.475324 | 0.739131 | 0.471472 |
| c12/c24 | 0.802850 | 0.302111 | 0.799764 | 0.304050 |
| c1/c24 | 0.632737 | 0.560459 | 0.632587 | 0.552708 |

The document's rounded raw table reproduces. Removing the 135 month-change observations leaves 2,782 observations and does not eliminate the mismatch. The reason to remove those observations is that continuous contract rank `c6` can refer to a different delivery month after a roll. A full calibration should track the actual delivery contract or correctly align rank changes.

The quantity above is `std(dlog(F_i) - dlog(F_j)) * sqrt(252)`, meaning volatility of changes in the log price ratio. It is not the log of a monetary spread, which can cross zero, and it is not volatility in EUR/MWh of `F_j - F_i`.

The sample is not homogeneous:

| Sample, excluding roll dates | c6/c12 correlation | Annualised log-ratio volatility |
|---|---:|---:|
| 2015–2020 | 0.592029 | 0.262370 |
| 2021–2023 | 0.776969 | 0.606419 |
| 2024–6 March 2026 | 0.904677 | 0.194273 |

The quoted pooled volatility is nearly twice the recent-window estimate and much less than the 2021–2023 estimate. A pricing calibration needs an explicit window, weighting scheme and treatment of seasonality and regime changes.

### Exploratory covariance fit

I fitted the annualised covariance matrix of c1–c24 log returns, excluding rolls, to:

`C_ij = sigma_chi^2 a_i a_j + sigma_xi^2 + rho sigma_chi sigma_xi (a_i + a_j)`, with `a_i = exp(-kappa tau_i)`.

This was a diagnostic nonlinear least-squares fit, not a Kalman estimate or risk-neutral pricing calibration. It used approximate maturities `tau_i = i/12`, multiple starting points and residuals scaled by observed marginal standard deviations. Volatilities were bounded above at 5 and correlation at +/-0.999.

| Pooled 2015 onward diagnostic | One-factor fit | Schwartz–Smith two-factor fit | Observed |
|---|---:|---:|---:|
| Normalised covariance RMSE | 0.10205 | 0.06801 | n/a |
| c6/c12 correlation | 1.00000 | 0.98190 | **0.76307** |
| c6/c12 log-ratio volatility | 0.09777 | 0.14030 | **0.37776** |

The two-factor fit improves the aggregate objective, but badly misses the central spread pair. Its solution approaches the volatility bound and an almost perfectly negative factor correlation; these are unstable diagnostic parameters, not a recommended parameter set. Repeated fits by period also show instability. More careful delivery averaging and a proper observation model are required before drawing a final model-selection conclusion.

The first two principal components explain approximately **86.1%** of pooled c1–c24 return covariance; the first three explain 89.2%. This does not prove that three instantaneous factors are necessary: seasonal loadings, regimes, measurement errors and pooled maturities can raise pooled covariance rank. It does show why the assertion that two plain factors will solve the problem requires testing.

**Recommended alternatives to compare:** one-factor baseline, the proposed Schwartz–Smith model, two factors with different non-zero mean-reversion speeds, and a model with explicit seasonal/curve-shape loadings. Choose by out-of-sample covariance, marginal volatility and target spread diagnostics, followed by valuation sensitivity. Historical covariance fit alone does not determine risk premia or establish traded value. The distinction between historical and pricing dynamics, and the relevance of seasonal extensions, are also explicit in [Peters et al., sections 2–3](https://arxiv.org/html/1105.5850v1).

### Data claims requiring correction

The maximum gap between numeric DA observations is **five calendar days**, with 34 intervals longer than three days. The date index itself has only one- and three-day gaps; treating that as the DA observation spacing overlooks missing DA cells. Holiday settlement rules and delivery-period averaging must be established for a backtest. Some far-forward columns also have hundreds of missing entries; 55 columns do not mean a complete 55-contract panel throughout the history.

## 6. The P4.1 economic argument needs revision

### 6.1 One factor allows spread movements

For the ideal one-factor exponential-affine model, with fixed delivery dates and constant parameters,

`d log F(t,T_i) = deterministic drift * dt + sigma_chi exp(-kappa (T_i-t)) dW_t`.

Instantaneous log-return innovations are perfectly correlated, but the diffusion of the log ratio is:

`sigma_log_ratio = sigma_chi |exp(-kappa tau_1) - exp(-kappa tau_2)|`.

This is non-zero at different maturities when kappa is positive. The EUR spread also moves because its diffusion loading is the difference between price-weighted forward loadings.

One state variable does not, by itself, imply Pearson correlation one for arbitrary nonlinear functions of that state. The precise claim belongs to the specified log-return innovations. The existing finite tree has boundary and discretisation effects and does not explicitly publish a full conditional forward-curve process.

**Correct diagnosis:** the current specification restricts the joint covariance structure too much. The statement that it cannot price any seasonal spread because the spread cannot move is false.

### 6.2 The long-term factor cancels from the log forward ratio

Subtract the two equations in the design:

`log F(t,T_1) - log F(t,T_2) = (a_1-a_2) chi_t + A(t,T_1) - A(t,T_2)`.

The `xi_t` term cancels. At fixed short-factor parameters, adding `sigma_xi` does not directly increase the log-ratio diffusion above. It changes marginal forward volatilities and their correlations and can affect monetary spreads. A joint calibration can consequently estimate different short-factor parameters. These mechanisms must be distinguished.

### 6.3 A stronger counterexample: the long factor need not add storage value

For zero fixed cash fees, proportional fuel loss, deterministic discounting, price-independent physical constraints and hard terminal conditions, storage cashflows are homogeneous of degree one in gas prices.

An independent small DP used 24 decision dates, 81 short-factor grid points, four inventory levels, an OU transition matrix and a recombining independent long-factor martingale. Full two-factor backward induction was compared with a separate one-factor recursion. The terminal constraint was hard.

| kappa | One-factor value | Two-factor value with long-factor volatility 0.8 |
|---:|---:|---:|
| 0.2 | 24.9155727342 | 24.9155727342 |
| 1.0 | 25.1009944499 | 25.1009944499 |
| 4.0 | 26.3719389308 | 26.3719389308 |

Values are EUR for the small three-MWh-capacity example. The maximum absolute difference in these zero-fee cases was below `1.4e-13`. Adding 1.5% proportional fuel preserved this equality. Introducing a fixed EUR 1/MWh fee on each leg broke the invariance, as expected.

This is an exact property of the constructed discrete model, not a claim that the example is a calibrated production Schwartz–Smith engine. It disproves a universal requirement that introducing a common long factor must increase storage extrinsic or reverse its response to kappa.

There is a corresponding continuous-time reduction. For a forward-fitted model write:

`S_t = F_0(t) exp(chi_t + xi_t - 0.5 Var(chi_t + xi_t))`,

where the stochastic components start at zero, `chi` is OU and `xi` is Brownian. Define the martingale `L_t = exp(xi_t - 0.5 Var(xi_t))` and the tilted measure with density `L_t`. Under this measure `chi_t` has mean `c(t) = Cov(chi_t, xi_t)`. The centred process `X_t = chi_t - c(t)` has the same OU mean-reversion and diffusion coefficients. Therefore:

`S_t / L_t = F_0(t) exp(X_t - 0.5 Var(chi_t))`.

Each price-proportional cashflow expectation becomes an expectation under the tilted measure with the one-factor price above. Additional observation of the long factor cannot improve this Markov control problem when physical admissibility is price-independent. This explains the reduction. Fixed cash fees, strikes, non-homogeneous terminal cashflows or other assumptions can prevent it.

**Implication:** investigate this reduction before committing to a full extra price-state dimension for the zero-fee storage case. A second factor can still be needed to describe market risks and to value other products, and joint calibration can materially change the short-factor estimate. The number of market factors and the minimum valuation-state dimension need not be identical.

### 6.4 The mean-reversion sign is not an acceptance test

At fixed `sigma_chi = 0.5`, with maturities 0.5 and 1 year, annualised log-ratio volatility is:

| kappa | Log-ratio volatility |
|---:|---:|
| 0.2 | 0.04305 |
| 1.0 | 0.11933 |
| 4.0 | 0.05851 |

It is non-monotonic. Higher kappa simultaneously changes persistence, maturity damping, conditional predictability and cycling opportunities. Storage value is sensitive to the operating contract and to whether sigma or stationary variance is held fixed.

The independent two-factor example above retains an increasing value across these kappa values. Its intrinsic benchmark is unchanged by kappa, so extrinsic increases as well. Requiring a sign flip would reject a valid model on a false economic premise.

### 6.5 The ninefold claim is not established

The design's interim formula divides the loading difference by `sqrt(2*kappa)`. That converts the OU factor's stationary standard deviation into a log-ratio level standard deviation. It is not the instantaneous annualised volatility of daily log-ratio changes.

For a finite observation horizon h, integrated log-ratio variance is instead:

`sigma_chi^2 (exp(-kappa (T_1-h)) - exp(-kappa (T_2-h)))^2 * (1-exp(-2*kappa*h)) / (2*kappa)`,

with times measured from the valuation date and fixed delivery dates after h. Annualising this quantity requires specifying h. Daily returns, finite-horizon option variance and stationary level variance cannot be interchanged.

For the illustrative c6/c12 point-maturity approximation, sigma 0.5 and kappa 1 give **0.11933**, rather than 0.041. The observed no-roll value is 0.37776, a ratio of about 3.17 for this specific comparison. This is not a corrected universal shortfall estimate: actual delivery periods and model parameters must first be pinned down. The document does not supply enough detail to reconstruct its 0.041 calculation.

One spread-volatility observation also supplies only one equation for sigma and kappa; it cannot identify both without another observation or a restriction.

### 6.6 The 4.2% baseline is not reproducibly specified

I did not reproduce the stated 4.2% baseline or the exact 1.1% / 10.5% endpoints from the current documented examples:

| Explicitly reproduced example | Extrinsic as share of total at kappa 1 |
|---|---:|
| Current 30/60 notebook configuration, including mild ratchets and 70% floor, 10% funding | 19.6205% |
| Flat-month 30/60 example without ratchets or bounds, 10% funding | 23.7662% |
| Seasonal stepped 30/60 example, 10% funding | 3.5544% |
| Smooth cosine 30/65 notebook configuration, zero funding | 1.4367% |

This does not disprove that another configuration gives 4.2%. It means that the value and the claimed causal explanation are unsupported as a reproducible baseline until the exact curve, dates, discount rate, grid, ratchets, costs and parameters are recorded with executable code.

## 7. Repricing identity: preserve it, but understand what it proves

Define signed traded quantity `Q_i`, positive for sales. For price-only cashflows under any admissible policy:

`V_policy = E[sum_i DF_i S_i Q_i]` and `delta_i = E[S_i Q_i]/F_i`.

Then `sum_i DF_i delta_i F_i = V_policy` follows by linearity of expectation. It does not require an exhaustive DP, an optimal policy, a lattice or one factor.

An independent 100,000-path Monte Carlo example with a deliberately simple non-optimal policy gives value **5.043195370050699**, against delta-based reconstruction **5.043195370050656**, an absolute difference of **4.35e-14**. An LSMC policy evaluated using the same cashflow accounting can satisfy the same sample identity. Regression and policy-estimation error must be tested separately, preferably using independent evaluation paths and meaningful benchmarks.

For cash fees the reconciliation is:

`V = sum_i DF_i delta_i F_i - expected discounted cash fees + terminal cashflows - expected penalties`,

with any other contractual legs included explicitly. A deterministic 100 MWh cycle with purchase price 20, sale price 30, and EUR 2/MWh fee on each leg has value **EUR 600**, while the gross delta-price sum is **EUR 1,000**. Subtracting **EUR 400** of fees closes exactly. The current notebook correctly notes that its own zero-fee identity has no cost leg; the design generalises it too far.

At a differentiable optimum, the interpretation as a local forward sensitivity additionally requires the envelope argument, fixed stochastic parameters and the specified multiplicative curve perturbation. Cashflow reconciliation by itself is not proof of a correct hedge. The optimised value is generally convex and piecewise differentiable in the forward vector, and homogeneous when the cashflows permit it; it is not jointly linear in all forward levels.

**Method choice:** retaining a lattice is reasonable for continuity, transparent finite-state policy enumeration and deterministic reproducibility. Losing this identity is not a valid reason to reject LSMC. Lattice policies are exact only for their finite discretised problem, which is particularly consequential given the ratchet results above.

## 8. Further design requirements before implementation

1. **Specify historical and pricing dynamics separately.** The drift, risk premia, seasonal deterministic component and delivery-averaged observation equation are missing from the bare equations. Reconcile historical estimation with a risk-neutral pricing process and exact initial-curve fitting. Do not treat a historical estimate near zero for `sigma_xi` as automatic vindication of the whole one-factor model.
2. **Do not assume Cholesky rotation creates two independent factor processes.** It decorrelates diffusion increments. Because one factor mean-reverts and the other does not, the transformed drift is generally coupled. Derive transition means, variances and covariance and demonstrate recombination with non-negative probabilities.
3. **Budget the actual inventory grid and forward pass.** The proposed 580 x 41 x 31 x 61 example has about 45 million nodes. A 240-clip inventory grid gives about 178 million; a 390-clip grid gives about 288 million. Two value slices help, but the current full probability array would still require about 1.42 GB or 2.31 GB respectively in float64. Stream the forward probability pass where possible. One additional Python loop is not a sufficient implementation estimate.
4. **Choose the action integer dtype from its range.** `int8` supports only -128 through 127. A refined 3,840-clip 30-day injection rate is already 128 clips/day. The suggested dtype would overflow unless the action representation changes or a wider dtype is selected.
5. **Handle the one-factor limit explicitly.** Mathematical convergence as `sigma_xi -> 0` is required. Machine-precision agreement with the existing tree requires an explicit reduction using the same one-factor discretisation, boundaries and conventions. A separately constructed lattice does not automatically have bitwise equivalence.
6. **Use appropriate acceptance gates.** Analytic moments and covariances; forward repricing; joint transition validity; hard physical feasibility; full cashflow reconciliation; controlled one-factor and zero-volatility limits; convergence in both factor grids and inventory/actions; independent small-contract valuation; finite-difference risk away from policy kinks; and empirical out-of-sample covariance/spread diagnostics. Neither larger extrinsic nor a prescribed kappa sign is a valid gate.

The absolute `1e-6` tie rule remains a scale-dependent approximation. The tests cited in the findings may justify leaving it low priority for the measured cases, but identical cash in particular examples does not prove that it cannot ever change value. Distinguish exact equal-value policy selection from discarding a small positive gain.

## 9. Recommended sequence

1. Repair the dated-bound implementation: conservative rounding, correct inventory accounting, pathwise checks and hard constraints. Add small independent counterexamples to the suite.
2. Establish ratchet convergence and preserve actual contractual MWh rates. Correct the findings and tests that currently label a coarse-grid artefact as physical infeasibility.
3. Create a versioned executable benchmark for the claimed 4.2% and volatility comparison. State the spread definition, option/observation horizon, delivery periods and annualisation.
4. Build a delivery-aware return/covariance diagnostic with seasonal and regime analysis. Compare the candidate loading structures before committing to a full Kalman implementation or numerical architecture.
5. Specify P/Q dynamics and calibration targets. Investigate the homogeneous-storage dimension reduction. Choose lattice, reduced-state DP or LSMC using accuracy, contract coverage and measured runtime/memory.
6. Implement the selected process separately, validate the process and numerical limits, then extend the physical valuation engine. Reopen the out-of-sample/backtest work explicitly for any claim of market improvement.

P4.1 remains a worthwhile model-development investigation. The current evidence supports better covariance modelling and calibration, but it does not support the present rationale for an immediate full two-factor lattice build.

## Evidence archive

The accompanying archive contains the independent scripts, numerical outputs, the complete pytest log and source hashes. It excludes the original project code and market workbook, which remain in the project. `REPRODUCTION.md` explains how to run the checks against the reviewed files. Results are labelled as reproduced implementation behaviour, independent constructions, analytic deductions or exploratory estimates above; none is presented as a completed market validation.
