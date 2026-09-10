# Reply to the review response and proposed course of action

10 September 2026. In reply to [REVIEW-RESPONSE-2026-09-10.md](https://drive.google.com/file/d/1k296nMP3l8XANUED6DSxGK2O_1fxXgsp/view), read alongside the [independent review](https://drive.google.com/file/d/1lhcQyy9lSOlJY9b3otdB1w2LXF0hL7kT/view).

This is a proposed implementation plan. It does not record completed fixes or authorise a production release.

## Reply

The response establishes substantial agreement on the defects and on why the original P4.1 design needs revision. Reconstructing the examples from the prose is useful independent evidence. I agree that the next work should address contractual feasibility and numerical accuracy before adding a price-state dimension.

The inventory-bound defects, the ratchet grid artefact, the fuel caveat and the objections to the repricing and mean-reversion acceptance arguments remain supported. The response also reports a 2.52% increase in the shipped notebook's value between 240 and 960 clips. I have read that additional result, but have not independently rerun those particular refinements. It strengthens the case for a convergence requirement if reproduced in the benchmark harness. The response and updated status explicitly say that nothing has been fixed yet.

Three qualifications should carry into the revised documents and roadmap.

**First, the homogeneity result narrows the case for the proposed Schwartz–Smith lattice; it does not eliminate the storage case for additional market factors.** The strongest version of the response's argument is correct: with the specified common multiplicative long factor, fixed short-factor parameters, deterministic discounting, price-independent constraints and homogeneous cashflows, the extra valuation dimension can integrate out. Proportional fuel loss preserves that result. Fixed cash fees, strikes and non-homogeneous terminal cashflows can break it. A further mean-reverting or seasonal factor that changes relative prices is also outside this particular reduction. Moreover, a joint market calibration can change the short-factor estimate even when valuation subsequently reduces to one state. Consequently, market-model selection and valuation-state reduction must remain separate decisions. Calling the outstanding problem exclusively a one-factor calibration problem is premature.

**Second, approximately 3.1 times is an illustrative comparison, not an established universal replacement for nine times.** At the stated point-maturity approximation and parameters, the model log-ratio volatility is 0.11933. Comparing this with the pooled raw estimate of about 0.373 gives approximately 3.13; the no-roll estimate of 0.37776 gives approximately 3.17. Actual delivery averaging, observation alignment, estimation window and parameter provenance still need to be specified. The original ninefold claim is unsupported as written; a definitive replacement remains unresolved.

**Third, the review did not dispute the observed increase in extrinsic value with mean reversion.** Section 6.4 explicitly demonstrated that increase in the independent two-factor example. The objections were to the unreproduced numerical baselines and to requiring a reversal of the sign as an acceptance test. The response's new sensitivity results are compatible with that position.

The 34.2% withdrawal-rate loss at 10% fullness is a useful additional example. The review's 16.7% figure concerned a different inventory level and explicitly allowed larger errors elsewhere. Also, a 70.59% increase from the coarse-grid value to the finer-grid value corresponds to a 41.38% shortfall measured against the finer-grid value. Neither figure measures error against a converged continuous-volume answer, which is still unknown.

I support pausing the proposed full two-factor lattice build. I would replace its immediate implementation scope with a reproducible calibration and model-comparison investigation, while retaining the possibility of additional curve-shape factors and the separate struck-swing experiment.

## Proposed course of action

### 1. Correct the record and establish reproducible benchmarks

Add explicit correction notices to FINDINGS and DESIGN, and update STATUS and the notebook narrative. Preserve the dated history and links to the review and response. Withdraw the physical interpretation of the 52.5% cap and 70% floor rejection, the general claim of answer-neutral refinement, the ninefold claim, the undocumented 4.2% baseline and the assertion that LSMC loses cashflow reconciliation. Retain the qualifications on fuel, data gaps and mean reversion.

Put the benchmark inputs in executable fixtures: capacity, actual daily MWh rates, ratchets, dates, opening and terminal inventory, bounds, curve construction, fuel, fees, discounting, price parameters and both grids. Record the source revision and environment with each result. Include the shipped 30/60 case, the harsher ratchet case, the three bound counterexamples and an unratcheted control.

**Completion criterion:** another run reconstructs each retained headline result from named inputs. Unreproduced historical numbers remain labelled as such. New tests target contractual behaviour or independent calculations, rather than preserving the old incorrect outputs.

### 2. Repair inventory bounds as the first code change

Treat the documented dated inventory requirements as hard opening-inventory constraints. This is the proposed default; confirm the existing date/index mapping in code before changing it. Preserve any deliberately soft operating target only through an explicitly separate penalty mechanism.

- Convert floors with conservative ceiling and ceilings with conservative floor, allowing only a documented floating-point tolerance around exact grid points. Report requested and effective bounds; reject empty admissible grid intervals.
- Enforce admissibility in the optimiser. Propagate infeasible continuation states correctly, including states from which a future hard bound or terminal requirement cannot be met. A finite economic penalty must not make a contractual breach purchasable.
- Verify inventory directly from the state distribution, including non-zero opening inventory. Check probability mass outside the permitted interval at each relevant date, separately for the intrinsic and stochastic policies.
- Use a documented numerical tolerance for accumulated probability error. It is not permission for an economically chosen breach, and must not be enlarged to make a failed case pass.
- Retain independent checks of flow balance, action limits and terminal inventory, so a correct bound result cannot conceal a different feasibility defect.

**Completion criterion:** the full-start example is accepted correctly; the 71% floor and 29% ceiling cannot relax to 70% and 30%; the adverse-price examples cannot return a policy with contractual breaches. A feasible alternative is selected or genuine infeasibility is reported. Small enumerated schedules agree with the optimiser, and both valuation passes satisfy the same contract.

### 3. Establish a ratchet discretisation policy

Keep physical capacity and MWh/day rates fixed while refining the inventory grid. Report the permitted physical rate and effective grid rate by inventory level, including both absolute MWh/day loss and relative loss. Retain the zero-rate guard, but do not present it as an accuracy certificate.

Reproduce the existing refinement ladder and the response's additional shipped-notebook ladder. Compare deterministic feasibility with an independent continuous-volume calculation where applicable. Replace the test that encodes the 52.5% artefact as a physical limit. Ensure action storage can represent the refined rates; the proposed `int8` representation cannot hold a positive action of 128 clips.

Start with grid refinement and explicit diagnostics. If this cannot meet the required accuracy at an acceptable runtime, assess fractional-volume actions or interpolation against an independent small-contract oracle before changing the numerical scheme.

**Proposed initial gate:** less than 0.5% change in both total and intrinsic value over each of two successive grid doublings on the named material-value benchmarks. Specify an absolute EUR tolerance for zero or near-zero values. These are proposed engineering thresholds, not achieved results or established commercial tolerances. Report runtime, inventory profiles and hedge changes alongside value; equal-value policy switches can move hedges without demonstrating a valuation error.

Feasibility needs its own gate. A coarse-grid rejection must be labelled as grid infeasibility until continuous-volume reachability or an equivalent independent argument establishes physical infeasibility. The known 70% example must cease to be reported as physically impossible.

**Completion criterion:** the published notebook uses a grid that passes the declared numerical gate, or clearly reports that convergence is unresolved. Accuracy is not inferred from an unratcheted control. A separate price-grid check must prevent inventory refinement from concealing price discretisation error.

### 4. Rescope P4.1 around market evidence

Combine the overlapping parameter-provenance and calibration items into one investigation with explicit alternatives. Do not commit in advance to either one factor or a Kalman implementation.

| Work | Required output |
|---|---|
| Data and conventions | Delivery identifiers and periods, roll treatment, missing observations, return definition, annualisation, estimation window and as-of date. The available workbook ends on 6 March 2026; it cannot support a current September calibration without newer data. |
| Reproducible diagnostics | Marginal volatilities, correlations and target spread volatilities, split by relevant periods. Reproduce or challenge the exploratory covariance fit and PCA calculations, which the response did not check. |
| Candidate comparison | One-factor OU baseline, Schwartz–Smith, and a parsimonious alternative with different mean-reversion speeds or seasonal curve loadings. Explain any excluded candidate. |
| Estimation | A delivery-aware observation equation and identifiable parameterisation. Use Kalman estimation where the selected state-space model warrants it; report fit stability, boundary solutions and sensitivity to the sample. |
| Pricing interpretation | Separate historical dynamics from pricing dynamics, state risk-premium assumptions and fit the initial curve. Historical covariance agreement alone does not establish a tradable valuation. |
| Selection | Held-out marginal and spread diagnostics, parameter stability and impact on the benchmark contracts. Higher storage value is not a model-selection criterion. |

**Completion criterion:** a documented comparison establishes what each candidate explains and misses. If the available data do not identify a robust model, that remains an explicit unresolved result. A full trading backtest can stay deferred, but held-out statistical validation belongs inside this calibration stage.

### 5. Decide whether an additional valuation dimension is justified

Run small controlled experiments before building a production engine. Compare the homogeneous zero-fee store, the same store with fixed cash fees, and a struck swing. Hold the initial curve, physical contract and short-factor parameters fixed when isolating the common long factor; show separately what changes after recalibration.

For any selected factor that changes curve shape, test its own implications rather than assuming the common-factor reduction applies. Check whether an exact reduction is available under the intended contract before choosing lattice, reduced-state DP or LSMC.

If another engine is justified, require analytic process moments and covariances, valid transition probabilities, initial-curve fit, hard feasibility, complete cashflow reconciliation, controlled one-factor and zero-volatility limits, grid or simulation convergence, independent small-contract comparisons and finite-difference risk checks. Evaluate LSMC policies on independent paths. Budget the probability pass and action representation as well as the backward value recursion.

**Completion criterion:** the chosen method meets the declared accuracy and contract-coverage requirements at measured runtime and memory cost. Neither a prescribed mean-reversion sign nor larger extrinsic value is a gate.

## Delivery order

Use separate reviewable changes for documentation and benchmarks, inventory-bound enforcement, ratchet accuracy, and calibration diagnostics. The first two numerical repairs have equal urgency as correctness issues; implement bounds first because the scope is smaller and the hard constraints must also govern the refined ratchet cases.

After each code change, run the existing suite and the relevant independent counterexamples. Explain intended movements in value and feasibility instead of merely replacing regression anchors. Update STATUS with completed work, remaining limitations and the exact benchmark revision. Keep the tie threshold and terminal-backstop optimisation at their existing lower priority unless the repairs expose a concrete dependency.

The next implementation task is therefore the bound repair, accompanied by the correction notices and reproducible fixtures. Numerical convergence follows; calibration and architecture decisions follow a trustworthy numerical baseline.
