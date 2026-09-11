# Storage project review after the evening repairs

10 September 2026. Reviewed against GitHub `main`, commit [`b66d0a4b9a99b3772258007401bceefa24142f33`](https://github.com/dmitry-goryunov/Storage/commit/b66d0a4b9a99b3772258007401bceefa24142f33), and the corresponding Drive working files.

**The project has made real progress: the three inventory-bound repairs reproduce, terminal infeasibility is handled more reliably, and the ratchet convergence results are now executable. However, calibration is not the only substantial item left. Physical input conversion still changes contracts silently, and the new calibration probe and convergence reporting contain errors.**

This review changed no production code or project configuration. The accompanying evidence archive contains the review checks and their outputs.

## Evidence and scope

- **Canonical source:** 37 inspected source, test, notebook, data and documentation files match the GitHub commit's blob hashes, allowing CRLF/LF normalisation for text. No mismatches were found. The commit's complete tree was checked; no `AGENTS.md` was present.
- **CI independently checked:** the [current GitHub Actions run](https://github.com/dmitry-goryunov/Storage/actions/runs/34538911785) succeeded. Its log reports **116 passed in 81.92 seconds**, using the lockfile dependencies.
- **Local suite:** **115 passed; one Git-dependent provenance test could not run successfully** because the extracted snapshot has no Git repository metadata. This is an environment limitation, not a demonstrated product regression. The property it checks, that `ttf q.xlsx` is tracked, was separately verified against the canonical Git tree.
- **Independent checks:** 48 exhaustive four-day schedule cases, the earlier adverse-price floor examples, physical-rate counterexamples, transition-moment calculations, cache substitution, convergence-gate counterexamples, and the complete `benchmarks.py` and `two_factor_probe.py` runs.
- **Environment:** Python 3.12.14, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1 and Numba 0.67.0. The principal numerical dependencies match the lockfile. Timing comparisons are indicative only; some runs overlapped.

This is software and numerical verification. It is not a market calibration, historical trading backtest, independent market-data verification or complete production-readiness assessment.

## Findings requiring action

| Priority | Finding | Evidence status |
|---|---|---|
| P1 | Days-based physical rates are still silently rounded into a different contract | Reproduced through the public conversion and valuation functions |
| P1 | The probe's terminal-variance anchor does not match its lattice variance | Reproduced from transition moments and repricing |
| P1 | Variance-allocation scenarios are described as calibration results | Unsupported generalisation; assumptions and counterexample identified |
| P1 | An explicit workbook path can return an unrelated global cache | Reproduced with a separate one-row workbook |
| P1 | Convergence reporting can certify the wrong configuration; the gate accepts invalid inputs | Source inspection and independent counterexamples |
| P2 | Monthly delivery averaging is incorrectly described as always lowering the model comparison | Analytic counterexample |
| P2 | Current conventions and status documents disagree with the implementation and each other | Confirmed source discrepancies |

### 1. Physical-rate conversion remains a contract-changing defect

In `storage_model.py`, `params_for_run_valuation()` still derives both rates using `max(1, round(n_states / days))` (lines 1607 and 1616 onward). The new ambiguity guard in `resolve_grid()` helps distinguish grid size from days to fill, but it does not ensure that the days-based contract survives conversion.

The following unratcheted deterministic examples all use 600,000 MWh capacity, the recorded monthly curve, the 2027 operating window and 10% discounting:

| Requested fill / withdrawal | Grid clips | Requested withdrawal, MWh/day | Actual withdrawal, MWh/day | Actual days to empty | Value, EUR |
|---|---:|---:|---:|---:|---:|
| 30 / 90 days | 30 | 6,666.67 | **20,000.00** | **30** | 2,514,008.52 |
| 30 / 90 days | 90 | 6,666.67 | 6,666.67 | 90 | 2,382,639.26 |
| 30 / 65 days | 60 | 9,230.77 | **10,000.00** | **60** | 2,447,597.82 |
| 30 / 65 days | 390 | 9,230.77 | 9,230.77 | 65 | 2,436,687.60 |

The coarse 30/90 case has three times the requested withdrawal capacity and is worth **EUR 131,369 more, or 5.51%**, than the exactly representable example. These are different physical contracts, not estimates of discretisation error for one contract.

All four cases report zero ratchet-rate loss. The diagnostic cannot detect the error because it takes the already-rounded base rate as the contractual rate. Marking the whole of P1.4 done is therefore premature.

**Remedy:** retain requested capacity and physical MWh/day rates separately from effective grid quantities. Validate both before valuation. Select an exactly compatible grid where practical, or reject an incompatible request with the required refinement. If conservative rate approximation is permitted, report and gate that change explicitly; never increase a contractual maximum by nearest rounding or a minimum-one-clip rule. Apply the same input-preservation discipline to rounded capacity and initial/terminal inventory.

**Acceptance:** the 30/90 and 30/65 examples either preserve their physical limits or fail clearly before valuation. Both notebook and workbook/API routes must obey the same rule.

### 2. The terminal-variance comparison does not hold terminal variance fixed

`two_factor_probe.py::_ou_lattice()` uses the Euler transition mean `(1 - kappa*DT)*chi` and innovation variance `sigma_chi**2*DT`. In contrast, `matched_sig_chi()` uses the continuous-time OU variance formula. It also defaults to a two-year horizon, while the 24 decision dates run from zero to **23/12 years**.

At `kappa=4`, `sigma_xi=0.1`, and baseline `sigma_chi=0.6`, the actual transition laws give:

| Quantity | Value |
|---|---:|
| Baseline log-price variance at the last decision | 0.0539999996 |
| Supposedly matched two-factor variance | 0.0491666637 |
| Variance mismatch | **8.95% lower** |
| Current matched `sigma_chi` | 0.4472135754 |
| `sigma_chi` matching the existing lattice at its actual last decision | 0.4818944088 |
| Baseline struck-swing value | 36.32895830 |
| Current terminal-anchor swing value | 32.61429691 |
| Correctly variance-matched value on the same lattice | 33.46227862 |
| Reported effect | **−10.2251%** |
| Corrected comparison on that lattice | **−7.8909%** |

The negative sign survives this correction. The quoted −10.23% is nevertheless not the result of the stated equal-terminal-variance experiment.

**Remedy:** use consistent process moments, decision dates and matching horizons. Either construct and validate an exact-moment OU discretisation, or match the actual discrete transition law and label that experiment accordingly. Add direct mean/variance/covariance checks and numerical convergence before interpreting percentage effects. The −7.89% above is a diagnostic correction within the existing lattice, not a replacement market-calibration result or a converged continuous-time valuation.

### 3. The probe demonstrates variance-allocation scenarios, not a calibrated TTF haircut

The useful result is that allocating a fixed variance budget differently can change value, and that a scalar anchor does not identify the economically relevant term structure. That supports doing a panel calibration before selecting a production architecture.

However, the probe fits no market observations. It fixes mean reversion and independence, chooses a long-factor volatility, and reduces the short-factor volatility algebraically to preserve one selected scalar. “A calibrated second factor costs a store value” and “loses up to 8%” overstate what this establishes.

The direction of parameter reallocation is itself conditional. In a correlated model, instantaneous spot variance is

`sigma_chi² + sigma_xi² + 2*rho*sigma_chi*sigma_xi`.

For a target variance of 0.36, `sigma_xi=0.3` and `rho=-0.9`, a positive solution is **sigma_chi=0.85557664**, above the one-factor value of 0.6. This is an analytic counterexample to the unconditional claim that adding a factor necessarily lowers the fitted short-factor volatility. It is not a proposed parameter set. A joint panel fit can also change mean reversion, so even fixing the correlation would not make the probe a general calibration result.

**Remedy:** call these “variance-allocation scenarios under fixed mean reversion and independent factors”. Present the model outputs as conditional illustrations. Whether an empirically fitted additional factor raises or lowers the project's storage valuation remains **unknown**. Retain the fixed-short-factor comparison as a valid control that isolates the common-factor effect; it answers a different question rather than an invalid one.

### 4. The workbook cache can substitute a different data source

`benchmarks.py::load_quote_matrix(path)` checks the single module-global `_PARQUET_CACHE`, even when a different workbook is supplied. If that cache is newer, it returns it without checking which workbook generated it.

I supplied a separate workbook containing **one row with TTFc1=999** and an older modification time. The loader returned **4,171 rows**, beginning with TTFc1=10, from the repository's cache instead. A request to analyse another dataset therefore silently analyses the original one.

The fetched default cache and canonical workbook do agree in their current values after normalising timestamp resolution. This finding does not invalidate the reproduced default correlations. It disproves the stronger claim that the cache and an explicitly requested workbook cannot disagree, and exposes a risk for the next calibration dataset.

**Remedy:** bind caches to the resolved source path and a content fingerprint, plus the cleaning/schema version. Bypass the default cache for explicit alternative inputs unless their identity matches. Preserve a no-cache path for reproducible research. Add a two-workbook test and a stale-cache test. The current test that the declared workbook is tracked does not test which bytes are actually consumed.

### 5. Convergence evidence is real, but the reporting is not reliable for changed inputs

The default soft-ratchet ladder reproduces:

| Inventory clips | Total value, EUR | Intrinsic, EUR | Total change from preceding row |
|---:|---:|---:|---:|
| 240 | 2,941,740.07 | 2,364,555.82 | n/a |
| 480 | 2,993,114.73 | 2,383,333.81 | 1.746% |
| 960 | 3,015,923.74 | 2,390,303.74 | 0.762% |
| 1,920 | 3,026,417.92 | 2,393,044.86 | 0.348% |
| 3,840 | 3,032,172.02 | 2,394,740.62 | 0.190% |

The 3,840 endpoint passes the stated two-successive-refinement gate for both total and intrinsic value. The shipped 1,920 setting is **0.1898% below the 3,840 result**. That is a measured difference to a finer grid, not a proven error bound against the continuous problem.

Two implementation issues remain:

1. **The notebook's convergence message is hard-coded.** `Storage_30_60.ipynb`, input cell 3, selects `_RESIDUAL` solely by `N_STATES`, before the current curve and dated bounds are even defined. At 3,840 it prints “converged” regardless of the edited contract. Yet the harsher ratchet at 960/1,920/3,840 gives EUR 2,019,762.98 / 2,134,479.76 / 2,182,561.32. Its last two changes are **5.680% and 2.253%**, and the executable gate correctly returns false. Changing to that profile and selecting 3,840 does not justify the notebook's unconditional certificate.
2. **The general gate fails open on invalid or unsuitable input.** `benchmarks.py::convergence_verdict()` returns true for three NaN-valued rows and for three repeated copies of the same grid. Its EUR 1,000 absolute fallback also passes values 10,000 → 9,500 → 9,000 despite approximately 5% changes. These are constructed gate inputs, not claimed outputs of the present benchmark valuations.

**Remedy:** compute the certificate from the current configuration, or bind stored benchmark results to a complete input/source fingerprint and label them as historical otherwise. Reject non-finite values and require the intended successive grid refinements without silently skipping failed rungs. Define explicitly when the absolute tolerance applies and scale it to the intended contract use. Preserve the distinction between the rate-loss gate and a valuation convergence gate.

The 10% default rate threshold is a useful screening choice, but neither it nor the name “expressible” establishes value accuracy. At a fixed inventory grid, the existing price-width ladder also reproduces; that is evidence for that benchmark, not a universal guarantee after changing the contract.

### 6. Delivery averaging is not a universal downward correction

`benchmarks.py` and the evening findings call the `tau_i=i/12` point-maturity model comparison an upper bound because monthly delivery allegedly lowers volatility. That direction cannot be asserted without defining what point in the delivery period `tau_i` represents and how the delivery forward is weighted.

For flat initial forwards within a delivery period `[A,B]`, the one-factor instantaneous log loading at the initial date is

`a_bar = (exp(-kappa*A) - exp(-kappa*B)) / (kappa*(B-A))`.

For equal one-month periods ending at 0.5 and 1 year, `kappa=1` and `sigma=0.5`, the log-ratio volatility is **0.12443854**, above the month-end point approximation **0.11932561**. This is a counterexample to the asserted upper bound, not an alternative empirical TTF estimate.

**Remedy:** remove the directional “upper bound” claim. Specify actual delivery start/end dates and the appropriate price weights. Keep the 3.1× comparison explicitly illustrative until this is done.

### 7. The current documentation needs reconciliation

The correction notices are useful, but several active sources still conflict:

- `MODEL-CONVENTIONS.md`, the document users are told to read before using a number, still states that dated bounds are a `1000*v_step` penalty rather than hard constraints (around lines 376–389 and its constants table).
- The notebook's dated-bound comment also still says “penalty rather than a hard constraint”.
- STATUS describes the current grid both as 0.53% short and as having a 0.19% residual. The response still quotes 0.53% where the current 1,920-grid comparison is 0.1898%.
- The roadmap retains the old open-bound description, ninefold comparison and three-day DA-gap claim under a broad correction notice. Its operational priorities no longer reliably identify completed work.
- Claims that all 34 files still match the original review snapshot need to be explicitly dated to the pre-repair verification. Source files have since changed; current provenance should identify the new commit.

**Remedy:** keep dated historical findings intact with clear correction notices, but make the conventions, current status and roadmap describe one current implementation and one current set of unresolved issues.

## Repairs that withstand this review

The inventory-bound work should be credited as a substantive correction. Across **48 independently enumerated four-day cases**, with non-zero starts, varying terminal inventory, daily rates, ratchets, dated floors/ceilings, fees and fuel, the engine matched all feasible optima and rejected all infeasible cases: **20 feasible, 28 infeasible**, with maximum value discrepancy **2.13e-14**.

At the earlier EUR 300, EUR 1,000 and EUR 10,000 adverse-price levels, the actual optimiser-generated policies now carry **zero probability below the two-clip floor**. They retain positive value. The existing tests also verify conservative rounding and the full-start inventory example.

The repricing/fuel benchmarks reproduce, as do the raw forward correlations and the common-long-factor invariance controls. The ratchet diagnostics reveal the sawtooth rate loss and the refinement ladder overturns the original physical-cap interpretation correctly. At 240 clips the soft-profile worst withdrawal-rate loss is **33.1%**; it falls to **5.83% at 1,920** and **3.00% at 3,840**.

These results support closing the specific bound defects. They do not support closing all physical-rate accuracy work or describing the numerical engine as universally verified. The kernel still represents infeasibility with large finite sentinels; the checks here support its behaviour at the tested scales, not the literal phrase “at any price”. An explicit admissibility representation would remove that remaining dependence on monetary scale.

## Recommended next work

1. **Finish physical input preservation.** Repair days-to-rate conversion and report requested versus effective capacity, rates and inventory. Reopen the relevant part of P1.4. Add the demonstrated 30/90 and 30/65 cases as acceptance tests.
2. **Make numerical evidence trustworthy.** Bind convergence results to live inputs; make invalid inputs fail the gate; repair the workbook cache identity. These are prerequisites for a calibration harness that can be trusted.
3. **Correct the probe and current documentation.** Align transition moments and matching horizons; relabel its outputs as conditional scenarios; remove the universal haircut and delivery-averaging claims. Keep the surviving sign comparison as a qualified experimental observation.
4. **Then perform delivery-aware panel calibration and model comparison.** Compare the one-factor baseline, Schwartz–Smith and a parsimonious curve-shape alternative, with historical/pricing assumptions, estimation windows, parameter stability and held-out diagnostics stated explicitly. Current market calibration still requires data beyond the workbook's 6 March 2026 endpoint.
5. **Choose production architecture from the selected process and contract scope.** Preserve the homogeneous-contract reduction where applicable, and test fixed-fee storage and struck swings separately. Do not use higher extrinsic value or a prescribed mean-reversion sign as acceptance criteria.

The next code task is the physical-rate conversion repair. The additional price factor remains a research question; the present probe does not establish that current storage values should be reduced by 8%.
