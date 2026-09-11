# Storage: detailed implementation and investigation guide

11 September 2026. Prepared for the Storage project following the findings, P4.1 design, review response and independent evening review.

**Recommended course: preserve the requested physical contract first, make data selection and numerical acceptance reliable, then complete a properly specified calibration investigation. Select a production two-factor valuation method only when those results justify it.**

This is an implementation specification and a tested acceptance pack. It does not claim that the proposed repairs have been implemented. The canonical source remains commit [`b66d0a4b9a99b3772258007401bceefa24142f33`](https://github.com/dmitry-goryunov/Storage/commit/b66d0a4b9a99b3772258007401bceefa24142f33), rechecked on 11 September. Proposed function names, record formats and additional files below are design choices, not existing APIs.

## 1. Reply to the review response

The inventory-bound repairs are accepted on the evidence. The independent exhaustive checks and the adverse-price examples support keeping those repairs. The executable benchmark configurations are also useful progress.

The remaining work is broader than calibration. The public input conversion still changes a requested 30/90-day store into a 30/30-day store on a coarse grid. The convergence gate can accept invalid results, and the notebook can attach an old convergence statement to new inputs. The quote loader can select a different dataset from the workbook requested. These affect what is priced and what the reported evidence means, so they precede economic interpretation of another factor.

The two-factor probe establishes conditional variance-allocation effects. It does not establish a calibrated TTF storage haircut. Its terminal anchor also mixes continuous-time moments with an Euler lattice and uses a horizon beyond the final decision date. The negative struck-swing effect survives a correction on the existing lattice, but that corrected number is neither a market estimate nor a converged continuous-time result.

I propose the eight delivery slices below. Each should contain the repair, a check against an independent requirement, and the corresponding current-document update. A green test suite should no longer preserve known contract-changing behaviour. P4.1 should become a calibration and model-selection investigation, with the valuation architecture chosen from its outcome.

## 2. Baseline: what is established and what remains open

| Item | Evidence | Treatment in this plan |
|---|---|---|
| Canonical source | 37 inspected files matched GitHub blobs, allowing text line-ending normalisation | Use the stated commit as the reproduction baseline |
| Existing suite | GitHub CI: 116 passed; extracted local snapshot: 115 passed and one Git-metadata-dependent failure | Keep the existing suite; do not count the extraction limitation as a pricing regression |
| Inventory bounds | 48 independent exhaustive cases: 20 feasible, 28 infeasible; maximum feasible-value discrepancy 2.14e-14 | Preserve the repairs and independent checks |
| Adverse-price floors | Breach probability zero at the three reviewed price levels | Keep these regression examples |
| Days-based rate preservation | Public valuation counterexamples reproduced | Open, first implementation priority |
| Data cache identity | Wrong-workbook and same-path/new-content cases reproduced | Open, before new calibration data are used |
| Numerical certification | Invalid tables accepted; hard-coded notebook status reused | Open |
| Two-factor terminal anchor | Actual final-date variance differs by 8.95% in the selected example | Open |
| Market effect of a fitted second factor | No joint market fit has been performed | Unknown |

The earlier full evidence is in `PROJECT-REVIEW-2026-09-10-evening.md` and its evidence archive in this directory. The new `ACCEPTANCE-PACK-2026-09-11.zip` contains 31 targeted checks, with **9 passing controls and 22 unmet acceptance requirements** on the unchanged baseline. These are deliberately concentrated on known gaps; 22 failures are not 22 independent defects or a project-wide failure rate.

Three additional details were reproduced while preparing this guide:

1. With capacity 600,000 MWh and 60 clips, a requested 5,000 MWh opening or terminal inventory becomes zero through nearest rounding. Exact initial and terminal quantities need the same protection as daily rates.
2. A synthetic workbook changed from TTFc1=10 to 999 at the same path, with its original modification time restored, still returns 10 from the cache. Path identity alone is insufficient.
3. A ratchet multiplier of 0.5 and daily base rates of one and two clips produce a 100% rounding loss on the one-clip day. `worst_ratchet_rate_loss()` reports zero because it examines only the maximum daily base rate. This is an isolated diagnostic example using model-shaped arrays, not a valuation of a market contract.

## 3. Delivery order and dependencies

| Slice | Deliverable | Main locations | Completion evidence |
|---|---|---|---|
| S1 | Strict physical-to-grid conversion | `storage_model.py`; conversion tests | Requested capacity, rates and boundary inventory preserved or rejected |
| S2 | One conversion route and complete rate diagnostics | Notebooks, `streamlit_app.py`, `benchmarks.py`, ratchet helpers | Entry-point parity; daily-rate diagnostics; no duplicate silent conversion |
| S3 | Source-bound quote loading | Proposed `quote_data.py`; `benchmarks.py`; `portfolio_app.py` | Cache hit/miss equivalence; alternate and changed sources read correctly |
| S4 | Configuration-bound convergence evidence | `benchmarks.py`; `Storage_30_60.ipynb` | Invalid inputs fail; evidence matches the current physical contract and model |
| S5 | Consistent two-factor research probe | `two_factor_probe.py`; probe tests | Moment matching, horizon identity and limiting-case checks |
| S6 | Historical panel and delivery observation specification | Proposed research module and recorded data manifest | Auditable delivery mapping, return construction and observation functions |
| S7 | Calibration comparison and decision note | Proposed calibration outputs; new P4.1 successor | Fit diagnostics, identification and held-out comparison; architecture decision |
| S8 | Current documentation and release reconciliation | `docs/STATUS.md`, `docs/MODEL-CONVENTIONS.md`, roadmap | Every current claim names its configuration, date and evidence |

S2 follows S1. S4 depends on S1 and S2 for a fixed physical contract. S6 depends on S3. S7 depends on S4, S5 and S6 before it can support valuation conclusions. S3 and S5 can be separate changes without waiting for S2. Update the affected documentation within every slice; S8 is the final cross-check, not a reason to postpone corrections.

Do not bundle a new multidimensional production kernel into S1 to S6. It would mix unresolved inputs, data and numerical assumptions with an architecture change and make differences difficult to attribute.

## 4. S1: preserve the physical contract before constructing a grid

### 4.1 Establish one explicit input contract

Treat capacity, physical rates, inventory quantities, dates, ratchets and fees as contract inputs. Treat inventory clip size and price-grid settings as numerical inputs. The implementation must retain both sets separately.

| Quantity | Physical meaning | Required handling |
|---|---|---|
| `capacity_mwh` | Working gas capacity | Positive, finite; exactly represented at the declared measurement precision |
| Injection and withdrawal MWh/day | Maximum movement into/out of working inventory per active day | Finite, non-negative; preserve separate directions |
| `inj_days`, `wdr_days` | Capacity divided by the corresponding unrestricted daily rate | Positive finite days in the physical interface; not an inventory-state count |
| Opening and terminal MWh | Exact initial quantity and exact terminal target under the existing storage product | Within capacity and representable; do not silently round |
| `n_states` | Number of inventory intervals; levels are 0 through `n_states` | Positive integer; report `n_states + 1` actual inventory levels |
| `v_step` | MWh per inventory interval | Positive finite; consistent with capacity and interval count |
| Ratchet factors | Multipliers evaluated at the documented inventory timing | Validate domain, interpolation and non-negative factors |

The current inventory rate convention is gas entering/leaving working inventory. Fuel adjusts purchased injection gas and cash flow. Preserve that convention: 1 MWh of inventory injection with fuel loss `f` purchases `1/(1-f)` MWh under the current model. Do not accidentally apply the same fuel adjustment to physical inventory movement twice.

For the existing workbook, keep the schema initially. Convert the days fields once, retain the resulting requested MWh/day rates, and use a normalised internal record. Supporting direct MWh/day inputs can be added through that same record. If both days and MWh/day are supplied, verify agreement; do not select one silently.

### 4.2 Implement the strict conversion in a pure helper

Proposed helper: `normalise_storage_contract(params)`, called by `params_for_run_valuation()` and the storage entry point. Keep it independent of Numba, plotting and workbook I/O so that incorrect inputs fail before allocation.

1. Copy the input mapping. Reject non-finite quantities, non-positive capacity, fractional state counts and impossible inventory targets. Permit zero rates only where a direct rate schedule deliberately specifies closure. A days-to-fill input cannot describe a zero rate with a finite positive day count.
2. Resolve the physical rates. For a days-based store, calculate `I = capacity / inj_days` and `W = capacity / wdr_days`. Preserve these values as the requested rates.
3. Resolve the grid from one unambiguous combination: capacity plus state count, or an explicit compatible clip size. If redundant inputs are provided, check all identities. In particular, capacity 600,000, clip size 10,000 and 90 intervals are contradictory.
4. Let `h = capacity / N`. Compute `I/h`, `W/h`, `opening/h` and `terminal/h`. Accept a quantity as integral only within the documented floating-point comparison tolerance. Round only after this representability test succeeds.
5. If a physical quantity is not representable, raise a targeted `ValueError` before constructing `Storage`. Include the requested quantity, current effective grid and a practical compatible-grid suggestion where one is available.
6. Check explicit clip fields against their physical equivalents. Supplying `initial_storage_mwh=20000` and `initial_inv_clips=1` at a 10,000 MWh clip must be rejected. An explicit field is not permission to discard the other field.
7. Return the physical record, grid record and derived integer clip quantities. Recheck the identities immediately before dispatch to `value_storage()` so that another caller cannot bypass them through a legacy parameter dictionary.

Suggested numerical identity tolerance: `abs(actual-requested) <= 1e-7 MWh + 1e-10*abs(requested)`. This is an engineering tolerance for floating-point representation, not a commercial allowance to alter capacity. Record it as a named constant. Validate business inputs at their declared decimal precision before this comparison.

### 4.3 Choose a compatible grid without hiding its cost

For integer fill/withdrawal days, exact base rates require `N` to be a multiple of `lcm(inj_days, wdr_days)`:

| Days in/out | Smallest base-rate-compatible N | Injection clips/day | Withdrawal clips/day |
|---|---:|---:|---:|
| 30/60 | 60 | 2 | 1 |
| 30/90 | 90 | 3 | 1 |
| 30/65 | 390 | 13 | 6 |
| 30/365 | 2,190 | 73 | 6 |

This condition only establishes exact base rates. Opening inventory, terminal inventory and ratchet approximation may require a finer grid. For example, 5,000 MWh opening inventory in a 600,000 MWh store requires `N` divisible by 120; a 30/90 store therefore needs at least `lcm(90,120)=360` intervals to express all three requirements.

For decimal rate or inventory inputs, use declared decimal units or rational ratios to find compatible denominators. Do not construct exact fractions from binary floating-point values and unexpectedly request billions of states. If the compatible grid is impractical, report that limitation and the memory estimate. Do not allocate it automatically.

Recommended default: strict rejection for an explicitly incompatible `N`. An optional automatic-grid mode may choose a compatible `N` before valuation if it reports the choice and respects a resource limit. The acceptance pack permits preservation or rejection; the additional strict-mode test must require rejection when a caller explicitly requests an incompatible grid.

A conservative approximation mode is a separate future option. It must floor maximum movements, disclose the loss and carry a distinct approximation status. Never use `max(1, round(...))` to create a movement that the contract forbids. Flooring a sub-clip rate to zero can make a feasible contract infeasible; the result must explain that numerical limitation.

### 4.4 Change the tests that currently preserve the defect

`tests/test_reconciliation.py::test_asymmetric_storage_rates_survive_the_days_to_rate_conversion` currently asserts rounded rates for incompatible grids, including a 30-state 30/60 request. Replace those expectations. Retain the compatible 60- and 120-interval examples and the end-to-end schedule check.

Add the following cases before changing the implementation:

| ID group | Inputs | Required result |
|---|---|---|
| R-compatible | The four day pairs above and a doubled 30/65 grid | Exact requested MWh/day preserved |
| R-incompatible | 30/90 at N=30; 30/65 at N=60 and 90; 30/60 at N=45 | Explicit refusal in strict mode |
| V-boundary | Opening/terminal 5,000 MWh at N=60, capacity 600,000 | Refusal, not zero inventory |
| V-conflict | Inconsistent physical and clip quantities or inconsistent capacity/step/N | Refusal naming the conflicting fields |
| V-capacity | Capacity 100 MWh with a declared 30 MWh clip | Refusal, not a 90 MWh store |
| R-binding | Short deterministic schedule with separate injection/withdrawal opportunities | Actual moves obey the original physical limits; both limits bind in at least one example |

Use the original requested numbers as the test oracle. A check that multiplies the newly rounded clip rate by the clip size and compares it with another derived field can pass while pricing the wrong contract.

The previously observed EUR 131,369 difference between 30/90 at N=30 and N=90 is an illustration of changed economics. It must not become a convergence target. After S1, the incompatible case should be refused rather than anchored to its old value.

**S1 completion:** conversion controls pass, incompatible requests fail clearly before valuation, and the existing feasible-bound and compatible-storage examples retain their behaviour.

## 5. S2: use the same conversion everywhere and measure all rate losses

### 5.1 Remove competing conversions at the call sites

1. In `params_for_run_valuation()`, delegate storage physical normalisation to S1. Preserve the distinct forced-cycle defaults for call and put swings. Do not impose storage opening/terminal defaults on them.
2. In `resolve_grid()`, validate redundant inputs rather than prioritising capacity and discarding `n_states`. Keep a clearly isolated legacy adapter if compatibility requires it. Do not allow the legacy `inj_days` state-count meaning to enter the physical interface.
3. In `benchmarks.storage_params()`, stop rounding requested physical rates independently. Route them through the shared normaliser and explicitly fail incompatible benchmark grids. Existing multiples-of-60 ladders remain valid base-rate examples.
4. In `Storage_30_65.ipynb`, replace `rates_for()` and its direct parameter construction with the shared conversion. Its current coarse-grid comparison intentionally demonstrates different contracts. Retain it only as a labelled historical illustration, or change it to a table of incompatible requests refused by the new API.
5. In `Storage_30_60.ipynb`, use the same physical record to construct every refinement. Remove the idea that an increased `REFINEMENT` alone proves adequacy.
6. Inspect `forward.ipynb`, `pricing.ipynb`, `Products.ipynb` and `Swing_new.ipynb` for direct legacy sizing. Migrate actual storage callers, and preserve genuine swing-specific contracts. Do not mechanically replace every `inj_days` occurrence.
7. In `streamlit_app.py`, the existing storage UI explicitly offers a symmetric legacy rate. Either expose physical capacity plus separate fill/withdrawal rates through the shared normaliser, or retain a clearly labelled symmetric product. Show requested versus effective physical quantities when a grid is selected. Keep grid internals in an advanced area, with plain error messages such as “This grid cannot represent a 90-day withdrawal rate; use at least 90 compatible inventory intervals.”

Parity acceptance: construct the same 30/90 storage through a workbook fixture, direct API input and the notebook helper. Compare their normalised physical records and deterministic values. Use a small in-memory workbook fixture; do not edit `products.xlsx` merely to make a test pass. Check the UI input-building function separately from an interactive app launch.

### 5.2 Separate base-rate alteration from ratchet discretisation

The current `describe_ratchet_rates()` starts from `curve.max()` and therefore loses both the original requested rate and smaller date-dependent rates. Replace its comparison inputs with the requested physical schedule and the actual admissible grid movements.

For each distinct active-day base-rate/profile combination and each inventory level `v = j*h`:

1. Compute the contractual rate `R(t,v) = requested_base_rate(t) * ratchet(v/capacity)` using the existing documented interpolation and opening-inventory timing.
2. Compute physical headroom: `capacity-v` for injection, `v` for withdrawal. The permitted one-day movement is `A(t,v) = min(R(t,v), headroom)`.
3. Compute the permitted grid movement conservatively: `G(t,v) = h*floor(A(t,v)/h)`, with a narrowly specified tolerance to avoid rounding an exact integer down through floating-point noise. Verify `G <= A` within that tolerance.
4. Report the absolute loss `A-G` and relative loss `(A-G)/A` when `A>0`. Do not call the lack of headroom a discretisation loss. Keep raw ratchet-rate loss as a separate diagnostic if it is useful.
5. Retain the worst date or rate bucket, direction and inventory level so that the reported maximum is reproducible. Grouping identical daily rates is sufficient; an enormous date-by-inventory report is unnecessary.
6. Gate the declared diagnostic and report it independently of value convergence. The existing 10% relative threshold is an engineering choice; it does not establish a 10% acceptable contractual or valuation error.

At an empty/full boundary, zero admissible movement is physical. At an interior state with a positive sub-clip rate, zero grid movement is numerical. Test both cases. Add the one-/two-clip daily-rate example from the new pack so that taking only the maximum rate cannot pass unnoticed.

Do not promise that every worst relative loss halves after refinement. Grid locations and the point at which the maximum occurs can change. Recompute the diagnostic. If a feasible grid is too expensive, investigate continuous-control interpolation or another inventory scheme as a separate method change with its own oracle.

**S2 completion:** all supported storage routes preserve the same physical record; every distinct active base rate contributes to the diagnostic; the report differentiates base-rate preservation, ratchet approximation and value refinement.

## 6. S3: make the requested data source authoritative

### 6.1 Consolidate loading without importing an app

Create a small proposed `quote_data.py` module containing cleaning and source-aware loading. `benchmarks.py` and `portfolio_app.py` should import it. Do not import `portfolio_app.py` into the library: it configures Streamlit and has application side effects at module import.

Suggested interface, to be finalised during implementation:

```python
quotes, provenance = load_quote_matrix(
    source_path,
    cache_dir=None,
    use_cache=True,
)
```

Preserve an adapter for callers that currently expect a DataFrame alone. Make provenance available to research outputs and valuation manifests. Explicit parquet inputs are sources in their own right; distinguish “requested parquet dataset” from “derived cache of a workbook”.

### 6.2 Implement source identity and cache validation

1. Resolve the requested source and read its bytes or calculate a SHA-256 from the exact byte stream used for parsing. For these workbooks, reading once into bytes is a straightforward way to avoid hashing one version and parsing another during an edit.
2. Record the resolved path, source SHA-256, input format, byte count and cleaning-schema version. The cache key must include source identity and cleaning version. Modification time may accelerate diagnostics; it is not proof of identity.
3. Load a cached result only if its metadata match. Validate expected columns and date/numeric types. Record whether the result was a hit or a rebuild. A metadata mismatch is a cache miss, not a reason to reuse plausible-looking data.
4. On a miss, parse the requested bytes. Share one cleaning function for local and uploaded workbooks. Preserve the existing successful cleaning behaviour: date-column identification, invalid numeric coercion, date sorting and explicit treatment of missing dates. Record duplicate-date and rejected-cell counts instead of silently inventing observations.
5. Write the derived cache and metadata using temporary files and replacement where supported. A partial write or corrupt cache should lead to rebuilding from the source. If writing is unavailable, return the freshly read data with a non-persisted-cache status. Do not catch source parse errors as though they were harmless cache-write failures.
6. Expose `use_cache=False` for reproducible comparisons. The same source and cleaner must produce equivalent data with caching enabled and disabled.

The second cache in the app also matters. `portfolio_app.load_quote_matrix_local` is decorated with `st.cache_data` and takes path strings. A change to bytes at the same path need not change those arguments. Pass the source fingerprint into the cached function, computed outside it, or pass immutable source bytes. Do not prefix that identity argument with an underscore: Streamlit excludes such arguments from hashing. This recommendation follows the documented argument-based caching behaviour; check the project's installed version when implementing it. [Streamlit API documentation](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data).

### 6.3 Test source consumption, not merely file tracking

| Case | Setup | Acceptance |
|---|---|---|
| Two sources | Warm cache from workbook A with value 10; request older workbook B with value 999 | Return B and record B's hash |
| Same path, new content | Replace 10 by 999 and restore the original mtime | Return 999 with a changed fingerprint |
| Cache/no cache | Same canonical workbook and cleaning version | Equivalent values, columns and normalised timestamp units |
| Cleaner change | Same source bytes; increment schema version | Rebuild and record new cleaner identity |
| Corrupt cache | Truncate a derived test cache | Rebuild from the source or report a precise source-independent cache error |
| Read-only persistence | Simulate cache write failure after successful parsing | Return valid source data; no substitution |
| In-memory app cache | Same path, changed fingerprint across calls | Second result reflects new bytes |

The new pack implements the first two cases with temporary synthetic files. It does not alter the repository workbook or cache. Additional cache-format and Streamlit integration cases should accompany the implementation.

The current canonical workbook and its retrieved default cache agreed after timestamp normalisation. Do not discard that evidence or describe all previous statistics as invalid. The repair protects source selection for subsequent work.

**S3 completion:** recorded data identity proves which bytes were consumed, every supported entry point uses the shared cleaner, and an explicit alternate workbook cannot read the default dataset.

## 7. S4: make convergence evidence specific and fail safely

### 7.1 Bind every row to a complete experiment

Create a serialisable benchmark specification before running a ladder. It must contain physical inputs, all dates, opening/terminal inventory, daily rate schedules, ratchets and their interpolation, bounds and their timing, curve values and dates, fuel, fees, discounting, price parameters, decision calendar and software revision.

Hash normalised content, not Python object identities or memory addresses. Sort mappings deterministically and preserve array shape and dates. Use separate identifiers:

- `contract_model_id`: the physical contract, market inputs, model conventions and decision calendar, excluding the one resolution parameter deliberately varied.
- `run_id`: the above plus inventory resolution, price resolution, code identity and numerical-policy version.
- `evidence_id`: the ordered set of run IDs plus acceptance thresholds and the gate version.

These are proposed fields. Every ladder row should carry the common contract/model identity and its own resolution and status. An input hash alone does not establish numerical correctness; it prevents unrelated evidence from being attached to the current request.

### 7.2 Replace the permissive verdict algorithm

1. Validate required columns, finite scalar results and positive integer resolutions. Reject NaN and infinity before computing differences.
2. Verify that all candidate rows describe the same physical contract and model. Recheck effective MWh/day limits, not just equality of input labels.
3. Require ordered, strictly increasing resolutions. For the stated two-doubling rule, the final three eligible rows must be `N`, `2N`, `4N`.
4. Keep failure/refusal rows. A refused intermediate level breaks the sequence. Do not filter it out and treat non-adjacent successes as two valid refinements. Earlier failed rows may remain in a longer report if a later complete three-row sequence succeeds.
5. Require both final steps to satisfy the declared acceptance rule for each tracked column. Report step differences and the exact reason for refusal or insufficient evidence.
6. Separate statuses such as `invalid`, `insufficient`, `outside_tolerance` and `within_declared_tolerance`. Do not present all non-successes as an infeasible physical contract.

The current unconditional EUR 1,000 alternative admits moves from EUR 10,000 to 9,500 to 9,000. Replace it with an explicit near-zero policy. A concrete **proposed engineering default** is: relative tolerance 0.5% for material values; use an absolute tolerance of EUR 1 only when both endpoints are at most EUR 100 in magnitude. For the relative denominator use `max(abs(low), abs(high))`. These proposed amounts need commercial calibration before a decision-grade acceptance policy is claimed. They are not existing project approvals.

For total and intrinsic, apply the rule independently. Extrinsic can be a small difference between two large numbers, so a total-value pass does not certify relative extrinsic accuracy. Report extrinsic movements in EUR and percentage where meaningful. Define separate tolerances for monthly hedge quantities if the output is to support hedging; do not imply that a smooth value establishes a stable optimal schedule.

### 7.3 Replace the notebook's `_RESIDUAL` lookup

The current `Storage_30_60.ipynb` dictionary is keyed only by `N_STATES` and evaluated before all experiment inputs are defined. Delete it as a source of current certification.

1. Build the complete current experiment specification after the curve, ratchets, bounds and other inputs are defined.
2. Load evidence only if its common configuration, code identity and gate version match. Otherwise display “No matching convergence evidence for these inputs”.
3. Offer an explicit ladder computation through the shared benchmark code. Cache its result under the full experiment identity.
4. Display the measured final two steps and the dimensions tested. If the current result uses N=1920 while the accepted endpoint is N=3840, state the measured difference to that endpoint; do not transfer the endpoint's acceptance status to the coarser run.
5. Changing any material curve point, ratchet, floor, rate, fee or price parameter must invalidate the status. Add at least one changed-ratchet and one changed-curve integration case.

Use the existing measurements carefully:

| Configuration | N=960 | N=1920 | N=3840 | Meaning |
|---|---:|---:|---:|---|
| Soft ratchets, recorded 70% floor, stochastic reference | EUR 3,015,923.74 | EUR 3,026,417.92 | EUR 3,032,172.02 | Total moves about 0.348% then 0.190%; both total and intrinsic passed the previous declared endpoint gate |
| Harsh ratchets, no floor, deterministic example | EUR 2,019,762.98 | EUR 2,134,479.76 | EUR 2,182,561.32 | Moves about 5.680% then 2.253% using the original preceding-value denominator; not within 0.5% |

These two rows are different configurations and are not a controlled estimate of the effect of changing only the ratchets. They show why N=3840 alone cannot certify arbitrary inputs. The percentages above describe the recorded calculation; a revised denominator must be labelled consistently in regenerated output.

### 7.4 Test the numerical dimensions separately

The current `price_grid_ladder()` changes `n_p`, principally the width of a tree whose time step and spacing remain fixed. Name this a price-boundary-width check. It is useful, but not a complete price-discretisation study.

Use this order:

1. Establish exact base-rate representation and contractual feasibility.
2. Widen the price domain at a fixed inventory grid; inspect value and boundary probability.
3. Refine inventory at a price width with acceptable boundary evidence.
4. Investigate price spacing and transition/time accuracy independently. The current production time step is daily; introduce a separate numerical-transition setting if this dimension is to be refined.
5. When substepping price transitions, retain the original decision dates and daily movement rights. Giving the optimiser additional exercise dates or applying a full daily rate at each substep changes the contract.
6. Rerun the representative joint setting after these studies and record runtime and peak memory. Run large ladders sequentially, releasing value/strategy arrays between cases. Retain summary evidence rather than every full model object.

There is no demonstrated continuous-control error bound from two finite inventory refinements. Describe a pass as “within the declared successive-refinement tolerance”. Keep the 48 independent small-contract oracle cases to test correctness, rather than substituting convergence for correctness.

**S4 completion:** invalid tables cannot pass, the notebook's status is tied to its full inputs, and each numerical dimension has an honest description of what was tested.

## 8. S5: repair and delimit the two-factor probe

### 8.1 Name the experiment correctly

Rename the probe's “calibration anchors” as variance-allocation scenarios unless market observations are actually fitted. Preserve three distinct comparisons:

| Comparison | What remains fixed | Question answered |
|---|---|---|
| Added independent common factor | Short-factor dynamics and physical contract | Does this factor add an independent valuation state for this payoff? |
| Scalar variance allocation | Selected variance target, fixed kappa and independence | How does value change under this particular variance redistribution? |
| Joint market fit | Dataset, observation definitions and fitting protocol | Which parameterised model better represents the observations, and what does it imply for value? |

The first comparison is a valid control. The second cannot substitute for the third. Rename `test_a_calibrated_second_factor_takes_volatility_out_of_the_short_one` to state its actual independence, fixed-kappa and instantaneous-variance assumptions. Its expected direction is a result for that scenario, not a universal model acceptance criterion.

### 8.2 Make the time axis explicit

Currently `N_T=24`, `DT=1/12`, with decisions at `0, DT, ..., 23*DT`. The final decision is at 23/12 years, not two years.

1. Introduce an explicit `decision_times` array and derive discount factors, forwards and transition counts from it.
2. If the intended contract instead has a decision at two years, change the schedule deliberately and report it as a different contract.
3. Require a terminal variance anchor to name the observation time it matches. Default to the actual last decision only if that is the documented experiment.
4. Keep valuation timing and post-exercise terminal inventory timing separate. Ending empty after the last exercise does not create another stochastic price observation one step later.

### 8.3 Use a coherent OU transition law

For `dchi = -kappa*chi*dt + sigma_chi*dW`, the exact one-step conditional moments over `d` years are:

```text
a       = exp(-kappa*d)
mean    = a*chi
q_chi   = sigma_chi^2 * (1-exp(-2*kappa*d)) / (2*kappa)
```

At `kappa=0`, use `a=1` and `q_chi=sigma_chi^2*d`. Use `-expm1(-2*kappa*d)` for numerical stability near zero. A zero volatility should create a deterministic transition without division by a zero grid spacing.

There are two defensible implementation steps:

1. **Immediate correction:** retain the existing Euler lattice but match its actual transition variance at the actual observation date. Label it a discrete-lattice scenario.
2. **Preferred research improvement:** construct a lattice matching the exact conditional mean and variance, then validate its finite-grid approximation separately. Exact one-step moments do not make all option values exact.

For the second step, use the exact innovation variance when choosing a trinomial spacing and probabilities. Check each reachable transition row's mass, non-negativity, conditional mean and variance. Boundary clipping can invalidate moment matching even if row sums equal one. Widen the domain and test it; do not “repair” negative probabilities by clipping and renormalising without measuring the resulting moment error.

If correlation is subsequently introduced with `dxi = mu*dt + sigma_xi*dW_xi`, the joint innovation covariance is:

```text
q_xi     = sigma_xi^2*d
q_chi_xi = rho*sigma_chi*sigma_xi*(1-exp(-kappa*d))/kappa
```

Its zero-kappa limit is `rho*sigma_chi*sigma_xi*d`. Check positive semidefiniteness of the complete covariance matrix. Correlated transitions cannot use the present product of independent transition matrices unchanged.

These formulas follow by integrating the linear SDEs. They are proposed independent moment targets, not a claim that the current kernels implement them.

### 8.4 Match the variance that is actually valued

Starting from the central initial state, propagate probability `q_next = q @ P`. At the chosen observation index calculate:

```text
mean     = sum(q * nodes)
variance = sum(q * nodes^2) - mean^2
```

Perform this for both factors. Under the probe's independence, log-price variance is the sum. Deterministic forward-fitting shifts do not change that variance. Compare the measured sum with the measured baseline, not merely with the formula used to choose a parameter.

For the current lattice at kappa 4, sigma_xi 0.1:

| Quantity | Current result | Corrected matching on the same lattice |
|---|---:|---:|
| Matched sigma_chi | 0.4472135754 | 0.4818944088 |
| Final-date log-price variance | 0.0491666637 | 0.0539999996, matching baseline |
| Struck-swing value | 32.61429691 | 33.46227862 |
| Difference from baseline 36.32895830 | -10.2251% | -7.8909% |

Keep this table as a dated diagnostic. An exact-moment transition repair may change all values, so do not freeze 33.46227862 as the acceptance price for a different transition law.

If unit-variance scaling is used to solve for sigma_chi, verify that the lattice construction preserves that scaling and that boundary effects remain negligible. Otherwise solve the measured variance equation numerically. An exhausted variance budget should return a clear infeasible-scenario status, not a price with NaN inputs. A zero residual is a valid deterministic-short-factor limit if the solver supports it.

### 8.5 Acceptance sequence for the probe

1. Test mean and variance against analytic OU targets at kappa 0, near zero, 0.2, 1 and 4. Test sigma zero separately. For a deliberately retained Euler mode, compare with its documented discrete recursion and quantify its difference from the exact target.
2. Test transition mass and non-negative probabilities on reachable nodes. Report boundary occupation and conditional-moment errors there.
3. Test `E[S_t]=F0[t]` on every decision date. This is a forward-fit check; it does not certify variance, serial dependence or option prices.
4. Match the actual target variance within a proposed tolerance `1e-10 + 1e-8*target_variance`. The new acceptance pack includes this case for the current terminal scenario.
5. Retain the zero-long-factor comparison and the fixed-short-factor zero-fee storage/unstruck-swing invariance controls. Their assumptions include independence, a common multiplicative factor with the appropriate martingale normalisation, price-independent constraints and homogeneous terminal/cash-flow rules. A correlated extension needs its own reduction argument.
6. Retain cash-fee storage and struck-swing examples as non-homogeneous controls. Fixed percentage fuel loss preserves price homogeneity under the current convention; fixed EUR/MWh fees and a strike generally do not.
7. Refine transition time and domain while keeping all original exercise dates, monthly quotas and movement rights unchanged. Record value changes. At numerical substeps permit transition only, not additional injection, withdrawal or exercise.
8. Rename printed headings and tests to describe the experiment actually run. Remove an unconditional “second factor lowers short volatility” or required mean-reversion sign flip from acceptance logic.

**S5 completion:** each variance-allocation row identifies its dynamics, horizon and measured variance, and claims remain conditional until S7 provides a market fit.

## 9. S6: construct an auditable panel and delivery observation model

### 9.1 Record the dataset before calculating returns

The reviewed workbook ends on 6 March 2026. It can support a historical investigation through that date. It cannot support a claim of calibration to the market on 11 September without later observations.

Create a data manifest containing the source fingerprint from S3, retrieval/as-of dates, quote-date range, column definitions, cleaning version, units, duplicates, missing observations and all filters. Preserve the raw workbook as the input record; write derived panels separately.

For each observation, construct at least:

```text
quote_date, contract_identifier, delivery_start, delivery_end,
price_eur_mwh, source_column, source_hash, validity_flag
```

The project maps TTFc1 to the next delivery month. Verify that convention against the workbook/provider definition before treating it as externally established. Make delivery identity explicit for every rank and date. Decide whether daily weighting means calendar days, delivery hours or the actual contract specification; do not infer it from the column number alone.

### 9.2 Build returns on aligned observations

1. Choose training and held-out windows before fitting. A historical replication can retain the recorded 2015-onward window, then show sensitivity to separately declared subperiods.
2. Identify the same delivery contract on consecutive observations. Continuous rank c6 across a roll is not automatically the same contract. Either follow fixed delivery identifiers or explicitly exclude roll transitions for a rank-based diagnostic.
3. Align the two contracts' return intervals before calculating correlation or spread statistics. Keep actual start/end dates and elapsed calendar time for each return.
4. Filter non-positive values before log transformation, record the exclusions, and avoid bridging missing rows while still labelling the result a one-day return. A Friday-to-Monday observation and a missing-week observation have different elapsed times even if both become adjacent after `dropna()`.
5. Declare the statistic: log-return difference, EUR/MWh spread change, or another observable. The log of a monetary spread is undefined when the spread is non-positive and is not the same as a log price ratio.
6. Retain the legacy `sqrt(252)` statistic as a clearly labelled diagnostic if useful. For a continuous-time likelihood, use actual elapsed year fractions consistently with the stochastic model. Do not compare trading-day annualisation with a calendar-time model variance without a stated conversion.
7. Fix the variance estimator convention (`ddof`, weighting and missing-data treatment), include observation counts, and reproduce the legacy statistic before changing it. A difference caused by filtering before versus after differencing is a specification change, not evidence of market-data drift.

A tiny synthetic panel should contain a month-end roll, a weekend, one missing quote and two delivery identifiers. Write out the expected retained return intervals by hand. This catches errors that a comparison of two vectorised implementations may share.

### 9.3 Model the delivered instrument

For a point-delivery forward, the one-factor instantaneous loading is `a(t,T)=exp(-kappa*(T-t))`. A monthly delivered forward is an average of forwards over its delivery period. Its local relative diffusion loading is the price-weighted average:

```text
a_bar(t; A,B) = integral_A^B F(t,u)*a(t,u) du / integral_A^B F(t,u) du
```

For flat forwards over the period, this becomes:

```text
a_bar = [exp(-kappa*(A-t)) - exp(-kappa*(B-t))] / [kappa*(B-A)]
```

Use the kappa-zero limit explicitly. For a discrete daily/hourly delivery specification, use the corresponding weighted sum. When comparing model and market, use the same delivery calendar and observation interval.

Do not claim averaging necessarily lowers volatility relative to the old `i/12` point approximation. At sigma 0.5, kappa 1, equal 1/12-year months ending at 0.5 and 1 years, the month-end point comparison is 0.11932561, while the flat-forward delivery average is 0.12443854. The direction depends on the point convention and weights.

A further modelling distinction matters: the logarithm of an average of exponential-affine forwards is generally not affine in the factors. A standard linear Gaussian observation equation for point log-forwards is therefore not automatically exact for log monthly-average quotes.

Implement and test the actual delivery observation function first. Then quantify the error of a point or linearised approximation over the fitted state range. Use a linear Kalman formulation only under its stated observation approximation; consider a nonlinear filter if that error is material. Do not introduce particle methods simply because they are available. Research on commodity panel models explicitly treats seasonality and nonlinear observation/filtering issues, but it does not establish which extension this TTF dataset needs. [Peters, Briers, Shevchenko and Doucet, 2011](https://arxiv.org/abs/1105.5850).

**S6 completion:** every fitted observation has a traceable source and delivery identity; returns have matched intervals; model observables reproduce the declared delivery weighting on synthetic examples.

## 10. S7: perform calibration as a model comparison

### 10.1 State the measures and parameters before fitting

Write separate transition and pricing specifications under the physical measure P and valuation measure Q. Historical time-series dynamics alone do not identify every risk-premium or Q-drift parameter. If the implementation equates P and Q mean reversion or uses a constant market price of risk, make that an explicit model restriction.

Fitting the initial curve with a deterministic shift establishes the time-zero forward fit. It does not identify the stochastic dynamics. Similarly, allocating one scalar variance budget does not identify kappa, two volatilities, correlation and seasonality.

Start with a small candidate set using the same cleaned panel and delivery observation rule:

| Candidate | Purpose | Key comparison |
|---|---|---|
| One-factor baseline | Establish what re-estimation alone changes | Fitted kappa/volatility versus the current illustrative values |
| Two-factor short/long model | Add a common long component with estimated correlation where identifiable | Covariance term structure, fit stability and observed delivery spreads |
| Restricted two-factor controls | Test independence or fixed kappa restrictions | Whether those restrictions materially drive the result |

Consider a second mean-reverting factor or richer seasonality only if residuals and identification diagnostics justify it. Comparing progressively richer models without a held-out check would reward complexity by construction.

For point log-forwards with short loadings `a_i`, the instantaneous covariance is:

```text
C_ij = sigma_chi^2*a_i*a_j + sigma_xi^2
       + rho*sigma_chi*sigma_xi*(a_i+a_j)
```

This is a useful independent algebraic check. The common long term cancels directly from a point log-ratio diffusion, leaving short-factor loading `sigma_chi*(a_i-a_j)`. It can still change a joint fit through other covariance entries, correlation and re-estimation. This is why a single pair's variance cannot establish the whole model.

### 10.2 Run a reproducible estimation sequence

1. Freeze the dataset, training dates, held-out dates, measurement convention and candidate restrictions in a configuration file.
2. Fit the one-factor baseline first. Record parameter values, transformed optimisation coordinates, starting points, objective value, convergence status and any boundary hits.
3. Fit the two-factor candidates from several dispersed starting points. Enforce positive volatilities and mean reversion, and correlation within its admissible range through an explicit parameterisation. Treat an optimiser's success flag as one diagnostic, not proof of identification.
4. Specify measurement noise separately from process shocks. Inspect whether the fit explains genuine maturity structure or simply moves discrepancies into a large observation-noise term.
5. Use profile likelihood, local curvature or an appropriately specified bootstrap to assess weak identification. Report parameter dependence, especially kappa/volatility trade-offs and short/long factor confounding. If a parameter is weakly identified, report a range or scenario rather than spurious precision.
6. Evaluate held-out likelihood or forecast errors using only training-fitted parameters. Compare delivery-pair covariance, seasonal spread changes and residual autocorrelation. Preserve the same observation treatment across candidates.
7. Repeat on predeclared alternative windows or regimes. Do not pick the window that generates the preferred storage value. Distinguish parameter instability from ordinary sampling uncertainty.
8. Save the fitted parameters, units, P/Q restrictions, initial-state treatment, seasonality, observation noise and full data/configuration identity. Record random seeds only for methods that use randomness.

Acceptance thresholds for statistical and commercial materiality should be written before interpreting the final ranking. The current evidence does not justify inventing a universal required correlation, fit improvement or valuation percentage. If the data cannot distinguish models, the conclusion is inconclusive and the next task is better identification or data, not a forced architecture choice.

This is held-out statistical validation of a calibration. It does not expand the current task into a historical trading or hedging backtest. Keep that deferred work separate.

### 10.3 Compare values without changing the deal

Use the same physically normalised, numerically assessed contracts across models: zero-fee storage, storage with cash fees, an unstruck swing and a struck swing. Keep curve, operating dates, exercise rights, fuel and discounting fixed unless a particular difference is the declared subject of the experiment.

Record three movements separately where practicable:

1. Re-estimating the one-factor parameters from the current illustrative baseline.
2. Adding a factor while holding the short-factor dynamics fixed, under a valid common-factor control.
3. Moving from the fitted one-factor model to the jointly fitted two-factor model.

The third movement combines changes in parameters and model structure. It is not necessarily the sum of independently calculated component effects because policies may change. Report total, intrinsic, extrinsic and monthly hedge differences, with numerical-refinement evidence and parameter uncertainty beside them.

An analytic counterexample already rules out a guaranteed downward short-volatility adjustment: for total instantaneous variance 0.36, sigma_xi 0.3 and rho -0.9, sigma_chi is 0.85557664, above 0.6. This parameter set is a logical counterexample, not a fitted TTF recommendation.

### 10.4 Decide the valuation architecture from the result

| Finding | Proposed next action |
|---|---|
| Added factor not identified or no robust held-out benefit | Keep the simpler model for the stated scope; record the limitation and investigate the residuals |
| Multi-factor market fit improves, but the target payoff permits a demonstrated reduction | Retain the richer calibration and derive/validate the reduced valuation state |
| Material fee/strike or relative-price effect requires another valuation state | Build a small independent two-factor valuation oracle, then compare production methods |
| Full lattice is within the measured resource envelope | Prototype rolling value layers and the actual required strategy/probability outputs |
| Lattice cost is excessive for required resolution | Compare LSMC or a hybrid with independent policy valuation and numerical diagnostics |

Before expanding the lattice, measure dimensions using the actual number of dates, factor nodes and inventory levels. Budget value, strategy, probability, transition and temporary arrays, not just one tensor. If storing actions in an integer dtype, size it against the maximum representable move; do not assume `int8` is adequate.

The repricing identity is not a method-selection advantage exclusive to dynamic programming. For a fixed adapted policy, linearity of expectation supports a cash-flow/hedge reconciliation under the stated definitions. LSMC needs independent policy evaluation and sampling-error assessment; the identity alone does not prove optimality for either method.

**S7 completion:** a reproducible comparison supports a documented model/architecture decision, or explicitly records insufficient evidence. No “8% calibrated haircut” is carried forward from the current probe.

## 11. S8: update current documents without rewriting history

| File | Required edit |
|---|---|
| `docs/STATUS.md` | Replace the blanket claim that code computes every stated input correctly; retain verified bound progress and list unresolved conversion/data/convergence issues. Tie test counts to the commit/run. Reconcile the 0.53% versus approximately 0.19% reference-grid statements. |
| `docs/MODEL-CONVENTIONS.md` | Replace the old soft-bound-penalty description with current admissibility behaviour and inventory timing. Describe strict physical normalisation, requested/effective quantities, fuel convention and each numerical gate. |
| `docs/DESIGN-P4.1-two-factor.md` | Preserve the dated original as history with its correction notice; link a new current calibration specification rather than leaving superseded implementation steps as the active plan. |
| `docs/REVIEW-RESPONSE-2026-09-10.md` | Date the 34-file hash match and 101-test statement as pre-repair evidence. Link the later 37-file review and current CI without implying the old snapshot is the present one. |
| Dated findings | Preserve original observations with precise correction links. Correct the delivery-averaging upper-bound interpretation where it is used as a current claim. |
| `.planning/ROADMAP.md` | Keep bounds repaired; keep rate preservation and calibration open until their stated checks pass. Replace stale soft-bound, ninefold-shortfall and DA-gap assumptions in active tasks. |
| Notebook comments and outputs | Use the shared conventions and live evidence; remove hard-coded convergence certification and current references to soft bounds. |
| `two_factor_probe.py` narrative/tests | Describe scalar variance scenarios and their assumptions; distinguish them from fitted market parameters. |

The kernel currently uses large finite exclusion/sentinel values with feasibility checks. Describe the behaviour precisely. The realistic adverse-price cases passed; this review does not establish a new live sentinel failure. A future explicit reachability mask could make admissibility structurally clearer, but should not displace the reproduced physical-input defects as the next priority.

Current status language after the documentation-only update can be:

> Inventory-bound repairs are independently reproduced. Physical input preservation, source-bound quote loading and configuration-specific convergence reporting remain open. The second-factor probe provides conditional numerical scenarios; a market calibration and its valuation implications remain to be established.

After a repair, replace only the corresponding open statement with a link to its tests and exact revision. A historical green suite remains valid evidence of what it tested; it cannot close requirements that its assertions did not enforce.

## 12. Execution checklist and review gates

The following items are pending production work. The guide and baseline acceptance pack are completed preparation, not evidence that these boxes are done.

| Order | Task | Depends on | Evidence required to close |
|---:|---|---|---|
| 1 | Add conversion counterexamples using original physical inputs | Baseline | Failing tests reproduce rate and boundary-volume changes |
| 2 | Implement strict normalisation and redundant-input checks | 1 | Compatible cases pass; incompatible/conflicting requests refused |
| 3 | Replace test expectations that preserve silent rounding | 2 | Tests express requested contracts rather than old clip output |
| 4 | Migrate benchmark and storage notebook constructors | 2 | Physical-record parity across callers |
| 5 | Migrate supported storage app/workbook routes | 2 | Same contract from each input surface; swings retain their own defaults |
| 6 | Extend ratchet diagnostics to distinct daily rates and headroom | 2 | Smaller-rate and boundary examples pass |
| 7 | Extract shared quote cleaner/loader | Baseline | Existing default cleaning reproduced |
| 8 | Add source fingerprints and validated cache metadata | 7 | Alternate source and restored-mtime tests pass |
| 9 | Pass identity through Streamlit's in-memory cache | 8 | Same-path content change refreshes app data |
| 10 | Add benchmark specification and experiment identifiers | 4, 6 | Every ladder row names one unchanged physical experiment |
| 11 | Implement validated convergence verdict | 10 | NaN, infinity, duplicate/reversed grids and refused gaps cannot pass |
| 12 | Replace notebook status lookup | 11 | Changed curve/ratchet invalidates prior evidence |
| 13 | Separate boundary width, inventory and transition refinement | 11 | Each numerical dimension has its own recorded evidence |
| 14 | Make probe dates and anchor time explicit | Baseline | Observation time equals the documented experiment |
| 15 | Correct measured terminal-variance matching | 14 | Actual propagated variance agrees with target |
| 16 | Validate exact OU moments and limiting cases | 15 | Conditional moments, mass and zero-volatility cases pass |
| 17 | Refine probe with unchanged exercise rights | 16 | Conditional value effects have recorded numerical stability |
| 18 | Rewrite probe claims and affected current documents | 15 | No unsupported calibration or universal sign claims |
| 19 | Freeze data and delivery mapping | 8 | Data manifest and synthetic roll/missing-date cases |
| 20 | Implement point and delivery-average observation checks | 19 | Analytic flat-forward example and general weighting agree |
| 21 | Declare candidates, windows and P/Q restrictions | 20 | Written fitting protocol before model ranking |
| 22 | Fit baseline and two-factor candidates | 21 | Optimisation, identification and held-out diagnostics |
| 23 | Compare fixed contracts across fitted models | 13, 17, 22 | Value/risk differences separated from numerical error |
| 24 | Decide whether another valuation state is justified | 23 | Written decision including uncertainty and resource evidence |
| 25 | Reconcile current status, conventions and roadmap | Each slice | No conflicting active claims; each completed item has evidence |

For each code slice, run its focused acceptance cases first, then the repository suite in a real Git checkout using `requirements-lock.txt`. The existing suite includes expensive notebook execution, so run the full required gate once per completed slice or combined reviewable change rather than repeatedly during prose edits. Rerun large valuation ladders only after a relevant model, contract or numerical-setting change.

Suggested review record for each slice:

```text
Problem and affected public behaviour:
Changed files and entry points:
Physical/model conventions preserved or deliberately changed:
Independent acceptance cases and observed results:
Repository test result and exact commit:
Numerical benchmark changes, with explanation:
Current documentation updated:
Remaining limitations and next dependent task:
```

Do not revise price anchors merely to restore a green test. First explain whether a difference comes from corrected physical inputs, a numerical method change, data bytes, convention changes or a regression. Each has a different interpretation.

## 13. Using the attached acceptance pack

Extract `ACCEPTANCE-PACK-2026-09-11.zip` and read its `README.md`. It contains a portable script, this baseline's detailed JSON results and source hashes. It does not contain the repository or market workbook.

In the project's locked Python environment, from the repository root:

```bash
python docs/acceptance-pack/check_acceptance.py --source . --output docs/acceptance-pack/current-results.json
```

Use the extraction path in place of `docs/acceptance-pack` if unpacked elsewhere. An exit code of 1 means an acceptance requirement remains unmet; 2 means a check encountered an unexpected error and needs investigation. The baseline returns 1, with 9 PASS and 22 FAIL. The JSON contains the exact requested and observed quantities for each case.

The pack tests the current API. If a repair moves a function, adapt only the thin calling portion and retain the independent expected property. In particular, update the cache adapter if loading moves into `quote_data.py`, and extend convergence tests to pass the newly required experiment identity. An API error is not a successful rejection of a bad contract.

The pack is intentionally narrower than the complete plan. It does not execute the full stochastic suite, a high-resolution ladder, a Streamlit session or a market fit. The implementation must add the integration and numerical checks identified in the relevant slice. Preserve the baseline JSON as dated evidence and write new results to a different file.

The work is complete when the requested contract survives every supported entry point, source identity is auditable, numerical statements refer to the current configuration, and P4.1 reaches an evidence-based calibration decision. More elaborate architecture is justified only by that evidence.

---

## 14. Verification of this guide — 2026-09-11

Added by the project, not by the guide's author. Sections 1–13 above are unchanged. This
records what was checked before acting on the guide, in the same way
[REVIEW-RESPONSE-2026-09-10.md](REVIEW-RESPONSE-2026-09-10.md) checked the first review.

### Method

Every numerical claim in this guide and in the [project review](PROJECT-REVIEW-2026-09-10-evening.md)
it rests on was reproduced from the **prose**, with independently written scripts, against
`main` at `b66d0a4` on the working machine (NumPy 2.4.3 / pandas 2.3.3 / Numba 0.65.1 — behind
the lockfile on every pin, as [STATUS.md](STATUS.md) now records). The acceptance pack was
then extracted and run on the same tree. The cited CI run (116 passed under the lockfile)
could not be checked from here; the count matches the local suite.

### Result

**Nothing in the guide or the review was found to be wrong.** Every claim reproduced, most to
the last digit.

| claim | reproduced |
|---|---|
| 30/90 at N = 30 silently prices a 30/**30** store: 20,000 MWh/day against 6,667 requested | exactly — 2,514,008.52 against 2,382,639.26 EUR, **+131,369.26 (+5.51 %)** |
| 30/65 at N = 60 prices 30/**60** | exactly — 2,447,597.82 against 2,436,687.60 |
| 5,000 MWh opening and terminal inventory round to **0 clips** at 60 clips | yes, both, through `params_for_run_valuation` |
| the probe's terminal anchor misses the lattice's own variance by 8.95 % | exactly — 0.0491666637 against 0.0539999996 at the last decision (23/12 y, not 2 y) |
| corrected struck-swing effect is −7.89 %, not −10.23 % | exactly — σ_χ 0.4818944088, value 33.46227862 |
| `convergence_verdict` passes three NaN rows, three copies of one grid, and 10,000 → 9,500 → 9,000 | all three fail open |
| `load_quote_matrix(path)` returns the repository cache for a different workbook | yes — 4,171 rows and TTFc1 = 10 returned for a one-row request with TTFc1 = 999 |
| same path, new bytes, restored mtime returns stale data | yes |
| delivery averaging is not an upper bound on the point comparison | yes — 0.12443854 against 0.11932561; a project claim, withdrawn |
| ρ = −0.9 with σ_ξ = 0.3 gives σ_χ = 0.8556 > 0.6 | arithmetic checks |
| `MODEL-CONVENTIONS.md`, the notebook, `STATUS.md` and the roadmap still carry penalty, 0.53 %, ninefold and three-day language | all confirmed by search |
| acceptance pack: 31 checks, 9 pass, 22 fail | **9 / 22** on this tree, no errors |

Three of the five P1 findings concern code written on 2026-09-10 — the workbook cache, the
convergence gate and the notebook's `_RESIDUAL` lookup. The conversion defect predates it.

### One sharpening

The guide's third new detail (§2, item 3) — a 0.5 multiplier on alternating one- and two-clip
days is a 100 % loss that `worst_ratchet_rate_loss()` misses because it reads `curve.max()` —
is **worse than stated**. `assert_ratchets_expressible()`, the zero-rate guard built
specifically to catch "cannot move at all", reads `curve.max()` too and **also passes**. A
store that cannot move on half its active days clears the guard designed for exactly that.
Both need the per-day treatment in §5.2, not only the diagnostic.

### Where the project pushes back

One place, and on shape rather than substance. §7.1's three identifiers
(`contract_model_id`, `run_id`, `evidence_id`) are more machinery than a notebook needs. The
requirement underneath — compute the certificate from the *current* inputs or show none — is
met by hashing the full parameter set, caching ladder results under that hash, and printing
"no matching convergence evidence" on a miss. Same guarantee, one identifier. The
implementation will do that and can grow the taxonomy if a second consumer ever needs it.

Two notes on the conversion defect, for the record rather than in disagreement:

- `max(1, round(n_states / days))` was a zero-rate guard of the same family as the ratchet
  one, and it went the wrong way — it **granted** a faster rate instead of refusing. The
  guide's rule that a minimum-one-clip rule must never raise a contractual maximum is the
  right one.
- The docstring of `params_for_run_valuation` already says *"check the derived rates rather
  than assuming the days you asked for survived"*. The contract change was a known trap with
  its warning in the one place a caller does not look. `Storage_30_65.ipynb` protects itself
  against it; the workbook and API routes do not — which is the entry-point parity §5.1 asks
  for.

### Order

Agreed as written. **S1 first**, because it is the only item where a public route silently
prices a different contract. S3 and S4 next — repairs to the previous day's cache and gate,
both small. S5 is small. S6–S7 stay blocked on data past 6 March 2026 whatever else happens.
S8's reconciliation goes inside each slice; `MODEL-CONVENTIONS.md` still describing the
penalty is the worst of those, because it is the file users are told to read first.

The guide, the project review, and both evidence archives are committed alongside this note.
