# Process and Progress Review

**Project:** Gas Storage / Swing Pricing Model  
**Review date:** 8 September 2026  
**Review basis:** The current Storage project in Google Drive, including `STATUS.md`, `code_review.md`, `finding.md`, `storage_model.py`, `streamlit_app.py`, `test_model.py`, the planning files and notebooks; and the current public GitHub `main` branch.

## Executive conclusion

The September review work is technically useful and substantially better controlled than the original notebook-led development process. Findings were reproduced numerically, regression tests were run against the defective baseline, fixes were separated from valuation changes, and the remaining work was converted into traceable requirements.

However, further implementation against the current Drive branch should pause. The review branch appears to have been created from an obsolete local version of the repository. Current GitHub `main` already contains material June changes that are absent from the Drive branch, including a split kernel module, portfolio valuation, asymmetric injection and withdrawal rates, a corrected Streamlit default, removal of `wdr_days`, annualised volatility labelling and an existing regression suite. The branch `review/model-fixes` is not visible on GitHub.

Consequently, the statement that the review branch is five commits ahead of `main` and fast-forwardable is not established relative to current GitHub `main`. It may be true only relative to a stale local `main`. Continuing the five-phase roadmap without first reconciling the repositories risks duplicating work and reverting features.

The immediate priority is therefore repository reconciliation, followed by a fresh review of the combined current codebase.

## Evidence and limitations

### Verified observations

- The Drive review branch contains the documented fixes to the price-per-MWh denominator, price-state indexing, fractional-ratchet validation, terminal-feasibility handling, volatility-grid sizing and curve-coverage errors.
- `test_model.py` contains 13 regression tests addressing those defects and several numerical invariants.
- The current GitHub `main` includes June changes not present in the Drive branch:
  - `n_p_full` is already 30;
  - the Streamlit `wdr_days` control has already been removed;
  - `sVol` is described as annualised;
  - the library supports distinct injection and withdrawal rates;
  - Numba kernels are separated into `storage_kernels.py`;
  - portfolio valuation and `tests/test_regression.py` are present.
- The three notebooks in the Drive copy were last modified before the September fixes. Their saved outputs and some parameter narratives therefore do not represent the reviewed branch.
- The Drive copy contains `.git`, `.venv` and `__pycache__` directories.
- `requirements.txt` lists dependencies without version constraints.

### Reported but not independently rerun in this review

- The 13/13 test result.
- The clean working tree.
- The exact five-commit distance from the local `main` branch.
- The before/after valuation equality reported for the six quoted products and additional cases.

These claims are supported by the project records, but the unpushed local branch was not available as an executable Git checkout in this review environment.

## Assessment of the review process

### Strengths

1. **Executed evidence rather than code-reading alone.** The review records concrete numerical reproductions for the delta profile, tree behaviour, convergence, curve error and infeasible cases.

2. **Tests were applied to the defective baseline.** Moving from 3/13 to 13/13 is stronger evidence than adding tests only after implementation, because it demonstrates that the tests detect the intended defects.

3. **Good separation of reporting and valuation effects.** The review distinguishes changes to displayed metrics from changes to the underlying DP value.

4. **The anomalous delta profile was investigated rather than cosmetically altered.** The January quota saturation and December amplification were tested and correctly retained as model behaviour.

5. **Traceability is unusually strong for a small quantitative project.** Requirements map back to numbered findings and code locations, and acceptance criteria are generally measurable.

6. **The proposed `exact=False` route is valuable.** It permits old and corrected curve construction to be compared in one environment, avoiding uncertainty caused by checking out different commits or dependency versions.

### Weaknesses and misleading framing

1. **The baseline is stale relative to GitHub.** This is the largest process defect. A rigorous review of the wrong baseline can still produce the wrong delivery plan.

2. **Progress reporting is internally confusing.** `STATE.md` reports 0%, while seven of the original twenty findings are recorded as fixed. The 0% describes only the newly created roadmap, not total remediation progress. Both measures should be shown explicitly.

3. **The valuation-impact classification is too categorical.** Phase 2 is described as the only valuation-changing phase, yet Phase 3 changes the default displayed extrinsic value by approximately 15%, and Phase 4 may change DP results through the tunnel penalty. These are different forms of impact, but all matter to users.

4. **Several conclusions overstate validation.** “The DP and tree are correct” should be replaced with “no defect was found in the tested domain”. The current tests are meaningful, but they do not prove correctness across all admissible curves, rates, grids and constraints.

5. **The master invariant is necessary but not sufficient.** A common error shared by the DP, forward probability pass and metric calculation could preserve the repricing identity. Independent finite differences and a small reference implementation are needed.

6. **Software verification is being conflated with model validation.** The reviewed materials use assumed volatility and mean-reversion inputs. No controlled calibration method, market-data provenance or parameter sensitivity standard was found in the reviewed artefacts. Calibration work elsewhere is unknown.

7. **Planning documentation is beginning to duplicate itself.** `STATUS`, `STATE`, `PROJECT`, `ROADMAP`, `REQUIREMENTS`, `SPEC` and the conflict report repeat overlapping claims. They have already drifted on commit counts, progress and valuation impact.

8. **Reproducibility is incomplete.** Numerical libraries are unpinned and no automated clean-environment test is recorded. This matters particularly for SciPy interpolation and Numba kernels.

## Assessment of progress

There are three different progress measures and they should not be collapsed into one percentage:

| Measure | Current position | Interpretation |
|---|---:|---|
| Original review findings | 7 of 20 fixed | 35% by count; the completed set includes several of the highest-impact defects |
| Major findings | 5 of 10 fixed | The denominator, indexing, ratchet, infeasibility and volatility-grid defects are addressed in the Drive branch |
| Newly defined v1 requirements | 0 of 18 completed | This measures only the post-review roadmap, not total project work |

Risk-weighted progress is better than the 35% count suggests because several serious silent-error paths have been addressed. Release readiness is nevertheless lower than the code-fix count suggests because the fixes have not been reconciled with current GitHub `main`, the notebooks are stale, and economic validation remains incomplete.

## Required next steps

### 1. Reconcile the repository before further model work

Freeze the current Drive branch and establish the actual Git relationship:

```bash
git fetch --all --prune
git status -sb
git log --oneline --graph --decorate --all -30
git rev-list --left-right --count origin/main...review/model-fixes
git diff --stat origin/main...review/model-fixes
```

Push the review branch to a temporary remote branch, or recreate it from a fresh clone of current GitHub `main`. Do not fast-forward `main` based on the current status statement. Port each September fix selectively, preserving the June architecture and portfolio functionality.

### 2. Re-run the review against current GitHub `main`

Classify every September finding as:

- still present;
- already fixed on `main`;
- changed by the June architecture;
- superseded by another feature; or
- not applicable.

The current roadmap should be regenerated only after this reconciliation. Known examples include APP-01 and APP-02, which are already complete on GitHub `main`, and the ratchet/tunnel findings, which need reassessment against `storage_kernels.py`.

### 3. Consolidate the regression suites

Move the September tests under `tests/` and combine them with the existing GitHub tests. Preserve the portfolio and product-workbook anchors. Add:

- finite-difference validation of daily and monthly deltas;
- an independent, deliberately simple solver for small grids;
- deterministic and zero-volatility analytical cases;
- single-thread versus parallel-result comparison;
- actual `curve.csv` and product-level regression cases;
- partial-month, steep-curve, isolated-spike, negative-price and extreme-volatility cases;
- sign and unit tests for put, call and storage deltas;
- scale-invariance tests for decision tolerances and penalties.

Tests that skip when data files are absent should be reported as skipped, not passed. Release validation should fail if required production fixtures are missing.

### 4. Rebuild the roadmap around risk and user impact

Recommended sequence:

1. Repository reconciliation and combined green baseline.
2. Silent-error safety fixes that remain applicable.
3. Exact curve repricing with controlled before/after results.
4. User-interface and notebook alignment.
5. API and dead-code cleanup.
6. Economic validation, calibration evidence and release controls.

Split pure refactoring from behavioural changes. In particular, tunnel deletion and tunnel redesign should not share one phase. If the current tunnels cannot bind, the cleaner option is to remove them and specify a genuine tunnel feature separately rather than replace one arbitrary penalty with another.

### 5. Strengthen exact-curve acceptance

The corrected curve should satisfy more than monthly-average equality:

- every input contract reprices to the stated tolerance;
- partial first and last months work correctly;
- the curve remains finite under flat, steep, spiked and negative input shapes;
- daily movements and overshoot remain within a stated bound;
- the solve has an iteration cap and explicit non-convergence error;
- `exact=False` reproduces the previous algorithm;
- valuation and monthly-delta before/after tables are generated automatically.

The existing phrase “no month-boundary discontinuity beyond the natural slope” is not sufficiently precise for a regression test and should be replaced with a numerical criterion.

### 6. Establish the model conventions explicitly

A short controlled document should define:

- delta as forward-equivalent hedge volume, including sign and units;
- the relationship between delta and expected physical exercise;
- discount-factor treatment and settlement timing;
- `sVol` as annualised instantaneous log-volatility;
- the units of `sMR`;
- valuation and risk-measure conventions;
- curve, volatility and mean-reversion calibration sources and as-of dates.

The delta convention should also be verified by finite differences. Restating the internal identity alone does not establish that the reported number is the correct hedge for a quoted monthly contract.

### 7. Introduce lightweight release controls

- Pin the Python version and numerical-library versions in a lock file.
- Add a minimal GitHub Actions workflow that runs the complete non-skipping suite.
- Generate a machine-readable valuation comparison artefact rather than maintaining numbers only in Markdown.
- Rerun or clear notebook outputs whenever model logic changes.
- Mark legacy notebooks clearly and remove stale controls and date descriptions.
- Keep the working Git clone outside Google Drive synchronisation; do not synchronise `.git`, `.venv` or `__pycache__`.

## Suggested status wording

Until reconciliation is complete, the project status should read substantially as follows:

> The September review identified and locally fixed several material implementation defects, with 13 regression tests reported green. These fixes were developed against a local baseline that is behind current GitHub `main`, which already contains additional architecture, application and test changes. The review branch must be reconciled with current `main` before its fixes or roadmap can be treated as the project’s authoritative state. Existing valuation results remain exploratory; implementation consistency has improved, but economic calibration and independent model validation are not yet complete.

## Overall judgement

The work completed is valuable and should be retained. The strongest achievement is not the number of closed findings, but the transition towards reproducible, test-led quantitative development. The immediate risk is version-control divergence, not another pricing formula.

After repository reconciliation, the September findings should be ported selectively, the two test suites combined, and the roadmap shortened. Only then should exact curve repricing and further model changes continue.
