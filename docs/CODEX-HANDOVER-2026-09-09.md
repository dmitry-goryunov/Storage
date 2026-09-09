# Codex handover: economic corrections and P0 verification

**Date:** 2026-09-09  
**Project:** `Storage`  
**Windows folder:** `H:\My Drive\Github\dmitry-goryunov\Storage`

## Purpose

This note records the economic review, the resulting code changes and the P0 verification
work so that development can continue safely in the Codex extension for Visual Studio Code.

## Current verified state

- The corrected files are in the Google Drive `Storage` folder. Existing Drive file IDs were
  preserved when files were replaced.
- The maintained suite passes: **71 tests passed**.
- `Products.ipynb` also completed a full clean-process run with its default four deal sizes.
- The review was committed in a separate scratch clone on branch
  `review/correct-rate-economics`:
  - `d82883a Correct rate economics and intrinsic attribution`
  - `9a839c3 Add P0 economic verification gates`
- Those commits were **not pushed to GitHub**. Only the source files were updated in Drive;
  the Drive folder's Git branch and index were not changed by this work. Their current state
  is therefore unknown until checked locally with `git status`.
- Whether Google Drive for desktop has finished synchronising the latest files is also
  unknown until the local folder is inspected.

## Economic conclusions implemented

1. **Daily discounting is internally consistent under exercise-day settlement.** Each daily
   cash flow is multiplied by its discount factor once. This remains a contractual
   convention, not a universal gas-market convention.
2. **A flat gas forward curve is not inconsistent with positive financial rates.** Storage
   costs, convenience yield, seasonality and physical constraints can offset financial
   carry. The documentation no longer presents the model identity `DF * F` as a general
   commodity-carry claim.
3. **Market discounting, treasury funding and hurdle rates are different economic views.**
   The selected rate must be identified by purpose.
4. **Borrow and invest rates are scenario inputs, not a full asymmetric-funding model.** A
   one-directional swing must explicitly select `funding_direction="borrow"` or
   `funding_direction="invest"`. Direction is not inferred from product type or average
   `F-K`.
5. **Storage cannot use one treasury funding direction.** Storage pays on injection and
   receives on withdrawal. It uses a market/valuation `discount_rate` or `d_curve` until a
   genuine cash-balance-dependent funding recursion is implemented.
6. **Intrinsic attribution has three components.** `intrinsic_attribution()` reports day
   selection, settlement timing and their interaction. The legacy two-component function
   remains available and assigns the interaction to its second result.
7. **Hedge conventions remain distinct.** `delta` is the undiscounted forward hedge volume;
   `delta_pv` is its discount-tailed equivalent for margined futures.

## Code and interface changes

| File | Change |
|---|---|
| `storage_model.py` | Added central `normalise_rate_parameters()` validation; corrected explicit zero-rate handling; retained explicit scenario selection; added three-way intrinsic attribution. |
| `streamlit_app.py` | Added a **Rate management** section with market/valuation and treasury-scenario modes; displays the rate actually applied by the model. |
| `Products.ipynb` | Added `RATE_MODE`, discount and treasury inputs; corrected the cash-balance calculation and intrinsic attribution; added a reduced smoke configuration for automated execution. Stored outputs are deliberately empty. |
| `tests/test_reconciliation.py` | Added the independent schedule oracle, rate-plumbing tests, zero-rate regression and clean notebook execution gate. |
| `products.xlsx` loader | Optional `discount_rate`, or `borrow_rate`, `invest_rate` and `funding_direction`, now pass through to valuation. The existing workbook itself was not given new columns. |
| Documentation | Updated `FINDINGS-2026-09-09.md`, `MODEL-CONVENTIONS.md`, `DELTA-CONVENTION.md`, `STATUS.md` and `.planning/ROADMAP.md`. |

## P0 verification completed

### Independent exhaustive oracle

For four-day deterministic contracts, the test enumerates every feasible physical schedule
and compares the best independently calculated PV with the dynamic programme. It covers:

- put swing, including strike and forced terminal volume;
- call swing, including strike and forced terminal volume;
- storage, including inventory transitions, injection and withdrawal costs, bounds and the
  terminal state.

### Zero-rate configuration regression

The earlier conflict test used the truth value of `discount_rate`. Consequently,
`discount_rate=0.0` was treated as absent and could silently coexist with a treasury rate
pair. Presence is now checked explicitly and the conflicting configuration raises.

### End-to-end rate plumbing

Tests now verify:

- workbook row to loader to parameter bridge to valuation;
- real Streamlit widgets to the rate applied by the model;
- `Products.ipynb` running under an explicit invest-rate scenario.

### Notebook execution gate

Every code section of `Products.ipynb` runs in a fresh Python process under a reduced-width
smoke configuration. The test fails on an exception or committed stale output. A separate
full default run also completed successfully for 10, 30, 90 and 180 exercise days.

## Where to change the rate

- **Streamlit:** sidebar, **Rate management**.
- **`Products.ipynb`:** section **2. Deal**. Use `RATE_MODE="discount"` with
  `DISCOUNT_RATE`, or `RATE_MODE="treasury_scenario"` with the borrow rate, invest rate and
  explicit direction.
- **Workbook:** add optional columns to the `products` sheet. Use `discount_rate`, or use
  all three of `borrow_rate`, `invest_rate` and `funding_direction`.
- **Python:** pass the same fields in the `params` dictionary supplied to `run_valuation()`.

Do not provide `discount_rate` together with treasury scenario fields.

## Continue safely in Visual Studio Code

Open this folder:

```text
H:\My Drive\Github\dmitry-goryunov\Storage
```

Then run these read-only checks first:

```powershell
git status
git branch --show-current
git diff --check
python -m pytest -q
```

Expected test result is **71 passed**. If the local Python environment is missing packages,
install the pinned dependencies in a virtual environment outside the Drive folder before
interpreting a test failure as a code failure.

Inspect the working-tree diff before committing. The files may appear as uncommitted changes
because Drive received the corrected source bytes, not the scratch clone's Git commits. Do
not discard them. Once the diff and tests are confirmed, create or select a review branch,
commit the files explicitly and only then decide whether to push and open a pull request.

## Remaining work

The following are unresolved, not disproved and not covered by P0:

- production source and construction of a non-flat discount curve;
- mapping contractual settlement dates and payment lags into discount factors;
- genuine asymmetric funding based on the evolving cash balance;
- calibration and provenance of volatility and mean reversion;
- the remaining P1 and P2 items in `.planning/ROADMAP.md`.

Software verification does not constitute market calibration or model validation.
