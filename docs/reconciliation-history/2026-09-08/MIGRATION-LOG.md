# Migration log — 2026-09-08

Required by section 5 of [STORAGE-RECONCILIATION-PROCEDURE.md](STORAGE-RECONCILIATION-PROCEDURE.md).
Records every substantive old-only item found when the reconciled snapshot replaced the
Drive review working tree.

**Method deviation from the procedure, deliberate.** Sections 6–7 promote the reconciled
snapshot by renaming Drive folders. That was not used. Instead the reconciled commits were
applied *inside* this repository:

```
git branch -f main origin/main                            # 4a1ca46 -> 2e1b5dd, fast-forward
git checkout -b review/reconcile-model-fixes main
git am --3way docs/RECONCILIATION-COMMITS.patch           # 16 commits, all applied cleanly
```

The result is byte-identical in content to `Storage_new` (verified with
`diff -r --strip-trailing-cr`, all 28 files, differences confined to CRLF line endings),
but keeps the git history, the configured remotes and the path
`H:\My Drive\Github\dmitry-goryunov\Storage`, so no shortcut or folder ID changes and the
branch can be pushed the moment the HTTP 403 is resolved. The review work remains on
branch `review/model-fixes`; `Storage_new` was not modified and remains available for the
folder-rename route if this one is rejected.

Verification: `python -m pytest -q` → **27 passed in 27.5 s** in this directory.

## Items reviewed

| Item | Old (Drive review copy) | New (reconciled) | Result |
|---|---|---|---|
| `curve.csv` | 38 × 3 | 38 × 3 | Identical. No action |
| `quotes.csv` | 6 × 9 | 6 × 9 | Identical. No action |
| `curve_work.xlsx` | sheet `curve`, 39 × 3 | same | Values identical (compared cell values, not bytes). No action |
| `ttf q.xlsx` | `Sheet1`, 4173 × 57 | same shape | **One row differs — see finding 1** |
| `Swing_new.ipynb` | 10 code cells | 10 | 1 cell rewritten; no analysis lost — see finding 2 |
| `forward.ipynb` | 6 code cells | 6 | Substantially rewritten; no analysis lost — see finding 2 |
| `pricing.ipynb` | 9 code cells | 9 | Source cells identical. No action |
| `README.md`, `requirements.txt`, `storage_model.py`, `streamlit_app.py` | stale | reconciled | Reconciled version authoritative, per procedure §4.1 |
| `debug_delta.py` … `debug_delta5.py` | present | absent | Not migrated (procedure §4.3). Retained in git on `review/model-fixes`; their reasoning is preserved in `code_review.md` and `review_checks.py` |

## Finding 1 — `ttf q.xlsx`: one quote row differs, unresolved

Row 4172 (the file's **last row**, quote date **2010-03-12**) differs in **52 of its 57
columns**. Every other cell in the 237,861-cell sheet matches.

| | DA | TTFc1 | TTFc2 | TTFc3 | TTFc4 | TTFc5 | TTFc6 |
|---|---|---|---|---|---|---|---|
| Drive copy | 12.6 | 12.5 | 10 | 10 | 10 | 12.5 | 12.5 |
| Reconciled | 10 | 10 | 10 | 10 | 10 | 10 | 11 |

Neither version is obviously correct — both carry runs of flat 10s that look unlike the
neighbouring row (2010-03-15: 12.7, 11.6, 11.375, 11.3, 11.55 …), so this may be a
data-entry artefact present in the source workbook in two different states.

**Impact today: none.** Both `load_quote_matrix` (app) and `review_checks.get_curve` sort
by `quote_date` before `quote_row_for_fd_date` takes `.iloc[-1]`, so the selected pricing
row is the one with the **latest date**, not the last row in file order. 2010-03-12 sorts
before 2010-03-15 and is never selected. The difference would matter to any time-series or
backtest use of the full matrix.

**Action: none taken.** Neither version was migrated over the other. Resolve from the
original data source, not by copying between directories.

## Finding 2 — notebooks rewritten, nothing lost

`forward.ipynb` cell 4 carries 168 old-only lines. Inspected rather than assumed:

- `def curve_df_for_storage(row, curve_start=None, include_da=True)` — an inline
  reimplementation. The function now lives in the library at `storage_model.py:323` as
  `curve_df_for_storage(row, contract_columns, curve_start=None, include_da=True)`.
- `def full_value_eur(curve_input)` — a local helper with no library equivalent; small,
  and preserved (below).
- The remainder is a parameter block (`product_type`, `FDDate`, `valDate`, `storageStart`,
  `storageEnd`, `days`) replaced by the reconciled notebook's own setup.

This is the duplication that `main` removed in commit `2e1b5dd`, "Kill the notebook/library
duplication trap". `Swing_new.ipynb`'s single old-only cell is the same pattern — an inline
valuation loop now expressed through `run_valuation`.

**Action:** the complete source cells of all three Drive notebooks are preserved, outputs
stripped, in [legacy-notebook-sources.py](legacy-notebook-sources.py) (43 KB). No notebook
was replaced on the strength of differing outputs, and no reconciled notebook was
overwritten.

## Evidence archived under this directory

`code_review.md`, `finding-drive-review.md`, `review_checks.py`, `test_model.py`,
`PROCESS-AND-PROGRESS-REVIEW.md`, `RECONCILIATION-PLAN.md`, `SPEC-remaining-work.md`,
`STATUS-drive-review-final.md`, `STATUS-drive-review-detailed.md`, `ingest-manifest.yaml`,
`STORAGE-RECONCILIATION-PROCEDURE.md`, `legacy-notebook-sources.py`, and the complete
former `.planning/` tree under `legacy-planning/` (14 files including the ingest
classifications and conflict report).

## Collection guard added

Archiving `test_model.py` under `docs/reconciliation-history/` made pytest collect it from
the repository root: 40 tests ran, 4 failed. The failures were not regressions but the API
changes the reconciliation plan predicted in its §7 — `flat()` now means the window-mean
forward, and the fractional-ratchet rejection is obsolete under `strat`-as-clip-count.
Notably 9 of the 13 archived tests pass unmodified against the reconciled model.

`pytest.ini` now sets `testpaths = tests`, so the maintained suite is the only one
collected and `python -m pytest -q` in CI stays at 27 passed. The archived suite remains
readable as evidence.

## Open items — not closed by this migration

1. **Publication.** `review/reconcile-model-fixes` has not been pushed; GitHub returned
   HTTP 403. Until it is pushed and merged, this is a local reconciliation only, and
   `origin/main` remains at `2e1b5dd` without the fixes.
2. **The delta convention contradicts its own decision record.** The reconciled code
   defines `delta` as a *discounted* forward-price sensitivity
   (`compute_all_metrics`: `pa * discount * exp_x`, with
   `test_delta_is_discounted_forward_sensitivity` asserting it). The decision taken at the
   ingest conflict gate and recorded in `legacy-planning/intel/decisions.md` (D-O2) and
   `SPEC-remaining-work.md` §R6 is the opposite: an *undiscounted physical hedge volume*,
   on the grounds that the discount factor cancels in `h = E[S·Q]/F`, so the reported
   number is the MWh actually traded. Both are internally consistent; they differ in what
   the number means to whoever hedges with it, and the app's table is labelled in MWh.
   No number changes while `d_curve` is all ones, which it is on every current path.
   **Requires a decision, then one of the two records must change.**
3. **`ttf q.xlsx` row 4172** — see finding 1.
4. **Legacy directory retention.** The procedure's §7 rename was not performed and no
   directory was deleted. `Storage_new` is now redundant but untouched.
