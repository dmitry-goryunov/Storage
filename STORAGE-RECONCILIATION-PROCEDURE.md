# Storage directory reconciliation procedure

Date prepared: 2026-09-08

## 1. Objective

Create one active Drive project at:

`H:\My Drive\Github\dmitry-goryunov\Storage`

The active project should contain the reconciled source from `Storage_new`, while
preserving the September review material from the existing `Storage` directory as
dated, read-only evidence. GitHub should remain the authoritative version-control
system once write access is restored.

This procedure does not treat the two directories as ordinary Git branches. The
existing `Storage` directory was built from a stale and divergent baseline and
contains untracked files. `Storage_new` is a reconciled file snapshot based on
canonical GitHub `main`, but deliberately contains no `.git` directory.

## 2. Current verified state

| Item | Current role | Drive folder |
|---|---|---|
| `Storage` | Legacy review working directory | [Open](https://drive.google.com/drive/folders/1E8yDc92FS-WvsXblX__Qw9PQkX58HxCK) |
| `Storage_new` | Proposed active reconciled snapshot | [Open](https://drive.google.com/drive/folders/1q9uWUzg9tzJyTexZPy510R6FJYor_18t) |
| GitHub `main` baseline | Canonical source baseline | Commit `2e1b5ddacd9649b211e7ce45cedd3bbb944a7f1a` |
| Reconciled local branch | Completed implementation | `review/reconcile-model-fixes` |
| Reconciliation patch | Recovery of the ordered local commits | `Storage/docs/RECONCILIATION-COMMITS.patch` |

Verified observations:

- `Storage_new` contains all 28 files tracked by the reconciled branch.
- The reconciled suite passes 27 tests with no failures or skips when repository
  data are present.
- A second clean environment installed from `requirements-lock.txt` also passes.
- Remote publication is unresolved. GitHub returned HTTP 403 when the branch was
  created through the connected integration.
- The old and new directories must not be overlaid or synchronised file by file.

## 3. Governing rules

1. Do not overwrite `Storage_new` source files with files of the same name from
   the old directory.
2. Copy historical material before any rename. Do not move or delete originals
   until verification is complete.
3. Do not copy `.git`, `.venv`, `__pycache__`, Numba caches or other generated
   files into the active Drive directory.
4. Do not use binary hashes alone to compare Excel workbooks or notebooks. Excel
   ZIP metadata and notebook outputs can differ without a substantive data or code
   change.
5. Do not delete the legacy directory in this procedure. Deletion is a separate,
   explicitly approved retention decision.
6. Git operations should run in a clean clone outside Google Drive. The Drive
   directory is a working snapshot and archive surface, not the Git database.

## 4. Classification and conflict policy

### 4.1 Files for which `Storage_new` is authoritative

Keep the `Storage_new` version of:

- `storage_model.py`
- `storage_kernels.py`
- `streamlit_app.py`
- `portfolio_app.py`
- `README.md`
- `requirements.txt`
- `requirements-lock.txt`
- `.github/workflows/test.yml`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- everything under `tests/`
- the notebooks and data files already copied from the reconciled branch

These files should not be replaced merely because the old directory contains a
file with the same name.

### 4.2 Legacy review evidence to preserve under a new name or archive path

Create this directory in `Storage_new`:

`docs/reconciliation-history/2026-09-08/`

Copy the following material from the old `Storage` directory:

| Old location | Archive location or treatment |
|---|---|
| `code_review.md` | `docs/reconciliation-history/2026-09-08/code_review.md` |
| `finding.md` | `docs/reconciliation-history/2026-09-08/finding-drive-review.md` |
| `review_checks.py` | `docs/reconciliation-history/2026-09-08/review_checks.py` |
| `test_model.py` | `docs/reconciliation-history/2026-09-08/test_model.py` |
| `docs/PROCESS-AND-PROGRESS-REVIEW.md` | Preserve with its current name |
| `docs/RECONCILIATION-PLAN.md` | Preserve with its current name |
| `docs/SPEC-remaining-work.md` | Preserve with its current name and mark historical |
| `docs/STATUS.md` | Preserve as `STATUS-drive-review-final.md` |
| `docs/ingest-manifest.yaml` | Preserve with its current name |
| old `.planning/` | Copy in full to `docs/reconciliation-history/2026-09-08/legacy-planning/` |

`RECONCILIATION-EXECUTION-REPORT.md` and `RECONCILIATION-COMMITS.patch` may be
copied into the active `docs/` directory because they describe and recover the
reconciled implementation rather than the stale source.

### 4.3 Files to exclude

Do not migrate:

- old `storage_model.py`, `streamlit_app.py`, README or notebooks as active files;
- `debug_delta.py` through `debug_delta5.py`;
- old `.git` contents;
- old `.venv` contents;
- `__pycache__`, `.pyc`, Numba cache or Streamlit cache files;
- temporary exports or duplicate files whose provenance is unknown.

The debug scripts may be retained only inside the dated legacy directory if their
retention is desired. Their relevant reasoning is already preserved in the review
documents and reconciled tests.

## 5. Data and notebook comparison gate

Before declaring the old directory redundant, compare files with matching names.

### Text and Python files

Use an ordinary line diff. A difference is expected for reconciled source; retain
the `Storage_new` version unless the old file contains clearly separate review
evidence, in which case archive it under a distinct name.

### Excel workbooks

For each workbook, compare:

- sheet names;
- row and column dimensions;
- header names;
- formulas and values in populated ranges;
- named ranges, if any;
- record counts and key identifiers.

If a substantive old-only data row or formula exists, document it and migrate it
through a reviewed change. Do not overwrite the complete reconciled workbook.

### Notebooks

Compare source cells separately from outputs and metadata. Preserve an old-only
analytical cell by copying it into an explicitly labelled archive notebook or by
porting the logic through a reviewed source change. Do not replace a reconciled
notebook solely because output cells differ.

Record every substantive old-only item in:

`docs/reconciliation-history/2026-09-08/MIGRATION-LOG.md`

For each item record the source, destination, reason, reviewer and result.

## 6. Verification before renaming folders

All of the following must be true:

1. `Storage_new` still contains the 28 reconciled tracked files.
2. `storage_model.py`, both files under `tests/`, `.github/workflows/test.yml`,
   `requirements-lock.txt` and `docs/RECONCILIATION-REPORT.md` match the reconciled
   source by size and content.
3. The legacy evidence listed in section 4.2 exists in the dated archive path.
4. `MIGRATION-LOG.md` records every substantive data or notebook difference, or
   states explicitly that none were found.
5. No `.git`, `.venv`, `__pycache__` or compiled cache appears in `Storage_new`.
6. The old `Storage` directory remains intact and recoverable.
7. The reconciliation report and patch can be opened from Drive.

Where a runnable clean clone is available outside Drive, also run:

```bash
python -m venv <external-environment-path>
<external-environment-path>/bin/pip install -r requirements-lock.txt
<external-environment-path>/bin/python -m pytest -q
```

On Windows, use the corresponding `Scripts\python.exe` and `Scripts\pip.exe`
paths. The expected result for the reconciled commit is 27 passing tests.

## 7. Folder promotion

Only after section 6 passes:

1. Rename the existing `Storage` directory to
   `Storage_legacy_review_2026-09-08`.
2. Confirm the renamed legacy directory is visible and retains its contents.
3. Rename `Storage_new` to `Storage`.
4. Confirm the active path is now:
   `H:\My Drive\Github\dmitry-goryunov\Storage`.
5. Open the active README, reconciliation report, model file and both test files.
6. Allow Google Drive synchronisation to finish before running or editing files.
7. Update any shortcuts that depended on the old folder ID or name.

Renaming a Drive folder normally preserves its Drive folder ID, but local Windows
paths and shortcuts change. Verify both rather than assuming synchronisation has
completed.

## 8. Restore GitHub as the source of truth

The active Drive snapshot is not a substitute for Git history. In a clean clone
outside Drive:

```bash
git clone https://github.com/dmitry-goryunov/Storage.git
cd Storage
git checkout -b review/reconcile-model-fixes
git am "<path-to>/RECONCILIATION-COMMITS.patch"
python -m pytest -q
git push -u origin review/reconcile-model-fixes
```

Before applying the patch, confirm that GitHub `main` still has baseline commit
`2e1b5dd` in its history. If `main` has moved, first inspect the new commits and
apply the patch on a new branch. Resolve conflicts deliberately; do not force-push
or discard the newer history.

Open a pull request and require the continuous-integration checks to pass. After
merge, record the pull-request URL and merge commit in the migration log and in
the active status file.

## 9. Rollback

Until the legacy folder is separately approved for deletion, rollback is:

1. Stop editing both directories.
2. Rename the promoted active `Storage` back to `Storage_new_failed_<date>`.
3. Rename `Storage_legacy_review_2026-09-08` back to `Storage`.
4. Record the reason and any changed files in `MIGRATION-LOG.md`.

Do not merge changes made independently in both directories during rollback. Diff
them and classify each change under section 4 first.

## 10. Definition of done

Reconciliation is complete only when:

- one active Drive directory is named `Storage`;
- the former directory is a dated, read-only legacy archive;
- all old-only review evidence is preserved under `docs/reconciliation-history/`;
- data and notebook differences have been reviewed and logged;
- the active directory contains no Git metadata, environments or caches;
- the 27-test suite passes in a clean external clone;
- the reconciled branch has been pushed and merged into GitHub;
- the status file identifies the GitHub merge commit as the canonical state.

Until the GitHub branch is published and merged, the directory reconciliation is
complete only as a Drive reorganisation. The software release remains unpublished.
