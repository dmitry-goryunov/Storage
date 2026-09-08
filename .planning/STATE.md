# Reconciled state

Date: 2026-09-08 (updated after merge)

- Canonical baseline: GitHub `main` at `8e7997c`, the merge of pull request #1
  (reconciliation baseline before it: `2e1b5dd`)
- Active branch: `main`. `review/reconcile-model-fixes` is merged;
  `review/model-fixes` is retained as the September review evidence
- Applicable review findings resolved: 12 of 19 at the time of the reconciliation report,
  plus two settled afterwards -- finding 15's guard was made reachable (it ran after
  `smoothen_curve`, so contract-curve gaps surfaced as SciPy's NaN error), and a new
  finding, `value_storage` silently discarding `wdr_days`, was verified and half fixed
- Test suite: 29 passing, 0 failing, 0 skipped with repository data present
- Clean lockfile reproduction: passing; CI runs it pinned on every push
- Curve correction: additive method retained pending model-governance criteria
- Release status: published and merged. Software verification only -- market calibration
  and independent model validation remain outstanding

The 12 resolved findings are 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15 and 18.
Finding 16 is partial. Finding 20 is not applicable to canonical `main`.
