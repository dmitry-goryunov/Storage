# Status: storage and swing pricing model

Updated: 2026-09-08

## Current state

The reconciliation plan has been executed against a fresh clone of canonical
GitHub `main` at `2e1b5dd`. The former Drive branch and its planning files are
historical review evidence only.

- Local branch: `review/reconcile-model-fixes`
- Local HEAD: `b0416078f22fb36364ffad8626822af147f6552f`
- Working tree: clean
- Reconciled suite: 27 passed, 0 failed, 0 skipped with repository data present
- Clean lockfile reproduction: 27 passed
- Streamlit app smoke tests: both entrypoints pass
- Portfolio MtM anchor: unchanged at approximately -56,901 EUR
- Remote publication: blocked by GitHub integration HTTP 403

Read [RECONCILIATION-EXECUTION-REPORT.md](https://drive.google.com/file/d/1Eb-XHqi4YreiWDeQyqSNR86C_iq0Wip8/view)
for the complete evidence, product impact, curve study and residual roadmap.

## Completed

- Removed exercise and delta aggregation rounding.
- Normalised swing metrics by expected physical exercise rather than hedge delta.
- Sized a variable-volatility tree from peak volatility and validated all live probabilities.
- Added a generic post-build terminal inventory reachability check.
- Defined delta as discounted forward-price sensitivity and included discount factors.
- Added convexity explanations to both apps.
- Consolidated regression, finite-difference, portfolio, app and notebook checks.
- Pinned direct dependencies and captured a tested transitive lock.
- Added continuous integration for Python 3.12.
- Assessed the additive curve correction against a knot solve and retained the additive method.

## Curve decision

The experimental knot solve reduced month-boundary daily steps, but changed storage
value by 0.75% on `curve.csv` and by 15% to 32% in shaped stress cases. It also increased
spline overshoot in some cases. This is a model-governance change, not a safe defect port,
so the existing additive correction remains in place pending explicit acceptance criteria.

## Release blocker

No remote branch or pull request exists. Direct HTTPS push had no credential, and the
connected GitHub integration returned `403 Resource not accessible by integration` when
creating the branch. Once write access is enabled, publish from the clean local clone with:

```bash
git push -u origin review/reconcile-model-fixes
```

The complete ordered commit series is preserved as
[RECONCILIATION-COMMITS.patch](https://drive.google.com/file/d/1QP746bU5_JY23UwtsIDifz4_iaVZ8bfQ/view)
and can be applied to canonical `main` with `git am` if the local clone is unavailable.

## Remaining roadmap

The remaining work is deliberately separate: hard versus soft tunnel semantics, safe
backstop shortening, a scale-aware exercise tie rule, disentangling `n_op_start`, dead-code
deprecation, unused probability arguments and review of the 0.9 default volatility.
