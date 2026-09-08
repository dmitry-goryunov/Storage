# Roadmap from the reconciled baseline

This replaces planning derived from the stale Drive working tree.

## Priority 1: behavioural specifications

1. Define hard versus soft tunnel semantics, including whether constraints apply
   before action, after action, or at both points. Then implement and test finding 13.
2. Define curve-shape acceptance criteria for continuity, overshoot, positivity and
   valuation stability before reconsidering the exact knot solve.
3. Agree the production discount-curve source and hedge reporting convention now
   that `delta` is explicitly a discounted forward sensitivity.
4. Decide how withdrawal capacity is expressed. Found 2026-09-08 and verified:
   `wdr_days` passed straight to `run_valuation` has no effect at all --
   30, 45, 90 and 365 all price a storage deal at 2.499672 EUR/MWh -- because
   `value_storage` reads `wdr_rate`, which only `params_for_run_valuation` derives.
   `pricing.ipynb` passes `wdr_days` directly and so silently ignores it, while
   `forward.ipynb` converts it correctly. Worse, the conversion itself collapses:
   `wdr_rate = max(1, round(n_states / wdr_days))` with 30 states gives rate 1 for
   30, 45 and 90 days alike, so any withdrawal slower than one clip per day is
   inexpressible on the current grid. "30 in, 90 out" is silently priced as
   "30 in, 30 out".

   PARTLY ADDRESSED: `value_storage` now rejects `wdr_days` without `wdr_rate`
   rather than dropping it, so the mis-specification cannot pass silently. No
   valuation changed -- the legacy path (no `wdr_days`) and the workbook path
   (which converts) both price exactly as before. Two decisions remain: whether
   `run_valuation` should accept days-based capacity natively and convert, and
   whether sub-clip daily rates need a finer `v_step` or fractional rates in the
   DP. Note also that `inj_days` carries two meanings -- the inventory-state
   count in `resolve_grid`, days-to-fill in `params_for_run_valuation` -- which
   is why the guard covers `wdr_days` only.

## Priority 2: performance and numerical policy

1. Shorten the terminal backstop only after proving that the final exercise day and
   terminal payoff indices remain unchanged. Benchmark finding 14 before and after.
2. Replace the absolute `1e-6` exercise tie threshold with a documented scale-aware
   rule, then run policy-stability and portfolio regressions for finding 17.

## Priority 3: API and maintenance

1. Separate inventory grid size from initial inventory state instead of overloading
   `n_op_start` (finding 19), with a compatibility period for notebooks and callers.
2. Deprecate and then remove unused `check_curve`, `valuation`, `get_exercise` and
   `get_delta` helpers (finding 11).
3. Remove unused tunnel arguments from `probabilities` when the public compatibility
   impact has been checked (finding 12).
4. Review the 0.9 default annualised volatility and document or change it with
   calibration evidence (remaining part of finding 16).

## Release discipline

- GitHub is the only writable source tree.
- Drive review material is read-only evidence, not a second branch.
- Every behavioural change starts with a failing test.
- Pull requests must pass model, portfolio, app and notebook gates from the lockfile.
- Report software verification separately from market calibration and model validation.
