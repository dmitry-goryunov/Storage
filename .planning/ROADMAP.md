# Roadmap from the reconciled baseline

This replaces planning derived from the stale Drive working tree.

## Priority 1: behavioural specifications

1. Define hard versus soft tunnel semantics, including whether constraints apply
   before action, after action, or at both points. Then implement and test finding 13.
2. Define curve-shape acceptance criteria for continuity, overshoot, positivity and
   valuation stability before reconsidering the exact knot solve.
3. Agree the production discount-curve source. The hedge reporting convention is
   SETTLED: `delta` is an undiscounted physical hedge volume (decision D-O2), the
   invariant carries the discount weights, and both are documented in
   docs/MODEL-CONVENTIONS.md. What remains is where a real `d_curve` comes from,
   and the settlement-timing convention behind it -- the model currently assumes
   cash settles on the exercise day.
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

5. DONE: the intrinsic split for struck swings. `flat_metric` is now net of the
   strike in both `value_call_swing` and `value_put_swing`, so it benchmarks the
   same quantity `profiled_metric` reports. The defect was symmetric and the fix
   had to be too -- it was found on the call (intrinsic 0.314, -9.686, -27.686 for
   K = 0, 10, 28) and the put turned out to be the mirror image (0.157, 10.157,
   20.157 for K = 0, 10, 20). Both now hold flat across strikes, `extrinsic` was
   never affected, and no deal without a strike changes. Covered by a test
   parametrised over both products.

## Priority 2: performance and numerical policy

1. Shorten the terminal backstop only after proving that the final exercise day and
   terminal payoff indices remain unchanged. Benchmark finding 14 before and after.
2. Replace the absolute `1e-6` exercise tie threshold with a documented scale-aware
   rule, then run policy-stability and portfolio regressions for finding 17.

## Priority 3: API and maintenance

1. DONE: `n_states` (grid size) and `initial_state` (starting inventory) replace the
   overloaded `n_op_start` (finding 19). Both are settable in one call as
   `set_volume_states(n_states, initial_state=...)`, and `n_op_start` remains as a
   deprecated alias so existing notebooks and callers keep working. `build()` now
   rejects a start outside the grid -- previously an out-of-range value indexed past
   the value array inside a Numba kernel, where that is undefined rather than an
   IndexError.
2. DONE: removed unused `check_curve`, `valuation`, `get_exercise` and `get_delta`
   (finding 11). Compatibility checked first -- no caller anywhere in the library,
   apps, notebooks or tests; `get_exercise`/`get_delta` were only re-exported.
3. DONE: `probabilities` no longer takes `i_curve`, `w_curve`, `i_ratch`, `w_ratch`,
   `mintunnel` or `max_tunnel` (finding 12). All six were unread -- `strat` has held
   the exact signed clip move since the reconciliation refactor.
4. Review the 0.9 default annualised volatility and document or change it with
   calibration evidence (remaining part of finding 16).

## Release discipline

- GitHub is the only writable source tree.
- Drive review material is read-only evidence, not a second branch.
- Every behavioural change starts with a failing test.
- Pull requests must pass model, portfolio, app and notebook gates from the lockfile.
- Report software verification separately from market calibration and model validation.
