# Roadmap from the reconciled baseline

This replaces planning derived from the stale Drive working tree.

## Priority 0: verification gates

Completed 2026-09-09. An independent exhaustive oracle now checks four-day put swing, call
swing and storage valuations. Rate configuration has one validation path across the library,
workbook, notebook and app; an explicit zero discount rate can no longer be mistaken for an
absent field. The app treasury scenario is exercised end to end, and every code section of
`Products.ipynb` runs in a fresh-process smoke test that also rejects stale stored output.

## Priority 1: behavioural specifications

1. Define hard versus soft tunnel semantics, including whether constraints apply
   before action, after action, or at both points. Then implement and test finding 13.

   PARTLY ADDRESSED 2026-09-10, and the earlier note that "the tunnels cannot bind
   as built" was wrong. They bind exactly: a 70 % floor on 1 April holds at 42.00 of
   60 clips, and a 30 % ceiling on 1 October at 18.00. What was missing was a way to
   reach them -- `value_storage` read no such parameter and `set_volume_states`
   resets the arrays -- so `min_inventory`/`max_inventory` now carry date to
   fraction through `run_valuation`, and the convention is documented: the bound
   applies to the balance the day OPENS with. Reported on the closing balance the
   same schedule looks a clip short, which is the ambiguity this item is really
   about.

   Two things still open. The choice of convention is documented, not decided --
   a contract may mean either balance. And the constraint remains a penalty of
   `1000 * v_step` per clip, so a large enough deal can pay it and breach the
   bound; `value_storage` now re-checks each bound on the built policy and raises
   rather than reporting a valuation of a different contract, but that is a
   detector, not a fix. The arbitrary scale is still arbitrary.
2. Define curve-shape acceptance criteria for continuity, overshoot, positivity and
   valuation stability before reconsidering the exact knot solve.
3. PARTLY ADDRESSED: `discount_rate` now builds `d_curve` (annual, continuously
   compounded) on `Storage` and through `run_valuation`, so time value is priced and
   the optimiser prefers early withdrawal; the flat benchmark is PV'd to match. What
   remains is the same question in its production form. Agree the market discount-curve
   source, contractual settlement-date mapping, and whether each reported rate is a market
   discount, treasury funding or internal hurdle rate. These are different economic views.
   `borrow_rate`/`invest_rate` are only explicit treasury scenarios; genuine asymmetric
   funding requires a cash-balance state or equivalent nonlinear recursion. The hedge reporting convention is
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

   ASYMMETRIC RATES WORK, and the docstring that said otherwise is fixed
   (2026-09-10). `params_for_run_valuation` derives inj_rate and wdr_rate from
   inj_days/wdr_days and `value_storage` reads both, so "30 in, 60 out" prices
   correctly as inj_rate=2, wdr_rate=1 on a 60-state grid, with the schedule
   honouring the 20,000 and 10,000 MWh/day caps. What remains open is the
   rounding: rates are whole clips per day, so 30/60 needs n_states to be a
   multiple of 60, and a coarser grid silently prices 30/30. Covered by a test.

   ALSO 2026-09-10: the same integer-clip limit reaches ratchets, and there it is
   worse. `int(rate * multiplier)` floors a positive multiplier to zero, so an
   ordinary profile -- withdrawal x0.30 near empty on a 1 clip/day rate -- froze the
   store and priced the deal at exactly 0 EUR against 5,299,882 unratcheted, with no
   error. `assert_ratchets_expressible` now refuses it and names the grid that would
   work; on 240 states the same profile prices at 2,807,291, so a realistic ratchet
   costs 47 % of value. A wrong rate misprices a deal; a wrong ratchet zeroed it.
   This strengthens the case for expressing rates as something other than whole
   clips, rather than making each caller find a grid that happens to divide.

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

## Priority 4: model capability

Not defects. The model computes what it claims and 91 tests say so. These are things it
cannot currently represent at all, found 2026-09-10 while pricing a 30/60 store, and ordered
by what they are worth against what they cost.

1. **A second factor, so the seasonal spread can move.** This is one-factor: every forward is
   driven by a single state variable, so any two forwards are correlated **exactly 1.000**
   and the summer/winter spread can only move as a fixed multiple of spot. Measured at
   sVol 0.5, sMR 1.0, from June 2026: Jul-27 carries log-vol 0.170 and Dec-27 0.112, giving
   the spread a log-vol of `|0.340 - 0.223| x 0.354 = 0.0412` -- about **1.03 EUR/MWh** of
   uncertainty on a spread 12.00 EUR/MWh wide.

   The consequence is visible in any storage valuation: extrinsic 0.465 against intrinsic
   10.533, a 4.2 % share. And it moves the wrong way with mean reversion --

   | sMR | intrinsic | extrinsic | share |
   |---:|---:|---:|---:|
   | 0.2 | 10.5333 | 0.1189 | 1.1 % |
   | 1.0 | 10.5333 | 0.4651 | 4.2 % |
   | 4.0 | 10.5333 | 1.2324 | 10.5 % |

   -- which shows what the extrinsic is actually measuring: short-term spot wiggle to cycle
   against, not seasonal spread optionality. Faster reversion gives more intra-month churn
   and does nothing for the spread. So the model prices a fast-cycling store competently and
   structurally cannot price the piece that dominates a seasonal one.

   The fix is a Schwartz-Smith style short-term deviation plus long-term equilibrium level.
   The cost is real: the DP gains a state dimension, from (time x price x inventory) to
   (time x short x long x inventory). **Cheaper interim** if that is too much: keep one
   factor but calibrate it to the *spread's* volatility rather than spot's, and say so in the
   outputs -- the wrong model with the right sensitivity, which is defensible only when
   documented.

2. **Volumetric fuel loss.** Not modelled at all. `value_storage` reads `inj_cost` and
   `wdr_cost` in EUR/MWh and nothing volumetric, so injecting 100 MWh always makes 100 MWh
   available to withdraw. Real storage retains 1-2 % of injected gas as fuel. On a 600,000
   MWh deal at ~25 EUR/MWh that is 150,000-300,000 EUR -- comparable to the entire extrinsic
   value of 282,000 EUR on the same deal, so it is first-order for storage and currently
   absent. Much cheaper than item 1: scale the volume that arrives in inventory against the
   volume injected, in the transition.

3. **Calibrate at what the product is sensitive to.** Related to P3.4 but sharper than it. A
   vol fitted to front-month spot returns is calibrated to the wrong quantity for a spread
   product, and the table above shows the answer swinging tenfold across plausible `sMR`.
   Whatever is estimated, record the target alongside the estimate.

Two smaller items already listed above were both hit in practice on 2026-09-10 and are worth
re-reading in that light: integer clip rates (P1.4 -- a 30/65 deal needs a 390-state grid to
be expressible at all) and the absolute `1e-6` tie threshold (P2.2).

## Release discipline

- GitHub is the only writable source tree.
- Drive review material is read-only evidence, not a second branch.
- Every behavioural change starts with a failing test.
- Pull requests must pass model, portfolio, app and notebook gates from the lockfile.
- Report software verification separately from market calibration and model validation.
