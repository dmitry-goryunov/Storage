"""Numba-compiled kernel for the averaged-reset swing's accumulating-month hot
path. Kept in a separate module from reset_swing_averaged.py for the same
reason storage_kernels.py is separate from storage_model.py: Numba's disk
cache (cache=True) is invalidated whenever the source file containing the
jitted function changes, so isolating the kernel here means editing the
wrapper/valuation code no longer triggers a recompile.

DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.13/Release 2: the pure-Python
`accumulate_step`/`_exercise_step_3d`/`_propagate_price_step_3d` triple, called
from `_run_month`'s own `for k in range(n_r)` Python loop, made a realistic
6-month deal take on the order of an hour at an n_r worth trusting -- measured,
not assumed. Two numpy-vectorised rewrites of just the interpolation step were
tried and reverted (see `reset_swing_averaged.accumulate_step`'s own
docstring): both gave a real speedup for some deal shapes and a real
regression for others, because the actual cost is Python/numpy per-call
dispatch overhead spread across many small array operations, not the
arithmetic itself. A single Numba-compiled kernel doing the WHOLE per-k month
recursion as one compiled unit -- explicit nested loops, not vectorised numpy
tricks, matching storage_kernels.py's own established pattern for exactly this
class of problem -- removes that dispatch overhead categorically rather than
trading one deal shape's speed for another's.

`run_month_accumulate_core` computes EXACTLY what `reset_swing_averaged.
_run_month_accumulate_reference` computes in pure Python for a non-last month
(the only case expensive enough to need this -- the deal's last month has no
r-axis interpolation in its own loop and was never the bottleneck).
tests/test_reset_swing_kernels.py checks this kernel's output against that
Python reference directly, before `_run_month` is ever wired to call it -- the
brute-force tests in tests/test_reset_swing_averaged.py then confirm the swap
did not change the end-to-end answer.

Accumulation and exercise are now over TWO SEPARATE calendars, `fixing_indices`
(every day of the ONE calendar month feeding the NEXT month's strike) and its
trailing `n_exercise_days`-long suffix (this month's own exercise days) --
2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-01/R-02 found the original
version silently conflated the two, which happens to be harmless only when the
current month is a COMPLETE calendar month (then they coincide exactly) and
silently wrong otherwise (a partial first delivery month, or a first month's
fixing window reaching back past `val_date` by more than one month). See
`reset_terms.DeliveryMonth`'s own docstring for the full account.
"""
import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True, cache=True)
def run_month_accumulate_core(x, p_u, p_m, p_d, d_curve, fixing_indices, n_exercise_days,
                              r_grid, quotes_for_next_month, terminal_value,
                              v_step, daily_max_clips):
    """One delivery month, every incoming strike bucket `k` at once, for the
    non-last month (accumulate=True) case. Mirrors
    `_run_month_accumulate_reference`'s Python k-loop day by day, backward
    over `fixing_indices` (the FULL fixing-observation window for the month
    that follows this one -- always the calendar month THIS delivery month
    itself falls in, so `fixing_indices` always ends exactly at this month's
    own last exercise day and BEGINS at or before this month's own first
    exercise day): fold today's quote into the running average
    (`accumulate_step`'s own uniform-grid interpolation, inlined) on every
    day, then -- ONLY on the trailing `n_exercise_days` of `fixing_indices`,
    i.e. this month's own actual exercise days -- the day's exercise decision
    (`_exercise_step_3d`, inlined), then always propagate one lattice step
    back in price (`_propagate_price_step_3d`, inlined). No separate
    gap-closing propagation afterwards: `fixing_indices` already reaches all
    the way back to this month's own fixing date by construction (it is the
    full calendar month, not just the days this deal happens to exercise on),
    so the backward walk lands there directly.

    `fixing_indices`: int64 array, ASCENDING, the full fixing-observation
    window (this kernel does its own reversed traversal, unlike the Python
    version's `reversed(...)` argument, since a plain Python list reversed()
    has no direct Numba equivalent worth fighting). `n_exercise_days`: how
    many of `fixing_indices`' LAST entries are also this month's own exercise
    days -- `fixing_indices[-n_exercise_days:]` equals this month's
    `exercise_dates`, as indices, by construction (the caller asserts this).

    `k` is independent across iterations (each starts from the SAME
    `terminal_value` and only ever reads/writes its own `results[k]` slice),
    so it is the `prange` dimension -- same reasoning as storage_kernels.py's
    own `k` prange over price nodes.

    Returns (n_r, width, n_l, n_r): `results[k]` is `_run_month`'s own
    `results[k]`, a plain (width, n_l, n_r) array.
    """
    n_r = r_grid.shape[0]
    width = terminal_value.shape[0]
    n_l = terminal_value.shape[1]
    n_fixing_days = fixing_indices.shape[0]
    r0 = r_grid[0]
    dr = (r_grid[n_r - 1] - r_grid[0]) / (n_r - 1)

    # Day-level quantities do not depend on k -- computed once, not per k.
    spot_by_day = np.empty((n_fixing_days, width), dtype=np.float64)
    quote_by_day = np.empty((n_fixing_days, width), dtype=np.float64)
    dc_by_day = np.empty(n_fixing_days, dtype=np.float64)
    for pos in range(n_fixing_days):
        i = fixing_indices[n_fixing_days - 1 - pos]
        spot_by_day[pos, :] = np.exp(x[i, :])
        quote_by_day[pos, :] = quotes_for_next_month[i, :]
        dc_by_day[pos] = d_curve[i]

    results = np.empty((n_r, width, n_l, n_r), dtype=np.float64)

    for k in prange(n_r):
        strike = r_grid[k]
        v = terminal_value.copy()

        for pos in range(n_fixing_days):
            weight_so_far = float(n_fixing_days - 1 - pos)
            new_weight = weight_so_far + 1.0

            # accumulate_step: fold today's own quote into the running
            # average, interpolate the continuation back onto r_grid --
            # EVERY fixing day, exercisable or not.
            v_acc = np.empty((width, n_l, n_r), dtype=np.float64)
            for j in range(width):
                q = quote_by_day[pos, j]
                for r in range(n_r):
                    r_new = (r_grid[r] * weight_so_far + q) / new_weight
                    idx_f = (r_new - r0) / dr
                    if idx_f < 0.0:
                        idx_f = 0.0
                    elif idx_f > n_r - 1:
                        idx_f = float(n_r - 1)
                    idx_lo = int(idx_f)
                    if idx_lo > n_r - 2:
                        idx_lo = n_r - 2
                    frac = idx_f - idx_lo
                    lo = 1.0 - frac
                    for l in range(n_l):
                        v_acc[j, l, r] = v[j, l, idx_lo] * lo + v[j, l, idx_lo + 1] * frac
            v = v_acc

            # _exercise_step_3d: best of 0..daily_max_clips clips today --
            # ONLY on this month's own trailing exercise days.
            if pos < n_exercise_days:
                v_ex = v.copy()
                for j in range(width):
                    add_per_clip = dc_by_day[pos] * v_step * (spot_by_day[pos, j] - strike)
                    for d in range(1, daily_max_clips + 1):
                        if d >= n_l:
                            break
                        add = d * add_per_clip
                        for l in range(n_l - d):
                            for r in range(n_r):
                                cand = v[j, l + d, r] + add
                                if cand > v_ex[j, l, r]:
                                    v_ex[j, l, r] = cand
                v = v_ex

            # _propagate_price_step_3d, one lattice step back (skip at i==0).
            i = fixing_indices[n_fixing_days - 1 - pos]
            if i > 0:
                v_prop = np.empty((width, n_l, n_r), dtype=np.float64)
                for j in range(width):
                    pm = p_m[i - 1, j]
                    for l in range(n_l):
                        for r in range(n_r):
                            acc = pm * v[j, l, r]
                            if j > 0:
                                acc += p_d[i - 1, j] * v[j - 1, l, r]
                            if j < width - 1:
                                acc += p_u[i - 1, j] * v[j + 1, l, r]
                            v_prop[j, l, r] = acc
                v = v_prop

        results[k, :, :, :] = v

    return results
