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
_run_month`'s Python k-loop computes when `accumulate=True` (the only branch
expensive enough to need this -- the deal's last month, `accumulate=False`,
has no r-axis interpolation in its own loop and was never the bottleneck).
tests/test_reset_swing_kernels.py checks this kernel's output against that
existing, already brute-force-verified Python path directly, before
`_run_month` is ever wired to call it -- the brute-force tests in
tests/test_reset_swing_averaged.py then confirm the swap did not change the
end-to-end answer.
"""
import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True, cache=True)
def run_month_accumulate_core(x, p_u, p_m, p_d, d_curve, exercise_indices, fixing_idx,
                              r_grid, quotes_for_next_month, terminal_value,
                              v_step, daily_max_clips):
    """One delivery month, every incoming strike bucket `k` at once, for the
    non-last month (accumulate=True) case. Mirrors `_run_month`'s Python
    k-loop day by day: fold today's quote into the running average
    (`accumulate_step`'s own uniform-grid interpolation, inlined), then the
    day's exercise decision (`_exercise_step_3d`, inlined), then propagate one
    lattice step back in price (`_propagate_price_step_3d`, inlined) -- for
    every exercise day, then the fixing-date gap-closing propagation.
    `exercise_indices`: int64 array, ASCENDING (the month's own exercise
    dates in calendar order -- this kernel does its own reversed traversal,
    unlike the Python version's `reversed(exercise_indices)` argument, since a
    plain Python list reversed() has no direct Numba equivalent worth fighting).

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
    n_days = exercise_indices.shape[0]
    r0 = r_grid[0]
    dr = (r_grid[n_r - 1] - r_grid[0]) / (n_r - 1)

    # Day-level quantities do not depend on k -- computed once, not per k.
    spot_by_day = np.empty((n_days, width), dtype=np.float64)
    quote_by_day = np.empty((n_days, width), dtype=np.float64)
    dc_by_day = np.empty(n_days, dtype=np.float64)
    for pos in range(n_days):
        i = exercise_indices[n_days - 1 - pos]
        spot_by_day[pos, :] = np.exp(x[i, :])
        quote_by_day[pos, :] = quotes_for_next_month[i, :]
        dc_by_day[pos] = d_curve[i]

    results = np.empty((n_r, width, n_l, n_r), dtype=np.float64)

    for k in prange(n_r):
        strike = r_grid[k]
        v = terminal_value.copy()

        for pos in range(n_days):
            weight_so_far = float(n_days - 1 - pos)
            new_weight = weight_so_far + 1.0

            # accumulate_step: fold today's own quote into the running
            # average, interpolate the continuation back onto r_grid.
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

            # _exercise_step_3d: best of 0..daily_max_clips clips today.
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
            i = exercise_indices[n_days - 1 - pos]
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

        # Gap-closing propagation from the first exercise day's predecessor
        # back to the month's own fixing date.
        first_i = exercise_indices[0]
        for i in range(first_i - 2, fixing_idx - 1, -1):
            v_prop = np.empty((width, n_l, n_r), dtype=np.float64)
            for j in range(width):
                pm = p_m[i, j]
                for l in range(n_l):
                    for r in range(n_r):
                        acc = pm * v[j, l, r]
                        if j > 0:
                            acc += p_d[i, j] * v[j - 1, l, r]
                        if j < width - 1:
                            acc += p_u[i, j] * v[j + 1, l, r]
                        v_prop[j, l, r] = acc
            v = v_prop

        results[k, :, :, :] = v

    return results
