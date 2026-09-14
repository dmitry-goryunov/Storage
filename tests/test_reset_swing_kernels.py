"""reset_swing_kernels.run_month_accumulate_core against
reset_swing_averaged._run_month_accumulate_reference -- the pure-Python
`for k in range(n_r)` loop `_run_month` itself used to run directly and now
only keeps around as an independent check, since `_run_month` calls this same
kernel in production (DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.13's
Release 2 investigation). Comparing against `_run_month` itself here would be
circular -- the compiled kernel checked against itself -- which is exactly
why `_run_month_accumulate_reference` exists: this is the bridge of trust
from "the decomposed Python version is correct"
(tests/test_reset_swing_averaged.py's brute-force enumeration, run before
`_run_month` was wired to the kernel) to "the compiled kernel computes the
identical thing, just faster."

Randomised rather than a single golden case: `terminal_value` and
`quotes_for_next_month` are random arrays (not derived from an actual
lattice projection), deliberately, so this exercises the recursion's own
arithmetic on arbitrary inputs rather than one specific, possibly
unrepresentative, economic scenario.
"""
import numpy as np
import pandas as pd
import pytest

import reset_forward as rf
import reset_swing_averaged as rsa
import reset_swing_kernels as rsk
import reset_terms as rt


def _scenario(n_p, n_l, n_r, n_days, daily_max_clips, seed):
    rng = np.random.default_rng(seed)
    val_date = pd.Timestamp("2026-01-01")
    fixing = pd.Timestamp("2026-02-28")
    exercise_dates = pd.date_range("2026-03-01", periods=n_days, freq="D")
    end_date = exercise_dates[-1] + pd.Timedelta(days=10)
    curve = pd.Series(25.0, index=pd.date_range("2020-01-01", "2030-12-31", freq="D"))

    lattice = rf.build_lattice(val_date, fixing, end_date, vol=0.4, sMR=1.0,
                              n_p=n_p, daily_curve=curve, discount_rate=0.05)
    date_span = lattice["date_span"]
    width = 2 * n_p + 1
    month = rt.DeliveryMonth(label=None, fixing_date=fixing, exercise_dates=tuple(exercise_dates))

    quotes_for_next_month = rng.uniform(15.0, 35.0, size=(len(date_span), width))
    r_grid = np.linspace(10.0, 40.0, n_r)
    terminal_value = rng.normal(scale=1000.0, size=(width, n_l, n_r))
    v_step = 500.0

    exercise_indices = np.array([date_span.get_loc(d) for d in month.exercise_dates], dtype=np.int64)
    fixing_idx = date_span.get_loc(month.fixing_date)
    return dict(lattice=lattice, month=month, r_grid=r_grid,
               quotes_for_next_month=quotes_for_next_month, terminal_value=terminal_value,
               v_step=v_step, daily_max_clips=daily_max_clips,
               exercise_indices=exercise_indices, fixing_idx=fixing_idx)


@pytest.mark.parametrize("n_p,n_l,n_r,n_days,daily_max_clips,seed", [
    (1, 2, 5, 1, 1, 0),    # smallest possible: single day, single clip
    (2, 3, 8, 2, 2, 1),
    (3, 4, 10, 3, 1, 2),
    (5, 6, 12, 5, 3, 3),   # daily_max_clips >= n_l - 1: exercises the `break` in the clip loop
    (8, 5, 15, 4, 6, 4),   # daily_max_clips > n_l: same break, more aggressively
    (1, 8, 6, 7, 1, 5),    # smallest width (n_p=1 -> 3), larger n_l and more days
])
def test_kernel_matches_the_python_reference_exactly(n_p, n_l, n_r, n_days, daily_max_clips, seed):
    s = _scenario(n_p, n_l, n_r, n_days, daily_max_clips, seed)

    # _run_month_accumulate_reference, NOT _run_month -- the latter now calls
    # this same kernel in production, which would make the comparison circular.
    results = rsa._run_month_accumulate_reference(
        s["lattice"], s["month"], s["r_grid"], s["quotes_for_next_month"],
        s["terminal_value"], s["v_step"], s["daily_max_clips"])
    python_stack = np.stack(results, axis=0)  # (n_r, width, n_l, n_r)

    lattice = s["lattice"]
    kernel_out = rsk.run_month_accumulate_core(
        lattice["x"], lattice["p_u"], lattice["p_m"], lattice["p_d"], lattice["d_curve"],
        s["exercise_indices"], s["fixing_idx"], s["r_grid"], s["quotes_for_next_month"],
        s["terminal_value"], s["v_step"], s["daily_max_clips"])

    assert kernel_out.shape == python_stack.shape
    np.testing.assert_allclose(kernel_out, python_stack, atol=1e-7, rtol=1e-9)


def test_kernel_is_substantially_faster_at_realistic_month_scale():
    """Not a tight timing assertion (machine-dependent, this repository's own
    standing caveat about timing claims) -- just confirms the kernel is not
    accidentally SLOWER than the Python path it exists to replace, at a scale
    representative of one real delivery month (n_p=15, ~monthly n_l, a 30-day
    month). n_r=50, not the several hundred a real deal would want: the
    Python reference is itself O(n_r^2) (sec.13's own finding), so this test
    pays that cost too -- n_r=50 already shows the speedup clearly without
    making the suite eat the Python path's full cost at n_r=200 (~1 minute)
    just to prove the kernel beats it."""
    import time

    s = _scenario(n_p=15, n_l=25, n_r=50, n_days=30, daily_max_clips=2, seed=7)
    lattice = s["lattice"]

    # Warm up the JIT compile (not timed) with the EXACT same call the timed block
    # below makes -- a smaller or sliced warm-up array can type as a DIFFERENT
    # Numba specialisation (e.g. a sliced array is no longer C-contiguous), which
    # compiles separately and silently leaves the real compile cost inside the
    # timed block instead. Found exactly this way: this test first showed a
    # "kernel" call taking 100+ seconds because its warm-up sliced the arrays.
    rsk.run_month_accumulate_core(
        lattice["x"], lattice["p_u"], lattice["p_m"], lattice["p_d"], lattice["d_curve"],
        s["exercise_indices"], s["fixing_idx"], s["r_grid"], s["quotes_for_next_month"],
        s["terminal_value"], s["v_step"], s["daily_max_clips"])

    t0 = time.perf_counter()
    rsk.run_month_accumulate_core(
        lattice["x"], lattice["p_u"], lattice["p_m"], lattice["p_d"], lattice["d_curve"],
        s["exercise_indices"], s["fixing_idx"], s["r_grid"], s["quotes_for_next_month"],
        s["terminal_value"], s["v_step"], s["daily_max_clips"])
    t_kernel = time.perf_counter() - t0

    t0 = time.perf_counter()
    rsa._run_month_accumulate_reference(
        s["lattice"], s["month"], s["r_grid"], s["quotes_for_next_month"],
        s["terminal_value"], s["v_step"], s["daily_max_clips"])
    t_python = time.perf_counter() - t0

    assert t_kernel < t_python, f"kernel ({t_kernel:.2f}s) should beat Python ({t_python:.2f}s)"
    assert t_kernel < t_python / 3.0, (
        f"expected at least a 3x speedup at this scale, got kernel={t_kernel:.2f}s "
        f"vs python={t_python:.2f}s")
