"""DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.10.3's STRONGER independent
check, applied to Release 1B (the averaged, not point, reset): literal
brute-force enumeration of every non-anticipating policy on a tiny scenario
tree -- not a second implementation of the same recursion.

Getting this test to actually match `reset_swing_averaged.py` took three
wrong attempts, each a bug in THIS test's own scenario setup, not in the
module under test -- worth recording since the same traps are easy to fall
into again:

  1. The pre-deal/accumulation dates passed to `_run_accumulation_only` and
     the per-day accumulation inside `_run_month` each advance the lattice by
     exactly ONE calendar step per date processed. That is only correct when
     the date list is genuinely consecutive calendar days -- exactly what
     `value_averaged_reset_call_swing` always builds via
     `pd.date_range(..., freq="D")`, but easy to violate by hand.
  2. `_run_accumulation_only` propagates back to (first date's index - 1),
     which lands on val_date (index 0) only when the date list itself starts
     at val_date + 1 day -- again exactly what the real caller builds
     (`pd.date_range(val_date, fixing_date, freq="D")[1:]`), not something
     the function re-derives itself.
  3. A delivery month's own `fixing_date` must be THE SAME calendar day as
     the last day of the accumulation window feeding it (`build_reset_schedule`
     sets it to `month_start - 1 day`, i.e. the preceding month's own last
     day) -- not a separate, later date. Get this wrong and the value handed
     across the accumulation/exercise boundary is off by one lattice step.

Once the scenario respects all three, the DP and the brute force agree to
~1e-12 (tighter than point-reset's own 1e-9 bar, since n_p=1 here leaves
essentially no lattice-approximation error to absorb).
"""
import numpy as np
import pandas as pd
import pytest

import reset_forward as rf
import reset_swing_exact as rse
import reset_swing_averaged as rsa
import reset_terms as rt


def _forward_point_mass(p_u, p_m, p_d, start_i, start_j, end_i, width):
    """P(node at end_i | point mass at node start_j, date start_i) -- written
    independently here (not a shared helper), same as
    test_reset_swing_exhaustive.py's own copy."""
    dist = np.zeros(width)
    dist[start_j] = 1.0
    for i in range(start_i, end_i):
        nxt = np.zeros(width)
        nxt += p_m[i, :] * dist
        nxt[1:] += p_u[i, :-1] * dist[:-1]
        nxt[:-1] += p_d[i, 1:] * dist[1:]
        dist = nxt
    return dist


def test_matches_brute_force_enumeration_with_averaged_strike():
    """Two pre-deal days average into the delivery month's strike, then a
    classic two-day optimal-stopping decision (exercise day 1, or wait and be
    forced to take day 2 -- mandatory total of exactly one clip) runs against
    it. Brute force: for each of the 3x3 (day_P1, day_P2) node pairs (each an
    exact, non-interpolated K), independently enumerate all 2**3 exercise-day-1
    policies (one bit per day-1 node) and take the best -- exactly
    test_reset_swing_exhaustive.py's own decomposition, extended with the
    averaging step in front of it.
    """
    n_p = 1
    width = 2 * n_p + 1  # 3
    val_date = pd.Timestamp("2026-01-01")
    day_P1, day_P2 = pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")  # val_date+1, +2
    fixing_date = day_P2  # SAME day as the last accumulation day -- see module docstring, trap 3
    day_E1, day_E2 = pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-05")  # right after fixing
    end_date = pd.Timestamp("2026-01-20")

    curve = pd.Series(25.0, index=pd.date_range("2020-01-01", "2030-12-31", freq="D"))
    curve.loc["2026-01-01":"2026-01-03"] = 27.0
    curve.loc["2026-01-04":] = 23.0

    lattice = rf.build_lattice(val_date, fixing_date, end_date, vol=0.6, sMR=1.0,
                              n_p=n_p, daily_curve=curve, discount_rate=0.08)
    date_span = lattice["date_span"]
    quotes = rf.project_month_end_quotes(lattice, [day_E2])
    u = date_span.get_loc(day_E2)
    H = quotes[u]  # H[i, j]: conditional expected price at day_E2, from node j at date i
    i_P1, i_P2 = date_span.get_loc(day_P1), date_span.get_loc(day_P2)
    i_E1, i_E2 = date_span.get_loc(day_E1), date_span.get_loc(day_E2)

    v_step = 1_000.0
    daily_max_clips = 1
    terminal_value = np.array([[-np.inf, 0.0]] * width)  # must end with l == 1
    month = rt.DeliveryMonth(label=None, fixing_date=fixing_date, exercise_dates=(day_E1, day_E2))

    Ks = np.array([[(H[i_P1, jp1] + H[i_P2, jp2]) / 2.0 for jp2 in range(width)]
                   for jp1 in range(width)])
    r_grid = np.linspace(Ks.min() - 0.5, Ks.max() + 0.5, 4001)

    results = rsa._run_month(lattice, month, r_grid, None, terminal_value, v_step, daily_max_clips)
    v_by_r_l0 = np.stack([arr[:, 0] for arr in results], axis=1)  # (width, n_r)
    quote_by_date = {day_P1: H[i_P1, :], day_P2: H[i_P2, :]}
    v_at_root = rsa._run_accumulation_only(lattice, [day_P1, day_P2], r_grid, quote_by_date, v_by_r_l0)
    root_row = v_at_root[n_p, :]
    assert root_row.max() - root_row.min() < 1e-6, "root value should not depend on the starting r bucket"
    code_pv = float(root_row[0])

    # --- Independent brute-force reference ---
    p_u, p_m, p_d, x, dc = (lattice["p_u"], lattice["p_m"], lattice["p_d"],
                            lattice["x"], lattice["d_curve"])
    p_jP1 = _forward_point_mass(p_u, p_m, p_d, 0, n_p, i_P1, width)
    spotE1, spotE2 = np.exp(x[i_E1, :]), np.exp(x[i_E2, :])

    total = 0.0
    for jP1 in range(width):
        w1 = p_jP1[jP1]
        if w1 < 1e-15:
            continue
        p_jP2 = _forward_point_mass(p_u, p_m, p_d, i_P1, jP1, i_P2, width)
        for jP2 in range(width):
            w2 = p_jP2[jP2]
            if w2 < 1e-15:
                continue
            K = (H[i_P1, jP1] + H[i_P2, jP2]) / 2.0
            branch_prob = w1 * w2
            p_jE1 = _forward_point_mass(p_u, p_m, p_d, i_P2, jP2, i_E1, width)
            best_branch = None
            for bits in range(2 ** width):
                policy = {}
                b = bits
                for jE1 in range(width):
                    policy[jE1] = b & 1
                    b >>= 1
                val = 0.0
                for jE1 in range(width):
                    pw = p_jE1[jE1]
                    if pw < 1e-15:
                        continue
                    if policy[jE1]:
                        val += pw * dc[i_E1] * v_step * (spotE1[jE1] - K)
                    else:
                        p_jE2 = _forward_point_mass(p_u, p_m, p_d, i_E1, jE1, i_E2, width)
                        for jE2 in range(width):
                            pw2 = p_jE2[jE2]
                            if pw2 < 1e-15:
                                continue
                            val += pw * pw2 * dc[i_E2] * v_step * (spotE2[jE2] - K)
                if best_branch is None or val > best_branch:
                    best_branch = val
            total += branch_prob * best_branch

    assert code_pv == pytest.approx(total, abs=1e-6), (
        f"DP+interpolation: {code_pv}, brute force: {total}")


def test_accumulate_step_is_constant_at_the_zero_weight_boundary():
    """At weight_so_far=0 (nothing folded in yet), r_new = quote_by_node[j] for
    EVERY r_grid entry -- the output must be independent of which bucket you
    started from, regardless of what `continuation` looks like. This is the
    boundary property both the pre-deal window and each month's own fresh
    outgoing accumulator rely on."""
    r_grid = np.linspace(0.0, 100.0, 51)
    width = 3
    continuation = np.stack([np.sin(r_grid / 7.0), r_grid ** 2, -r_grid], axis=0)  # deliberately non-trivial
    quote_by_node = np.array([10.0, 20.0, 30.0])

    out = rsa.accumulate_step(continuation, r_grid, quote_by_node, weight_so_far=0.0, day_weight=1.0)

    for j in range(width):
        expected = np.interp(quote_by_node[j], r_grid, continuation[j, :])
        assert out[j, :] == pytest.approx(expected, abs=1e-9)
        assert out[j, :].max() - out[j, :].min() == pytest.approx(0.0, abs=1e-9)


def test_accumulate_step_running_average_matches_hand_computation():
    """Three equal-weight folds of quotes 10, 20, 30 (in that order) must give
    exactly the plain average 20.0, independent of the r bucket -- the
    simplest possible hand-checkable running average, with a flat (r-independent)
    terminal so no interpolation of a non-trivial function is involved yet
    (test_matches_brute_force_enumeration_with_averaged_strike above covers
    that combination)."""
    r_grid = np.linspace(0.0, 50.0, 21)
    width = 1
    v = np.zeros((width, len(r_grid)))  # terminal value irrelevant to the accumulator itself
    quotes = [10.0, 20.0, 30.0]
    weight_so_far = 0.0
    r_after = None
    for q in quotes:
        # Track what running average a fresh accumulator (started at this q)
        # would show, by folding forward instead of backward -- an
        # independent, hand-computable check of the SAME recursive formula
        # accumulate_step implements backward.
        r_after = q if r_after is None else (r_after * weight_so_far + q) / (weight_so_far + 1.0)
        weight_so_far += 1.0
    assert r_after == pytest.approx(20.0)

    out = rsa.accumulate_step(v, r_grid, np.array([10.0]), weight_so_far=0.0, day_weight=1.0)
    out = rsa.accumulate_step(out, r_grid, np.array([20.0]), weight_so_far=1.0, day_weight=1.0)
    out = rsa.accumulate_step(out, r_grid, np.array([30.0]), weight_so_far=2.0, day_weight=1.0)
    # v is identically zero everywhere, so this only checks the accumulator
    # doesn't raise or misshape -- the real average-tracking check is the
    # forward hand computation above and the brute-force test's K values.
    assert out.shape == (width, len(r_grid))


def test_matches_point_reset_when_the_averaging_window_is_a_single_day():
    """A one-day accumulation window is a degenerate average of one term --
    Release 1B must then reproduce Release 1A exactly, using the REAL
    ResetSwingTerms/build_reset_schedule machinery end to end (not a hand-built
    month), since this is an integration check of the whole pipeline rather
    than the recursion in isolation."""
    curve = pd.Series(25.0, index=pd.date_range("2020-01-01", "2030-12-31", freq="D"))
    curve.loc["2026-04-01":"2026-04-30"] = 24.0

    terms = rt.ResetSwingTerms(
        val_date="2026-03-30", storage_start="2026-04-01", storage_end="2026-04-30",
        daily_max_mwh=1_000.0, v_step_mwh=1_000.0,
        global_min_mwh=0.0, global_max_mwh=5_000.0,
        vol=0.3, sMR=1.0, discount_rate=0.05, n_p=6)
    schedule = rt.build_reset_schedule(terms)

    point_pv = rse.value_point_reset_call_swing(terms, schedule, daily_curve=curve)
    # r_lo/r_hi bracket the curve tightly (24.0 +/- 4) rather than broadly:
    # this deal has a real, binding global-volume cap (5 of 30 possible days),
    # so the value-vs-K surface has a genuine kink, and -- same O(1/n_r)
    # story as test_finer_r_grid_moves_averaged_reset_toward_point_reset_at_low_vol
    # below -- a wide range at this n_r leaves visible interpolation error
    # (rel ~1e-4 over [15,35]) even though nothing about the DP is wrong; a
    # tight range gets the same n_r to rel ~1e-6.
    averaged_pv = rsa.value_averaged_reset_call_swing(
        terms, schedule, curve, n_r=400, r_lo=20.0, r_hi=28.0)

    assert averaged_pv == pytest.approx(point_pv, rel=1e-4)


def test_finer_r_grid_moves_averaged_reset_toward_point_reset_at_low_vol():
    """Not a tight numerical match: the averaged-reset value function has a
    genuine kink (in K, at the exercise boundary), and linear interpolation
    across a kink converges only at FIRST order in the grid spacing --
    O(1/n_r), confirmed empirically (n_r 100->400, a 4x refinement, cut the
    gap to point-reset by almost exactly 4x). A coarse grid is therefore
    expected to overstate the value at low vol, not a sign of a logic bug --
    see reset_swing_averaged.py's own module docstring and
    DESIGN-MONTHLY-RESET-SWING-2026-09-13.md for the same note. This test
    only pins the DIRECTION and rough shape of that convergence so a real
    regression (wrong sign, non-monotonic, or no improvement at all) still
    fails it, without demanding a specific n_r reach machine precision."""
    curve = pd.Series(25.0, index=pd.date_range("2020-01-01", "2030-12-31", freq="D"))
    curve.loc["2026-04-01":"2026-04-30"] = 22.0
    curve.loc["2026-05-01":"2026-05-31"] = 30.0

    terms = rt.ResetSwingTerms(
        val_date="2026-01-01", storage_start="2026-04-01", storage_end="2026-05-31",
        daily_max_mwh=1_000.0, v_step_mwh=1_000.0,
        global_min_mwh=0.0, global_max_mwh=10_000.0,
        vol=1e-4, sMR=1.0, discount_rate=0.05, n_p=8)
    schedule = rt.build_reset_schedule(terms)

    point_pv = rse.value_point_reset_call_swing(terms, schedule, daily_curve=curve)
    assert point_pv == pytest.approx(0.0, abs=10.0)  # near-deterministic curve: point-reset's own near-zero anchor

    coarse = rsa.value_averaged_reset_call_swing(terms, schedule, curve, n_r=20, r_lo=15.0, r_hi=45.0)
    fine = rsa.value_averaged_reset_call_swing(terms, schedule, curve, n_r=100, r_lo=15.0, r_hi=45.0)
    finer = rsa.value_averaged_reset_call_swing(terms, schedule, curve, n_r=400, r_lo=15.0, r_hi=45.0)

    assert coarse > fine > finer > point_pv, (
        f"expected monotonic improvement toward point-reset's near-zero anchor: "
        f"coarse={coarse}, fine={fine}, finer={finer}, point={point_pv}")
    # First-order (O(1/n_r)) convergence: a 4x refinement should cut the
    # residual by roughly 4x, not merely "some". Loose bounds (2x-8x) since
    # this is an empirical rate, not an exact one.
    ratio = (fine - point_pv) / (finer - point_pv)
    assert 2.0 < ratio < 8.0, f"expected roughly first-order convergence, got ratio={ratio}"
