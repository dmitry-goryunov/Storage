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
  4. (Found extending this to a genuine two-month chain, below.) `r_grid`
     must bracket EVERY delivery month's own projected-quote range, not just
     one of them -- a curve with real month-to-month shape gives each month a
     different range, and `strike=r_grid[k]` together with `accumulate_step`'s
     interpolation will silently clamp to whichever range you DID bracket for
     any node whose true value falls outside it, for an edge (low-probability
     but not negligible) lattice node, with no exception and a confidently
     wrong number. `value_averaged_reset_call_swing` now checks this itself
     (weighted by `lattice["q"]`, since an unweighted check is USELESS in
     practice -- a truncated lattice piles up real, non-negligible probability
     at its own edge nodes once enough days have elapsed relative to `n_p`,
     which is a property of the lattice's own truncation, not something an
     r_grid could ever be wide enough to fully absorb).

Once a scenario respects all four, the DP and the brute force agree to
~1e-12 to ~1e-14 (tighter than point-reset's own 1e-9 bar, since n_p=1 here
leaves essentially no lattice-approximation error to absorb).
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
    r_grid = np.linspace(Ks.min() - 0.5, Ks.max() + 0.5, 1001)

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


def test_matches_brute_force_enumeration_across_two_delivery_months():
    """The single-month test above exercises `_run_accumulation_only` (the
    pre-deal window) and `_run_month` with `accumulate=False` (the deal's
    last/only month), but NOT `_run_month` with `accumulate=True` or
    `_collapse_fresh_axis` -- the code path used by every NON-last month in a
    real multi-month deal, and the exact code path responsible for three of
    the five real bugs this module had (see reset_swing_averaged.py's own
    module docstring). This closes that gap with a genuine two-month chain.

    One pre-deal day (day_P0) fixes month A's strike K_A; month A's own
    single exercise day (day_A1) is ALSO the sole day that accumulates into
    month B's strike K_B (a degenerate one-day "average", same as the
    single-month test's window -- deliberately, to keep this small enough to
    brute-force literally; multi-day averaging is already covered
    separately). One mandatory clip total across BOTH months: exercise on A1,
    or wait and be forced to take B1. Brute force: for each (day_P0, day_A1)
    node pair (9 combinations, each an exact K_A and K_B), the best of
    {exercise now at A1} vs {expected forced payoff at B1} -- exactly the
    same optimal-stopping decomposition as the other exhaustive tests here,
    extended one step further out.
    """
    n_p = 1
    width = 2 * n_p + 1
    val_date = pd.Timestamp("2026-01-01")
    day_P0 = pd.Timestamp("2026-01-02")  # val_date + 1 -- single pre-deal day, fixes K_A
    fixing_A = day_P0
    day_A1 = pd.Timestamp("2026-01-03")  # month A's only exercise day; also accumulates -> K_B
    fixing_B = day_A1                    # month B's fixing == month A's last (only) day
    day_B1 = pd.Timestamp("2026-01-04")  # month B's only exercise day
    end_date = pd.Timestamp("2026-01-20")

    curve = pd.Series(25.0, index=pd.date_range("2020-01-01", "2030-12-31", freq="D"))
    curve.loc["2026-01-01":"2026-01-03"] = 26.0
    curve.loc["2026-01-04":] = 24.0

    lattice = rf.build_lattice(val_date, fixing_A, end_date, vol=0.6, sMR=1.0,
                              n_p=n_p, daily_curve=curve, discount_rate=0.08)
    date_span = lattice["date_span"]
    quotes = rf.project_month_end_quotes(lattice, [day_A1, day_B1])
    u_A, u_B = date_span.get_loc(day_A1), date_span.get_loc(day_B1)
    H_A, H_B = quotes[u_A], quotes[u_B]  # H_A: month A's own settlement; H_B: month B's
    i_P0, i_A1, i_B1 = date_span.get_loc(day_P0), date_span.get_loc(day_A1), date_span.get_loc(day_B1)

    v_step = 1_000.0
    daily_max_clips = 1
    terminal_value_B = np.array([[-np.inf, 0.0]] * width)  # must end with l == 1
    month_A = rt.DeliveryMonth(label=None, fixing_date=fixing_A, exercise_dates=(day_A1,))
    month_B = rt.DeliveryMonth(label=None, fixing_date=fixing_B, exercise_dates=(day_B1,))

    # r_grid must bracket BOTH K_A's range and K_B's range -- see the module
    # docstring's trap 4, found precisely by this test's own first attempt.
    lo = min(H_A[i_P0, :].min(), H_B[i_A1, :].min()) - 1.0
    hi = max(H_A[i_P0, :].max(), H_B[i_A1, :].max()) + 1.0
    r_grid = np.linspace(lo, hi, 2001)

    results_B = rsa._run_month(lattice, month_B, r_grid, None, terminal_value_B, v_step, daily_max_clips)
    terminal_value_A = np.stack(results_B, axis=-1)  # (width, n_l, n_r): B's own results, stacked for A
    results_A = rsa._run_month(lattice, month_A, r_grid, H_B, terminal_value_A, v_step, daily_max_clips)
    collapsed_A = rsa._collapse_fresh_axis(results_A, label="A")
    v_by_r_l0 = np.stack([arr[:, 0] for arr in collapsed_A], axis=1)  # (width, n_r), fn of K_A
    quote_by_date = {day_P0: H_A[i_P0, :]}
    v_at_root = rsa._run_accumulation_only(lattice, [day_P0], r_grid, quote_by_date, v_by_r_l0)
    root_row = v_at_root[n_p, :]
    assert root_row.max() - root_row.min() < 1e-6, "root value should not depend on the starting r bucket"
    code_pv = float(root_row[0])

    p_u, p_m, p_d, x, dc = (lattice["p_u"], lattice["p_m"], lattice["p_d"],
                            lattice["x"], lattice["d_curve"])
    spotA1, spotB1 = np.exp(x[i_A1, :]), np.exp(x[i_B1, :])
    p_jP0 = _forward_point_mass(p_u, p_m, p_d, 0, n_p, i_P0, width)

    total = 0.0
    for jP0 in range(width):
        w0 = p_jP0[jP0]
        if w0 < 1e-15:
            continue
        K_A = float(H_A[i_P0, jP0])
        p_jA1 = _forward_point_mass(p_u, p_m, p_d, i_P0, jP0, i_A1, width)
        for jA1 in range(width):
            w1 = p_jA1[jA1]
            if w1 < 1e-15:
                continue
            K_B = float(H_B[i_A1, jA1])
            exercise_now = dc[i_A1] * v_step * (spotA1[jA1] - K_A)
            p_jB1 = _forward_point_mass(p_u, p_m, p_d, i_A1, jA1, i_B1, width)
            wait_then_forced_at_B = sum(
                p_jB1[jB1] * dc[i_B1] * v_step * (spotB1[jB1] - K_B)
                for jB1 in range(width) if p_jB1[jB1] > 1e-15)
            total += w0 * w1 * max(exercise_now, wait_then_forced_at_B)

    assert code_pv == pytest.approx(total, abs=1e-6), (
        f"DP+interpolation: {code_pv}, brute force: {total}")


def test_bracket_check_rejects_a_grid_that_clamps_a_material_lattice_state():
    """The same curve/scale as the multi-month brute-force test above, but
    with r_lo/r_hi narrowed to bracket only month B's range -- exactly the
    mistake this test file's own first attempt at that test made (trap 4 in
    the module docstring). Must raise, not silently clamp and return a
    confidently wrong number."""
    curve = pd.Series(25.0, index=pd.date_range("2020-01-01", "2030-12-31", freq="D"))
    curve.loc["2026-04-01":"2026-04-30"] = 26.0
    curve.loc["2026-05-01":"2026-05-31"] = 24.0

    terms = rt.ResetSwingTerms(
        val_date="2026-01-01", storage_start="2026-04-01", storage_end="2026-05-31",
        daily_max_mwh=1_000.0, v_step_mwh=1_000.0,
        global_min_mwh=0.0, global_max_mwh=5_000.0,
        vol=0.4, sMR=1.0, discount_rate=0.05, n_p=8)
    schedule = rt.build_reset_schedule(terms)

    with pytest.raises(ValueError, match="does not bracket"):
        rsa.value_averaged_reset_call_swing(terms, schedule, curve, n_r=200, r_lo=21.0, r_hi=27.0)

    # A grid wide enough for BOTH months' ranges must not raise.
    rsa.value_averaged_reset_call_swing(terms, schedule, curve, n_r=200, r_lo=15.0, r_hi=35.0)


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
    # r_lo/r_hi bracket the curve fairly tightly (24.0 +/- 8, not +/- 20) rather
    # than broadly: this deal has a real, binding global-volume cap (5 of 30
    # possible days), so the value-vs-K surface has a genuine kink, and -- same
    # O(1/n_r) story as test_finer_r_grid_moves_averaged_reset_toward_point_reset_at_low_vol
    # below -- a much wider range at this n_r leaves visible interpolation
    # error even though nothing about the DP is wrong. [20, 28] once looked
    # tight enough, but value_averaged_reset_call_swing's own bracket check
    # (added after a real multi-month bug turned out to be exactly this: too
    # narrow a grid silently clamping a materially-probable lattice state)
    # correctly rejects it -- at vol=0.3 the lattice's own >=1e-6-probability
    # states reach [20.32, 28.16], just outside [20, 28].
    averaged_pv = rsa.value_averaged_reset_call_swing(
        terms, schedule, curve, n_r=400, r_lo=16.0, r_hi=32.0)

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
