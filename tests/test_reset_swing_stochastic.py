"""The point-reset exact benchmark at REAL volatility -- everything in
test_reset_swing_point.py deliberately uses near-zero volatility so the answer
is hand-computable, which never exercises the genuinely stochastic part of the
engine (the actual optionality a swing exists to price). This file is that
missing check: monotonicity in volatility, an independently-coded (non-
vectorised) reference cross-check, and grid-convergence evidence.
"""
import numpy as np
import pandas as pd
import pytest

import reset_forward as rf
import reset_swing_exact as rse
import reset_terms as rt


def _shaped_curve():
    curve = pd.Series(25.0, index=pd.date_range("2020-01-01", "2030-12-31", freq="D"))
    curve.loc["2026-03-01":"2026-03-15"] = 32.0
    curve.loc["2026-03-16":"2026-03-31"] = 22.0
    return curve


def _terms(**overrides):
    defaults = dict(
        val_date="2026-01-01", storage_start="2026-03-01", storage_end="2026-03-31",
        daily_max_mwh=1_000.0, v_step_mwh=1_000.0,
        global_min_mwh=0.0, global_max_mwh=8_000.0,
        vol=0.5, sMR=1.0, discount_rate=0.0, n_p=15)
    defaults.update(overrides)
    return rt.ResetSwingTerms(**defaults)


def test_value_rises_with_volatility_for_optional_volume():
    """Standard option-pricing sanity check: more uncertainty in the month-
    ahead reset and the underlying spot can only make optional exercise more
    valuable, never less -- the one property every test up to now (all at
    near-zero vol) could not have exercised."""
    curve = pd.Series(25.0, index=pd.date_range("2020-01-01", "2030-12-31", freq="D"))
    values = []
    for vol in (0.1, 0.3, 0.5, 0.8, 1.2):
        terms = _terms(vol=vol, storage_end="2026-03-31", global_max_mwh=10_000.0, n_p=12)
        schedule = rt.build_reset_schedule(terms)
        values.append(rse.value_point_reset_call_swing(terms, schedule, daily_curve=curve))
    assert values == sorted(values), values
    assert values[-1] > values[0] * 5, "expected a large, not marginal, rise across this range"


def _independent_reference_value(terms, schedule, daily_curve):
    """Same recursion as reset_swing_exact.value_point_reset_call_swing, but
    with plain nested Python loops over every (j, l, d) instead of NumPy
    shift/broadcast operations -- an independent implementation of the same
    algorithm, so a bug specific to the vectorised version would show up as a
    disagreement here. Deliberately not shared code with reset_swing_exact.py."""
    lattice = rf.build_lattice(terms.val_date, terms.storage_start, terms.storage_end,
                               vol=terms.vol, sMR=terms.sMR, n_p=terms.n_p,
                               daily_curve=daily_curve, discount_rate=terms.discount_rate)
    quotes = rf.project_month_end_quotes(lattice, schedule.month_end_dates)
    date_span = lattice["date_span"]
    x, p_u, p_m, p_d, dc = (lattice["x"], lattice["p_u"], lattice["p_m"],
                            lattice["p_d"], lattice["d_curve"])
    width = 2 * terms.n_p + 1
    v_step = terms.v_step_mwh
    n_l = int(round(terms.global_max_mwh / v_step)) + 1
    lo = int(round(terms.global_min_mwh / v_step))
    hi = n_l - 1
    daily_max_clips = int(round(terms.daily_max_mwh / v_step))

    def propagate(v, i):
        out = [[0.0] * n_l for _ in range(width)]
        for j in range(width):
            for l in range(n_l):
                total = p_m[i, j] * v[j][l]
                if j + 1 < width:
                    total += p_u[i, j] * v[j + 1][l]
                if j - 1 >= 0:
                    total += p_d[i, j] * v[j - 1][l]
                out[j][l] = total
        return out

    terminal = [[0.0 if lo <= l <= hi else -np.inf for l in range(n_l)] for _ in range(width)]

    for month, month_end in zip(reversed(schedule.months), reversed(schedule.month_end_dates)):
        u = date_span.get_loc(month_end)
        H = quotes[u]
        fixing_idx = date_span.get_loc(month.fixing_date)
        exercise_indices = [date_span.get_loc(d) for d in month.exercise_dates]

        collapsed = [[None] * n_l for _ in range(width)]
        for j_fix in range(width):
            strike = float(H[fixing_idx, j_fix])
            v = [row[:] for row in terminal]
            for i in reversed(exercise_indices):
                spot_i = [float(np.exp(x[i, j])) for j in range(width)]
                v_new = [[None] * n_l for _ in range(width)]
                for j in range(width):
                    for l in range(n_l):
                        best = None
                        for d in range(0, min(daily_max_clips, n_l - 1 - l) + 1):
                            cont = v[j][l + d]
                            if cont == -np.inf:
                                continue
                            candidate = d * v_step * dc[i] * (spot_i[j] - strike) + cont
                            if best is None or candidate > best:
                                best = candidate
                        v_new[j][l] = best if best is not None else -np.inf
                v = v_new
                if i > 0:
                    v = propagate(v, i - 1)
            first_i = exercise_indices[0]
            for i in range(first_i - 2, fixing_idx - 1, -1):
                v = propagate(v, i)
            for l in range(n_l):
                collapsed[j_fix][l] = v[j_fix][l]
        terminal = collapsed

    v = terminal
    fixing_idx0 = date_span.get_loc(schedule.months[0].fixing_date)
    for i in range(fixing_idx0 - 1, -1, -1):
        v = propagate(v, i)
    return v[terms.n_p][0]


@pytest.mark.parametrize("vol,sMR,global_min,global_max", [
    (0.5, 1.0, 0.0, 5_000.0),      # optional volume
    (0.5, 1.0, 3_000.0, 3_000.0),  # mandatory volume
    (0.9, 0.3, 0.0, 8_000.0),      # slow mean reversion, high vol
    (0.2, 4.0, 2_000.0, 6_000.0),  # fast mean reversion, partly-optional volume
])
def test_matches_an_independently_coded_reference_at_real_volatility(
        vol, sMR, global_min, global_max):
    """DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.10.3's independent-exact-
    test requirement, at genuinely stochastic parameters -- not the near-zero-
    vol limit every other test in this file's sibling uses."""
    terms = _terms(vol=vol, sMR=sMR, global_min_mwh=global_min,
                   global_max_mwh=global_max, n_p=4)
    schedule = rt.build_reset_schedule(terms)
    curve = _shaped_curve()

    fast = rse.value_point_reset_call_swing(terms, schedule, daily_curve=curve)
    independent = _independent_reference_value(terms, schedule, curve)
    assert fast == pytest.approx(independent, abs=1e-6), (vol, sMR, global_min, global_max)


def test_n_p_converges_as_the_tree_widens():
    """sec.10.4's price-tree-boundary convergence, on a real-vol scenario.
    Not a specific tolerance claim (that belongs to a numbered benchmark, per
    this repository's own convention of not asserting a brittle number where
    a trend is the actual claim) -- just that successive relative moves
    shrink, which a boundary-truncated tree would not do cleanly."""
    curve = _shaped_curve()
    values = []
    for n_p in (6, 10, 15, 20, 28):
        terms = _terms(n_p=n_p)
        schedule = rt.build_reset_schedule(terms)
        values.append(rse.value_point_reset_call_swing(terms, schedule, daily_curve=curve))

    moves = [abs(values[i + 1] - values[i]) / abs(values[i]) for i in range(len(values) - 1)]
    assert moves == sorted(moves, reverse=True), (
        f"expected shrinking relative moves as n_p widens, got {moves}")
    assert moves[-1] < 1e-4, f"not converged by n_p=28: last move {moves[-1]:.2%}"


def test_v_step_refinement_does_not_move_the_price_once_the_daily_rate_is_exact():
    """sec.10.4's volume-grid convergence. Once daily_max_mwh is already exact
    on the coarsest grid tested, refining v_step further adds no new
    achievable daily choice -- the optimal policy here is bang-bang (use the
    full daily rate or none), so this is a genuine finding, not a weak test:
    the price should not move AT ALL under refinement, not just converge."""
    curve = _shaped_curve()
    values = []
    for v_step in (1000.0, 500.0, 250.0, 125.0):
        terms = _terms(v_step_mwh=v_step, n_p=15)
        schedule = rt.build_reset_schedule(terms)
        values.append(rse.value_point_reset_call_swing(terms, schedule, daily_curve=curve))
    assert values[0] == pytest.approx(values[-1], abs=1e-6), values
