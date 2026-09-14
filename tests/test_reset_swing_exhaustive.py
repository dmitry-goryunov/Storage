"""DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.10.3's STRONGER independent
check: literal brute-force enumeration of every non-anticipating policy on a
tiny scenario tree, not a second implementation of the same recursion (that
weaker-but-real check is tests/test_reset_swing_stochastic.py).

Deliberately tiny (n_p=1: three price nodes only) and hand-built rather than
routed through ResetSwingTerms/build_reset_schedule -- this targets the exact
recursion reset_swing_exact.py's month-processing loop implements, not the
calendar-month schedule machinery (already covered on its own in
tests/test_reset_terms.py).

Scenario: one fixing date, then exactly two exercise days, mandatory total
volume of exactly one clip. With daily_max = one clip/day, the only real
decision is a classic optimal-stopping one -- exercise on the first day, or
wait and be forced to take the second. Because waiting forces the second
day's action, a "policy" reduces to one binary choice per (j_fix, j_day1)
pair: 3 x 3 = 9 pairs, 2**9 = 512 policies, each one's exact expected
discounted value computable by hand-summed path probabilities -- small enough
to enumerate literally, not approximate.
"""
import numpy as np
import pandas as pd
import pytest

import reset_forward as rf
import reset_swing_exact as rse
import reset_terms as rt


def _forward_point_mass(p_u, p_m, p_d, start_i, start_j, end_i, width):
    """P(node at end_i | point mass at node start_j, date start_i) -- a plain
    forward sweep of the SAME transition arrays reset_forward.py and
    reset_swing_exact.py both read, but written here independently (not a
    shared helper) since this is the probability side of the cross-check,
    not the thing test_reset_forward.py already separately verifies."""
    dist = np.zeros(width)
    dist[start_j] = 1.0
    for i in range(start_i, end_i):
        nxt = np.zeros(width)
        nxt += p_m[i, :] * dist
        nxt[1:] += p_u[i, :-1] * dist[:-1]
        nxt[:-1] += p_d[i, 1:] * dist[1:]
        dist = nxt
    return dist


def test_matches_brute_force_enumeration_of_every_stopping_policy():
    n_p = 1
    width = 2 * n_p + 1  # 3
    val_date, fixing_date = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-10")
    day1, day2 = pd.Timestamp("2026-01-13"), pd.Timestamp("2026-01-16")
    end_date = pd.Timestamp("2026-01-20")

    curve = pd.Series(25.0, index=pd.date_range("2020-01-01", "2030-12-31", freq="D"))
    curve.loc["2026-01-13":"2026-01-15"] = 28.0
    curve.loc["2026-01-16":] = 22.0

    lattice = rf.build_lattice(val_date, fixing_date, end_date, vol=0.6, sMR=1.0,
                              n_p=n_p, daily_curve=curve, discount_rate=0.08)
    quotes = rf.project_month_end_quotes(lattice, [day2])
    date_span = lattice["date_span"]
    u = date_span.get_loc(day2)
    H = quotes[u]  # H[i, j]: conditional expected price at day2, from node j at date i
    fixing_idx = date_span.get_loc(fixing_date)

    v_step = 1_000.0
    daily_max_clips = 1
    n_l = 2  # l in {0, 1}; mandatory total = 1 clip
    terminal_value = np.array([[-np.inf, 0.0]] * width)  # must end with l == 1

    month = rt.DeliveryMonth(label=None, fixing_date=fixing_date, exercise_dates=(day1, day2))
    collapsed = np.empty((width, n_l))
    strikes = np.empty(width)
    for j_fix in range(width):
        strike = float(H[fixing_idx, j_fix])
        strikes[j_fix] = strike
        v = rse._run_month_for_one_fixing_node(
            lattice, month, strike, terminal_value, v_step, daily_max_clips)
        collapsed[j_fix, :] = v[j_fix, :]

    v = collapsed
    for i in range(fixing_idx - 1, -1, -1):
        v = rse._propagate_one_step(v, lattice["p_u"][i, :], lattice["p_m"][i, :], lattice["p_d"][i, :])
    dp_pv = float(v[n_p, 0])

    # --- Independent brute-force reference ---
    p_u, p_m, p_d, x, dc = (lattice["p_u"], lattice["p_m"], lattice["p_d"],
                            lattice["x"], lattice["d_curve"])
    i_fix, i_day1, i_day2 = fixing_idx, date_span.get_loc(day1), date_span.get_loc(day2)

    # P(j_fix) starting from the lattice's own single root node at i=0.
    p_jfix = _forward_point_mass(p_u, p_m, p_d, 0, n_p, i_fix, width)
    spot_day1 = np.exp(x[i_day1, :])
    spot_day2 = np.exp(x[i_day2, :])

    best_value, best_policy = None, None
    for bits in range(2 ** (width * width)):
        policy = {}
        b = bits
        for j_fix in range(width):
            for j1 in range(width):
                policy[(j_fix, j1)] = b & 1
                b >>= 1

        total = 0.0
        for j_fix in range(width):
            if p_jfix[j_fix] < 1e-15:
                continue
            strike = strikes[j_fix]
            p_j1_given_jfix = _forward_point_mass(p_u, p_m, p_d, i_fix, j_fix, i_day1, width)
            for j1 in range(width):
                p1 = p_j1_given_jfix[j1]
                if p1 < 1e-15:
                    continue
                exercise_day1 = policy[(j_fix, j1)]
                if exercise_day1:
                    payoff = dc[i_day1] * v_step * (spot_day1[j1] - strike)
                    total += p_jfix[j_fix] * p1 * payoff
                else:
                    p_j2_given_j1 = _forward_point_mass(p_u, p_m, p_d, i_day1, j1, i_day2, width)
                    for j2 in range(width):
                        p2 = p_j2_given_j1[j2]
                        if p2 < 1e-15:
                            continue
                        payoff = dc[i_day2] * v_step * (spot_day2[j2] - strike)
                        total += p_jfix[j_fix] * p1 * p2 * payoff
        if best_value is None or total > best_value:
            best_value, best_policy = total, policy

    assert best_value == pytest.approx(dp_pv, rel=1e-9), (
        f"DP: {dp_pv}, brute force over all {2**(width*width)} policies: {best_value}")
