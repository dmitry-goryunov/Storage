"""DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.5.1's projection acceptance tests,
against `reset_forward.py`. No reset-swing valuation exists yet -- this pins the
one piece everything else in that design depends on: the conditional month-ahead
quote, computed by exact backward induction through the lattice's own transition
law rather than a separately-derived closed form.
"""
import numpy as np
import pandas as pd
import pytest

import reset_forward as rf


def _seasonal_curve():
    days = pd.date_range("2025-01-01", "2029-12-31", freq="D")
    values = 25.0 + 5.0 * np.sin(np.arange(len(days)) * 2.0 * np.pi / 365.25)
    return pd.Series(values, index=days)


@pytest.mark.parametrize("sMR", [0.0, 1e-6, 0.3, 1.0, 4.0])
def test_one_step_recursion_matches_explicit_neighbour_sum(sMR):
    """Q_i^M(j) = sum_k P_i(j,k) * Q_(i+1)^M(k) -- checked by an explicit,
    non-vectorised neighbour sum at several (i,j), independent of the
    shift-and-add implementation being tested."""
    curve = _seasonal_curve()
    lattice = rf.build_lattice("2026-01-01", "2026-02-01", "2026-11-30",
                               vol=0.5, sMR=sMR, n_p=15, daily_curve=curve)
    month_end = pd.Timestamp("2026-08-31")
    u = lattice["date_span"].get_loc(month_end)
    H = rf.project_month_end_quotes(lattice, [month_end])[u]

    p_u, p_m, p_d = lattice["p_u"], lattice["p_m"], lattice["p_d"]
    width = H.shape[1]
    for i in (0, u // 3, u // 2, u - 1):
        for j in (0, width // 4, width // 2, width - 1):
            expected = p_m[i, j] * H[i + 1, j]
            if j + 1 < width:
                expected += p_u[i, j] * H[i + 1, j + 1]
            if j - 1 >= 0:
                expected += p_d[i, j] * H[i + 1, j - 1]
            assert H[i, j] == pytest.approx(expected, abs=1e-9), (sMR, i, j)


@pytest.mark.parametrize("sMR", [0.0, 1e-6, 0.3, 1.0, 4.0])
def test_tower_identity_reproduces_the_input_forward(sMR):
    """sum_j q[i,j] * Q_i^M(j) == Q_0^M at every date i, and Q_0^M itself must
    equal the input forward curve's own value at the delivery date -- the
    model's own forward-fitting guarantee, not a new claim."""
    curve = _seasonal_curve()
    lattice = rf.build_lattice("2026-01-01", "2026-02-01", "2026-11-30",
                               vol=0.5, sMR=sMR, n_p=15, daily_curve=curve)
    month_end = pd.Timestamp("2026-08-31")
    u = lattice["date_span"].get_loc(month_end)
    H = rf.project_month_end_quotes(lattice, [month_end])[u]
    q, fwd = lattice["q"], lattice["fwd"]

    root_value = float(H[0, :] @ q[0, :])
    assert root_value == pytest.approx(float(fwd[u]), rel=1e-9)

    for i in (0, u // 4, u // 2, u - 1, u):
        tower = float(H[i, :] @ q[i, :])
        assert tower == pytest.approx(float(fwd[u]), rel=1e-6), (sMR, i)


def test_zero_mean_reversion_conditional_quote_tracks_the_current_node():
    """At sMR=0 and a FLAT curve there is no mean reversion to correct for and
    nothing for the daily forward-fit to distort between dates, so the model's
    own conditional view of a future date's price should track the current
    node's own price closely -- not a separate claim, a sanity check that the
    recursion is not silently doing something else entirely."""
    flat = pd.Series(25.0, index=pd.date_range("2025-01-01", "2029-12-31", freq="D"))
    lattice = rf.build_lattice("2026-01-01", "2026-02-01", "2026-11-30",
                               vol=0.5, sMR=0.0, n_p=15, daily_curve=flat)
    month_end = pd.Timestamp("2026-08-31")
    u = lattice["date_span"].get_loc(month_end)
    H = rf.project_month_end_quotes(lattice, [month_end])[u]
    x = lattice["x"]

    mid = u // 2
    spot = np.exp(x[mid, :])
    # Compare only near the centre of the distribution -- the rarely-visited
    # outer nodes carry a genuinely larger Jensen's-inequality drift correction
    # even at sMR=0, which is not what this sanity check is about.
    q = lattice["q"][mid, :]
    reachable = q > 1e-3
    relative_gap = np.abs(H[mid, reachable] - spot[reachable]) / spot[reachable]
    assert np.max(relative_gap) < 0.05


def test_root_conditional_quote_matches_across_two_tree_widths():
    """sec.5.1's boundary check: a tower identity can pass on an inadequately
    narrow tree. Confirm the root-date projection is stable as n_p widens, and
    that outer-node occupation is small at the width actually used."""
    curve = _seasonal_curve()
    month_end = pd.Timestamp("2026-08-31")

    narrow = rf.build_lattice("2026-01-01", "2026-02-01", "2026-11-30",
                              vol=0.5, sMR=1.0, n_p=20, daily_curve=curve)
    wide = rf.build_lattice("2026-01-01", "2026-02-01", "2026-11-30",
                            vol=0.5, sMR=1.0, n_p=35, daily_curve=curve)

    u_n = narrow["date_span"].get_loc(month_end)
    u_w = wide["date_span"].get_loc(month_end)
    H_n = rf.project_month_end_quotes(narrow, [month_end])[u_n]
    H_w = rf.project_month_end_quotes(wide, [month_end])[u_w]

    assert float(H_n[0, 20]) == pytest.approx(float(H_w[0, 35]), rel=1e-6)

    # At n_p=12 (deliberately too narrow for sVol=0.5 over this ~8-month
    # projection) this same check would still pass -- the point of sec.5.1's
    # boundary check is that a tower/root identity is not sufficient on its
    # own, so this asserts occupancy directly rather than inferring it.
    q_n = narrow["q"]
    boundary_mass = q_n[u_n, 0] + q_n[u_n, -1]
    assert boundary_mass < 1e-2, f"outer nodes carry {boundary_mass:.2e} probability at n_p=20"


def test_multiple_delivery_months_are_independent_projections():
    """Projecting several month-ends at once must give the same per-month result
    as projecting them one at a time -- no cross-month state leakage."""
    curve = _seasonal_curve()
    lattice = rf.build_lattice("2026-01-01", "2026-02-01", "2026-11-30",
                               vol=0.5, sMR=1.0, n_p=12, daily_curve=curve)
    ends = [pd.Timestamp("2026-05-31"), pd.Timestamp("2026-08-31")]
    combined = rf.project_month_end_quotes(lattice, ends)
    for end in ends:
        u = lattice["date_span"].get_loc(end)
        solo = rf.project_month_end_quotes(lattice, [end])[u]
        assert np.allclose(combined[u], solo)
