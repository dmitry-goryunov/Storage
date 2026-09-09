"""Regression tests for the September 2026 reconciliation findings."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import storage_model as sm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _seasonal_daily_curve():
    days = pd.date_range("2026-01-01", "2027-12-31", freq="D")
    values = 25.0 + 5.0 * np.sin(np.arange(len(days)) * 2.0 * np.pi / 365.25)
    return pd.Series(values, index=days)


def _mandatory_put(n_p=20, run_intrinsic=False):
    params = {
        "product_type": "put_swing",
        "valDate": "2026-01-01",
        "storageStart": "2026-02-01",
        "storageEnd": "2026-04-30",
        "vol": 0.5,
        "sMR": 1.0,
        "n_p_full": n_p,
        "run_intrinsic": run_intrinsic,
        "daily_max": 1000.0,
        "clips_per_day": 1,
        "capacity_mwh": 10_000.0,
        "strike": 0.0,
        "daily_curve": _seasonal_daily_curve(),
    }
    return sm.run_valuation(None, params)


def _direct_put(n_p=20, days=10, discount_rate=0.0, injection_ratchet=1.0):
    model = sm.Storage(
        "2026-01-01",
        "2026-02-01",
        "2026-04-30",
        daily_curve=_seasonal_daily_curve(),
        n_p=n_p,
        v_step=1000.0,
        sVol=0.5,
        clips_per_day=1,
    )
    _, active = sm.active_masks(model)
    model.i_curve = active
    model.w_curve = np.zeros(len(model.date_span))
    model.set_volume_states(days)
    model.i_ratch[:] = injection_ratchet
    model.n_op_start = 0
    model.t_p_curve = np.full(model.n_op + 2, -1e9)
    model.t_p_curve[days] = 0.0
    model.d_curve = np.exp(-discount_rate * np.arange(model.n_t) / 365.25)
    return model.build()


def test_expected_exercise_is_not_rounded_before_aggregation():
    """The expected exercise schedule must exactly reprice the contract value."""
    for n_p in (0, 20):
        model, _ = _mandatory_put(n_p=n_p)
        value = float(model.v[0, model.n_p, model.n_op_start])
        repriced = float(np.dot(model.delta[:model.n_t], model.fwd))

        assert abs(repriced - value) / abs(value) < 1e-9
        assert abs(-sum(model.exp_ex) - 10_000.0) < 1e-9


def test_profiled_metrics_use_physical_expected_exercise_volume():
    """Per-MWh metrics use expected physical exercise, not hedge delta."""
    model, result = _mandatory_put()
    value = float(model.v[0, model.n_p, model.n_op_start])
    expected = value / float(sum(model.exp_ex))

    assert abs(sum(model.exp_ex) - sum(model.delta)) > 1e-3
    np.testing.assert_allclose(model.profiled(), expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result["stochastic_metric"], expected, rtol=0.0, atol=1e-12)


def test_time_varying_volatility_produces_valid_tree_probabilities():
    """A later volatility spike must not create negative transition mass."""
    n_t = 60
    n_p = 5
    forwards = np.full(n_t, 25.0)
    volatility = np.r_[0.1, np.full(n_t - 1, 1.5)]
    mean_reversion = np.ones(n_t)

    _, _, q, p_u, p_m, p_d = sm.build_tree(
        forwards, n_t, n_p, volatility, mean_reversion
    )

    for i in range(n_t):
        live = slice(max(n_p - i, 0), min(n_p + i, 2 * n_p) + 1)
        transitions = np.column_stack((p_u[i, live], p_m[i, live], p_d[i, live]))
        assert np.isfinite(transitions).all()
        assert transitions.min() >= -1e-12
        assert transitions.max() <= 1.0 + 1e-12
        np.testing.assert_allclose(transitions.sum(axis=1), 1.0, atol=1e-12)

    assert np.isfinite(q).all()
    assert q.min() >= -1e-12
    np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-12)


def test_flat_volatility_tree_is_bit_identical_to_legacy_spacing():
    n_t = 30
    n_p = 5
    forwards = np.linspace(25.0, 30.0, n_t)
    volatility = np.full(n_t, 0.5)
    mean_reversion = np.ones(n_t)
    new = sm.build_tree(forwards, n_t, n_p, volatility, mean_reversion)

    dt = 1.0 / 365.25
    dx = volatility[0] * np.sqrt(3.0 * dt)
    x = np.zeros((n_t, 2 * n_p + 1))
    p_u = np.zeros_like(x)
    p_m = np.zeros_like(x)
    p_d = np.zeros_like(x)
    q = sm._tree_core(
        x, p_u, p_m, p_d, forwards.copy(), volatility, mean_reversion,
        n_t, n_p, dx, dt,
    )
    legacy = (forwards, x, q, p_u, p_m, p_d)

    for actual, expected in zip(new, legacy):
        np.testing.assert_array_equal(actual, expected)


def test_stochastic_tree_rejects_zero_volatility():
    """A zero-width stochastic lattice is rejected with a useful error."""
    with np.testing.assert_raises_regex(ValueError, "positive volatility"):
        sm.build_tree(np.full(5, 25.0), 5, 2, np.zeros(5), np.ones(5))


def test_unstable_tree_configuration_raises_instead_of_returning_nans():
    n_t = 120
    with np.testing.assert_raises_regex(ValueError, "Invalid transition probabilities"):
        sm.build_tree(
            np.full(n_t, 25.0), n_t, 20,
            np.linspace(0.1, 1.2, n_t), np.ones(n_t),
        )


def test_post_build_feasibility_check_accounts_for_ratchets():
    """A nominally feasible terminal state can be unreachable under ratchets."""
    model = sm.Storage(
        "2026-01-01",
        "2026-01-02",
        "2026-01-10",
        daily_curve=_seasonal_daily_curve(),
        n_p=0,
        v_step=1000.0,
        sVol=0.5,
        clips_per_day=1,
    )
    _, active = sm.active_masks(model)
    model.i_curve = active
    model.w_curve = np.zeros(len(model.date_span))
    model.set_volume_states(2)
    model.apply_ratchets([0.0, 1.0], [0.0, 0.0], [1.0, 1.0])
    model.n_op_start = 0
    model.t_p_curve = np.full(model.n_op + 2, -1e9)
    model.t_p_curve[2] = 0.0

    with np.testing.assert_raises_regex(ValueError, "terminal inventory"):
        model.build()


def test_delta_is_an_undiscounted_hedge_volume():
    """`delta` is the forward MWh to trade, not a PV sensitivity (decision D-O2).

    Hedging day i with h forwards gives PV = h*DF_i*F_i*eps against
    dV/deps = DF_i*E[S_i*Q_i], so the discount factor cancels and h = E[S_i*Q_i]/F_i.
    The identity therefore carries the discount weights, and reduces to
    sum(delta*fwd) == V0 while d_curve is all ones.
    """
    model = _direct_put(discount_rate=0.08)
    value = float(model.v[0, model.n_p, model.n_op_start])
    delta = np.asarray(model.delta[:model.n_t])
    discounted = float(np.dot(model.d_curve[:model.n_t] * delta, model.fwd))
    assert abs(discounted - value) / abs(value) < 1e-9

    # ...and the reported number must carry no discount factor of its own. Checked
    # against a direct recomputation from the model's own policy, so it does not
    # depend on the policy: the DP maximises PV, so WHICH days it exercises does
    # legitimately shift with d_curve (~0.26 % of total delta at 8 %). What must not
    # happen is the reported figure being scaled by DF on top of that.
    action = model.strat[:model.n_t] * model.v_step
    pa = model.prob[:model.n_t] * action
    undiscounted = -(pa * np.exp(model.x)[:, :, None]).sum(axis=(1, 2)) / model.fwd
    assert np.allclose(delta, undiscounted, rtol=1e-12, atol=1e-9), (
        "delta is not the plain E[S*Q]/F hedge volume")
    scaled = undiscounted * model.d_curve[:model.n_t]
    assert not np.allclose(delta, scaled, rtol=1e-6), (
        "delta still looks scaled by the discount factor")


def test_multi_clip_ratchet_keeps_policy_probability_and_metrics_consistent():
    model = _direct_put(days=10, injection_ratchet=2.0)
    assert abs(-sum(model.exp_ex) - 10_000.0) < 1e-9
    value = float(model.v[0, model.n_p, model.n_op_start])
    repriced = float(np.dot(model.delta[:model.n_t], model.fwd))
    assert abs(repriced - value) / abs(value) < 1e-9


def test_tree_reprices_forward_curve_at_every_time_step():
    n_t = 60
    n_p = 10
    forwards = 25.0 + 3.0 * np.sin(np.arange(n_t) / 9.0)
    fwd, x, q, *_ = sm.build_tree(
        forwards, n_t, n_p, np.full(n_t, 0.5), np.ones(n_t)
    )
    expected_spot = np.sum(q * np.exp(x), axis=1)
    np.testing.assert_allclose(expected_spot, fwd, rtol=1e-12, atol=1e-12)


def test_smoothed_curve_reprices_each_monthly_contract():
    stepped = _seasonal_daily_curve()
    smoothed = sm.smoothen_curve(stepped)
    expected = stepped.resample("ME").mean()
    actual = smoothed.resample("ME").mean()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_monthly_delta_matches_independent_finite_difference():
    """Local monthly deltas agree with symmetric 5 bp price bumps."""
    base = _direct_put(n_p=20)
    dates = pd.DatetimeIndex(base.date_span[:base.n_t])
    delta = pd.Series(base.delta[:base.n_t], index=dates)
    fwd = pd.Series(base.fwd, index=dates)

    for month in sorted(set(dates.to_period("M"))):
        mask = dates.to_period("M") == month
        analytic = float((delta[mask] * fwd[mask]).sum())

        def bumped_value(epsilon):
            model = _direct_put(n_p=20)
            bumped = model.price_curve.copy()
            bumped.loc[bumped.index.to_period("M") == month] *= 1.0 + epsilon
            model.price_curve = bumped
            model.build()
            return float(model.v[0, model.n_p, model.n_op_start])

        epsilon = 0.00005
        finite_difference = (bumped_value(epsilon) - bumped_value(-epsilon)) / (2.0 * epsilon)
        np.testing.assert_allclose(finite_difference, analytic, rtol=0.01, atol=5.0)


@pytest.mark.parametrize("app_path", ["streamlit_app.py", "portfolio_app.py"])
def test_streamlit_app_starts_without_exceptions(app_path):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(os.path.join(ROOT, app_path), default_timeout=60).run()
    assert not app.exception


@pytest.mark.parametrize(
    "notebook",
    ["Swing_new.ipynb", "forward.ipynb", "portfolio.ipynb", "pricing.ipynb"],
)
def test_notebook_is_valid_json(notebook):
    with open(os.path.join(ROOT, notebook), encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload.get("cells"), list)
    assert payload.get("nbformat") == 4


def test_contract_curve_gap_reports_the_coverage_problem():
    """A contract curve that stops short of the backstop must say so.

    The coverage guard in Storage.__init__ carries a message naming the missing
    days and the backstop date, but it runs *after* smoothen_curve, which raises
    SciPy's "`y` must contain only finite values" first. On the contract-curve
    path -- the one the guard was written for -- the useful message was
    unreachable.
    """
    starts = pd.date_range("2026-01-01", "2026-03-01", freq="MS")
    short = pd.DataFrame({
        "contractStart": starts,
        "contractEnd": starts + pd.offsets.MonthEnd(0),
        "value": 25.0,
    })
    with np.testing.assert_raises_regex(ValueError, "missing day"):
        sm.Storage("2026-01-01", "2026-02-01", "2026-04-30", curve=short,
                   n_p=0, v_step=1000.0, sVol=0.5)


def test_storage_rejects_capacity_expressed_in_days():
    """run_valuation reads inj_rate/wdr_rate, never inj_days/wdr_days.

    A caller who describes storage capacity in days -- the natural way, and the
    way the product workbook does it -- used to have those inputs silently
    discarded and get the default symmetric clip rate instead. On a 2026-2027
    deal, wdr_days of 30, 45, 90 and 365 all priced at 2.499672 EUR/MWh while
    the rate itself moves the value from 2.449229 (1 clip/day) to 2.514549 (10).
    """
    params = dict(product_type="storage", valDate="2026-01-01",
                  storageStart="2026-04-01", storageEnd="2027-03-31", days=30,
                  vol=0.6, n_p_full=3, run_intrinsic=False, v_step=1000,
                  inj_days=30, wdr_days=90, inj_cost=0.5, wdr_cost=0.5,
                  daily_curve=_seasonal_daily_curve())
    with np.testing.assert_raises_regex(ValueError, "wdr_rate"):
        sm.run_valuation(None, params)


def _grid_model(**kw):
    return sm.Storage("2026-01-01", "2026-02-01", "2026-04-30",
                      daily_curve=_seasonal_daily_curve(), n_p=0, v_step=1000.0,
                      sVol=0.5, clips_per_day=1, **kw)


def test_grid_size_and_initial_state_are_separate_names():
    """`n_op_start` meant the grid size to set_volume_states and the initial
    inventory state to build(), so every caller had to set it twice. The two
    meanings now have their own names, and both can be set in one call."""
    m = _grid_model()
    m.set_volume_states(10, initial_state=3)
    assert m.n_states == 10 and m.n_op == 11
    assert m.initial_state == 3
    assert m.n_op_start == 3        # compatibility alias reads the initial state

    m.n_op_start = 4                # ...and writing it still sets the initial state
    assert m.initial_state == 4

    # Grid size unchanged by touching the initial state
    assert m.n_states == 10 and m.n_op == 11


def test_initial_state_outside_the_grid_is_rejected():
    """Nothing checked this: an out-of-range start indexed past the value array."""
    m = _grid_model()
    m.set_volume_states(10)
    m.initial_state = 11            # grid holds states 0..10
    m.t_p_curve = np.full(m.n_op + 2, -1e9)
    m.t_p_curve[0] = 0.0
    with np.testing.assert_raises_regex(ValueError, "initial_state"):
        m.build()


def test_set_volume_states_defaults_the_start_to_a_full_grid():
    """Unchanged behaviour: with no initial_state, the start is the full grid."""
    m = _grid_model()
    m.set_volume_states(7)
    assert m.initial_state == 7 and m.n_states == 7


def _struck(product_type, strike, run_intrinsic=True):
    return sm.run_valuation(None, dict(
        product_type=product_type, valDate="2026-01-01",
        storageStart="2026-02-01", storageEnd="2026-04-30",
        capacity_mwh=10_000, daily_max=1_000, clips_per_day=1,
        vol=0.5, sMR=1.0, n_p_full=10, run_intrinsic=run_intrinsic,
        strike=strike, daily_curve=_seasonal_daily_curve()))[1]


def _struck_call(strike, run_intrinsic=True):
    return _struck("call_swing", strike, run_intrinsic)


@pytest.mark.parametrize("product_type", ["call_swing", "put_swing"])
def test_struck_swing_intrinsic_is_benchmarked_net_of_the_strike(product_type):
    """`intrinsic` compared a strike-net value against a raw forward average.

    A mandatory swing takes the N best days whatever the strike -- a constant
    per-MWh amount cannot reorder them -- so the intrinsic spread it captures is
    the same for every K. It used to shift by K instead, in opposite directions:
    the call gave 0.314, -9.686, -27.686 for K = 0, 10, 28, and the put gave
    0.157, 10.157, 20.157 for K = 0, 10, 20.
    """
    base = _struck(product_type, 0.0)
    for K in (10.0, 20.0):
        r = _struck(product_type, K)
        assert abs(r["intrinsic"] - base["intrinsic"]) < 1e-9, (
            f"K={K}: intrinsic {r['intrinsic']:.4f} vs {base['intrinsic']:.4f} at K=0")
        assert abs(r["extrinsic"] - base["extrinsic"]) < 1e-9, "extrinsic was never affected"
        # The benchmark must move with the strike, since the payoff does.
        assert abs(r["flat_metric"] - (base["flat_metric"] - K)) < 1e-9
        # ...and the decomposition must still add up on the reported figures.
        spread = (r["profiled_metric"] - r["flat_metric"]) if product_type == "call_swing"             else (r["flat_metric"] - r["profiled_metric"])
        assert abs(spread - r["intrinsic"]) < 1e-9
        assert abs((r["intrinsic"] + r["extrinsic"]) - r["total"]) < 1e-9


# ── Time value of money ───────────────────────────────────────────────────────
# Cash from an early withdrawal can be redeployed, so the model must prefer
# earlier exercise when a discount rate is supplied. The machinery existed
# (`d_curve` is applied to every cash flow in the DP) but nothing ever set it.

def _flat_daily_curve(price=25.0):
    days = pd.date_range("2026-01-01", "2027-06-30", freq="D")
    return pd.Series(float(price), index=days)


def _timed_call(discount_rate=0.0, **kw):
    params = dict(product_type="call_swing", valDate="2026-01-01",
                  storageStart="2026-02-01", storageEnd="2026-12-31",
                  capacity_mwh=30_000, daily_max=1_000, clips_per_day=1,
                  vol=0.5, sMR=1.0, n_p_full=0, run_intrinsic=True,
                  discount_rate=discount_rate, daily_curve=_flat_daily_curve())
    params.update(kw)
    return sm.run_valuation(None, params)


def _mean_exercise_day(model):
    ex = np.abs(np.asarray(model.exp_ex[:model.n_t]))
    return float(np.dot(np.arange(model.n_t), ex) / ex.sum())


def test_discount_rate_pulls_exercise_earlier():
    """On a flat curve, timing is indifferent at 0 % and worth something at 10 %."""
    flat, _ = _timed_call(0.0)
    disc, _ = _timed_call(0.10)
    assert _mean_exercise_day(disc) < _mean_exercise_day(flat) - 1.0, (
        f"mean exercise day {_mean_exercise_day(disc):.1f} at 10 % vs "
        f"{_mean_exercise_day(flat):.1f} at 0 % — the rate did not move the schedule")


def test_flat_curve_intrinsic_is_zero_without_a_rate_and_positive_with_one():
    """With no price shape, the only thing left to optimise is WHEN you sell.

    At 0 % that is worth nothing, so intrinsic must be 0. At 10 % selling early
    beats selling evenly, and that timing gain is exactly what intrinsic should
    now report -- it used to be invisible.
    """
    _, flat = _timed_call(0.0)
    assert abs(flat["intrinsic"]) < 1e-9, f"flat curve, no rate: {flat['intrinsic']}"

    _, disc = _timed_call(0.10)
    assert disc["intrinsic"] > 0.05, f"10 % rate on a flat curve gave {disc['intrinsic']}"


def test_discounted_benchmark_keeps_the_decomposition_comparable():
    """`flat_metric` must be PV'd too, or intrinsic just measures the discount."""
    s, r = _timed_call(0.10)
    df = np.exp(-0.10 * np.arange(s.n_t) / 365.25)
    win = slice(s.Dt, s._active)
    expected = float(np.mean(df[win] * s.fwd[win]))
    assert abs(r["flat_metric"] - expected) < 1e-9, (
        f"flat_metric {r['flat_metric']:.4f} is not the PV-weighted average forward "
        f"{expected:.4f}")
    assert abs((r["profiled_metric"] - r["flat_metric"]) - r["intrinsic"]) < 1e-9
    assert abs((r["intrinsic"] + r["extrinsic"]) - r["total"]) < 1e-9


def test_discounting_lowers_the_value_of_a_positive_deal():
    a, _ = _timed_call(0.0)
    b, _ = _timed_call(0.10)
    va = float(a.v[0, a.n_p, a.initial_state]); vb = float(b.v[0, b.n_p, b.initial_state])
    assert 0 < vb < va, f"value {vb:,.0f} at 10 % should sit below {va:,.0f} at 0 %"


def test_storage_starts_where_the_reported_value_says_it_does():
    """`value_storage` reads v[0, ., init_inv] but must also START the forward pass there.

    A refactor dropped the assignment, so the reported value was for a store
    starting empty while exp_ex/delta/prob described one starting full: it
    withdrew 66,041 MWh having injected 6,041, emptying a store it never filled.
    An empty-to-empty deal must move the same volume in as out.
    """
    params = dict(product_type="storage", valDate="2026-01-01",
                  storageStart="2026-04-01", storageEnd="2027-03-31",
                  capacity_mwh=60_000, daily_max=1_000, clips_per_day=1,
                  vol=0.5, sMR=1.0, n_p_full=3, run_intrinsic=False,
                  inj_cost=0.5, wdr_cost=0.5, daily_curve=_seasonal_daily_curve())
    s, _ = sm.run_valuation(None, params)
    assert s.initial_state == 0, (
        f"initial_state={s.initial_state}, but the value is read at init_inv=0")

    moved = s.prob[:s.n_t] * s.strat[:s.n_t] * s.v_step
    injected = float(np.clip(moved, 0, None).sum())
    withdrawn = float(-np.clip(moved, None, 0).sum())
    assert abs(injected - withdrawn) < 1e-6, (
        f"empty-to-empty storage injected {injected:,.0f} and withdrew {withdrawn:,.0f}")


def _rising_daily_curve(lo=22.0, hi=30.0):
    days = pd.date_range("2026-01-01", "2027-06-30", freq="D")
    return pd.Series(np.linspace(lo, hi, len(days)), index=days)


def _timed(product_type, discount_rate, curve, **kw):
    params = dict(product_type=product_type, valDate="2026-01-01",
                  storageStart="2026-02-01", storageEnd="2026-12-31",
                  capacity_mwh=30_000, daily_max=1_000, clips_per_day=1,
                  vol=0.5, sMR=1.0, n_p_full=0, run_intrinsic=True,
                  discount_rate=discount_rate, daily_curve=curve)
    params.update(kw)
    s, r = sm.run_valuation(None, params)
    ex = np.abs(np.asarray(s.exp_ex[:s.n_t]))
    return s, r, float(np.dot(np.arange(s.n_t), ex) / ex.sum())


def test_a_put_swing_defers_where_a_call_swing_accelerates():
    """The buy side is the mirror: paying later is the gain, not receiving sooner.

    On a rising curve a buyer wants the cheap early days and a seller the dear
    late ones, so price and time value pull against each other. Raise the rate
    far enough and each flips to the other end of the window -- in opposite
    directions.
    """
    curve = _rising_daily_curve()
    _, _, put_cheap = _timed("put_swing", 0.0, curve)
    _, _, put_dear = _timed("put_swing", 0.40, curve)
    _, _, call_cheap = _timed("call_swing", 0.0, curve)
    _, _, call_dear = _timed("call_swing", 0.40, curve)

    assert put_cheap < 100 and put_dear > 300, (
        f"put should buy early at 0 % ({put_cheap:.0f}) and defer at 40 % ({put_dear:.0f})")
    assert call_cheap > 300 and call_dear < 100, (
        f"call should sell late at 0 % ({call_cheap:.0f}) and accelerate at 40 % "
        f"({call_dear:.0f})")
    assert (put_dear - put_cheap) * (call_dear - call_cheap) < 0, (
        "the two sides must move in opposite directions")


def test_both_sides_book_a_timing_gain_on_a_flat_curve():
    """With no price shape, timing is the only edge, and both sides have one."""
    curve = _flat_daily_curve()
    for product_type in ("put_swing", "call_swing"):
        _, flat, _ = _timed(product_type, 0.0, curve)
        _, disc, _ = _timed(product_type, 0.10, curve)
        assert abs(flat["intrinsic"]) < 1e-9, f"{product_type} at 0 %: {flat['intrinsic']}"
        assert disc["intrinsic"] > 0.5, f"{product_type} at 10 %: {disc['intrinsic']}"


def test_delta_pv_is_the_tailed_hedge_and_reprices_directly():
    """Two hedge ratios, because there are two hedge instruments.

    `delta` is the physical forward volume: correct against an OTC forward that
    settles with the deal, where the discount factor cancels. `delta_pv` is that
    tailed by DF: correct against margined futures, whose variation margin moves
    today while the gas settles later, and the right number for PV risk.

    On the 10-day put swing 2027 at 10 %, tailing takes December from -2,481 to
    -2,036 MWh, and the book from -9,213 to -7,819.
    """
    model = _direct_put(discount_rate=0.10)
    n = model.n_t
    delta = np.asarray(model.delta[:n])
    delta_pv = np.asarray(model.delta_pv[:n])

    np.testing.assert_allclose(delta_pv, delta * model.d_curve[:n], rtol=0, atol=1e-12)
    assert abs(delta_pv).sum() < abs(delta).sum(), "tailing must shrink a forward-dated book"

    # The tailed series reprices the contract without carrying the weights
    # separately -- the identity in its simplest form.
    value = float(model.v[0, model.n_p, model.initial_state])
    assert abs(float(np.dot(delta_pv, model.fwd)) - value) / abs(value) < 1e-9

    # With no rate the two series coincide.
    plain = _direct_put(discount_rate=0.0)
    np.testing.assert_allclose(np.asarray(plain.delta_pv[:n]),
                               np.asarray(plain.delta[:n]), rtol=0, atol=1e-12)


def _quote_row(quote_date="2026-03-06", da=52.0, n=12, level=50.0):
    """A single synthetic quote row, shaped like a row of `ttf q.xlsx`."""
    cols = [f"TTFc{i + 1}" for i in range(n)]
    row = {"quote_date": pd.Timestamp(quote_date), "DA": da}
    row.update({c: level - i for i, c in enumerate(cols)})
    return pd.Series(row), cols


def test_a_curve_cannot_start_before_the_quote_it_is_built_from():
    """Valuing before the quote date is look-ahead, and it back-fills silently.

    `curve_start` and the quote date were independent inputs, so moving the
    as-of date forward while leaving the valuation date behind produced a curve
    whose front stub was the *later* quote's day-ahead price stamped flat over
    the months in between -- two months of 52.00 across Jan and Feb 2026 for a
    2026-03-06 quote valued from 2026-01-01. It now raises.
    """
    row, cols = _quote_row("2026-03-06", da=52.0)

    with pytest.raises(ValueError, match="before the quote"):
        sm.curve_df_for_storage(row, cols, curve_start="2026-01-01", include_da=True)

    # The same gap without a DA column is equally look-ahead, and equally rejected.
    row_no_da = row.drop(labels=["DA"])
    with pytest.raises(ValueError, match="before the quote"):
        sm.curve_df_for_storage(row_no_da, cols, curve_start="2026-01-01", include_da=True)

    # On or after the quote date is fine, and the stub starts where it is told.
    same = sm.curve_df_for_storage(row, cols, curve_start="2026-03-06", include_da=True)
    assert same["contractStart"].min() == pd.Timestamp("2026-03-06")
    assert float(same.iloc[0]["value"]) == 52.0

    # Valuing *after* the quote is allowed. The day-ahead stub still reaches back
    # to the quote date -- harmless, because the model only maps the curve from
    # the valuation date forward, so those days are never read.
    later = sm.curve_df_for_storage(row, cols, curve_start="2026-03-20", include_da=True)
    assert later["contractStart"].min() == pd.Timestamp("2026-03-06")

    # Defaulting curve_start to the quote date must not trip its own guard.
    default = sm.curve_df_for_storage(row, cols, curve_start=None, include_da=True)
    assert default["contractStart"].min() == pd.Timestamp("2026-03-06")


def test_the_as_of_date_moves_the_curve():
    """The knob that started this: a different as-of must select a different quote.

    `Products.ipynb` had `AS_OF` read only on the `quotes` branch while the
    source stayed `csv`, so changing the date left the curve on curve.csv's
    stored March 2026 contract of 28.00. The library half is asserted here; the
    notebook half is the source/AS_OF guard in section 1.
    """
    quotes = pd.DataFrame([
        {"quote_date": pd.Timestamp("2026-01-05"), "DA": 20.0, "TTFc1": 21.0, "TTFc2": 22.0},
        {"quote_date": pd.Timestamp("2026-03-06"), "DA": 52.0, "TTFc1": 53.0, "TTFc2": 54.0},
    ])
    cols = ["TTFc1", "TTFc2"]

    early = sm.quote_row_for_fd_date(quotes, cols, "2026-01-31")
    late = sm.quote_row_for_fd_date(quotes, cols, "2026-03-06")
    assert pd.Timestamp(early["quote_date"]) == pd.Timestamp("2026-01-05")
    assert pd.Timestamp(late["quote_date"]) == pd.Timestamp("2026-03-06")

    front_early = float(sm.curve_df_for_storage(early, cols, include_da=False)
                        .sort_values("contractStart").iloc[0]["value"])
    front_late = float(sm.curve_df_for_storage(late, cols, include_da=False)
                       .sort_values("contractStart").iloc[0]["value"])
    assert front_early == 21.0
    assert front_late == 53.0


def test_with_no_volatility_every_euro_of_value_is_financing():
    """The sharpest flat-curve sense check: kill the vol and only timing is left.

    A flat curve removes day-selection, so intrinsic can only be the discount
    rate. Removing the volatility as well removes optionality, so extrinsic must
    vanish outright and the buyer must defer to the very end of the window --
    with no dip left to wait for, when you pay is the only thing to optimise.
    With vol switched back on the schedule sits in between, balancing the
    financing pull against the option to catch a dip.
    """
    curve = _flat_daily_curve(40.0)
    quiet, res_quiet, day_quiet = _timed("put_swing", 0.10, curve, n_p_full=20)

    assert abs(res_quiet["extrinsic"]) > 1e-9, "sanity: vol 50 % must carry optionality"

    still, res_still, day_still = _timed("put_swing", 0.10, curve, vol=1e-6, n_p_full=20)
    assert abs(res_still["extrinsic"]) < 1e-9, (
        f"no vol must leave no optionality, got {res_still['extrinsic']:.3e}")
    assert res_still["intrinsic"] > 0.1, (
        f"the rate alone must still be worth something, got {res_still['intrinsic']:.4f}")

    # Deferred to the back of the window, and further than when vol competes.
    assert day_still > still.Dt + 0.95 * (still._active - still.Dt), (
        f"mean exercise day {day_still:.1f} is not at the end of "
        f"[{still.Dt}, {still._active})")
    assert day_still > day_quiet + 10.0, (
        f"no-vol schedule {day_still:.1f} should sit later than the vol one "
        f"{day_quiet:.1f} -- optionality is what pulls exercise forward")

    # And with neither vol nor rate there is nothing to gain at all.
    _, res_none, _ = _timed("put_swing", 0.0, curve, vol=1e-6, n_p_full=20)
    assert abs(res_none["total"]) < 1e-4, (
        f"no shape, no vol, no rate must be worth nothing, got {res_none['total']:.3e}")


def _deterministic(curve, discount_rate, **kw):
    """The n_p = 0 run whose schedule produces `profiled_metric`."""
    params = dict(product_type="put_swing", valDate="2026-01-01",
                  storageStart="2026-02-01", storageEnd="2026-12-31",
                  capacity_mwh=30_000, daily_max=1_000, clips_per_day=1,
                  vol=0.5, sMR=1.0, n_p_full=0, run_intrinsic=False,
                  discount_rate=discount_rate, daily_curve=curve)
    params.update(kw)
    return sm.run_valuation(None, params)[0]


def test_on_a_flat_curve_every_euro_of_intrinsic_is_financing():
    """Why a shapeless curve still shows intrinsic once a rate is on.

    A flat *forward* curve is not flat once discounted. The optimiser sees
    `DF * F`, which at 10 % slopes downwards across the window, so there is
    something to choose even with no price shape at all -- and the day-selection
    term is exactly zero while the financing term carries the whole of it.

    The two terms must also reconstruct `intrinsic` on a curve that does have
    shape, where both are non-zero.
    """
    flat = _flat_daily_curve(40.0)

    shape, financing = sm.intrinsic_components(_deterministic(flat, 0.10))
    assert abs(shape) < 1e-9, f"a flat curve has no day-selection gain, got {shape:.3e}"
    assert financing > 0.1, f"the rate must carry all of it, got {financing:.4f}"

    # It reconstructs what run_valuation reports.
    _, res, _ = _timed("put_swing", 0.10, flat, n_p_full=20)
    assert abs(shape + financing - res["intrinsic"]) < 1e-9, (
        f"{shape:.6f} + {financing:.6f} != reported {res['intrinsic']:.6f}")

    # With no rate there is nothing to choose at all, on either leg.
    zero = sm.intrinsic_components(_deterministic(flat, 0.0))
    assert max(abs(z) for z in zero) < 1e-9, zero

    # On a sloped curve both terms are live and still add up.
    sloped = _seasonal_daily_curve()
    for rate in (0.0, 0.10):
        det = _deterministic(sloped, rate, valDate="2026-01-01",
                             storageStart="2026-02-01", storageEnd="2026-12-31")
        shape, financing = sm.intrinsic_components(det)
        _, res, _ = _timed("put_swing", rate, sloped, n_p_full=20)
        assert shape > 0.1, f"a seasonal curve must offer day selection, got {shape:.4f}"
        assert abs(shape + financing - res["intrinsic"]) < 1e-9, (
            f"rate {rate}: {shape:.6f} + {financing:.6f} != {res['intrinsic']:.6f}")
        if rate == 0.0:
            assert abs(financing) < 1e-9, f"no rate, no financing gain: {financing:.3e}"
        else:
            assert financing > 0.0, f"a buyer gains by deferring: {financing:.4f}"


def test_the_intrinsic_split_refuses_a_two_sided_deal():
    """Storage buys and sells, so value per net MWh -- and the split -- is undefined."""
    params = {
        "product_type": "storage", "valDate": "2026-01-01",
        "storageStart": "2026-02-01", "storageEnd": "2026-12-31",
        "vol": 0.5, "sMR": 1.0, "n_p_full": 0, "run_intrinsic": False,
        "discount_rate": 0.10, "daily_curve": _seasonal_daily_curve(),
        "capacity_mwh": 30_000.0, "daily_max": 1_000.0, "clips_per_day": 1,
        "inj_cost": 0.0, "wdr_cost": 0.0,
    }
    model, _ = sm.run_valuation(None, params)
    with pytest.raises(ValueError, match="zero-net-volume"):
        sm.intrinsic_components(model)


def test_the_intrinsic_split_is_net_of_the_strike():
    """Both legs the split divides out are net of the strike, so it must be passed.

    `profiled_metric` is the effective price after the strike leg and
    `flat_metric` is the forward average net of it, so the discount factors
    recovered from them are only right if `intrinsic_components` is given the
    same strike the run used. Getting it wrong is silent -- the numbers still
    look like prices -- so it is asserted both ways.
    """
    curve = _seasonal_daily_curve()
    strike = 20.0

    det = _deterministic(curve, 0.10, strike=strike)
    _, res, _ = _timed("put_swing", 0.10, curve, n_p_full=20, strike=strike)

    shape, financing = sm.intrinsic_components(det, strike=strike)
    assert abs(shape + financing - res["intrinsic"]) < 1e-9, (
        f"{shape:.6f} + {financing:.6f} != reported {res['intrinsic']:.6f}")
    assert shape > 0.1, shape

    # Forgetting the strike does not raise. It answers a different question, and
    # the giveaway is that the two terms stop adding up to the reported intrinsic.
    wrong = sm.intrinsic_components(det)
    assert abs(sum(wrong) - res["intrinsic"]) > 0.5, (
        f"dropping the strike should break the reconciliation: {sum(wrong):.6f} "
        f"against {res['intrinsic']:.6f}")


def test_a_strike_reorders_the_days_only_once_there_is_a_rate():
    """A constant per-MWh amount is not neutral once cash flows are discounted.

    Undiscounted, the cost is `sum (P_i - K) q_i` and the `K` leg is a constant
    times a fixed volume, so it cannot reorder anything: schedule and split are
    identical struck or not. Discounted it becomes `sum DF_i (P_i - K) q_i`, and
    the `-K sum DF_i q_i` term rewards days with *large* discount factors -- it
    pulls exercise earlier, against the deferral the rate otherwise buys.

    The financing gain scales with the net cash actually moving, not the gross
    index, so a deep strike all but removes it: on this curve a 20.00 strike
    against a ~25 average takes financing from 0.481 to 0.002 and returns the
    schedule to its undiscounted optimum.
    """
    curve = _seasonal_daily_curve()

    flat_plain = sm.intrinsic_components(_deterministic(curve, 0.0))
    flat_struck = sm.intrinsic_components(_deterministic(curve, 0.0, strike=20.0), strike=20.0)
    np.testing.assert_allclose(flat_struck, flat_plain, rtol=0, atol=1e-9,
                               err_msg="with no rate a strike must be neutral")

    plain = _deterministic(curve, 0.10)
    struck = _deterministic(curve, 0.10, strike=20.0)
    _, fin_plain = sm.intrinsic_components(plain)
    _, fin_struck = sm.intrinsic_components(struck, strike=20.0)
    assert fin_plain > 0.4, fin_plain
    assert fin_struck < 0.05, (
        f"a deep strike leaves almost no cash to defer, got {fin_struck:.6f}")

    assert _mean_exercise_day(struck) < _mean_exercise_day(plain) - 1.0, (
        f"the strike must pull exercise earlier: {_mean_exercise_day(struck):.1f} "
        f"vs {_mean_exercise_day(plain):.1f}")

    # Extrinsic follows the same rule, and it is the cleaner statement of it:
    # optionality is worth the same struck or not while nothing discounts, and
    # stops being so the moment something does.
    def extrinsic(rate, strike):
        _, res, _ = _timed("put_swing", rate, curve, n_p_full=20, strike=strike)
        return res["extrinsic"]

    assert abs(extrinsic(0.0, 20.0) - extrinsic(0.0, 0.0)) < 1e-9, (
        f"with no rate a strike must not touch optionality: "
        f"{extrinsic(0.0, 20.0):.6f} vs {extrinsic(0.0, 0.0):.6f}")
    assert abs(extrinsic(0.10, 20.0) - extrinsic(0.10, 0.0)) > 1e-3, (
        f"with a rate it must, through the -K*sum(DF*q) leg: "
        f"{extrinsic(0.10, 20.0):.6f} vs {extrinsic(0.10, 0.0):.6f}")


def test_a_strike_on_the_curve_zeroes_intrinsic_and_defeats_the_split():
    """K on a flat curve: intrinsic is exactly 0 at any rate, and the split is lost.

    With `F - K == 0` on every day the deterministic cash flow is zero whatever
    the schedule, so `sum DF_i (P_i - K) q_i` is zero for all of them: the rate
    has no lever and no financing gain can exist. `intrinsic` is well defined
    and exactly 0.

    The *split* is a different matter. It recovers discount factors by dividing
    by the benchmark and by the price the schedule pays, both of which are zero
    here, so `intrinsic_components` raises rather than returning 0/0. Callers
    that want a table rather than an exception catch it -- `Products.ipynb`
    reports n/a.
    """
    curve = _flat_daily_curve(40.0)
    for rate in (0.0, 0.10):
        _, res, _ = _timed("put_swing", rate, curve, n_p_full=20, strike=40.0)
        assert abs(res["flat_metric"]) < 1e-12, res["flat_metric"]
        assert abs(res["intrinsic"]) < 1e-9, (
            f"a strike on the curve leaves no intrinsic at {rate:.0%}: {res['intrinsic']:.3e}")
        assert res["extrinsic"] > 0.5, res["extrinsic"]

        det = _deterministic(curve, rate, strike=40.0)
        with pytest.raises(ValueError, match="strike sits on the curve"):
            sm.intrinsic_components(det, strike=40.0)


def test_financing_scales_with_the_net_cash_not_the_index():
    """The financing gain is linear in `(level - K) / level` on a flat curve.

    Deferring is worth a fraction of what actually moves. On a flat curve at
    `level` the schedule pays `level - K` per MWh whatever days it picks, so
    the whole of intrinsic -- which is financing there, day selection being
    zero -- must scale exactly with the net. It does, to machine precision:
    at 40.00 a K of 30.00 gives a quarter of the unstruck gain, and a K of
    39.00 gives a fortieth. This is why a deep strike all but removes the
    rate's effect, and why the schedule reverts to its undiscounted optimum.
    """
    level = 40.0
    curve = _flat_daily_curve(level)

    def financing(strike):
        det = _deterministic(curve, 0.10, strike=strike)
        shape, fin = sm.intrinsic_components(det, strike=strike)
        assert abs(shape) < 1e-9, f"K={strike}: a flat curve has no day selection, {shape:.3e}"
        return fin

    base = financing(0.0)
    assert base > 0.1, base
    for strike in (10.0, 20.0, 30.0, 35.0, 39.0):
        expected = (level - strike) / level * base
        assert abs(financing(strike) - expected) < 1e-12 * max(abs(expected), 1.0), (
            f"K={strike}: {financing(strike):.9f} != {expected:.9f}")


def test_the_split_reports_an_unsigned_zero():
    """`-0.0` in a results table reads like a defect. It is not one; nor is it wanted.

    On a flat curve the day-selection term is `sign * df_bench * 0.0`, and for a
    buyer `sign` is -1, so the raw product is negative zero and formats as
    "-0.000". Both components are normalised.
    """
    shape, financing = sm.intrinsic_components(_deterministic(_flat_daily_curve(40.0), 0.10))
    assert shape == 0.0
    assert not np.signbit(shape), "day selection came back as -0.0"
    assert f"{shape:.3f}" == "0.000", f"{shape:.3f}"
    assert not np.signbit(financing) and financing > 0.0, financing

    # And with no rate at all, where both terms are zero.
    for term in sm.intrinsic_components(_deterministic(_flat_daily_curve(40.0), 0.0)):
        assert term == 0.0 and not np.signbit(term), term


def test_a_put_swing_is_an_obligation_not_an_option_on_the_strike():
    """Value is linear in K with no kink, so `intrinsic` cannot mean moneyness.

    An option's value is convex in the strike with a kink at the money, and its
    intrinsic value is `max(.,0)` of the moneyness. A `put_swing` here is the
    obligation to buy: total volume is fixed, so at a zero rate the value is
    `K * volume - sum P_i q_i` and `dV/dK` is exactly the volume at every strike,
    in or out of the money.

    Discounting adds a little real convexity -- the slope becomes `sum DF_i q_i`,
    which the schedule can raise by exercising earlier as K grows. It is the
    schedule responding, not an option payoff.
    """
    curve = _flat_daily_curve(40.0)
    strikes = [0.0, 15.0, 30.0, 45.0, 60.0]

    def value(strike, rate):
        model, _ = sm.run_valuation(None, dict(
            product_type="put_swing", valDate="2026-01-01", storageStart="2026-02-01",
            storageEnd="2026-12-31", capacity_mwh=30_000, daily_max=1_000,
            clips_per_day=1, vol=0.5, sMR=1.0, n_p_full=20, run_intrinsic=False,
            discount_rate=rate, strike=strike, daily_curve=curve))
        return float(model.v[0, model.n_p, model.initial_state])

    flat_slopes = [(value(b, 0.0) - value(a, 0.0)) / (b - a)
                   for a, b in zip(strikes, strikes[1:])]
    for s in flat_slopes:
        assert abs(s - 30_000.0) < 1e-3, (
            f"dV/dK must be the fixed volume at every strike, got {s:,.3f}")

    disc_slopes = [(value(b, 0.10) - value(a, 0.10)) / (b - a)
                   for a, b in zip(strikes, strikes[1:])]
    assert all(s < 30_000.0 for s in disc_slopes), disc_slopes
    assert all(b > a for a, b in zip(disc_slopes, disc_slopes[1:])), (
        f"discounting should make the value convex in K, got {disc_slopes}")
    assert disc_slopes[-1] / disc_slopes[0] - 1 < 0.05, (
        f"but only slightly -- {disc_slopes[-1]/disc_slopes[0]-1:.1%} is too much")


def test_the_reported_metrics_compose_into_the_price():
    """The four reported numbers nest; they are not terms to add side by side.

        flat  =  price  +/-  (intrinsic + extrinsic),   intrinsic = shape + financing

    `flat` is the benchmark and `price` is what the deal actually pays, so the
    gain between them must be exactly what the decomposition claims. The two are
    computed by different routes -- one off the DP's value, one off the split --
    so their agreeing ties the reported metrics to the prices.
    """
    curve = _flat_daily_curve(40.0)
    for product, strike, rate in (("put_swing", 30.0, 0.10), ("put_swing", 0.0, 0.10),
                                  ("call_swing", 30.0, 0.10), ("put_swing", 30.0, 0.0)):
        _, res, _ = _timed(product, rate, curve, n_p_full=20, strike=strike)
        flat, price = res["flat_metric"], res["stochastic_metric"]
        gain = (flat - price) if product == "put_swing" else (price - flat)
        assert abs(gain - (res["intrinsic"] + res["extrinsic"])) < 1e-9, (
            f"{product} K={strike} r={rate}: gain {gain:.9f} != "
            f"{res['intrinsic']:.9f} + {res['extrinsic']:.9f}")
        assert abs(res["total"] - (res["intrinsic"] + res["extrinsic"])) < 1e-12

        det = _deterministic(curve, rate, product_type=product, strike=strike)
        shape, financing = sm.intrinsic_components(det, strike=strike)
        assert abs(shape + financing - res["intrinsic"]) < 1e-9, (
            f"{product} K={strike} r={rate}: {shape:.9f} + {financing:.9f} != "
            f"{res['intrinsic']:.9f}")


def test_financing_is_interest_on_cash_the_deal_has_not_paid_out():
    """The financing gain is a real cash flow, not a discount-factor artefact.

    The deterministic schedule and the flat benchmark move the same gas at the
    same prices, so their nominal totals are identical and only the timing
    differs. The balance between them is money still in hand; the present value
    of the interest it earns at the discount rate is the financing number.
    """
    curve = _flat_daily_curve(40.0)
    rate, strike = 0.10, 30.0
    det = _deterministic(curve, rate, strike=strike)
    _, financing = sm.intrinsic_components(det, strike=strike)

    n = det.n_t
    win = slice(det.Dt, det._active)
    net = np.asarray(det.price_curve, dtype=float)[:n] - strike
    volume = np.abs(np.asarray(det.exp_ex[:n]))
    total = volume.sum()

    cash_deal = volume * net
    cash_bench = np.zeros(n)
    cash_bench[win] = total / (det._active - det.Dt) * net[win]
    assert abs(cash_deal.sum() - cash_bench.sum()) < 1e-6 * total, (
        "the two schedules must move the same nominal cash, only at different times")

    balance = np.cumsum(cash_bench - cash_deal)
    assert balance.max() > 0, "the deal should be holding cash the benchmark has paid"
    pv_interest = float(np.dot(det.d_curve[:n], rate * balance / 365.25))

    assert abs(pv_interest - financing * total) < 0.001 * abs(financing * total), (
        f"PV of interest {pv_interest:,.2f} != financing {financing * total:,.2f}")


def test_a_curve_in_contango_at_the_discount_rate_leaves_no_timing_gain():
    """Charging financing on the gas needs no new input -- the curve does it.

    A flat forward curve beside a positive rate is internally inconsistent: it
    says gas costs the same in December as in January while money costs 10 %.
    The financing gain is the model reporting that inconsistency as free money.
    Put the curve in contango at the same rate and `DF * F` -- the only curve the
    optimiser ever sees -- is flat, so the timing gain vanishes exactly. Day
    selection and financing are then equal and opposite: buying early is cheaper
    on the curve by precisely what paying early costs in funding.

    Optionality is untouched, because that comes from volatility, not slope.
    """
    rate, level = 0.10, 40.0
    days = pd.date_range("2026-01-01", "2027-12-31", freq="D")
    t = (days - pd.Timestamp("2026-01-01")).days.values / 365.25

    def at(carry):
        curve = pd.Series(level * np.exp(carry * t), index=days)
        _, res, _ = _timed("put_swing", rate, curve, n_p_full=20)
        shape, financing = sm.intrinsic_components(_deterministic(curve, rate))
        return res, shape, financing

    flat_res, flat_shape, flat_fin = at(0.0)
    assert abs(flat_shape) < 1e-9 and flat_fin > 0.5, (flat_shape, flat_fin)

    res, shape, financing = at(rate)
    assert abs(res["intrinsic"]) < 1e-9, (
        f"contango at the discount rate must leave no intrinsic, got {res['intrinsic']:.3e}")
    assert abs(shape + financing) < 1e-9, (shape, financing)
    assert shape < -0.5 and financing > 0.5, (
        f"and the two halves should be large and opposite, got {shape:.4f}, {financing:.4f}")
    assert res["extrinsic"] > 0.5, (
        f"optionality comes from vol, not slope, and must survive: {res['extrinsic']:.4f}")


def test_borrow_and_invest_rates_follow_the_deal_direction():
    """One rate assumes spare cash earns what borrowed cash costs. It need not.

    Given the pair, the rate follows the sign of the deal's own cash: a net payer
    funds at the borrow rate, a net receiver places cash at the invest rate. The
    direction comes from the forward net of strike, not the product type, because
    a strike flips it -- a put swing struck above the curve receives.
    """
    curve = _flat_daily_curve(40.0)
    borrow, invest = 0.12, 0.03

    def run(product, strike, **rates):
        params = dict(product_type=product, valDate="2026-01-01",
                      storageStart="2026-02-01", storageEnd="2026-12-31",
                      capacity_mwh=30_000, daily_max=1_000, clips_per_day=1,
                      vol=0.5, sMR=1.0, n_p_full=20, run_intrinsic=False,
                      strike=strike, daily_curve=curve)
        params.update(rates)
        model, _ = sm.run_valuation(None, params)
        return float(model.v[0, model.n_p, model.initial_state]), model.discount_rate

    for product, strike, expected in (("put_swing", 30.0, borrow),    # pays: K below
                                      ("put_swing", 50.0, invest),    # receives: K above
                                      ("call_swing", 30.0, invest),   # receives
                                      ("call_swing", 50.0, borrow)):  # pays
        value, used = run(product, strike, borrow_rate=borrow, invest_rate=invest)
        assert used == expected, f"{product} K={strike}: used {used}, expected {expected}"
        single, _ = run(product, strike, discount_rate=expected)
        assert abs(value - single) < 1e-9, (value, single)

    with pytest.raises(ValueError, match="not.*both"):
        run("put_swing", 30.0, discount_rate=0.10, borrow_rate=borrow, invest_rate=invest)
    with pytest.raises(ValueError, match="together"):
        run("put_swing", 30.0, borrow_rate=borrow)
    with pytest.raises(ValueError, match="no single funding direction"):
        sm.run_valuation(None, dict(
            product_type="storage", valDate="2026-01-01", storageStart="2026-02-01",
            storageEnd="2026-12-31", capacity_mwh=30_000, daily_max=1_000,
            clips_per_day=1, vol=0.5, sMR=1.0, n_p_full=0, run_intrinsic=False,
            daily_curve=curve, inj_cost=0.0, wdr_cost=0.0,
            borrow_rate=borrow, invest_rate=invest))


def test_the_pnl_bridge_sums_to_the_model_value():
    """Total P&L decomposes into moneyness, discounting, and the three gains.

    With nothing paid for the structure the whole P&L is the deal. The lines are

        obligation at the forward  (F - K over the volume, signed by side)
      + effect of discounting it   (it settles during the window, not today)
      + intrinsic                  (day selection)
      + NPV cash                   (financing on cash not yet paid out)
      + extrinsic                  (optionality)

    and they must reproduce the DP's value. On the 10-day put swing, flat 40,
    K = 30: -100,000 nominal, +12,350 discounting, 0 day selection, +4,192
    financing and +35,345 optionality give -48,112 at 10 %; at 0 % the two
    middle lines vanish and -100,000 + 41,809 gives -58,191.
    """
    curve = _flat_daily_curve(40.0)
    strike = 30.0
    for product, sign in (("put_swing", -1.0), ("call_swing", 1.0)):
        for rate in (0.0, 0.10):
            model, res, _ = _timed(product, rate, curve, n_p_full=20, strike=strike)
            det = _deterministic(curve, rate, product_type=product, strike=strike)
            shape, financing = sm.intrinsic_components(det, strike=strike)

            volume = float(np.abs(np.asarray(model.exp_ex[:model.n_t])).sum())
            win = slice(model.Dt, model._active)
            nominal = sign * (float(np.mean(np.asarray(model.price_curve, dtype=float)[win]))
                              - strike)
            bridge = (nominal
                      + (sign * res["flat_metric"] - nominal)
                      + shape + financing + res["extrinsic"]) * volume

            value = float(model.v[0, model.n_p, model.initial_state])
            assert abs(bridge - value) < 1e-6 * max(abs(value), 1.0), (
                f"{product} r={rate}: bridge {bridge:,.4f} != value {value:,.4f}")

            if rate == 0.0:
                assert abs(sign * res["flat_metric"] - nominal) < 1e-9, (
                    "with no rate there is nothing to discount")
                assert abs(financing) < 1e-9, financing
