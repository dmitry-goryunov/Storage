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
