"""Regression tests for the September 2026 reconciliation findings."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import storage_model as sm  # noqa: E402


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


def test_expected_exercise_is_not_rounded_before_aggregation():
    """The expected exercise schedule must exactly reprice the contract value."""
    model, _ = _mandatory_put()
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


def test_stochastic_tree_rejects_zero_volatility():
    """A zero-width stochastic lattice is rejected with a useful error."""
    with np.testing.assert_raises_regex(ValueError, "positive volatility"):
        sm.build_tree(np.full(5, 25.0), 5, 2, np.zeros(5), np.ones(5))
