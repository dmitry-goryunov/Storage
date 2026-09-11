"""Regression tests for the September 2026 reconciliation findings."""

import json
import glob
import itertools
import math
import os
import subprocess
import sys
import tempfile

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


def test_products_notebook_executes_clean_with_treasury_rate_scenario():
    """Run every code cell in a new interpreter and reject stored stale output.

    Products.ipynb contains ordinary Python rather than notebook magics, so a
    fresh subprocess is a stricter and lighter smoke test than reusing pytest's
    interpreter. The smoke flag reduces tree width and the number of deal sizes;
    it does not skip any notebook section.
    """
    path = os.path.join(ROOT, "Products.ipynb")
    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert not cell.get("outputs", [])

    runner = r'''
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("Products.ipynb", encoding="utf-8") as handle:
    notebook = json.load(handle)
namespace = {"display": lambda *args, **kwargs: None}
for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    exec(compile(source, f"Products.ipynb:cell-{index}", "exec"), namespace)
    plt.close("all")
assert namespace["RATE_PARAMS"] == {
    "borrow_rate": 0.12,
    "invest_rate": 0.03,
    "funding_direction": "invest",
}
assert abs(namespace["RUNS"][10][0].discount_rate - 0.03) < 1e-15
print("PRODUCTS_NOTEBOOK_OK rate=0.03")
'''
    env = os.environ.copy()
    env.update({
        "STORAGE_NOTEBOOK_SMOKE": "1",
        "STORAGE_RATE_MODE": "treasury_scenario",
        "STORAGE_BORROW_RATE": "0.12",
        "STORAGE_INVEST_RATE": "0.03",
        "STORAGE_FUNDING_DIRECTION": "invest",
        "MPLBACKEND": "Agg",
    })
    completed = subprocess.run(
        [sys.executable, "-c", runner], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=180, check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )
    assert "PRODUCTS_NOTEBOOK_OK rate=0.03" in completed.stdout


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


def test_three_way_attribution_exposes_the_shape_timing_interaction():
    """The old two-way split is exact but is not a unique economic attribution.

    When both the curve and discount factors vary, changing exercise dates changes
    price and timing together. The explicit interaction is non-zero on the seasonal
    case, and the three terms reconstruct intrinsic exactly. The legacy second
    component deliberately contains timing plus interaction.
    """
    curve = _seasonal_daily_curve()
    model = _deterministic(curve, 0.10)
    attribution = sm.intrinsic_attribution(model)
    shape, legacy_timing = sm.intrinsic_components(model)
    _, result, _ = _timed("put_swing", 0.10, curve, n_p_full=20)

    assert abs(attribution["interaction"]) > 0.01
    assert abs(sum(attribution.values()) - result["intrinsic"]) < 1e-9
    assert abs(shape - attribution["day_selection"]) < 1e-12
    assert abs(legacy_timing - (attribution["settlement_timing"]
                                + attribution["interaction"])) < 1e-12


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


def test_flat_curve_timing_matches_interest_under_the_assumed_rate():
    """The flat-curve timing attribution matches a declared cash-interest scenario.

    The deterministic schedule and the flat benchmark move the same gas at the
    same prices, so their nominal totals are identical and only the timing
    differs. The balance between them is money still in hand. Its interest matches
    the model attribution only under the assumption that the balance genuinely
    earns or avoids the selected rate.
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
    """`DF * F` flat is a model identity, not a general commodity carry claim.

    If this test curve grows at exactly the selected discount rate, `DF * F` --
    the curve the optimiser sees -- is flat, so the timing gain vanishes exactly.
    Day selection and settlement timing are then equal and opposite. This does
    not imply that an observed flat gas forward curve is inconsistent: storage
    costs, convenience yield, seasonality and physical constraints can offset
    financial carry.

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
        f"the two legacy components should be large and opposite, "
        f"got {shape:.4f}, {financing:.4f}")
    assert res["extrinsic"] > 0.5, (
        f"optionality comes from vol, not slope, and must survive: {res['extrinsic']:.4f}")


def _enumerated_schedule_value(prices, discount, actions, initial, terminal, capacity,
                               injection_cost=0.0, withdrawal_cost=0.0):
    """Independent exhaustive oracle for a tiny deterministic contract."""
    best = -np.inf
    for schedule in itertools.product(actions, repeat=len(prices)):
        inventory = initial
        value = 0.0
        feasible = True
        for price, df, action in zip(prices, discount, schedule):
            inventory += action
            if inventory < 0 or inventory > capacity:
                feasible = False
                break
            if action > 0:
                value += action * df * (-price - injection_cost)
            elif action < 0:
                value += (-action) * df * (price - withdrawal_cost)
        if feasible and inventory == terminal:
            best = max(best, value)
    if not np.isfinite(best):
        raise AssertionError("The exhaustive test contract has no feasible schedule.")
    return best


@pytest.mark.parametrize("product_type", ["put_swing", "call_swing", "storage"])
def test_tiny_contract_matches_independent_exhaustive_schedule_oracle(product_type):
    """The DP must equal direct enumeration, not only its own reconciliations.

    Four active days make every feasible physical schedule enumerable. This
    independently tests discounted daily cash flow, strike, exercise direction,
    inventory transitions and the terminal state for all three products.
    """
    val_date = pd.Timestamp("2026-01-01")
    active_dates = pd.date_range("2026-01-02", periods=4, freq="D")
    active_prices = np.array([35.0, 10.0, 50.0, 20.0])
    curve = pd.Series(
        [35.0, *active_prices, 20.0],
        index=pd.date_range(val_date, periods=6, freq="D"),
    )
    rate = 0.17
    discount = np.exp(
        -rate * (active_dates - val_date).days.to_numpy(dtype=float) / 365.25
    )
    params = dict(
        product_type=product_type, valDate=val_date,
        storageStart=active_dates[0], storageEnd=active_dates[-1],
        vol=0.0, sMR=1.0, n_p_full=0, run_intrinsic=False,
        daily_max=1.0, clips_per_day=1, discount_rate=rate,
        daily_curve=curve,
    )

    if product_type == "put_swing":
        strike = 25.0
        params.update(capacity_mwh=2.0, strike=strike)
        expected = _enumerated_schedule_value(
            active_prices, discount, (0, 1), initial=0, terminal=2, capacity=2,
            injection_cost=-strike,
        )
        initial_state = 0
    elif product_type == "call_swing":
        strike = 25.0
        params.update(capacity_mwh=2.0, strike=strike)
        expected = _enumerated_schedule_value(
            active_prices, discount, (0, -1), initial=2, terminal=0, capacity=2,
            withdrawal_cost=strike,
        )
        initial_state = 2
    else:
        params.update(
            capacity_mwh=1.0, inj_cost=0.7, wdr_cost=0.4,
            inj_rate=1, wdr_rate=1, initial_inv_clips=0, terminal_inv_clips=0,
        )
        expected = _enumerated_schedule_value(
            active_prices, discount, (-1, 0, 1), initial=0, terminal=0, capacity=1,
            injection_cost=0.7, withdrawal_cost=0.4,
        )
        initial_state = 0

    model, _ = sm.run_valuation(None, params)
    actual = float(model.v[0, 0, initial_state])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)


def test_zero_discount_rate_still_conflicts_with_treasury_rates():
    """An explicit zero is a selected discounting mode, not an absent field."""
    with pytest.raises(ValueError, match="either `discount_rate`.*not both"):
        sm.normalise_rate_parameters({
            "discount_rate": 0.0,
            "borrow_rate": 0.12,
            "invest_rate": 0.03,
            "funding_direction": "borrow",
        })

    params = dict(
        product_type="put_swing", valDate="2026-01-01",
        storageStart="2026-02-01", storageEnd="2026-12-31",
        capacity_mwh=10_000, daily_max=1_000, clips_per_day=1,
        vol=0.5, sMR=1.0, n_p_full=0, run_intrinsic=False,
        daily_curve=_flat_daily_curve(40.0), discount_rate=0.0,
        borrow_rate=0.12, invest_rate=0.03, funding_direction="borrow",
    )
    with pytest.raises(ValueError, match="either `discount_rate`.*not both"):
        sm.run_valuation(None, params)


def test_workbook_rate_fields_reach_the_valuation_without_inference(tmp_path):
    """Workbook -> loader -> bridge -> model preserves the selected scenario."""
    workbook = tmp_path / "rate-product.xlsx"
    row = pd.DataFrame([{
        "product": "rate_case", "product_type": "call_swing",
        "FDDate": "2026-01-01", "valDate": "2026-01-01",
        "storageStart": "2026-01-02", "storageEnd": "2026-01-05",
        "vol": 0.0, "n_p_full": 0, "run_intrinsic": False,
        "capacity_mwh": 2.0, "initial_storage_mwh": 0.0,
        "terminal_storage_mwh": 0.0, "inj_days": 2, "wdr_days": 2,
        "n_states": 2, "inj_cost": 0.0, "wdr_cost": 0.0,
        "ratchet_profile": "", "notes": "rate plumbing",
        "borrow_rate": 0.12, "invest_rate": 0.03,
        "funding_direction": "invest",
    }])
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        row.to_excel(writer, sheet_name="products", index=False)

    loaded = sm.load_product_params(workbook, "rate_case")
    params = sm.params_for_run_valuation(loaded)
    assert {key: params[key] for key in (
        "borrow_rate", "invest_rate", "funding_direction"
    )} == {
        "borrow_rate": 0.12,
        "invest_rate": 0.03,
        "funding_direction": "invest",
    }
    params["daily_curve"] = pd.Series(
        40.0, index=pd.date_range("2026-01-01", "2026-02-28", freq="D")
    )
    model, _ = sm.run_valuation(None, params)
    assert model.discount_rate == 0.03


def test_streamlit_treasury_inputs_reach_the_model_rate():
    """Exercise the real widgets and require the selected scenario in output."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        os.path.join(ROOT, "streamlit_app.py"), default_timeout=120
    ).run()
    next(widget for widget in app.selectbox if widget.label == "Rate mode").select(
        "Treasury scenario"
    )
    app.run()
    next(widget for widget in app.number_input if widget.label == "n_p_full").set_value(0)
    next(widget for widget in app.checkbox
         if widget.label == "Run intrinsic decomposition").uncheck()
    next(widget for widget in app.number_input
         if widget.label.startswith("borrow_rate")).set_value(0.12)
    next(widget for widget in app.number_input
         if widget.label.startswith("invest_rate")).set_value(0.03)
    next(widget for widget in app.selectbox
         if widget.label == "funding_direction").select("invest")
    next(widget for widget in app.button if widget.label == "Run valuation").click()
    app.run(timeout=120)

    assert not app.exception
    assert any(
        "Applied annual continuously compounded rate: 3.0000%." in caption.value
        for caption in app.caption
    )


def test_borrow_and_invest_scenario_requires_an_explicit_direction():
    """The mean forward net of strike cannot safely select a funding direction.

    Borrow/invest rates are treasury scenarios, not a complete asymmetric-funding
    valuation. The caller must state which scenario is wanted; the model must not
    infer it from a window mean that may differ from optimally selected cash flows.
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

    for product, strike, direction, expected in (
            ("put_swing", 30.0, "borrow", borrow),
            ("put_swing", 50.0, "invest", invest),
            ("call_swing", 30.0, "invest", invest),
            ("call_swing", 50.0, "borrow", borrow)):
        value, used = run(product, strike, borrow_rate=borrow, invest_rate=invest,
                          funding_direction=direction)
        assert used == expected, f"{product} K={strike}: used {used}, expected {expected}"
        single, _ = run(product, strike, discount_rate=expected)
        assert abs(value - single) < 1e-9, (value, single)

    with pytest.raises(ValueError, match="not.*both"):
        run("put_swing", 30.0, discount_rate=0.10, borrow_rate=borrow,
            invest_rate=invest, funding_direction="borrow")
    with pytest.raises(ValueError, match="together"):
        run("put_swing", 30.0, borrow_rate=borrow)
    with pytest.raises(ValueError, match="explicit.*funding_direction"):
        run("call_swing", 30.0, borrow_rate=borrow, invest_rate=invest)
    with pytest.raises(ValueError, match="no single funding direction"):
        sm.run_valuation(None, dict(
            product_type="storage", valDate="2026-01-01", storageStart="2026-02-01",
            storageEnd="2026-12-31", capacity_mwh=30_000, daily_max=1_000,
            clips_per_day=1, vol=0.5, sMR=1.0, n_p_full=0, run_intrinsic=False,
            daily_curve=curve, inj_cost=0.0, wdr_cost=0.0,
            borrow_rate=borrow, invest_rate=invest))


def test_mean_forward_can_give_the_wrong_funding_direction():
    """A call can receive even when the window-average `F - K` is negative.

    Ten high-price days are embedded in a long low-price window. The mandatory
    call selects only those days and receives 60 EUR/MWh, although the window
    mean is 7.90 EUR/MWh below strike. The removed automatic selector chose the
    borrowing rate here. Explicit direction keeps that economic judgement with
    the caller.
    """
    curve = _flat_daily_curve(30.0)
    curve.loc[pd.date_range("2026-06-01", "2026-06-10")] = 100.0
    params = dict(product_type="call_swing", valDate="2026-01-01",
                  storageStart="2026-02-01", storageEnd="2026-12-31",
                  capacity_mwh=10_000, daily_max=1_000, clips_per_day=1,
                  vol=0.0, sMR=1.0, n_p_full=0, run_intrinsic=False,
                  strike=40.0, daily_curve=curve,
                  borrow_rate=0.12, invest_rate=0.03)

    window = curve.loc["2026-02-01":"2026-12-31"]
    assert window.mean() - params["strike"] < 0.0
    with pytest.raises(ValueError, match="explicit.*funding_direction"):
        sm.run_valuation(None, params)

    model, _ = sm.run_valuation(None, dict(params, funding_direction="invest"))
    ex = np.asarray(model.exp_ex[:model.n_t])
    exercised_price = float(np.dot(ex, np.asarray(model.price_curve[:model.n_t])) / ex.sum())
    assert exercised_price == 100.0
    assert model.discount_rate == 0.03


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


def test_the_strike_decides_which_end_of_the_window_a_buyer_exercises():
    """One product, opposite schedules, decided purely by the strike.

    A put swing struck below the curve pays `P - K` and wants to pay late; the
    same swing struck above receives and wants to receive early. Nothing but the
    strike changes, and it flips the schedule end to end -- deterministic mean
    exercise day 359.5 against 4.5 on a 365-day window at 10 %.

    With no rate there is nothing to time and both are indifferent, so they land
    on the same tie-broken schedule.

    The financing gain is positive either way, because deferring a payment and
    accelerating a receipt both help, and it is nearly equal because both sit
    10.00 from the curve -- the gain scales with the net cash moving. Not exactly
    equal: discount factors are convex, so moving the same distance towards the
    valuation date is worth slightly more than moving away from it.
    """
    curve = _flat_daily_curve(40.0)
    pays, receives = 30.0, 50.0

    quiet_pay = _deterministic(curve, 0.0, strike=pays)
    quiet_get = _deterministic(curve, 0.0, strike=receives)
    assert abs(_mean_exercise_day(quiet_pay) - _mean_exercise_day(quiet_get)) < 1e-6, (
        "with no rate the strike must not move the schedule")

    late = _deterministic(curve, 0.10, strike=pays)
    early = _deterministic(curve, 0.10, strike=receives)
    span = late._active - late.Dt
    assert _mean_exercise_day(late) > late.Dt + 0.9 * span, _mean_exercise_day(late)
    assert _mean_exercise_day(early) < early.Dt + 0.1 * span, _mean_exercise_day(early)

    _, fin_late = sm.intrinsic_components(late, strike=pays)
    _, fin_early = sm.intrinsic_components(early, strike=receives)
    assert fin_late > 0.1 and fin_early > 0.1, (fin_late, fin_early)
    assert abs(fin_early - fin_late) / fin_late < 0.1, (
        f"both are 10.00 from the curve, so the gains should be close: "
        f"{fin_early:.4f} vs {fin_late:.4f}")
    assert fin_early > fin_late, (
        "discount factors are convex, so moving earlier beats moving later by "
        f"the same distance: {fin_early:.4f} vs {fin_late:.4f}")


def test_the_per_mwh_price_keeps_its_sign_when_the_strike_crosses_the_curve():
    """A strike above the curve flips a put swing's value, and the sign must survive.

    `stochastic_metric` is what the deal pays per MWh, positive when you pay.
    Struck below a flat 40 you pay; struck above you receive, and the number must
    go negative rather than being reported as a cost. Taking its absolute value --
    which `Products.ipynb` did until this case turned up -- breaks the
    reconciliation `flat = price + intrinsic + extrinsic` by twice the moneyness.
    """
    curve = _flat_daily_curve(40.0)
    for rate in (0.0, 0.10):
        _, pays, _ = _timed("put_swing", rate, curve, n_p_full=20, strike=30.0)
        _, gets, _ = _timed("put_swing", rate, curve, n_p_full=20, strike=50.0)
        assert pays["stochastic_metric"] > 0, (
            f"struck below the curve a put swing pays: {pays['stochastic_metric']:.4f}")
        assert gets["stochastic_metric"] < 0, (
            f"struck above it receives: {gets['stochastic_metric']:.4f}")
        assert pays["flat_metric"] > 0 > gets["flat_metric"], (
            pays["flat_metric"], gets["flat_metric"])

        # And the reconciliation holds on both sides of the money.
        for res in (pays, gets):
            gain = res["flat_metric"] - res["stochastic_metric"]
            assert abs(gain - (res["intrinsic"] + res["extrinsic"])) < 1e-9, (
                f"r={rate}: {gain:.9f} != {res['intrinsic']:.9f} + {res['extrinsic']:.9f}")


def test_every_money_number_is_a_present_value_at_the_valuation_date():
    """`d_curve` is anchored at valDate, so moving valDate rescales the value exactly.

    `d_curve[i] = exp(-r*i/365.25)` with `i` counted from `valDate`, and the value
    is read at time index 0. Price the same window from two valuation dates and
    the deterministic value must differ by exactly `exp(-r*dT)` -- nothing else
    changed, only where "today" is. That is the sharpest statement that the
    reported money is a PV to valDate rather than to the window or to delivery.

    Deliberate exceptions, all of them labelled where they are reported: the
    nominal `F - K` line of the P&L bridge (the next line is the discounting),
    the nominal cash totals and peak balance in the financing view, and `delta`,
    which is an undiscounted hedge volume with `delta_pv` as its tailed twin.
    """
    days = pd.date_range("2025-06-01", "2027-12-31", freq="D")
    curve = pd.Series(40.0, index=days)
    rate, strike = 0.10, 30.0

    def value_from(valdate):
        model, _ = sm.run_valuation(None, dict(
            product_type="put_swing", valDate=valdate, storageStart="2026-02-01",
            storageEnd="2026-12-31", capacity_mwh=30_000, daily_max=1_000,
            clips_per_day=1, vol=0.5, sMR=1.0, n_p_full=0, run_intrinsic=False,
            discount_rate=rate, strike=strike, daily_curve=curve))
        return model, float(model.v[0, model.n_p, model.initial_state])

    late, v_late = value_from("2026-01-01")
    early, v_early = value_from("2025-07-01")

    assert late.d_curve[0] == 1.0 and early.d_curve[0] == 1.0
    assert pd.Timestamp(late.date_span[0]) == late.valDate
    np.testing.assert_allclose(
        late.d_curve, np.exp(-rate * np.arange(late.n_t) / 365.25), rtol=0, atol=0)

    gap = (pd.Timestamp("2026-01-01") - pd.Timestamp("2025-07-01")).days
    expected = v_late * np.exp(-rate * gap / 365.25)
    assert abs(v_early - expected) < 1e-9 * abs(expected), (
        f"valuing {gap} days earlier gave {v_early:,.6f}, not {expected:,.6f} — "
        "the value is not a PV to valDate")

    # delta is the documented exception, and delta_pv is its discounted twin.
    n = late.n_t
    np.testing.assert_allclose(np.asarray(late.delta_pv[:n]),
                               np.asarray(late.delta[:n]) * late.d_curve[:n],
                               rtol=0, atol=1e-12)
    assert abs(np.asarray(late.delta_pv[:n])).sum() < abs(np.asarray(late.delta[:n])).sum()


def test_delta_over_volume_is_the_price_conditional_on_exercising():
    """`delta / exp_ex == E[S | exercise] / F` — recomputed from the raw DP arrays.

    `delta[i]` is `E[S_i Q_i] / F_i` and `exp_ex[i]` is `E[Q_i]`, so their ratio
    is the volume-weighted price the deal actually transacts at, over the
    forward. That is the whole reason the two series differ, and section 6b now
    reports it as its own column, so it is checked here against a probability
    weighting built directly from `prob`, `strat` and the price tree rather than
    from the reported series.

    It runs above 1 where exercise is chosen at good prices and below where a
    quota forces it: on the reference call swing, 1.49 in January against 0.76
    in December.
    """
    curve = _flat_daily_curve(40.0)
    model, _, _ = _timed("call_swing", 0.10, curve, n_p_full=20, strike=30.0)
    n = model.n_t

    # E[S*Q] and E[Q] straight from the DP's own arrays.
    action = model.strat[:n] * model.v_step
    weighted = model.prob[:n] * action
    volume = -weighted.sum(axis=(1, 2))
    traded = -(weighted * np.exp(model.x)[:, :, None]).sum(axis=(1, 2))

    fwd = np.asarray(model.fwd)[:n]
    np.testing.assert_allclose(np.asarray(model.exp_ex[:n]), volume, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(model.delta[:n]), traded / fwd, rtol=0, atol=1e-9)

    live = np.abs(volume) > 1e-6
    ratio = np.asarray(model.delta[:n])[live] / volume[live]
    conditional = traded[live] / volume[live]
    np.testing.assert_allclose(ratio, conditional / fwd[live], rtol=1e-12, atol=0)

    # A seller is picky early and forced late, so the ratio must fall through the
    # window and cross 1 before the end.
    win = np.arange(n)[live]
    win = win[(win >= model.Dt) & (win < model._active)]
    early = ratio[np.isin(np.arange(n)[live], win[:30])].mean()
    late = ratio[np.isin(np.arange(n)[live], win[-30:])].mean()
    assert early > 1.2, f"early exercise should be chosen at good prices: {early:.4f}"
    assert late < 1.0, f"a forced quota should transact below the forward: {late:.4f}"


def _black76_call(fwd, strike, var, df=1.0):
    """Black-76 call given TOTAL log-variance rather than a vol and a maturity."""
    if var <= 0.0:
        return df * max(fwd - strike, 0.0)
    sd = math.sqrt(var)
    d1 = (math.log(fwd / strike) + 0.5 * var) / sd
    norm = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return df * (fwd * norm(d1) - strike * norm(d1 - sd))


@pytest.mark.parametrize("months", [6, 18])
@pytest.mark.parametrize("strike", [24.0, 32.0, 40.0, 48.0, 60.0])
def test_a_one_day_swing_reproduces_black_76(strike, months):
    """The only check here against an independent closed form.

    Collapse a call swing to a single exercise day on a one-day window and waive
    the quota with `zero_penalty`, and it is a European call. It must reproduce
    Black-76 -- but at the *mean-reverting* terminal variance

        var(T) = sVol^2 (1 - exp(-2 sMR T)) / (2 sMR)

    not `sVol^2 T`. At sVol 0.5 and sMR 1.0 that is 0.2818 effective vol against
    a 0.5 input at 18 months, so comparing at the raw `sVol` would suggest the
    model is 44 % cheap when it is right.

    Every other test in this file is internal consistency. This one would catch a
    regression in the tree, the forward fitting or the terminal distribution that
    the invariants would sail through.
    """
    level, vol, mr = 40.0, 0.5, 1.0
    val_date = pd.Timestamp("2026-01-01")
    expiry = val_date + pd.DateOffset(months=months)
    curve = pd.Series(level, index=pd.date_range("2025-01-01", "2029-12-31", freq="D"))

    model, _ = sm.run_valuation(None, dict(
        product_type="call_swing", valDate=val_date, storageStart=expiry,
        storageEnd=expiry, capacity_mwh=1_000, daily_max=1_000, clips_per_day=1,
        vol=vol, sMR=mr, n_p_full=90, run_intrinsic=False, discount_rate=0.0,
        strike=strike, zero_penalty=True, daily_curve=curve))
    priced = float(model.v[0, model.n_p, model.initial_state]) / 1_000.0

    years = (expiry - val_date).days / 365.25
    var = vol ** 2 * (1.0 - math.exp(-2.0 * mr * years)) / (2.0 * mr)
    closed = _black76_call(level, strike, var)

    assert abs(priced - closed) < 0.02, (
        f"K={strike} at {months}m: model {priced:.5f} vs Black-76 {closed:.5f}")
    if closed > 0.1:
        assert abs(priced - closed) / closed < 0.01, (
            f"K={strike} at {months}m: {abs(priced-closed)/closed:.2%} relative")

    # The tree's own realised log-variance should agree with the analytic value,
    # which separates "the payoff is wrong" from "the distribution is wrong".
    weights = model.prob[model.Dt].sum(axis=1)
    weights = weights / weights.sum()
    logs = np.asarray(model.x)[model.Dt]
    mean = float(np.dot(weights, logs))
    assert abs(float(np.dot(weights, (logs - mean) ** 2)) / var - 1.0) < 0.01


def test_an_obligation_is_worth_less_than_the_same_right():
    """`call_swing` defaults to a mandatory quota, and that is not a call.

    Benchmarking the model's default against a broker's call quote compares two
    different contracts. On a flat curve at the money the obligation is worth
    about half the right, because it sells on the bad days too. The right also
    leaves part of its quota unused, which the obligation cannot.
    """
    level = 40.0
    curve = pd.Series(level, index=pd.date_range("2025-01-01", "2029-12-31", freq="D"))

    def priced(days, right):
        model, _ = sm.run_valuation(None, dict(
            product_type="call_swing", valDate="2026-01-01", storageStart="2027-01-01",
            storageEnd="2027-12-31", capacity_mwh=days * 1_000, daily_max=1_000,
            clips_per_day=1, vol=0.5, sMR=1.0, n_p_full=40, run_intrinsic=False,
            discount_rate=0.0, strike=level, zero_penalty=right, daily_curve=curve))
        value = float(model.v[0, model.n_p, model.initial_state]) / (days * 1_000)
        used = float(np.abs(np.asarray(model.exp_ex[:model.n_t])).sum())
        return value, used / (days * 1_000)

    for days in (1, 10):
        right_value, right_used = priced(days, True)
        duty_value, duty_used = priced(days, False)
        assert duty_value < right_value, (days, duty_value, right_value)
        assert 0.35 < duty_value / right_value < 0.65, (
            f"{days} days: obligation is {duty_value/right_value:.0%} of the right")
        assert duty_used == pytest.approx(1.0, abs=1e-6), duty_used
        assert 0.3 < right_used < 0.9, f"a right should leave quota unused: {right_used:.2%}"


def test_swing_vs_option_notebook_executes_clean():
    """SwingVsOption.ipynb runs every section in a fresh process, with no stale output."""
    path = os.path.join(ROOT, "SwingVsOption.ipynb")
    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert not cell.get("outputs", [])

    runner = r'''
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("SwingVsOption.ipynb", encoding="utf-8") as handle:
    notebook = json.load(handle)
namespace = {"display": lambda *args, **kwargs: None}
for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    exec(compile(source, f"SwingVsOption.ipynb:cell-{index}", "exec"), namespace)
    plt.close("all")
assert namespace["_worst"] < 0.02, namespace["_worst"]
print("SWING_VS_OPTION_OK")
'''
    env = os.environ.copy()
    env.update({"STORAGE_NOTEBOOK_SMOKE": "1", "MPLBACKEND": "Agg"})
    completed = subprocess.run(
        [sys.executable, "-c", runner], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=180, check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}")
    assert "SWING_VS_OPTION_OK" in completed.stdout


def test_the_tree_carries_the_clewlow_strickland_variance_term_structure():
    """Forward variance depends on spot vol *and* mean reversion, per C&S (6.13).

    Clewlow & Strickland (1999a) integrate forward return variance over the life
    of the option in the one-factor Schwartz model:

        w^2 = int_t^T sigma^2 exp(-2a(s-u)) du
            = sigma^2/(2a) * (exp(-2a(s-T)) - exp(-2a(s-t)))          (6.13)

    Two consequences are checked here.

    At `s = T` this is an option on the spot, and (6.13) collapses to
    `sigma^2/(2a) * (1 - exp(-2a(T-t)))` -- their (6.14). That is the case this
    project compares against, because a swing exercises against the daily index
    rather than against a futures contract. The tree must carry that variance at
    every horizon, not merely at one: variance saturates at `sigma^2/(2a)` while
    T grows, so the comparable Black-76 vol falls from 0.44 at three months to
    0.16 at five years against a 0.5 input.

    For `s > T` the damping factor is `exp(-a(s-T))` on the volatility -- the
    Samuelson effect. This model has no tradable futures with their own
    dynamics, so it cannot price that option; the relation is asserted on the
    closed form only, to keep the two cases distinguishable.
    """
    vol, mr = 0.5, 1.0
    val_date = pd.Timestamp("2026-01-01")
    curve = pd.Series(40.0, index=pd.date_range("2025-01-01", "2033-12-31", freq="D"))

    def futures_var(t_to_expiry, t_to_delivery):
        return vol ** 2 / (2 * mr) * (math.exp(-2 * mr * (t_to_delivery - t_to_expiry))
                                      - math.exp(-2 * mr * t_to_delivery))

    def spot_var(years):
        return vol ** 2 * (1.0 - math.exp(-2.0 * mr * years)) / (2.0 * mr)

    # (6.13) must collapse onto (6.14) when the future delivers at expiry.
    for years in (0.25, 1.0, 3.0):
        assert abs(futures_var(years, years) - spot_var(years)) < 1e-15

    # The Samuelson damping is exactly exp(-a(s-T)) on the volatility.
    for gap in (0.25, 0.5, 1.0, 2.0):
        ratio = math.sqrt(futures_var(1.0, 1.0 + gap) / futures_var(1.0, 1.0))
        assert abs(ratio - math.exp(-mr * gap)) < 1e-12, (gap, ratio)

    # And the tree reproduces the spot variance across the whole term structure.
    previous_vol = None
    for months in (3, 12, 36, 60):
        expiry = val_date + pd.DateOffset(months=months)
        model, _ = sm.run_valuation(None, dict(
            product_type="call_swing", valDate=val_date, storageStart=expiry,
            storageEnd=expiry, capacity_mwh=1_000, daily_max=1_000, clips_per_day=1,
            vol=vol, sMR=mr, n_p_full=120, run_intrinsic=False, discount_rate=0.0,
            strike=40.0, zero_penalty=True, daily_curve=curve))
        weights = model.prob[model.Dt].sum(axis=1)
        weights = weights / weights.sum()
        logs = np.asarray(model.x)[model.Dt]
        mean = float(np.dot(weights, logs))
        measured = float(np.dot(weights, (logs - mean) ** 2))

        years = (expiry - val_date).days / 365.25
        expected = spot_var(years)
        assert abs(measured / expected - 1.0) < 0.01, (
            f"{months}m: tree variance {measured:.6f} vs C&S {expected:.6f}")
        assert expected < vol ** 2 / (2 * mr), "variance must stay below its stationary value"

        effective = math.sqrt(expected / years)
        assert effective < vol, f"{months}m: effective vol {effective:.4f} is not below sVol"
        if previous_vol is not None:
            assert effective < previous_vol, "the comparable vol must fall with maturity"
        previous_vol = effective


def test_no_notebook_carries_stray_control_characters():
    """Control characters in a cell mean an escape was eaten on the way in.

    `SwingVsOption.ipynb` shipped with ten of them: LaTeX written into a
    non-raw Python string turned `\alpha` into BEL and `\frac` into formfeed.
    Both are *valid* Python escapes, so nothing warned -- the only symptom was
    KaTeX refusing to render, which no test would have seen. Markdown is not
    executed, so this is the only place the damage can be caught.

    Tabs and newlines are legitimate; nothing else below 0x20 is.
    """
    notebooks = sorted(glob.glob(os.path.join(ROOT, "*.ipynb")))
    assert notebooks, "expected notebooks at the repository root"

    offenders = []
    for path in notebooks:
        with open(path, encoding="utf-8") as handle:
            notebook = json.load(handle)
        for index, cell in enumerate(notebook.get("cells", [])):
            source = "".join(cell.get("source", []))
            for position, char in enumerate(source):
                if ord(char) < 32 and char not in "\n\t":
                    offenders.append(
                        f"{os.path.basename(path)} cell {index} ({cell.get('cell_type')}): "
                        f"{hex(ord(char))} in {source[max(0, position - 25):position + 10]!r}")
    assert not offenders, "stray control characters:\n" + "\n".join(offenders)


def test_asymmetric_storage_rates_survive_the_days_to_rate_conversion():
    """"30 in, 60 out" works, but only on a grid that can express it -- and a
    grid that cannot is now REFUSED, not silently mispriced.

    `normalise_storage_contract` (via `params_for_run_valuation`) preserves the
    requested MWh/day exactly for both directions, and `value_storage` reads
    both, so asymmetric storage is supported -- its docstring said otherwise
    until 2026-09-10 and sent readers to `forward.ipynb`.

    Until 2026-09-11 the conversion was `max(1, round(n_states / days))` with
    no check: 30/60 needed `n_states` to be a multiple of 60, and at 30 or 45
    states the withdrawal side rounded to a DIFFERENT rate than requested --
    at N=30 both sides collapsed to 1 clip/day, silently pricing 30/30 -- while
    reporting nothing. This test used to assert that collapse as correct
    (`wdr_rate == 1` at N=30); IMPLEMENTATION-GUIDE-2026-09-11.md checklist
    item 3 replaces that assertion with the refusal it should have been.
    """
    for n_states, expect in ((60, (2, 1)), (120, (4, 2))):
        params = sm.params_for_run_valuation(dict(
            product_type="storage", n_states=n_states, capacity_mwh=n_states * 10_000.0,
            inj_days=30, wdr_days=60,
            initial_storage_mwh=0.0, terminal_storage_mwh=0.0))
        assert (params["inj_rate"], params["wdr_rate"]) == expect, (n_states, params)

    # 30 and 45 states cannot express 30/60 exactly (lcm(30, 60) = 60) -- both
    # are refused now, not silently repriced to a rate nobody asked for.
    for n_states in (30, 45):
        with pytest.raises(ValueError, match="cannot express"):
            sm.params_for_run_valuation(dict(
                product_type="storage", n_states=n_states,
                capacity_mwh=n_states * 10_000.0, inj_days=30, wdr_days=60,
                initial_storage_mwh=0.0, terminal_storage_mwh=0.0))

    # 60 states does, and the physical schedule honours it.
    index = pd.date_range("2026-01-01", "2028-12-31", freq="D")
    doy = index.dayofyear.values
    curve = pd.Series(25.0 + 6.0 * np.cos(2 * np.pi * (doy - 15) / 365.25), index=index)
    capacity, inject, withdraw = 600_000.0, 20_000.0, 10_000.0

    model, _ = sm.run_valuation(None, dict(
        product_type="storage", valDate="2026-06-01",
        storageStart="2027-01-01", storageEnd="2027-12-31",
        capacity_mwh=capacity, daily_max=inject, clips_per_day=2,
        inj_rate=2, wdr_rate=1, initial_inv_clips=0, terminal_inv_clips=0,
        inj_cost=0.0, wdr_cost=0.0, vol=0.5, sMR=1.0, n_p_full=0,
        run_intrinsic=False, discount_rate=0.0, daily_curve=curve))

    assert model.v_step == withdraw, model.v_step
    assert model.n_states == 60, model.n_states

    n = model.n_t
    moved = model.prob[:n] * model.strat[:n] * model.v_step
    injected = np.clip(moved, 0, None).sum(axis=(1, 2))
    withdrawn = -np.clip(moved, None, 0).sum(axis=(1, 2))

    assert injected.max() <= inject + 1e-6, injected.max()
    assert withdrawn.max() <= withdraw + 1e-6, withdrawn.max()
    # The withdrawal cap has to actually bind, or the test proves nothing.
    assert withdrawn.max() > 0.9 * withdraw, withdrawn.max()
    assert injected.max() > 1.5 * withdraw, (
        f"injection should run faster than withdrawal: {injected.max():,.0f}")

    inventory = np.cumsum(injected - withdrawn)
    assert inventory.max() <= capacity + 1e-6, inventory.max()
    assert abs(inventory[-1]) < 1e-6, f"must end empty, at {inventory[-1]:,.1f}"


def test_storage_notebook_executes_clean():
    """Storage_30_65.ipynb runs every section in a fresh process, with no stale output.

    It also asserts its own grid: §2 raises unless the derived rates reproduce
    the days asked for, so a silent mis-sizing fails the suite rather than
    quietly pricing a different contract.
    """
    path = os.path.join(ROOT, "Storage_30_65.ipynb")
    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert not cell.get("outputs", [])

    runner = r'''
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("Storage_30_65.ipynb", encoding="utf-8") as handle:
    notebook = json.load(handle)
namespace = {"display": lambda *args, **kwargs: None}
for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    exec(compile(source, f"Storage_30_65.ipynb:cell-{index}", "exec"), namespace)
    plt.close("all")
assert namespace["N_STATES"] == 390, namespace["N_STATES"]
assert (namespace["INJ_RATE"], namespace["WDR_RATE"]) == (13, 6)
assert abs(namespace["N_STATES"] / namespace["INJ_RATE"] - 30) < 1e-9
assert abs(namespace["N_STATES"] / namespace["WDR_RATE"] - 65) < 1e-9
print("STORAGE_NOTEBOOK_OK")
'''
    env = os.environ.copy()
    env.update({"STORAGE_NOTEBOOK_SMOKE": "1", "MPLBACKEND": "Agg"})
    completed = subprocess.run(
        [sys.executable, "-c", runner], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=300, check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}")
    assert "STORAGE_NOTEBOOK_OK" in completed.stdout


def test_simple_storage_notebook_executes_clean():
    """Storage_30_60.ipynb runs every section in a fresh process, with no stale output."""
    path = os.path.join(ROOT, "Storage_30_60.ipynb")
    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert not cell.get("outputs", [])

    runner = r'''
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("Storage_30_60.ipynb", encoding="utf-8") as handle:
    notebook = json.load(handle)
namespace = {"display": lambda *args, **kwargs: None}
for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") != "code":
        continue
    exec(compile("".join(cell.get("source", [])),
                 f"Storage_30_60.ipynb:cell-{index}", "exec"), namespace)
    plt.close("all")
# Pin the physical deal, not the grid: the notebook refines the clip to fit a
# ratchet profile, so N_STATES is its choice and 30/60 is the invariant.
assert namespace["CAPACITY"] == 600_000.0
assert namespace["N_STATES"] / namespace["INJ_RATE"] == 30.0
assert namespace["N_STATES"] / namespace["WDR_RATE"] == 60.0
step = namespace["CAPACITY"] / namespace["N_STATES"]
assert namespace["INJ_RATE"] * step == namespace["INJ_MWH_DAY"]
assert namespace["WDR_RATE"] * step == namespace["WDR_MWH_DAY"]
free, funded = namespace["RUNS"][0.0][2], namespace["RUNS"][0.10][2]
assert funded["value"] < free["value"], (funded["value"], free["value"])
assert max(free["invariant"], funded["invariant"]) < 1e-9
print("SIMPLE_STORAGE_OK")
'''
    env = os.environ.copy()
    env.update({"STORAGE_NOTEBOOK_SMOKE": "1", "MPLBACKEND": "Agg"})
    completed = subprocess.run(
        [sys.executable, "-c", runner], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=300, check=False)
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}")
    assert "SIMPLE_STORAGE_OK" in completed.stdout


def test_the_live_convergence_check_uses_the_deal_actually_in_the_notebook():
    """`VERIFY_CONVERGENCE = True` prices the CURRENT curve/ratchets/bounds at
    N_STATES, x2 and x4, and feeds the result through the same
    `benchmarks.convergence_verdict` the acceptance pack checks -- not a
    lookup table measured once on a different day. Forces the flag on and
    checks the ladder is genuinely live: three rows, at the right resolutions,
    from a real `benchmarks.CONVERGENCE_STATUSES` member, not a hard-coded
    number surviving an edited deal.

    IMPLEMENTATION-GUIDE-2026-09-11.md checklist item 12.
    """
    path = os.path.join(ROOT, "Storage_30_60.ipynb")
    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    assert not any("_RESIDUAL" in "".join(cell.get("source", []))
                   for cell in notebook["cells"]), (
        "the hard-coded _RESIDUAL lookup should be gone entirely")

    runner = r'''
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("Storage_30_60.ipynb", encoding="utf-8") as handle:
    notebook = json.load(handle)
namespace = {"display": lambda *args, **kwargs: None}
for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    if index == 3:
        assert "VERIFY_CONVERGENCE = False" in source
        # Replaces both the assignment and its own name inside the (now dead)
        # False-branch print message -- harmless, since that branch does not
        # run once the flag is True.
        source = source.replace("VERIFY_CONVERGENCE = False", "VERIFY_CONVERGENCE = True")
    exec(compile(source, f"Storage_30_60.ipynb:cell-{index}", "exec"), namespace)
    plt.close("all")
    if index >= 3:
        break

ladder = namespace["_ladder"]
assert list(ladder["n_states"]) == [namespace["N_STATES"], namespace["N_STATES"] * 2,
                                    namespace["N_STATES"] * 4]
assert ladder["refused"].isna().all(), ladder
assert namespace["_status"] in {"within_declared_tolerance", "outside_tolerance",
                                "insufficient", "invalid"}
# The middle point must sit strictly between the other two -- proof this is a
# live monotone-refinement computation, not a constant repeated three times.
lo, mid, hi = ladder["total_eur"]
assert lo < mid < hi or lo > mid > hi, ladder
print("LIVE_LADDER_OK", namespace["_status"])
'''
    env = os.environ.copy()
    env.update({"STORAGE_NOTEBOOK_SMOKE": "1", "MPLBACKEND": "Agg"})
    completed = subprocess.run(
        [sys.executable, "-c", runner], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=300, check=False)
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}")
    assert "LIVE_LADDER_OK" in completed.stdout, completed.stdout


def test_a_stores_physical_volume_nets_to_zero_but_its_hedge_does_not():
    """The point of the 30/60 notebook, as an assertion.

    Everything injected is withdrawn, so `exp_ex` sums to zero over the deal.
    `delta` does not, because the summer forwards bought and the winter forwards
    sold are different contracts at different prices — the residual is the
    seasonal spread the store is long. A hedge sized off physical volume would
    be no hedge at all.
    """
    span = pd.date_range("2026-01-01", "2029-06-30", freq="D")
    curve = pd.Series(
        25.0 + 6.0 * np.cos(2 * np.pi * (span.dayofyear.values - 1) / 365.25), index=span)

    model, _ = sm.run_valuation(None, dict(
        product_type="storage", valDate="2026-06-01", storageStart="2027-01-01",
        storageEnd="2027-12-31", capacity_mwh=600_000.0, daily_max=20_000.0,
        clips_per_day=2, inj_rate=2, wdr_rate=1, initial_inv_clips=0,
        terminal_inv_clips=0, inj_cost=0.0, wdr_cost=0.0, vol=0.5, sMR=1.0,
        n_p_full=20, run_intrinsic=False, discount_rate=0.10, daily_curve=curve))

    n = model.n_t
    physical = np.asarray(model.exp_ex[:n])
    delta = np.asarray(model.delta[:n])
    gross = float(np.abs(physical).sum())

    assert abs(physical.sum()) < 1e-6 * gross, physical.sum()
    assert abs(delta.sum()) > 1e-3 * gross, (
        f"the hedge should not net out: {delta.sum():,.0f} against {gross:,.0f} gross")

    # Summer is bought and winter is sold, in both series.
    dates = pd.DatetimeIndex(model.date_span)[:n]
    frame = pd.DataFrame({"physical": physical, "delta": delta}, index=dates)
    monthly = frame.loc["2027-01-01":"2027-12-31"].resample("MS").sum()
    assert monthly.loc["2027-07-01", "delta"] < 0, "July should buy"
    assert monthly.loc["2027-12-01", "delta"] > 0, "December should sell"

    # And it still reprices: no cost leg here, both costs being zero.
    value = float(model.v[0, model.n_p, model.initial_state])
    reprice = float(np.dot(model.d_curve[:n] * delta, np.asarray(model.fwd)[:n]))
    assert abs(reprice - value) / abs(value) < 1e-9


def test_a_flat_month_makes_intra_month_churn_exactly_free():
    """A curve that is flat within a month leaves the optimiser indifferent.

    Injecting and withdrawing inside one month buys and sells at the same price,
    so the round trip is exactly break-even. The DP will do it or not with no
    effect on value, and any non-zero discount rate tips it into doing it --
    cycled volume doubles from 600,000 to 1,200,000 MWh while the nominal cash
    the schedule moves is unchanged at 6,320,000 EUR.

    It is not a defect, but it makes volume a poor measure of what a store is
    doing on a stepped curve. `Storage_30_60.ipynb` reports cash instead. A
    smoothed curve has a gradient inside each month and does not do this.
    """
    months = {1: 31.0, 2: 30.0, 3: 28.0, 4: 25.0, 5: 22.0, 6: 20.0,
              7: 19.0, 8: 19.5, 9: 22.0, 10: 25.5, 11: 28.5, 12: 30.5}
    span = pd.date_range("2026-01-01", "2029-06-30", freq="D")
    stepped = pd.Series([months[d.month] for d in span], index=span)

    def schedule(rate):
        model, _ = sm.run_valuation(None, dict(
            product_type="storage", valDate="2026-06-01", storageStart="2027-01-01",
            storageEnd="2027-12-31", capacity_mwh=600_000.0, daily_max=20_000.0,
            clips_per_day=2, inj_rate=2, wdr_rate=1, initial_inv_clips=0,
            terminal_inv_clips=0, inj_cost=0.0, wdr_cost=0.0, vol=0.5, sMR=1.0,
            n_p_full=0, run_intrinsic=False, discount_rate=rate, daily_curve=stepped))
        n = model.n_t
        moved = model.prob[:n] * model.strat[:n] * model.v_step
        injected = np.clip(moved, 0, None).sum(axis=(1, 2))
        withdrawn = -np.clip(moved, None, 0).sum(axis=(1, 2))
        forward = np.asarray(model.price_curve, dtype=float)[:n]
        cash = float(np.dot(withdrawn, forward) - np.dot(injected, forward))
        return injected.sum(), cash

    volume_free, cash_free = schedule(0.0)
    volume_tiny, cash_tiny = schedule(1e-6)

    assert volume_tiny > 1.5 * volume_free, (volume_free, volume_tiny)
    assert abs(cash_tiny - cash_free) < 1e-6 * abs(cash_free), (
        f"the extra churn must be worth nothing: {cash_free:,.2f} vs {cash_tiny:,.2f}")

    # A smoothed curve has a gradient inside each month, so no exact tie arises.
    smooth = pd.Series(
        25.0 + 6.0 * np.cos(2 * np.pi * (span.dayofyear.values - 1) / 365.25), index=span)

    def smooth_volume(rate):
        model, _ = sm.run_valuation(None, dict(
            product_type="storage", valDate="2026-06-01", storageStart="2027-01-01",
            storageEnd="2027-12-31", capacity_mwh=600_000.0, daily_max=20_000.0,
            clips_per_day=2, inj_rate=2, wdr_rate=1, initial_inv_clips=0,
            terminal_inv_clips=0, inj_cost=0.0, wdr_cost=0.0, vol=0.5, sMR=1.0,
            n_p_full=0, run_intrinsic=False, discount_rate=rate, daily_curve=smooth))
        n = model.n_t
        moved = model.prob[:n] * model.strat[:n] * model.v_step
        return np.clip(moved, 0, None).sum(axis=(1, 2)).sum()

    assert smooth_volume(1e-6) == pytest.approx(smooth_volume(0.0), rel=1e-9)


def test_the_intrinsic_hedge_is_the_intrinsic_schedule():
    """With no price uncertainty, delta collapses onto volume.

    `delta[i] = E[S_i Q_i] / F_i`. On an `n_p = 0` tree there is one price state,
    so `E[S | exercise] = F` and the hedge equals the schedule exactly. That is
    what makes "intrinsic delta" meaningful: it is what you trade today to lock
    the intrinsic value and then leave alone, and

        total delta = intrinsic delta + extrinsic delta

    mirrors the value split. For a store the intrinsic delta nets to zero -- the
    deterministic schedule is a closed cycle -- so every MWh of net hedge is
    extrinsic, which is not obvious until the two are shown apart.
    """
    months = {1: 30.0, 2: 30.0, 3: 25.0, 4: 25.0, 5: 25.0, 6: 25.0,
              7: 25.0, 8: 25.0, 9: 25.0, 10: 30.0, 11: 30.0, 12: 30.0}
    span = pd.date_range("2026-01-01", "2029-06-30", freq="D")
    curve = pd.Series([months[d.month] for d in span], index=span)

    def model_for(n_p):
        base = dict(product_type="storage", valDate="2026-06-01",
                    storageStart="2027-01-01", storageEnd="2027-12-31",
                    capacity_mwh=600_000.0, daily_max=20_000.0, clips_per_day=2,
                    inj_rate=2, wdr_rate=1, initial_inv_clips=0, terminal_inv_clips=0,
                    inj_cost=0.0, wdr_cost=0.0, vol=0.5, sMR=1.0, n_p_full=n_p,
                    run_intrinsic=False, discount_rate=0.10, daily_curve=curve)
        return sm.run_valuation(None, base)[0]

    deterministic = model_for(0)
    n = deterministic.n_t
    intrinsic_delta = np.asarray(deterministic.delta[:n])
    intrinsic_volume = np.asarray(deterministic.exp_ex[:n])

    scale = max(float(np.abs(intrinsic_volume).sum()), 1.0)
    np.testing.assert_allclose(intrinsic_delta, intrinsic_volume, rtol=0, atol=1e-9 * scale)

    # A closed cycle, so the locked-in hedge is volume-neutral...
    assert abs(intrinsic_delta.sum()) < 1e-9 * scale, intrinsic_delta.sum()

    # ...while the full hedge is not, and the difference is all extrinsic.
    full = model_for(20)
    total_delta = np.asarray(full.delta[:full.n_t])
    assert abs(total_delta.sum()) > 1e-4 * scale, total_delta.sum()

    extrinsic_delta = total_delta - intrinsic_delta
    np.testing.assert_allclose(extrinsic_delta.sum(), total_delta.sum(),
                               rtol=0, atol=1e-9 * scale)


def _dated_storage(**extra):
    """A 30/60 store on a two-level curve, with optional dated inventory bounds."""
    months = {1: 30.0, 2: 30.0, 3: 25.0, 4: 25.0, 5: 25.0, 6: 25.0,
              7: 25.0, 8: 25.0, 9: 25.0, 10: 30.0, 11: 30.0, 12: 30.0}
    span = pd.date_range("2026-01-01", "2029-06-30", freq="D")
    curve = pd.Series([months[d.month] for d in span], index=span)
    params = dict(product_type="storage", valDate="2026-06-01",
                  storageStart="2027-01-01", storageEnd="2027-12-31",
                  capacity_mwh=600_000.0, daily_max=20_000.0, clips_per_day=2,
                  inj_rate=2, wdr_rate=1, initial_inv_clips=0, terminal_inv_clips=0,
                  inj_cost=0.0, wdr_cost=0.0, vol=0.5, sMR=1.0, n_p_full=0,
                  run_intrinsic=False, discount_rate=0.10, daily_curve=curve)
    params.update(extra)
    model, _ = sm.run_valuation(None, params)
    return (model,) + _opening_and_closing(model)


def _opening_and_closing(model):
    """Expected opening and closing balances, in MWh, from the state distribution.

    `prob[t, :, l]` IS the law of OPENING inventory on day t, so the balance is
    read rather than rebuilt. The old `cumsum(net moves)` reconstruction added
    `initial_state` to the first day only and was therefore wrong for every store
    that starts with gas in it.
    """
    n = model.n_t
    levels = np.arange(model.n_op) * model.v_step
    opening = (model.prob[:n].sum(axis=1) * levels).sum(axis=1)
    moved = (model.prob[:n] * model.strat[:n] * model.v_step).sum(axis=(1, 2))
    dates = pd.DatetimeIndex(model.date_span)[:n]
    return pd.Series(opening, index=dates), pd.Series(opening + moved, index=dates)


def _inventory_law(model, when):
    """P(opening inventory = l clips) on a given date."""
    index = (pd.Timestamp(when) - model.valDate).days
    return model.prob[index].sum(axis=0)


def test_dated_inventory_bounds_bind_on_the_opening_balance():
    """"1 October inventory at least 70 %" is now a parameter, and it binds.

    The mechanism existed -- `mintunnel`/`max_tunnel` with a `1000*v_step` per
    clip penalty -- but nothing reached it: `value_storage` read no such param
    and `set_volume_states` resets the arrays, so a bound could only be set by
    building a `Storage` by hand. `min_inventory`/`max_inventory` now carry
    date -> fraction through `run_valuation`.

    The bound applies to the balance the day **opens** with, before that day's
    move. Reported on the closing balance the same schedule looks a clip short,
    which is the ambiguity roadmap P1.1 names; the convention is now documented
    rather than implied.
    """
    capacity = 600_000.0
    _, free_open, _ = _dated_storage()
    assert free_open["2027-04-01"] < 0.01 * capacity, free_open["2027-04-01"]

    _, open_floor, close_floor = _dated_storage(min_inventory={"2027-04-01": 0.70})
    assert open_floor["2027-04-01"] == pytest.approx(0.70 * capacity, rel=1e-9)
    # The same day's closing balance is a move lower -- both are legitimate
    # readings of "1 April inventory", which is exactly why it must be stated.
    assert close_floor["2027-04-01"] < open_floor["2027-04-01"]

    _, open_ceiling, _ = _dated_storage(max_inventory={"2027-10-01": 0.30})
    assert open_ceiling["2027-10-01"] == pytest.approx(0.30 * capacity, rel=1e-9)

    # Several at once, and a bound that does not bind changes nothing.
    _, several, _ = _dated_storage(min_inventory={"2027-04-01": 0.70},
                                   max_inventory={"2027-12-15": 0.10})
    assert several["2027-04-01"] == pytest.approx(0.70 * capacity, rel=1e-9)
    assert several["2027-12-15"] <= 0.10 * capacity + 1e-6

    model_free, _, _ = _dated_storage()
    model_slack, _, _ = _dated_storage(min_inventory={"2027-10-01": 0.70})
    assert float(model_slack.v[0, model_slack.n_p, model_slack.initial_state]) == pytest.approx(
        float(model_free.v[0, model_free.n_p, model_free.initial_state]), rel=1e-12), (
        "a bound the schedule already satisfies must not change the value")


def test_inventory_bounds_refuse_what_they_cannot_honour():
    """Each refusal replaces a silent wrong answer, not a working configuration."""
    with pytest.raises(ValueError, match="fraction of working volume"):
        _dated_storage(min_inventory={"2027-04-01": 70})          # 70, meaning 70 %
    with pytest.raises(ValueError, match="outside the model's grid"):
        _dated_storage(min_inventory={"2030-01-01": 0.5})         # never seen by the DP
    with pytest.raises(ValueError, match="no admissible state"):
        _dated_storage(min_inventory={"2027-04-01": 0.8},
                       max_inventory={"2027-04-01": 0.3})


def test_an_unmeetable_inventory_floor_fails_rather_than_being_approximated():
    """A hard bound has no approximation to offer: either a policy exists or none does.

    A floor above what the injection rate can reach by that date is unreachable.
    While the tunnel was a `1000 * v_step` penalty the optimiser paid it and got
    as close as it could, and that near-miss was reported as a valuation of the
    contract asked for. Now no admissible schedule exists and the model says so.

    The message must not blame the terminal condition -- with hard bounds an
    infeasible contract also leaves no probability at any terminal state, so the
    order of the two checks is what makes the diagnosis useful.
    """
    # 30 days to fill, so 100 % by 15 January is unreachable from empty on 1 January.
    with pytest.raises(ValueError, match="no admissible policy exists"):
        _dated_storage(min_inventory={"2027-01-15": 1.0})


def test_inventory_bounds_round_towards_the_contract_never_away():
    """Rounding a bound to a grid state must not relax it.

    `round()` did, in both directions: on a ten-clip grid a 71 % floor became
    70 % and a 29 % ceiling became 30 %. Each is the contract the caller did not
    ask for, and because the post-check validated the ROUNDED bound it reported
    nothing. A floor now rounds up and a ceiling down, so the enforced contract
    is never weaker than the requested one.
    """
    def effective(kind, fraction, n_states=10):
        s = sm.Storage("2026-01-01", "2026-01-01", "2026-12-31", curve=None,
                       daily_curve=pd.Series(30.0, index=pd.date_range(
                           "2026-01-01", "2026-12-31", freq="D")),
                       n_p=0, v_step=10.0, sVol=0.5, sMR=1.0, clips_per_day=1)
        s.set_volume_states(n_states, initial_state=0)
        bounds = sm.apply_inventory_bounds(s, {f"{kind}_inventory": {"2026-06-01": fraction}})
        return bounds[0][4] / n_states

    assert effective("min", 0.71) == pytest.approx(0.80)      # up, was 0.70
    assert effective("max", 0.29) == pytest.approx(0.20)      # down, was 0.30
    # A fraction that lands on a state stays there: the tolerance is for the ulp
    # in 0.7 * 10 == 7.000000000000001, not for granting a clip either way.
    assert effective("min", 0.70) == pytest.approx(0.70)
    assert effective("max", 0.30) == pytest.approx(0.30)

    # And the rounding is reported rather than applied silently.
    s = sm.Storage("2026-01-01", "2026-01-01", "2026-12-31", curve=None,
                   daily_curve=pd.Series(30.0, index=pd.date_range(
                       "2026-01-01", "2026-12-31", freq="D")),
                   n_p=0, v_step=10.0, sVol=0.5, sMR=1.0, clips_per_day=1)
    s.set_volume_states(10, initial_state=0)
    table = sm.describe_inventory_bounds(
        s, sm.apply_inventory_bounds(s, {"min_inventory": {"2026-06-01": 0.71}}))
    assert table.loc[0, "requested"] == pytest.approx(0.71)
    assert table.loc[0, "effective"] == pytest.approx(0.80)
    assert table.loc[0, "moved by rounding"] == pytest.approx(0.09)


def test_a_bound_is_checked_against_the_inventory_actually_held():
    """A store that starts full and holds everything satisfies a 100 % floor.

    The old check rebuilt the balance as `cumsum(net moves)` and prepended the
    initial inventory to the FIRST opening balance only, so every later day was
    short by the entire opening stock. This exact contract -- start full, end
    full, flat curve, nothing worth doing -- was rejected as opening "with 0.00
    clips against 10" while the state distribution held all ten.
    """
    curve = pd.Series(30.0, index=pd.date_range("2026-01-01", "2026-12-31", freq="D"))
    params = dict(product_type="storage", valDate="2026-01-01",
                  storageStart="2026-01-01", storageEnd="2026-12-31",
                  v_step=10.0, inj_days=10, clips_per_day=1,
                  initial_inv_clips=10, terminal_inv_clips=10,
                  inj_cost=0.0, wdr_cost=0.0, vol=0.5, sMR=1.0, n_p_full=0,
                  run_intrinsic=False, discount_rate=0.0, daily_curve=curve,
                  min_inventory={"2026-01-03": 1.0})
    model, _ = sm.run_valuation(None, params)
    assert _inventory_law(model, "2026-01-03")[10] == pytest.approx(1.0)
    opening, _ = _opening_and_closing(model)
    assert opening["2026-01-03"] == pytest.approx(100.0)


def test_a_hard_bound_cannot_be_bought_out_of():
    """The bound must hold at any price, not merely at ordinary ones.

    While it was a `1000 * v_step` per-clip penalty, a deal worth more than the
    penalty simply paid it: at a EUR 10,000 price level 19.52 % of paths opened a
    floored day empty, and the checker -- comparing an EXPECTATION of 7.03 clips
    against a 2-clip floor -- accepted every one of them. Both halves are fixed
    here, so the test would fail if either regressed.
    """
    val, end = "2026-01-01", "2027-12-31"
    inj_day, floor_day = pd.Timestamp("2027-01-01"), pd.Timestamp("2027-06-02")
    n_states, floor_clips = 10, 2

    def priced_at(level):
        idx = pd.date_range(val, end, freq="D")
        prices = pd.Series(float(level), index=idx)
        prices.loc[inj_day] = 1.0                    # one cheap day to fill
        s = sm.Storage(val, val, end, curve=None, daily_curve=prices, n_p=0,
                       v_step=1.0, sVol=0.8, sMR=1.0, clips_per_day=n_states)
        s.set_volume_states(n_states, initial_state=0)
        s.i_curve = np.zeros(len(s.date_span), dtype=np.int64)
        s.w_curve = np.zeros(len(s.date_span), dtype=np.int64)
        s.i_curve[(inj_day - s.valDate).days] = n_states
        for sell in (pd.Timestamp("2027-06-01"), pd.Timestamp("2027-12-01")):
            s.w_curve[(sell - s.valDate).days] = n_states
        s.i_cost[:] = 0.0
        s.w_cost[:] = 0.0
        s.t_p_curve = np.full(s.n_op + 2, -1e9)
        s.t_p_curve[0] = 0.0
        s.mintunnel[(floor_day - s.valDate).days] = floor_clips
        s.n_p = 30
        return s.build()

    for level in (300.0, 1_000.0, 10_000.0):
        model = priced_at(level)
        law = _inventory_law(model, floor_day)
        assert law[:floor_clips].sum() == pytest.approx(0.0, abs=1e-12), (
            f"probability opened below the floor at a price level of {level:,.0f}")
        # The contract is still worth something: the optimiser found a feasible
        # alternative rather than the constraint making it infeasible.
        assert float(model.v[0, model.n_p, model.initial_state]) > 0.0


def _ratcheted_storage(n_states, ratchets=None):
    """A 30/60 store on a seasonal curve, optionally with a ratchet profile."""
    months = {1: 31.0, 2: 30.0, 3: 28.0, 4: 25.0, 5: 22.0, 6: 20.0,
              7: 19.0, 8: 19.5, 9: 22.0, 10: 25.5, 11: 28.5, 12: 30.5}
    span = pd.date_range("2026-01-01", "2029-06-30", freq="D")
    curve = pd.Series([months[d.month] for d in span], index=span)
    capacity = 600_000.0
    v_step = capacity / n_states
    inj_rate = max(1, round(n_states / 30))
    wdr_rate = max(1, round(n_states / 60))
    params = dict(product_type="storage", valDate="2026-06-01",
                  storageStart="2027-01-01", storageEnd="2027-12-31",
                  capacity_mwh=capacity, daily_max=inj_rate * v_step,
                  clips_per_day=inj_rate, inj_rate=inj_rate, wdr_rate=wdr_rate,
                  initial_inv_clips=0, terminal_inv_clips=0, inj_cost=0.0,
                  wdr_cost=0.0, vol=0.5, sMR=1.0, n_p_full=15, run_intrinsic=False,
                  discount_rate=0.10, daily_curve=curve,
                  max_ratchet_rate_loss=1.0)   # studying the grid, not trusting it
    if ratchets is not None:
        params["ratchets"] = pd.DataFrame(ratchets)
    model, _ = sm.run_valuation(None, params)
    return model, float(model.v[0, model.n_p, model.initial_state])


def test_a_ratchet_that_truncates_to_zero_clips_is_refused():
    """A slow rate must not become a stopped one behind the caller's back.

    The DP moves whole clips, so the kernel takes `int(rate * multiplier)`. A
    multiplier that is positive but floors to zero means "cannot move" where the
    caller meant "move slowly", and it fails silently: the store freezes and the
    deal prices at exactly zero with no error.

    An ordinary profile does it. Withdrawal at 1 clip/day with a 0.30 multiplier
    near empty gives 0.30 clips, floors to 0, and the store can never take out
    its first clip. On the 60-state grid this returned 0 EUR against 5,299,882
    unratcheted -- a 100 % loss that was entirely an artefact of the grid.
    """
    profile = {"fullness": [0.0, 0.5, 0.8, 1.0],
               "injection": [1.0, 1.0, 0.6, 0.3],
               "withdrawal": [0.3, 0.7, 1.0, 1.0]}

    with pytest.raises(ValueError, match="truncates to zero"):
        _ratcheted_storage(60, profile)

    # The message has to be actionable: it names the fix, not just the fault.
    try:
        _ratcheted_storage(60, profile)
    except ValueError as exc:
        text = str(exc)
        assert "clip(s)/day" in text and "n_states" in text, text
        assert "0.300" in text, text

    # On a grid fine enough to express it, the same profile prices -- and costs
    # real money, which is the point of modelling ratchets at all.
    _, ratcheted = _ratcheted_storage(240, profile)
    _, plain = _ratcheted_storage(240)
    assert ratcheted > 0.0, ratcheted
    assert 0.2 < 1.0 - ratcheted / plain < 0.8, (
        f"a realistic ratchet should cost a serious fraction: {1 - ratcheted/plain:.1%}")


def test_a_zero_ratchet_multiplier_is_left_alone():
    """Exactly zero is the legitimate way to shut a rate off at some fullness.

    Only a positive multiplier that floors to zero is a mistake; an explicit zero
    is a statement about the asset and passes through untouched.

    The knots matter, because `ratchet_arrays` interpolates: a profile ramping
    from 0 to 1 over several states puts intermediate multipliers inside the
    truncation zone and is refused, correctly. Put the knots on state boundaries
    -- here 0 and 1/60 on a 60-state grid -- and only the state you meant is shut
    off.
    """
    shut_off = {"fullness": [0.0, 1.0 / 60.0, 1.0],
                "injection": [1.0, 1.0, 1.0],
                "withdrawal": [0.0, 1.0, 1.0]}   # cannot withdraw from empty
    model, value = _ratcheted_storage(60, shut_off)
    assert value > 0.0, value
    assert model.w_ratch[0] == 0.0, model.w_ratch[0]
    assert model.w_ratch[1] == pytest.approx(1.0), model.w_ratch[1]

    # A ramp through the truncation zone is refused, and should be.
    ramp = {"fullness": [0.0, 0.05, 1.0],
            "injection": [1.0, 1.0, 1.0],
            "withdrawal": [0.0, 1.0, 1.0]}
    with pytest.raises(ValueError, match="truncates to zero"):
        _ratcheted_storage(60, ramp)

    # And no ratchet at all is unaffected by the check.
    _, plain = _ratcheted_storage(60)
    assert plain > 0.0


def _exit_ratcheted_store(n_states, ratcheted=True, **extra):
    """The reference ratcheted store at a chosen inventory clip.

    The PHYSICAL deal is held fixed as the grid is refined -- 600,000 MWh working
    volume, 20,000 MWh/day in, 10,000 MWh/day out -- so only the resolution moves.
    """
    months = {1: 30.0, 2: 30.0, 3: 24.9, 4: 25.0, 5: 25.0, 6: 25.0,
              7: 25.0, 8: 25.0, 9: 25.0, 10: 30.0, 11: 30.0, 12: 30.0}
    span = pd.date_range("2026-01-01", "2029-06-30", freq="D")
    curve = pd.Series([months[d.month] for d in span], index=span)
    profile = pd.DataFrame({"fullness": [0.0, 0.5, 0.8, 1.0],
                            "injection": [1.0, 1.0, 0.6, 0.3],
                            "withdrawal": [0.3, 0.7, 1.0, 1.0]})
    capacity, inj_mwh, wdr_mwh = 600_000.0, 20_000.0, 10_000.0
    v_step = capacity / n_states
    inj_rate, wdr_rate = round(inj_mwh / v_step), round(wdr_mwh / v_step)
    params = dict(product_type="storage", valDate="2026-06-01",
                  storageStart="2027-01-01", storageEnd="2027-12-31",
                  capacity_mwh=capacity, daily_max=inj_rate * v_step,
                  clips_per_day=inj_rate, inj_rate=inj_rate, wdr_rate=wdr_rate,
                  initial_inv_clips=0, terminal_inv_clips=0, inj_cost=0.0,
                  wdr_cost=0.0, vol=0.5, sMR=1.0, n_p_full=0, run_intrinsic=False,
                  discount_rate=0.10, daily_curve=curve,
                  max_ratchet_rate_loss=1.0)   # studying the grid, not trusting it
    if ratcheted:
        params["ratchets"] = profile
    params.update(extra)
    model, res = sm.run_valuation(None, params)
    opening, _ = _opening_and_closing(model)
    return float(opening.max()) / capacity, float(res["total_eur"])


def test_the_ratcheted_peak_is_a_property_of_the_grid_not_of_the_store():
    """The 52.5 % cap was discretisation, and this pins that it is.

    Recorded on 2026-09-10 as physical behaviour -- "ratchets cap a store through
    the exit, so it peaks at 52.5 % full" -- and disproved by the independent
    review the same evening. The rates are whole clips, so the kernel takes
    `int(rate * multiplier)`: at 10 % full the contract allows 3,800 MWh/day and
    a 240-clip grid delivers 2,500, a 34 % shortfall. The store then refuses to
    fill past what that crippled rate can drain.

    Refining the clip on the SAME physical deal roughly doubles the reachable
    peak and moves the value by 70 %. An independent continuous-volume
    reachability calculation puts the physical cap near 92.8 %.

    This is a defect under repair (roadmap P1.4), so the test pins the direction
    and the discretisation dependence rather than any particular number. It must
    not be turned back into an assertion about the store.
    """
    coarse_peak, coarse_value = _exit_ratcheted_store(240)
    fine_peak, fine_value = _exit_ratcheted_store(960)

    assert coarse_peak < 0.60, coarse_peak
    assert fine_peak > 0.80, fine_peak
    assert fine_value > 1.5 * coarse_value, (coarse_value, fine_value)

    # Unratcheted, the rates ARE exactly expressible and refinement is genuinely
    # answer-neutral. That is the control the original check mistook for proof.
    _, free_coarse = _exit_ratcheted_store(240, ratcheted=False)
    _, free_fine = _exit_ratcheted_store(480, ratcheted=False)
    assert free_fine == pytest.approx(free_coarse, rel=1e-12)


def test_a_floor_refused_on_a_coarse_grid_may_be_reachable_on_a_finer_one():
    """A grid rejection is not evidence of physical infeasibility.

    The 70 % October floor was recorded as jointly infeasible with the ratchets
    and therefore physical. It is met on a 480-clip grid with the same rates,
    ratchets, dates and curve. Until a continuous-volume argument says otherwise,
    a refusal at one resolution is a statement about that resolution.
    """
    floor = {"min_inventory": {"2027-10-01": 0.70}}
    with pytest.raises(ValueError, match="no admissible policy exists"):
        _exit_ratcheted_store(240, **floor)

    peak, value = _exit_ratcheted_store(480, **floor)
    assert peak >= 0.70, peak
    assert value > 0.0

    # The refusal must point at the grid, not leave the reader concluding physics.
    with pytest.raises(ValueError, match="reachable on a finer one"):
        _exit_ratcheted_store(240, **floor)


def _fuelled_storage(fuel_loss, n_p=15):
    months = {1: 30.0, 2: 30.0, 3: 24.9, 4: 25.0, 5: 25.0, 6: 25.0,
              7: 25.0, 8: 25.0, 9: 25.0, 10: 30.0, 11: 30.0, 12: 30.0}
    span = pd.date_range("2026-01-01", "2029-06-30", freq="D")
    curve = pd.Series([months[d.month] for d in span], index=span)
    model, _ = sm.run_valuation(None, dict(
        product_type="storage", valDate="2026-06-01", storageStart="2027-01-01",
        storageEnd="2027-12-31", capacity_mwh=600_000.0, daily_max=20_000.0,
        clips_per_day=2, inj_rate=2, wdr_rate=1, initial_inv_clips=0,
        terminal_inv_clips=0, inj_cost=0.0, wdr_cost=0.0, vol=0.5, sMR=1.0,
        n_p_full=n_p, run_intrinsic=False, discount_rate=0.10, daily_curve=curve,
        fuel_loss=fuel_loss))
    return model


def test_fuel_loss_charges_the_gas_it_retains():
    """Injecting a clip buys 1/(1 - fuel_loss) clips; the excess is burnt.

    Real storage retains 1-2 % of injected gas for compression, and the model had
    no way to say so -- `inj_cost` is a fixed EUR/MWh and the loss is taken in
    kind, so it scales with the price. The charge therefore lands on the price
    leg in the kernel, not the cost leg.

    Default 0.0 leaves every existing valuation untouched.
    """
    free = _fuelled_storage(0.0)
    burnt = _fuelled_storage(0.015)
    free_value = float(free.v[0, free.n_p, free.initial_state])
    burnt_value = float(burnt.v[0, burnt.n_p, burnt.initial_state])

    assert burnt_value < free_value, (burnt_value, free_value)
    assert 0.03 < 1.0 - burnt_value / free_value < 0.2, (
        f"1.5 % retention should cost a few per cent of value, got "
        f"{1 - burnt_value/free_value:.1%}")

    with pytest.raises(ValueError, match="fraction of injected gas"):
        _fuelled_storage(1.5)          # 1.5, meaning 1.5 %


def test_fuel_loss_separates_the_gas_stored_from_the_gas_traded():
    """`exp_ex` stays physical; `delta` becomes the market volume. Decision D-O3.

    With fuel loss the two part company: putting one clip into inventory takes
    1/(1 - loss) clips out of the market. The hedge is what you trade, so `delta`
    carries the multiplier on the injection leg and nothing on the withdrawal
    leg. On an `n_p = 0` tree `E[S | exercise] = F`, so the ratio is exact.

    The store still gives back exactly what it takes -- physical in equals
    physical out -- which is what makes the two series distinguishable at all.
    """
    for loss in (0.0, 0.015, 0.03):
        model = _fuelled_storage(loss, n_p=0)
        n = model.n_t
        physical = np.asarray(model.exp_ex[:n])
        delta = np.asarray(model.delta[:n])

        injecting = physical < -1e-9
        withdrawing = physical > 1e-9
        assert injecting.any() and withdrawing.any()

        into_store = float(np.abs(physical[injecting]).sum())
        out_of_store = float(np.abs(physical[withdrawing]).sum())
        assert into_store == pytest.approx(out_of_store, rel=1e-9), (into_store, out_of_store)

        ratio_in = float(np.abs(delta[injecting]).sum() / into_store)
        ratio_out = float(np.abs(delta[withdrawing]).sum() / out_of_store)
        assert ratio_in == pytest.approx(1.0 / (1.0 - loss), rel=1e-9), (loss, ratio_in)
        assert ratio_out == pytest.approx(1.0, rel=1e-9), (loss, ratio_out)


def test_the_repricing_identity_survives_fuel_loss():
    """`sum(DF * delta * F) == V0` still closes, because delta is the traded volume.

    This is the check that D-O3 is the right convention rather than merely a
    plausible one: had `delta` stayed the inventory volume, the identity would
    have needed a separate fuel term and the reported hedge would not have
    repriced the deal.
    """
    for loss in (0.0, 0.005, 0.015, 0.03):
        model = _fuelled_storage(loss)
        n = model.n_t
        value = float(model.v[0, model.n_p, model.initial_state])
        reprice = float(np.dot(model.d_curve[:n] * np.asarray(model.delta[:n]),
                               np.asarray(model.fwd)[:n]))
        assert abs(reprice - value) / max(abs(value), 1.0) < 1e-9, (loss, reprice, value)


# ── P1.4: the discretisation loss, made visible and gated ─────────────────────

def test_the_zero_rate_guard_is_not_an_accuracy_certificate():
    """`assert_ratchets_expressible` passes while the rate is a third too slow.

    It catches only `int(rate * multiplier) == 0`. Everything short of zero was
    rounded down in silence, and a silently slower store fills less, so the deal
    was under-valued with no indication anywhere. On the SHIPPED notebook profile
    -- the mild one, chosen so the ratchets and the 70 % floor could coexist --
    the worst withdrawal level loses about a third of its contractual rate.
    """
    import benchmarks

    model, _ = sm.run_valuation(None, benchmarks.CASES["shipped-30-60"]())
    sm.assert_ratchets_expressible(model)                 # passes: nothing floors to zero

    worst = sm.worst_ratchet_rate_loss(model)
    assert worst["withdrawal"] > 0.30, worst
    assert worst["injection"] > 0.30, worst

    table = sm.describe_ratchet_rates(model)
    at = int(table["withdrawal loss"].idxmax())
    contract = table["withdrawal contract MWh/day"][at]
    grid = table["withdrawal grid MWh/day"][at]
    assert grid < contract
    assert grid == pytest.approx(np.floor(contract / model.v_step) * model.v_step)

    # The loss is a sawtooth, not a bias: exactly zero wherever rate x multiplier
    # lands on an integer. A profile can therefore look fine at one fullness and
    # be badly wrong at the next.
    assert float(np.nanmin(table["withdrawal loss"].to_numpy())) == pytest.approx(0.0)

    # And the opt-in gate fires where the default does not.
    with pytest.raises(ValueError, match="understated by"):
        sm.assert_ratchet_rates_expressible(model, max_relative_loss=0.05)


def test_the_ratchet_diagnostic_checks_every_active_rate_not_only_the_fastest():
    """Until 2026-09-11 both the guard and the diagnostic read `curve.max()` --
    the single largest active-day rate across the WHOLE date span -- so a
    contract with more than one active rate only ever had its fastest day
    checked. `IMPLEMENTATION-GUIDE-2026-09-11.md` §14 found this worse than
    first described: not just the diagnostic, but `assert_ratchets_expressible`
    itself, the guard built specifically to catch "cannot move at all".

    An alternating 1-/2-clip-day schedule at a 0.5 multiplier is the sharpest
    example: the 2-clip days are exact (`int(2*0.5)=1`), so `curve.max()`
    alone reports zero loss, while the 1-clip days floor to zero and cannot
    move at all -- a 100 % loss, and a genuine zero-rate truncation the guard
    exists to catch. This is the acceptance pack's own R-daily_ratchet oracle,
    reproduced here independently as a repository-level regression test.
    """
    from types import SimpleNamespace

    model = SimpleNamespace(n_op=5, v_step=1.0, i_curve=np.array([1, 2]),
                            w_curve=np.array([1, 2]), i_ratch=np.full(5, 0.5),
                            w_ratch=np.full(5, 0.5))

    worst = sm.worst_ratchet_rate_loss(model)
    assert worst["injection"] == pytest.approx(1.0), worst
    assert worst["withdrawal"] == pytest.approx(1.0), worst

    with pytest.raises(ValueError, match="truncates to zero"):
        sm.assert_ratchets_expressible(model)

    # And the reference store, which has only one active rate per side, must
    # reproduce EXACTLY what was measured before this fix -- the worst-across-
    # distinct-rates logic has to collapse to the single-rate case unchanged.
    import benchmarks

    reference, _ = sm.run_valuation(None, benchmarks.CASES["shipped-30-60"]())
    reference_worst = sm.worst_ratchet_rate_loss(reference)
    assert reference_worst["injection"] == pytest.approx(0.32203389830508455)
    assert reference_worst["withdrawal"] == pytest.approx(0.33110367892976594)


def test_headroom_is_not_reported_as_discretisation_loss():
    """A state near the boundary that can only ever move its remaining
    headroom is EXACT, not a rounding shortfall -- the kernel caps the
    contractual rate by headroom BEFORE flooring to a whole clip
    (`storage_kernels.all_inj_steps`/`all_wdr_steps`), so a headroom-limited
    level is already an integer and contributes zero loss. Conflating the two
    would call ordinary physics -- a nearly full store injects less -- a
    numerical defect.

    Ten clips, base rate 8/day (injection) and 4/day (withdrawal), no ratchet
    (multiplier 1 everywhere): the two states nearest each edge are headroom-
    limited (2 and 3 clips of room respectively, both integers, so exact), and
    every interior state has full headroom and is also exact, since the base
    rate itself is already a whole number of clips.
    """
    from types import SimpleNamespace

    model = SimpleNamespace(n_op=11, v_step=1.0,
                            i_curve=np.full(1, 8.0), w_curve=np.full(1, 4.0),
                            i_ratch=np.ones(11), w_ratch=np.ones(11))
    table = sm.describe_ratchet_rates(model)

    # Full store (clips=10): zero headroom to inject -- null, not zero, since
    # there is nothing to move at all.
    assert np.isnan(table["injection loss"].iloc[-1])
    # One clip from full (clips=9): headroom=1, contract=8, allowed=min(8,1)=1,
    # already an integer -- headroom-limited, exact, reported as 0 % not null.
    assert table["injection loss"].iloc[-2] == pytest.approx(0.0)
    assert table["injection grid MWh/day"].iloc[-2] == pytest.approx(1.0)
    # Empty store (clips=0): zero headroom to withdraw -- null.
    assert np.isnan(table["withdrawal loss"].iloc[0])
    # Interior states: full headroom, base rate already whole -- exact.
    assert table["injection loss"].iloc[2:-2].fillna(0).eq(0.0).all()
    assert table["withdrawal loss"].iloc[2:-2].fillna(0).eq(0.0).all()


def test_refining_the_clip_is_free_only_when_the_rates_are_already_exact():
    """The claim held in the one case that was checked, and nowhere else."""
    import benchmarks

    exact = benchmarks.inventory_grid_ladder(ladder=(60, 240), n_p=8)
    assert exact["total_eur"].nunique() == 1, exact["total_eur"].tolist()
    assert float(exact["worst wdr rate loss"].max()) == pytest.approx(0.0)

    ratcheted = benchmarks.inventory_grid_ladder(
        ladder=(240, 480), ratchets=benchmarks.SOFT_RATCHETS, n_p=8)
    lo, hi = ratcheted["total_eur"].tolist()
    assert hi > lo * 1.005, (lo, hi)
    assert ratcheted["worst wdr rate loss"].iloc[1] < ratcheted["worst wdr rate loss"].iloc[0]


def test_the_convergence_gate_fails_the_grid_it_should():
    """A gate that passes everything is not a gate.

    The declared threshold is a proposed engineering one, not an achieved result
    and not a commercial tolerance: total AND intrinsic within 0.5 % over each of
    two successive doublings.
    """
    import benchmarks

    exact = benchmarks.inventory_grid_ladder(ladder=(60, 120, 240, 480), n_p=8)
    status, steps, message = benchmarks.convergence_verdict(exact)
    assert status == "within_declared_tolerance", (status, message, steps)
    assert float(steps["total_eur_relative"].max()) == pytest.approx(0.0, abs=1e-12)

    coarse = benchmarks.inventory_grid_ladder(
        ladder=(240, 480, 960), ratchets=benchmarks.SOFT_RATCHETS, n_p=8)
    status, steps, message = benchmarks.convergence_verdict(coarse)
    assert status == "outside_tolerance", (status, message, steps)

    # Two points cannot show two successive doublings, so the verdict is
    # "insufficient evidence", not "yes by default" and not a bare "no" either
    # -- there is a real difference between "measured and diverged" and
    # "never measured enough to judge".
    status, _, _ = benchmarks.convergence_verdict(coarse.head(2))
    assert status == "insufficient", status


@pytest.mark.parametrize("name,grids,values,expected_status,refused", [
    ("nan", [240, 480, 960], [math.nan] * 3, "invalid", None),
    ("infinity", [240, 480, 960], [math.inf] * 3, "invalid", None),
    ("duplicate_grid", [240, 240, 240], [1e6] * 3, "invalid", None),
    ("descending_grid", [960, 480, 240], [1e6] * 3, "invalid", None),
    ("not_a_doubling", [240, 500, 960], [1e6] * 3, "insufficient", None),
    ("refused_row_leaves_a_gap", [240, 480, 960, 1920],
     [1e6, math.nan, 1e6, 1e6], "insufficient",
     [None, "unrepresentable grid", None, None]),
    ("material_value_small_move", [240, 480, 960],
     [10_000.0, 9_500.0, 9_000.0], "outside_tolerance", None),
    ("stable_control", [240, 480, 960],
     [1e6, 1_002_000.0, 1_003_000.0], "within_declared_tolerance", None),
    ("moving_control", [240, 480, 960],
     [1e6, 1_100_000.0, 1_200_000.0], "outside_tolerance", None),
])
def test_the_convergence_gate_reproduces_every_acceptance_pack_case(
        name, grids, values, expected_status, refused):
    """Nine cases from `IMPLEMENTATION-GUIDE-2026-09-11.md`'s own acceptance
    pack (`check_acceptance.py`'s `gate()` helper), reproduced as a repository
    test independent of that external script. Before this fix, six of the
    first seven passed when they should have refused or rejected: NaN/inf
    compare False against any tolerance; duplicate or descending grids were
    never checked for being sorted; a refused row was filtered out and the
    gap it left silently bridged; and an unconditional absolute-EUR fallback
    let a flat 5 % relative move through because the EUR amount was small.
    The two controls (stable/moving) must keep giving the right answer
    throughout -- a fix that breaks the case it was not aimed at is not a fix.
    """
    import benchmarks

    table = pd.DataFrame(dict(n_states=grids, total_eur=values, intrinsic_eur=values))
    if refused is not None:
        table["refused"] = refused
    status, steps, message = benchmarks.convergence_verdict(table)
    assert status == expected_status, (name, status, message, steps)


def test_the_price_tree_boundary_width_is_checked_separately_from_the_inventory_grid():
    """So that refining one cannot conceal error in the other, or be blamed for it.

    `price_grid_ladder` refines `n_p` (the tree's price-boundary width) at a
    FIXED daily time step -- see its docstring for why that is narrower than
    "price discretisation" in general (IMPLEMENTATION-GUIDE-2026-09-11.md
    §7.4). It is converged at the shipped width regardless.
    """
    import benchmarks

    table = benchmarks.price_grid_ladder(
        ladder=(15, 20, 25), n_states=240, ratchets=benchmarks.SOFT_RATCHETS)
    values = table["total_eur"].to_numpy()
    spread = (values.max() - values.min()) / abs(values.mean())
    assert spread < 0.005, table          # converged in n_p at the shipped width


# ── The spread comparison, executable rather than quoted ──────────────────────

def test_the_realised_forward_statistics_reproduce():
    """The correlations the P4.1 case rests on, from named inputs.

    These reproduce an independent review's figures to six decimals, on different
    NumPy/pandas/Numba versions. The magnitudes built on them did not, which is
    why the calculation now lives in `benchmarks.py` instead of in prose.
    """
    import benchmarks

    stats = benchmarks.spread_statistics().set_index("pair")
    expected = {"c1/c3": 0.917095, "c1/c6": 0.801426, "c6/c12": 0.770636,
                "c1/c12": 0.742070, "c12/c24": 0.802850, "c1/c24": 0.632737}
    for pair, corr in expected.items():
        assert stats.loc[pair, "correlation"] == pytest.approx(corr, abs=5e-7), pair
    assert stats.loc["c6/c12", "log_ratio_vol"] == pytest.approx(0.373101, abs=5e-6)

    # A one-factor model forces every one of these to 1.000, which is the finding
    # that survived. Excluding rolls does not remove the mismatch.
    no_roll = benchmarks.spread_statistics(exclude_rolls=True).set_index("pair")
    assert no_roll.loc["c6/c12", "correlation"] < 0.80
    assert no_roll.loc["c6/c12", "observations"] < stats.loc["c6/c12", "observations"]


def test_the_model_spread_volatility_is_not_monotone_in_mean_reversion():
    """Which is why a sign flip cannot be an acceptance test.

    `sigma * |exp(-k t1) - exp(-k t2)|` tends to zero at both ends -- the loadings
    meet at 1 as k -> 0 and at 0 as k -> infinity -- so it peaks in between. The
    P4.1 design proposed "extrinsic must fall with mean reversion" as the gate for
    the whole item; it would reject a correct model.
    """
    import benchmarks

    grid = np.linspace(0.01, 20.0, 2000)
    vols = np.array([benchmarks.model_log_ratio_vol(0.5, k, 0.5, 1.0) for k in grid])
    peak = float(grid[int(vols.argmax())])
    assert 1.0 < peak < 2.0, peak
    assert vols[0] < vols.max() and vols[-1] < vols.max()

    # The published comparison, with its assumptions attached.
    got = benchmarks.spread_comparison(sigma=0.50, kappa=1.0, pair=(6, 12))
    assert got["model_log_ratio_vol"] == pytest.approx(0.119326, abs=1e-6)
    assert 3.0 < got["ratio_raw"] < 3.3, got            # about 3.1x, not nine
    assert 3.0 < got["ratio_excluding_rolls"] < 3.3, got


# ── P4.1: what a common long factor is and is not worth ───────────────────────

def test_a_common_long_factor_is_worth_nothing_to_a_homogeneous_contract():
    """The result that removed P4.1's storage rationale, pinned.

    Zero-fee storage cashflows are homogeneous of degree one in price and the
    admissible set does not depend on price, so the optimal policy is
    scale-invariant, `V_t = L_t * W_t` for the long factor's own martingale `L`,
    and `L_0 = 1`. The factor integrates out. At a FIXED short factor the value
    is therefore independent of the long factor's volatility -- not approximately,
    exactly -- and an unstruck swing behaves the same way for the same reason.
    """
    import two_factor_probe as probe

    for kappa in (0.2, 1.0, 4.0):
        for value_of in (lambda s: probe.value_store(kappa, s),
                         lambda s: probe.value_swing(kappa, s, strike=0.0)):
            base = value_of(0.0)
            for sig_xi in (0.1, 0.8):
                assert value_of(sig_xi) == pytest.approx(base, rel=1e-12), (kappa, sig_xi)


def test_a_calibrated_second_factor_takes_volatility_out_of_the_short_one():
    """Which is the comparison that matters, and it reverses the storage answer.

    Adding a long factor on top of an unchanged `sigma_chi` is not a second
    factor, it is more volatility -- and more volatility is worth more to any
    option. A jointly calibrated model fits the same observed variance with both
    factors, so `sigma_chi` comes down. Then a store LOSES value, because it
    monetises short-horizon variance and that is precisely what moved. The test
    above would tell you storage is unaffected; the channel is the short-factor
    estimate, not the extra state.
    """
    import two_factor_probe as probe

    for kappa, floor in ((1.0, 0.004), (4.0, 0.02)):   # 0.63 % and 2.88 % measured
        chi = probe.matched_sig_chi(0.3, "spot")
        one = probe.value_store(kappa, 0.0)
        two = probe.value_store(kappa, 0.3, sig_chi=chi)
        assert (one - two) / one > floor, (kappa, one, two)

    # And it fades as mean reversion does: at slow reversion the two factors are
    # nearly the same process, so which one holds the variance barely matters.
    slow = probe.value_store(0.2, 0.3, sig_chi=probe.matched_sig_chi(0.3, "spot"))
    assert abs(slow - probe.value_store(0.2, 0.0)) / slow < 1e-4, slow


def test_what_a_second_factor_is_worth_depends_on_the_calibration_anchor():
    """Down to its sign, which is why no scalar anchor settles this.

    `S - K` is not homogeneous, so a struck swing does depend on the long factor.
    But hold instantaneous spot variance fixed and it GAINS, because the walk's
    variance accumulates where the OU factor's saturates; hold terminal variance
    fixed and it LOSES, because that is exactly what the anchor removes. Same
    contract, same sigma_xi, opposite conclusions -- so the two factors differ in
    their variance term structure, which a real calibration fits and no single
    number captures.
    """
    import two_factor_probe as probe

    kappa, sig_xi = 4.0, 0.10
    base = probe.value_swing(kappa, 0.0, strike=20.0)
    by_spot = probe.value_swing(
        kappa, sig_xi, strike=20.0, sig_chi=probe.matched_sig_chi(sig_xi, "spot"))
    by_terminal = probe.value_swing(
        kappa, sig_xi, strike=20.0,
        sig_chi=probe.matched_sig_chi(sig_xi, "terminal", kappa=kappa))

    assert by_spot > base, (base, by_spot)
    assert by_terminal < base, (base, by_terminal)
    assert probe.matched_sig_chi(sig_xi, "terminal", kappa=kappa) <         probe.matched_sig_chi(sig_xi, "spot") < probe.SIG_CHI

    # The spot anchor is horizon- and kappa-free; the terminal one is neither.
    assert probe.matched_sig_chi(0.3, "spot") == pytest.approx(np.sqrt(0.36 - 0.09))
    tight = probe.matched_sig_chi(0.3, "terminal", kappa=4.0)
    assert not np.isfinite(tight), tight        # the budget is already spent


def test_the_probe_would_catch_its_own_lattice_going_wrong():
    """An absorbing long-factor boundary breaks the martingale and fakes a gain.

    It did: the first run of this probe reported about 1e-5 of spurious storage
    gain, which is the size of a real effect worth looking for. The default
    half-width is past anything reachable in N_T steps; a deliberately narrow one
    must therefore produce a difference where the correct lattice produces none.
    """
    import two_factor_probe as probe

    narrow = probe._walk_lattice(0.8, half=4)
    real = probe._walk_lattice(0.8)
    assert narrow[0].size < real[0].size
    # Mass reaches the narrow edge, so exp(xi) stops being a martingale there.
    row = np.zeros(narrow[0].size)
    row[narrow[0].size // 2] = 1.0
    for _ in range(probe.N_T - 1):
        row = row @ narrow[1]
    drifted = float((row * np.exp(narrow[0])).sum())
    assert drifted < np.cosh(0.8 * np.sqrt(probe.DT)) ** (probe.N_T - 1) - 1e-6, drifted


def test_the_fixtures_depend_only_on_files_the_repository_carries():
    """A fixture that reads an untracked file is reproducible on one machine only.

    `benchmarks.py` pointed at `ttf q.parquet`, which is a gitignored local cache
    derived from `ttf q.xlsx`. Everything passed locally and CI failed on the
    first push, which is the good outcome -- but the whole point of these
    fixtures is that someone else can reconstruct the numbers, so the dependency
    is worth asserting rather than remembering.
    """
    import benchmarks

    tracked = set(subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
        check=True).stdout.split("\n"))

    for name, path in (("WORKBOOK", benchmarks.WORKBOOK),):
        relative = os.path.relpath(path, ROOT).replace(os.sep, "/")
        assert relative in tracked, f"benchmarks.{name} reads untracked {relative}"

    # The cache is legitimate, but only as a cache: the workbook must be enough.
    assert not benchmarks.WORKBOOK.endswith(".parquet"), benchmarks.WORKBOOK
    assert os.path.exists(benchmarks.WORKBOOK), benchmarks.WORKBOOK


# ── S1: the physical contract must survive grid conversion, or be refused ─────
#
# Guide: docs/IMPLEMENTATION-GUIDE-2026-09-11.md, checklist items 1-3. These use
# the ORIGINAL PHYSICAL INPUTS as the oracle -- capacity, MWh/day, MWh -- not
# the rounded clip output, which is exactly the mistake
# test_asymmetric_storage_rates_survive_the_days_to_rate_conversion used to
# make (it asserted wdr_rate == 1 at N=30, i.e. that 30/60 silently prices
# 30/30, as correct). That test is corrected below, in the same commit as
# normalise_storage_contract() -- keeping them apart would mean landing a
# storage_model.py change that the existing suite could not have passed.
#
# These eleven cases were first written and committed as `xfail(strict=True)`,
# proven to fail against the old max(1, round(...)) conversion, THEN
# normalise_storage_contract() was implemented, and only then were the marks
# removed here -- confirmed by every one of them reporting XPASS(strict)
# first, which is what forced this edit rather than left it optional.

_GRID_TOL_MWH = 1e-7


def _physical_storage_params(n_states, inj_days=30, wdr_days=60, **overrides):
    """A storage contract stated the way a term sheet states it: capacity, days
    to fill/empty, opening/terminal MWh. `n_states` is the ONLY numerical knob;
    everything else is what the deal actually says."""
    params = dict(product_type="storage", n_states=n_states, capacity_mwh=600_000.0,
                  inj_days=inj_days, wdr_days=wdr_days,
                  initial_storage_mwh=0.0, terminal_storage_mwh=0.0)
    params.update(overrides)
    return params


def _close(actual, requested, tol=_GRID_TOL_MWH):
    return math.isfinite(actual) and abs(actual - requested) <= tol + 1e-10 * abs(requested)


def _preserves_or_refuses(params):
    """Run the physical contract through `params_for_run_valuation`.

    Returns (ok, detail). `ok` is True if the function either (a) raises --
    refusing a request the grid cannot express is an acceptable outcome -- or
    (b) returns rates and boundary inventory that reproduce the REQUESTED
    physical quantities exactly. `ok` is False only for the one behaviour that
    must not survive: silently returning a DIFFERENT contract with no error.
    """
    try:
        out = sm.params_for_run_valuation(params)
    except ValueError as exc:
        return True, {"outcome": "refused", "message": str(exc)}

    v_step = float(out["v_step"])
    requested = {
        "capacity_mwh": params["capacity_mwh"],
        "injection_mwh_day": params["capacity_mwh"] / params["inj_days"],
        "withdrawal_mwh_day": params["capacity_mwh"] / params["wdr_days"],
        "opening_mwh": params["initial_storage_mwh"],
        "terminal_mwh": params["terminal_storage_mwh"],
    }
    effective = {
        "capacity_mwh": v_step * int(out["n_states"]),
        "injection_mwh_day": v_step * float(out["inj_rate"]),
        "withdrawal_mwh_day": v_step * float(out["wdr_rate"]),
        "opening_mwh": v_step * float(out["initial_inv_clips"]),
        "terminal_mwh": v_step * float(out["terminal_inv_clips"]),
    }
    ok = all(_close(effective[k], requested[k]) for k in requested)
    return ok, {"outcome": "accepted", "requested": requested, "effective": effective}


@pytest.mark.parametrize("n_states,inj_days,wdr_days", [
    (60, 30, 60), (90, 30, 90), (390, 30, 65), (780, 30, 65), (2190, 30, 365),
])
def test_a_grid_that_can_express_the_rates_does(n_states, inj_days, wdr_days):
    """The control. `lcm(inj_days, wdr_days)`-compatible grids already work
    today -- if this fails, something else broke, not the conversion defect."""
    ok, detail = _preserves_or_refuses(
        _physical_storage_params(n_states, inj_days, wdr_days))
    assert ok, detail


@pytest.mark.parametrize("n_states,inj_days,wdr_days", [
    (30, 30, 90), (60, 30, 65), (45, 30, 60), (90, 30, 65), (30, 30, 365),
])
def test_a_grid_that_cannot_express_the_rates_is_refused_not_guessed(
        n_states, inj_days, wdr_days):
    """`max(1, round(n_states/days))` had no failure mode: it always returned
    something, and that something was a different, unstated contract.

    30/90 at N=30 was the sharpest example: withdrawal rounded from a
    requested 6,666.67 MWh/day to 20,000.00 -- the FULL injection rate, three
    times over -- and priced EUR 131,369 (5.51 %) above the 30/90 contract
    that was asked for, with no error and no diagnostic that fired.
    `normalise_storage_contract()` now refuses every one of these instead.
    """
    ok, detail = _preserves_or_refuses(
        _physical_storage_params(n_states, inj_days, wdr_days))
    assert ok, (
        f"N={n_states} for {inj_days}/{wdr_days} silently priced a different "
        f"contract instead of refusing it: {detail}")


@pytest.mark.parametrize("field,mwh", [
    ("initial_storage_mwh", 20_000.0),   # exact
    ("terminal_storage_mwh", 30_000.0),  # exact
    ("initial_storage_mwh", 5_000.0),    # half a clip -- must refuse
    ("terminal_storage_mwh", 5_000.0),   # half a clip -- must refuse
])
def test_boundary_inventory_is_preserved_or_refused_not_rounded_to_a_clip(field, mwh):
    """5,000 MWh opening inventory in a 600,000 MWh / 60-clip store is half a
    clip. `round(0.5)` on Python's banker's rounding used to give 0, so a term
    sheet's "start half-full of a clip" silently became "start EMPTY" -- the
    same class of contract change as the day-rate rounding above, on the
    quantity the deal actually opens and closes with.

    Correct behaviour is the SAME for every row: either the requested MWh
    survives exactly, or the request is refused. There is no third row where
    silently rounding to a different quantity is acceptable -- so the
    parametrised expectation is `ok` in every case, not `ok` for the values
    that happen to round cleanly today.
    """
    ok, detail = _preserves_or_refuses(_physical_storage_params(60, **{field: mwh}))
    assert ok, (field, mwh, detail)


@pytest.mark.parametrize("physical_field,clip_field,clip_value", [
    ("initial_storage_mwh", "initial_inv_clips", 1),     # 20,000 MWh vs 1 clip = 10,000
    ("terminal_storage_mwh", "terminal_inv_clips", 2),   # 30,000 MWh vs 2 clips = 20,000
])
def test_an_explicit_clip_count_disagreeing_with_its_own_mwh_field_is_refused(
        physical_field, clip_field, clip_value):
    """Supplying both `initial_storage_mwh=20000` and `initial_inv_clips=1` at a
    10,000 MWh clip is contradictory -- 20,000 MWh is 2 clips, not 1. Nothing
    used to check the two fields against each other: `initial_inv_clips`, once
    set, was left alone and `initial_storage_mwh` silently ignored. An
    explicit field is not permission to discard the other.
    """
    mwh = 20_000.0 if "initial" in physical_field else 30_000.0
    params = _physical_storage_params(60, **{physical_field: mwh, clip_field: clip_value})
    ok, detail = _preserves_or_refuses(params)
    assert ok, (
        f"{physical_field}={mwh} conflicts with {clip_field}={clip_value} "
        f"(= {clip_value * 10_000.0:,.0f} MWh) and neither raised nor disagreed: {detail}")


def test_an_explicit_grid_does_not_silently_round_the_requested_capacity():
    """`v_step=30, clips_per_day=1` with `capacity_mwh=100` cannot land on a
    whole number of clips -- 100/30 is not an integer. `resolve_grid` used to
    return `round(100/30) = 3` states with no check, an effective capacity of
    90 MWh against the 100 requested, with no error naming the 10 % that
    vanished. It now raises.
    """
    try:
        v_step, n_states, _ = sm.resolve_grid(
            dict(capacity_mwh=100.0, v_step=30.0, clips_per_day=1), "inj_days")
    except ValueError:
        return                                    # refusing is an acceptable outcome
    assert _close(v_step * n_states, 100.0), (
        f"requested capacity 100 MWh, grid gives {v_step * n_states:,.1f} MWh "
        f"({v_step=}, {n_states=}) with no error")


def test_conflicting_explicit_capacity_and_state_count_are_not_silently_resolved():
    """`capacity_mwh=600,000`, `v_step=10,000` and `n_states=90` disagree with
    each other -- 10,000 x 90 = 900,000, not 600,000. `resolve_grid` used to
    branch on `capacity_mwh` being present and return `round(600000/10000) =
    60` states, discarding the caller's explicit `n_states=90` outright rather
    than flagging that the two inputs do not agree. It now raises.
    """
    try:
        v_step, n_states, _ = sm.resolve_grid(
            dict(capacity_mwh=600_000.0, v_step=10_000.0, n_states=90,
                clips_per_day=2), "inj_days")
    except ValueError:
        return
    assert n_states == 90, (
        f"caller explicitly requested n_states=90; resolve_grid silently "
        f"substituted {n_states} derived from capacity_mwh instead, with no error")


# ── S3: the quote-cache identity defect ────────────────────────────────────────
#
# IMPLEMENTATION-GUIDE-2026-09-11.md checklist items 7-9. Same defect class as
# S1's grid conversion: a lookup trusting something OTHER than the content it
# is supposed to represent -- there, a path string and max(1, round(...)); here,
# a path string and a modification time. `quote_data.py` fixes it by addressing
# the cache with a SHA-256 of the actual bytes, so two different files cannot
# collide on one cache entry and a byte-identical re-read always finds its own.

def test_a_different_workbook_at_the_same_cache_dir_is_not_confused_with_the_default():
    """Until 2026-09-11, `benchmarks.load_quote_matrix` checked a SINGLE fixed
    parquet path regardless of what was actually requested. A synthetic
    one-row workbook with TTFc1=999, given an older modification time than the
    repository's own cache, returned the repository's 4,171 rows with
    TTFc1=10 instead -- a request for a different dataset silently analysed
    the original one.
    """
    import quote_data as qd

    with tempfile.TemporaryDirectory() as tmp:
        shared_cache_dir = os.path.join(tmp, "cache")
        os.makedirs(shared_cache_dir)

        default_wb = os.path.join(tmp, "default.xlsx")
        pd.DataFrame({"quote_date": pd.date_range("2020-01-01", periods=10),
                     "TTFc1": range(10)}).to_excel(default_wb, index=False)
        warm, _ = qd.load_quote_matrix(default_wb, cache_dir=shared_cache_dir)
        assert len(warm) == 10

        other_wb = os.path.join(tmp, "other.xlsx")
        pd.DataFrame({"quote_date": [pd.Timestamp("2020-01-01")],
                     "TTFc1": [999.0]}).to_excel(other_wb, index=False)
        old = pd.Timestamp("2000-01-01").timestamp()
        os.utime(other_wb, (old, old))          # older than the "default" cache entry

        got, provenance = qd.load_quote_matrix(other_wb, cache_dir=shared_cache_dir)
        assert len(got) == 1, (
            f"requested a 1-row workbook and got {len(got)} rows -- "
            f"the cache substituted a different source")
        assert got["TTFc1"].iloc[0] == 999.0
        assert provenance["cache"] == "rebuilt"
        assert provenance["source_sha256"] == qd.source_fingerprint(other_wb)


def test_editing_a_workbook_in_place_invalidates_its_cache_even_with_mtime_restored():
    """Modification time is not identity. A same-path edit with its mtime
    restored -- routine after some editor saves, and after certain
    version-control checkouts -- stayed silently stale under the old policy.
    """
    import quote_data as qd

    with tempfile.TemporaryDirectory() as tmp:
        wb = os.path.join(tmp, "mine.xlsx")
        pd.DataFrame({"quote_date": [pd.Timestamp("2020-01-01")],
                     "TTFc1": [10.0]}).to_excel(wb, index=False)
        first, prov_first = qd.load_quote_matrix(wb)
        assert first["TTFc1"].iloc[0] == 10.0
        assert prov_first["cache"] == "rebuilt"

        stat_before = os.stat(wb)
        pd.DataFrame({"quote_date": [pd.Timestamp("2020-01-01")],
                     "TTFc1": [999.0]}).to_excel(wb, index=False)
        os.utime(wb, (stat_before.st_atime, stat_before.st_mtime))

        second, prov_second = qd.load_quote_matrix(wb)
        assert second["TTFc1"].iloc[0] == 999.0, (
            "same path, restored mtime, changed content -- got stale data")
        assert prov_second["source_sha256"] != prov_first["source_sha256"]

        # And re-reading the FIRST content again (a third distinct file, same
        # bytes as the original) correctly hits the cache entry THAT content
        # made, not the second one -- content addressing means both coexist.
        again_wb = os.path.join(tmp, "again.xlsx")
        pd.DataFrame({"quote_date": [pd.Timestamp("2020-01-01")],
                     "TTFc1": [10.0]}).to_excel(again_wb, index=False)
        third, prov_third = qd.load_quote_matrix(again_wb, cache_dir=os.path.dirname(wb))
        assert third["TTFc1"].iloc[0] == 10.0
        assert prov_third["cache"] == "hit", "identical bytes should hit the first cache entry"


def test_cache_enabled_and_disabled_give_equivalent_data():
    """`use_cache=False` is the reproducible-comparison escape hatch: same
    source, same cleaner, same answer, whether or not caching is involved."""
    import quote_data as qd

    with tempfile.TemporaryDirectory() as tmp:
        wb = os.path.join(tmp, "ttf q.xlsx")
        pd.DataFrame({"quote_date": pd.date_range("2020-01-01", periods=5),
                     "TTFc1": [1.0, 2.0, "Retrieving...", 4.0, 4.0]}
                    ).to_excel(wb, index=False)

        cached, prov_cached = qd.load_quote_matrix(wb, use_cache=True)
        assert prov_cached["cache"] == "rebuilt"
        assert os.listdir(tmp), "use_cache=True should have written a cache file"

        uncached_dir = os.path.join(tmp, "no_cache_dir")
        uncached, prov_uncached = qd.load_quote_matrix(
            wb, cache_dir=uncached_dir, use_cache=False)
        assert not os.path.exists(uncached_dir), "use_cache=False must not write anything"

        pd.testing.assert_frame_equal(cached, uncached)
        # "Retrieving..." is the rejected-cell case; the stats say so rather
        # than silently vanishing.
        assert prov_cached["rejected_cells"] == 1
        assert prov_cached["duplicate_dates"] == 0


def test_a_parquet_source_is_a_source_not_a_cache_of_something_else():
    """Explicitly requesting a `.parquet` file reads it directly and
    fingerprints ITS OWN bytes -- it is a dataset in its own right, not
    silently treated as a cache entry for some other workbook."""
    import quote_data as qd

    with tempfile.TemporaryDirectory() as tmp:
        direct = os.path.join(tmp, "standalone.parquet")
        pd.DataFrame({"quote_date": [pd.Timestamp("2020-01-01")],
                     "TTFc1": [42.0]}).to_parquet(direct)
        quotes, provenance = qd.load_quote_matrix(direct)
        assert quotes["TTFc1"].iloc[0] == 42.0
        assert provenance["format"] == "parquet"
        assert provenance["source_sha256"] == qd.source_fingerprint(direct)
        assert "n/a" in provenance["cache"]


def test_streamlit_in_memory_cache_observes_a_content_change_at_the_same_path():
    """The other half of the defect: even with `quote_data`'s on-disk cache
    fixed, Streamlit's OWN `@st.cache_data` sat in front of it -- and until
    2026-09-11 `portfolio_app.load_quote_matrix_local` was keyed only on a
    path STRING, which `@st.cache_data` hashes as an unchanged argument no
    matter what the file's bytes do. It now also takes the fingerprint as an
    explicit (non-underscore-prefixed) argument, so Streamlit's own hash
    changes when the content does.

    Exercises Streamlit's actual caching decorator directly, not
    `AppTest.from_file` (which never reaches this code path -- it is gated
    behind the form's submit button and does not click it).
    """
    import streamlit as st
    import quote_data as qd

    @st.cache_data(show_spinner=False)
    def load_quote_matrix_local(xlsx_path, source_fingerprint):
        quotes, _ = qd.load_quote_matrix(xlsx_path)
        return quotes

    with tempfile.TemporaryDirectory() as tmp:
        wb = os.path.join(tmp, "ttf q.xlsx")
        pd.DataFrame({"quote_date": [pd.Timestamp("2020-01-01")],
                     "TTFc1": [10.0]}).to_excel(wb, index=False)
        first = load_quote_matrix_local(wb, qd.source_fingerprint(wb))
        assert first["TTFc1"].iloc[0] == 10.0

        stat_before = os.stat(wb)
        pd.DataFrame({"quote_date": [pd.Timestamp("2020-01-01")],
                     "TTFc1": [999.0]}).to_excel(wb, index=False)
        os.utime(wb, (stat_before.st_atime, stat_before.st_mtime))

        second = load_quote_matrix_local(wb, qd.source_fingerprint(wb))
        assert second["TTFc1"].iloc[0] == 999.0, (
            "Streamlit's in-memory cache returned stale data for a changed file "
            "at an unchanged path")

    # Negative control: the pre-fix single-argument signature IS stale on this
    # exact scenario, so the assertion above is not vacuously true.
    @st.cache_data(show_spinner=False)
    def load_quote_matrix_local_pre_fix(xlsx_path):
        quotes, _ = qd.load_quote_matrix(xlsx_path)
        return quotes

    with tempfile.TemporaryDirectory() as tmp:
        wb = os.path.join(tmp, "ttf q.xlsx")
        pd.DataFrame({"quote_date": [pd.Timestamp("2020-01-01")],
                     "TTFc1": [10.0]}).to_excel(wb, index=False)
        first = load_quote_matrix_local_pre_fix(wb)
        stat_before = os.stat(wb)
        pd.DataFrame({"quote_date": [pd.Timestamp("2020-01-01")],
                     "TTFc1": [999.0]}).to_excel(wb, index=False)
        os.utime(wb, (stat_before.st_atime, stat_before.st_mtime))
        second = load_quote_matrix_local_pre_fix(wb)
        assert second["TTFc1"].iloc[0] == 10.0, (
            "the negative control did not reproduce the staleness it is meant "
            "to demonstrate -- something about the test scenario changed")
