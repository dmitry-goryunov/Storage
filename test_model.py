"""
Regression tests for storage_model.

Run with `python test_model.py` (no pytest needed) or `pytest test_model.py`.

The suite is built around one master invariant: the value of a swing contract is
linear in the price level, so the reported delta profile must reprice the
contract exactly --

    sum_i  delta[i] * fwd[i]  ==  v[0, n_p, n_op_start]

That single identity ties together the DP (`run_model`), the forward pass
(`probabilities`) and the reported metrics (`compute_all_metrics`); if any of
the three disagrees about what the optimal strategy does, it breaks.
"""
import sys
import traceback

import numpy as np
import pandas as pd

from storage_model import (Storage, build_tree, daily_arithmetic_flat_metric,
                           run_valuation)

# Short-dated contract: the whole suite stays under a few seconds once Numba has
# compiled (kernels are cached on disk between runs).
VALDATE, START, END = "2026-01-01", "2026-02-01", "2026-04-30"
WINDOW = (pd.Timestamp(END) - pd.Timestamp(START)).days + 1      # 89 days
V_STEP, VOL = 1000, 0.5


def curve(last_month="2027-12-01"):
    """Flat-ish seasonal monthly curve covering valDate .. last_month."""
    starts = pd.date_range("2026-01-01", last_month, freq="MS")
    value = 25 + 3 * np.cos(2 * np.pi * (starts.month - 1) / 12)
    return pd.DataFrame({"contractStart": starts,
                         "contractEnd": starts + pd.offsets.MonthEnd(0),
                         "value": value})


def put_swing(n_p=20, days=10, storageEnd=END, ratch=None, c=None, vol=VOL):
    """Mandatory buy on `days` days -- same wiring as value_put_swing."""
    s = Storage(VALDATE, START, storageEnd, curve=curve() if c is None else c,
                n_p=0, v_step=V_STEP, sVol=vol)
    n = len(s.date_span)
    active = np.ones(n)
    active[s._active:] = 0.0
    active[:s.Dt] = 0.0
    s.i_curve = active.copy()
    s.w_curve = np.zeros(n)
    s.set_volume_states(days)
    s.n_op_start = 0
    if ratch is not None:
        s.i_ratch[:] = ratch
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    s.t_p_curve[days] = 0.0
    s.n_p = n_p
    return s.build()


def repricing_gap(s):
    """|sum(delta_i * fwd_i) - V0| relative to |V0|."""
    v0 = s.v[0, s.n_p, s.n_op_start]
    repriced = float(np.sum(np.array(s.delta[:s.n_t]) * s.fwd))
    return abs(repriced - v0) / max(abs(v0), 1.0)


# ── the master invariant ──────────────────────────────────────────────────────

def test_delta_reprices_the_contract():
    for n_p in (0, 20):
        assert repricing_gap(put_swing(n_p=n_p)) < 1e-9, f"delta does not reprice V0 at n_p={n_p}"


def test_delta_reprices_with_a_ratchet_of_two():
    # Exercises the ratchet path through DP, forward pass and metrics at once.
    assert repricing_gap(put_swing(days=10, ratch=2)) < 1e-9


# ── volumes and prices ────────────────────────────────────────────────────────

def test_mandatory_quota_is_met_exactly():
    for days in (5, 10, 30):
        s = put_swing(days=days)
        assert abs(-np.sum(s.exp_ex) - days * V_STEP) < 1e-6, f"quota not met for days={days}"


def test_quota_equal_to_window_prices_at_the_flat_price():
    """Buying every day of the window must cost the arithmetic average forward."""
    s = put_swing(n_p=20, days=WINDOW)
    price = s.v[0, s.n_p, s.n_op_start] / np.sum(s.exp_ex)
    assert abs(price - daily_arithmetic_flat_metric(s)) < 1e-6


def test_price_methods_work_on_a_full_tree():
    """flat()/profiled() must not read the dead price state 0 when n_p > 0."""
    s = put_swing(n_p=20)
    expected = s.v[0, s.n_p, s.n_op_start] / np.sum(s.exp_ex)
    assert s.v[0, 0, s.n_op_start] == 0.0, "price state 0 is unused at t=0 -- premise of this test"
    assert abs(s.flat() - expected) < 1e-9, "flat() reads the wrong price state or denominator"
    assert abs(s.profiled() - expected) < 1e-9, "profiled() reads the wrong price state or denominator"


def test_reported_price_per_mwh_uses_volume_not_delta():
    """stochastic_metric must be value / MWh exercised, not value / sum(delta)."""
    params = dict(product_type="put_swing", valDate=VALDATE, storageStart=START,
                  storageEnd=END, days=10, vol=VOL, n_p_full=20, run_intrinsic=False,
                  v_step=V_STEP)
    s, res = run_valuation(curve(), params)
    volume = -np.sum(s.exp_ex)
    assert abs(volume - 10 * V_STEP) < 1e-6
    expected = -s.v[0, s.n_p, s.n_op_start] / volume
    assert abs(res["stochastic_metric"] - expected) < 1e-6, (
        f"stochastic_metric={res['stochastic_metric']:.4f} but value/volume={expected:.4f}; "
        f"sum(delta)={np.sum(s.delta):.1f} != volume={-volume:.1f}")


def test_extrinsic_is_non_negative_and_converged():
    params = dict(product_type="put_swing", valDate=VALDATE, storageStart=START,
                  storageEnd=END, days=10, vol=VOL, n_p_full=20, run_intrinsic=True,
                  v_step=V_STEP)
    _, res20 = run_valuation(curve(), params)
    _, res30 = run_valuation(curve(), dict(params, n_p_full=30))
    assert res20["extrinsic"] >= -1e-9, "optionality cannot be worth less than zero"
    assert abs(res30["extrinsic"] - res20["extrinsic"]) < 0.01, (
        f"not converged in n_p: {res20['extrinsic']:.4f} (20) vs {res30['extrinsic']:.4f} (30)")


# ── the tree ──────────────────────────────────────────────────────────────────

def _tree_is_valid(n_p, vol_curve, s):
    fwd, x, q, p_u, p_m, p_d = build_tree(s.price_curve, s.n_t, n_p, vol_curve, [1.0] * s.n_t)
    live = np.zeros_like(p_m, dtype=bool)
    for i in range(s.n_t):
        live[i, max(n_p - i, 0):min(n_p + i, 2 * n_p) + 1] = True
    for name, p in (("p_u", p_u), ("p_m", p_m), ("p_d", p_d)):
        assert np.all(p[live] >= -1e-12), f"{name} has negative probabilities (min {p[live].min():.4f})"
        assert np.all(p[live] <= 1 + 1e-12), f"{name} exceeds 1 (max {p[live].max():.4f})"
    assert np.abs((p_u + p_m + p_d)[live] - 1).max() < 1e-12
    assert np.abs(q.sum(1) - 1).max() < 1e-9, "state probabilities do not sum to 1"
    expected = np.array([np.dot(q[i], np.exp(x[i])) for i in range(n_p, s.n_t)])
    assert np.abs(expected / fwd[n_p:] - 1).max() < 1e-12, "tree does not reprice the forward curve"


def test_tree_is_valid_for_flat_vol():
    s = Storage(VALDATE, START, END, curve=curve(), n_p=0, sVol=VOL)
    _tree_is_valid(20, [VOL] * s.n_t, s)


def test_tree_is_valid_for_a_vol_term_structure():
    """dx must be sized from max(sVol); sizing it from sVol[0] blows up the high-vol steps."""
    s = Storage(VALDATE, START, END, curve=curve(), n_p=0, sVol=VOL)
    _tree_is_valid(20, list(np.linspace(0.45, 0.90, s.n_t)), s)


def test_unstable_vol_term_structure_raises_instead_of_returning_nans():
    """A vol range too wide for one grid spacing must fail loudly, not silently NaN."""
    s = Storage(VALDATE, START, END, curve=curve(), n_p=0, sVol=VOL)
    _raises(lambda: build_tree(s.price_curve, s.n_t, 40, list(np.linspace(0.10, 1.20, s.n_t)),
                               [1.0] * s.n_t), "unstable", "svol")


# ── failing loudly ────────────────────────────────────────────────────────────

def _raises(fn, *needles):
    try:
        fn()
    except Exception as exc:                                   # noqa: BLE001
        msg = str(exc).lower()
        missing = [n for n in needles if n.lower() not in msg]
        assert not missing, f"error message is missing {missing}: {exc!r}"
        return
    raise AssertionError("expected an exception, none raised")


def test_infeasible_quota_raises_instead_of_returning_the_sentinel():
    _raises(lambda: put_swing(n_p=5, days=WINDOW + 10), "terminal")


def test_curve_that_stops_short_raises_a_useful_message():
    _raises(lambda: put_swing(n_p=5, c=curve(last_month="2026-03-01")), "curve")


def test_fractional_ratchets_are_rejected():
    """DP truncates, the forward pass used to round, metrics kept the float."""
    _raises(lambda: put_swing(n_p=5, ratch=1.5), "ratch")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:                                       # noqa: BLE001
            failed.append(name)
            print(f"FAIL  {name}")
            print("      " + traceback.format_exc().strip().splitlines()[-1])
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
