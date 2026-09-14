"""DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.9.3's three delta measures,
against reset_swing_exact.compute_deltas."""
import pandas as pd
import pytest

import reset_swing_exact as rse
import reset_terms as rt


def _terms(**overrides):
    defaults = dict(
        val_date="2026-01-01", storage_start="2026-03-01", storage_end="2026-04-30",
        daily_max_mwh=1_000.0, v_step_mwh=500.0,
        global_min_mwh=0.0, global_max_mwh=10_000.0,
        vol=0.5, sMR=1.0, discount_rate=0.05, n_p=10)
    defaults.update(overrides)
    return rt.ResetSwingTerms(**defaults)


def _shaped_curve():
    curve = pd.Series(25.0, index=pd.date_range("2020-01-01", "2030-12-31", freq="D"))
    curve.loc["2026-03-01":"2026-03-15"] = 32.0
    curve.loc["2026-03-16":"2026-03-31"] = 22.0
    return curve


def test_total_delta_matches_a_manual_central_difference():
    """compute_deltas' own bump construction, reproduced by calling
    value_point_reset_call_swing directly -- catches a bug in how the
    function assembles its bumped curves, independent of trusting its
    internal arithmetic."""
    terms = _terms()
    schedule = rt.build_reset_schedule(terms)
    curve = _shaped_curve()
    bump = 0.10

    deltas = rse.compute_deltas(terms, schedule, curve, bump_eur_mwh=bump)

    pv_up = rse.value_point_reset_call_swing(terms, schedule, daily_curve=curve + bump)
    pv_down = rse.value_point_reset_call_swing(terms, schedule, daily_curve=curve - bump)
    manual_total = (pv_up - pv_down) / (2.0 * bump)

    assert deltas["total"] == pytest.approx(manual_total, rel=1e-9)


def test_legs_have_the_expected_sign_and_dominate_the_small_net_total():
    """A call swing: bumping the delivery price alone (strike frozen) must
    raise value; bumping the strike alone (delivery frozen) must lower it.
    For a month-ahead INDEXED structure specifically, a parallel curve shift
    moves both roughly together, so the two legs should be large and mostly
    offsetting, leaving a much smaller net total -- the point of indexing
    the strike at all, and a property worth pinning, not just the signs."""
    terms = _terms()
    schedule = rt.build_reset_schedule(terms)
    curve = _shaped_curve()

    deltas = rse.compute_deltas(terms, schedule, curve, bump_eur_mwh=0.10)

    assert deltas["physical_leg"] > 0
    assert deltas["index_leg"] < 0
    assert abs(deltas["total"]) < 0.05 * abs(deltas["physical_leg"]), (
        "expected the indexed strike to cancel most of a parallel shift's "
        f"physical-leg exposure: {deltas}")


def test_legs_sum_to_the_total_to_first_order():
    """sec.9.3: not asserted exact (the split is only additive to first order
    and the reset composes nonlinearly through which node the fixing lands
    on) -- but it should be close, and this pins how close."""
    terms = _terms()
    schedule = rt.build_reset_schedule(terms)
    curve = _shaped_curve()

    deltas = rse.compute_deltas(terms, schedule, curve, bump_eur_mwh=0.10)
    combined = deltas["physical_leg"] + deltas["index_leg"]
    assert combined == pytest.approx(deltas["total"], rel=1e-3)


def test_mandatory_volume_has_a_larger_physical_leg_than_optional():
    """Optional volume can decline to exercise on an unfavourable bump;
    mandatory volume cannot -- so the physical-leg sensitivity for a
    mandatory contract should be at least as large."""
    curve = _shaped_curve()

    mandatory = _terms(global_min_mwh=5_000.0, global_max_mwh=5_000.0)
    optional = _terms(global_min_mwh=0.0, global_max_mwh=5_000.0)

    d_mandatory = rse.compute_deltas(
        mandatory, rt.build_reset_schedule(mandatory), curve, bump_eur_mwh=0.10)
    d_optional = rse.compute_deltas(
        optional, rt.build_reset_schedule(optional), curve, bump_eur_mwh=0.10)

    assert d_mandatory["physical_leg"] >= d_optional["physical_leg"] - 1e-6
