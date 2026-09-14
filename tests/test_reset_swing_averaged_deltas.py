"""DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.9.3's three delta measures,
against reset_swing_averaged.compute_deltas -- the same measures
tests/test_reset_swing_deltas.py pins for point-reset, extended here to the
averaged strike.

n_r kept modest (60) throughout: compute_deltas runs six full valuations,
and tests/test_reset_swing_averaged.py's own convergence test already
separately pins the O(1/n_r) interpolation-bias story, so a large n_r here
would only slow the suite down without checking anything new. Tolerances
are correspondingly a bit looser than the point-reset equivalents -- grid
interpolation adds its own small noise on top of the same "only additive to
first order" nonlinearity point-reset already has, and that noise is not
negligible next to `total`, which is itself a near-cancellation of two much
larger legs.
"""
import pandas as pd
import pytest

import reset_swing_averaged as rsa
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


_GRID = dict(n_r=60, r_lo=5.0, r_hi=55.0)  # wide margin: the shaped curve's own
# range is [22, 32], but April's strike is a genuine average over March's varied
# daily quotes plus real lattice spread at vol=0.5 -- checked empirically to clear
# value_averaged_reset_call_swing's own bracket check with comfortable room.


def test_total_delta_matches_a_manual_central_difference():
    """compute_deltas' own bump construction, reproduced by calling
    value_averaged_reset_call_swing directly -- catches a bug in how the
    function assembles its bumped curves, independent of trusting its
    internal arithmetic."""
    terms = _terms()
    schedule = rt.build_reset_schedule(terms)
    curve = _shaped_curve()
    bump = 0.10

    deltas = rsa.compute_deltas(terms, schedule, curve, bump_eur_mwh=bump, **_GRID)

    pv_up = rsa.value_averaged_reset_call_swing(terms, schedule, curve + bump, **_GRID)
    pv_down = rsa.value_averaged_reset_call_swing(terms, schedule, curve - bump, **_GRID)
    manual_total = (pv_up - pv_down) / (2.0 * bump)

    assert deltas["total"] == pytest.approx(manual_total, rel=1e-9)


def test_legs_have_the_expected_sign_and_dominate_the_small_net_total():
    """A call swing: bumping the delivery price alone (strike frozen) must
    raise value; bumping the strike alone (delivery frozen) must lower it.
    For a month-ahead INDEXED structure specifically, a parallel curve shift
    moves both roughly together, so the two legs should be large and mostly
    offsetting, leaving a much smaller net total -- the point of indexing
    the strike at all, same as point-reset's own version of this property."""
    terms = _terms()
    schedule = rt.build_reset_schedule(terms)
    curve = _shaped_curve()

    deltas = rsa.compute_deltas(terms, schedule, curve, bump_eur_mwh=0.10, **_GRID)

    assert deltas["physical_leg"] > 0
    assert deltas["index_leg"] < 0
    assert abs(deltas["total"]) < 0.05 * abs(deltas["physical_leg"]), (
        "expected the indexed strike to cancel most of a parallel shift's "
        f"physical-leg exposure: {deltas}")


def test_legs_sum_to_the_total_to_first_order():
    """sec.9.3: not asserted exact (the split is only additive to first order,
    the reset composes nonlinearly, and here there is also grid-interpolation
    noise on top) -- but the residual should be small relative to the legs
    themselves, not relative to `total`, which is itself a near-cancellation
    and so amplifies any relative comparison against it."""
    terms = _terms()
    schedule = rt.build_reset_schedule(terms)
    curve = _shaped_curve()

    deltas = rsa.compute_deltas(terms, schedule, curve, bump_eur_mwh=0.10, **_GRID)
    combined = deltas["physical_leg"] + deltas["index_leg"]
    assert abs(combined - deltas["total"]) < 0.01 * abs(deltas["physical_leg"]), (
        f"combined leg sum {combined} vs total {deltas['total']}: {deltas}")


def test_mandatory_volume_has_a_larger_physical_leg_than_optional():
    """Optional volume can decline to exercise on an unfavourable bump;
    mandatory volume cannot -- so the physical-leg sensitivity for a
    mandatory contract should be at least as large, same property as
    point-reset's own version of this test."""
    curve = _shaped_curve()

    mandatory = _terms(global_min_mwh=5_000.0, global_max_mwh=5_000.0)
    optional = _terms(global_min_mwh=0.0, global_max_mwh=5_000.0)

    d_mandatory = rsa.compute_deltas(
        mandatory, rt.build_reset_schedule(mandatory), curve, bump_eur_mwh=0.10, **_GRID)
    d_optional = rsa.compute_deltas(
        optional, rt.build_reset_schedule(optional), curve, bump_eur_mwh=0.10, **_GRID)

    assert d_mandatory["physical_leg"] >= d_optional["physical_leg"] - 1e-3
