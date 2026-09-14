"""DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.10.2's hand-computable tests for
the Phase 1 / Release 1A point-reset exact benchmark (reset_swing_exact.py).

At near-zero volatility the model is deterministic (forward-fitted to the input
curve, no randomness to average over), so the month-ahead strike each month
resets to is just that month's own known forward price -- turning the whole
problem into a hand-computable schedule optimisation. This is exactly
sec.10.2's "zero volatility gives the hand-computable deterministic reset and
PV" case, done properly: three independently-reasoned scenarios (single month,
cross-month chaining, optional volume), not one lucky match.

A note on scope: this covers the values sec.10.2 asks for. It does not yet cover
every fixture sec.14.7 eventually wants (sec.10.3's exhaustive-enumeration cross
check against a scenario tree, sec.10.1's chronology-under-valuation tests, or
sec.10.4's numerical convergence ladders) -- those are follow-up, not silently
declared done here.
"""
import pandas as pd
import pytest

import reset_swing_exact as rse
import reset_terms as rt

NEAR_ZERO_VOL = 1e-7  # ResetSwingTerms requires vol > 0; this is deterministic
                      # to well under the tolerances used below.


def _flat_curve(default=30.0):
    return pd.Series(default, index=pd.date_range("2020-01-01", "2030-12-31", freq="D"))


def test_single_month_mandatory_volume_takes_the_best_margin_days():
    """May's strike resets to May 31's own price (20 here). Mandatory 5 clips
    out of 31 possible days: the optimiser must use the five +5 EUR/MWh days,
    never the zero-margin ones, giving an exactly computable PV."""
    terms = rt.ResetSwingTerms(
        val_date="2026-01-01", storage_start="2026-05-01", storage_end="2026-05-31",
        daily_max_mwh=1_000.0, v_step_mwh=1_000.0,
        global_min_mwh=5_000.0, global_max_mwh=5_000.0,
        vol=NEAR_ZERO_VOL, sMR=1.0, discount_rate=0.0, n_p=6)
    schedule = rt.build_reset_schedule(terms)

    curve = _flat_curve()
    curve.loc["2026-05-01":"2026-05-15"] = 35.0  # +5 vs the 30.0 month-end strike
    curve.loc["2026-05-16":"2026-05-31"] = 30.0  # includes May 31 itself: K = 30.0

    pv = rse.value_point_reset_call_swing(terms, schedule, daily_curve=curve)
    assert pv == pytest.approx(5 * 1_000.0 * 5.0, rel=1e-4)


def test_cross_month_chaining_defers_a_global_quota_to_the_better_month():
    """A GLOBAL (deal-wide, not per-month) mandatory quota of 2 clips, spanning
    May (flat -> zero margin all month, K_May = 25) and June (K_June = June
    30's own price, 20, against a rich June 1-15 at 40 -> +20 margin). The
    optimiser must recognise, while still inside May, that carrying the
    obligation into June is worth more -- the cumulative-volume state has to
    survive the month rotation correctly for that to be visible at all."""
    terms = rt.ResetSwingTerms(
        val_date="2026-01-01", storage_start="2026-05-01", storage_end="2026-06-30",
        daily_max_mwh=1_000.0, v_step_mwh=1_000.0,
        global_min_mwh=2_000.0, global_max_mwh=2_000.0,
        vol=NEAR_ZERO_VOL, sMR=1.0, discount_rate=0.0, n_p=6)
    schedule = rt.build_reset_schedule(terms)
    assert len(schedule.months) == 2

    curve = _flat_curve()
    curve.loc["2026-05-01":"2026-05-31"] = 25.0
    curve.loc["2026-06-01":"2026-06-15"] = 40.0
    curve.loc["2026-06-16":"2026-06-30"] = 20.0  # includes June 30: K_June = 20.0

    pv = rse.value_point_reset_call_swing(terms, schedule, daily_curve=curve)
    assert pv == pytest.approx(2 * 1_000.0 * 20.0, rel=1e-4)


def test_optional_volume_only_exercises_strictly_positive_margin_days():
    """global_min=0 makes exercise optional. K_May = May 31's own price (20,
    since May 31 falls in the second block below). The +20-margin block
    (10 days) must be fully used; the zero-margin block must not be touched
    at all -- not "touched a little", exactly zero incremental value from it."""
    terms = rt.ResetSwingTerms(
        val_date="2026-01-01", storage_start="2026-05-01", storage_end="2026-05-31",
        daily_max_mwh=1_000.0, v_step_mwh=1_000.0,
        global_min_mwh=0.0, global_max_mwh=31_000.0,
        vol=NEAR_ZERO_VOL, sMR=1.0, discount_rate=0.0, n_p=6)
    schedule = rt.build_reset_schedule(terms)

    curve = _flat_curve()
    curve.loc["2026-05-01":"2026-05-10"] = 40.0   # +20 margin vs K=20.0
    curve.loc["2026-05-11":"2026-05-31"] = 20.0   # includes May 31: K = 20.0, zero margin

    pv = rse.value_point_reset_call_swing(terms, schedule, daily_curve=curve)
    assert pv == pytest.approx(10 * 1_000.0 * 20.0, rel=1e-4)


def test_curve_or_daily_curve_but_not_both_or_neither():
    terms = rt.ResetSwingTerms(
        val_date="2026-01-01", storage_start="2026-05-01", storage_end="2026-05-31",
        daily_max_mwh=1_000.0, v_step_mwh=1_000.0,
        global_min_mwh=0.0, global_max_mwh=5_000.0,
        vol=0.5, sMR=1.0, discount_rate=0.0, n_p=6)
    schedule = rt.build_reset_schedule(terms)
    with pytest.raises(ValueError, match="exactly one"):
        rse.value_point_reset_call_swing(terms, schedule)
    with pytest.raises(ValueError, match="exactly one"):
        rse.value_point_reset_call_swing(
            terms, schedule, daily_curve=_flat_curve(), curve=_flat_curve())


def test_detailed_wrapper_matches_the_plain_call_and_reports_the_hand_computable_strike():
    """R-08 (2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING): the `_detailed`
    wrapper must be pure plumbing -- same PV as the plain call, and a
    `reset_strikes` entry that matches the hand-computable value this file's
    own near-zero-vol convention already establishes (K_May = May 31's own
    price), not a new independent numerical claim."""
    terms = rt.ResetSwingTerms(
        val_date="2026-01-01", storage_start="2026-05-01", storage_end="2026-05-31",
        daily_max_mwh=1_000.0, v_step_mwh=1_000.0,
        global_min_mwh=5_000.0, global_max_mwh=5_000.0,
        vol=NEAR_ZERO_VOL, sMR=1.0, discount_rate=0.0, n_p=6)
    schedule = rt.build_reset_schedule(terms)

    curve = _flat_curve()
    curve.loc["2026-05-01":"2026-05-15"] = 35.0
    curve.loc["2026-05-16":"2026-05-31"] = 30.0  # includes May 31 itself: K = 30.0

    plain_pv = rse.value_point_reset_call_swing(terms, schedule, daily_curve=curve)
    result = rse.value_point_reset_call_swing_detailed(terms, schedule, daily_curve=curve)

    assert result.pv == plain_pv
    assert set(result.reset_strikes) == {schedule.months[0].label}
    assert result.reset_strikes[schedule.months[0].label] == pytest.approx(30.0, rel=1e-4)
    assert result.deltas is None


def test_detailed_wrapper_with_deltas_matches_compute_deltas_exactly():
    terms = rt.ResetSwingTerms(
        val_date="2026-01-01", storage_start="2026-05-01", storage_end="2026-05-31",
        daily_max_mwh=1_000.0, v_step_mwh=1_000.0,
        global_min_mwh=0.0, global_max_mwh=5_000.0,
        vol=0.3, sMR=1.0, discount_rate=0.05, n_p=6)
    schedule = rt.build_reset_schedule(terms)
    curve = _flat_curve()

    expected_deltas = rse.compute_deltas(terms, schedule, curve, bump_eur_mwh=0.10)
    result = rse.value_point_reset_call_swing_detailed(
        terms, schedule, daily_curve=curve, with_deltas=True, bump_eur_mwh=0.10)

    assert result.deltas == expected_deltas


def test_detailed_wrapper_with_deltas_requires_daily_curve_not_curve():
    """`compute_deltas` only accepts the `daily_curve` form (see its own
    docstring); the wrapper must refuse `with_deltas=True` together with the
    `curve=` alternative rather than let it fail downstream with a confusing
    error. `curve` here is a minimal valid contract-strip DataFrame -- the
    shape `Storage.__init__`/`map_curve_to_dates` actually expect (columns
    contractStart/contractEnd/value spanning the whole valuation window), not
    a flat daily Series -- so the `pv` computation ahead of the guard clause
    succeeds and the guard clause itself is what is actually being tested."""
    terms = rt.ResetSwingTerms(
        val_date="2026-01-01", storage_start="2026-05-01", storage_end="2026-05-31",
        daily_max_mwh=1_000.0, v_step_mwh=1_000.0,
        global_min_mwh=0.0, global_max_mwh=5_000.0,
        vol=0.3, sMR=1.0, discount_rate=0.05, n_p=6)
    schedule = rt.build_reset_schedule(terms)
    curve = pd.DataFrame([{"contractStart": pd.Timestamp("2026-01-01"),
                          "contractEnd": pd.Timestamp("2027-01-01"), "value": 30.0}])
    with pytest.raises(ValueError, match="with_deltas"):
        rse.value_point_reset_call_swing_detailed(
            terms, schedule, curve=curve, with_deltas=True)
