"""DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.14.7's named, committed acceptance
fixtures -- specifically the two genuinely new ones out of the 14 the list names
(2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-10). The other 12 are not
duplicated here: 8 already exist under different names (4 in
tests/test_reconciliation.py -- test_zero_rate_mandatory_strike_affinity,
test_common_settlement_strike_affinity, test_daily_discounting_can_break_affinity,
test_optional_volume_strike_convexity -- and 4 more split across
test_reset_swing_exhaustive.py, test_reset_swing_averaged.py, test_reset_forward.py
and test_reset_swing_stochastic.py), 2 are partial evidence for an already-named
open question (R-03's same-day ordering; near-zero-vol testing that cannot reach
literal vol=0 since ResetSwingTerms refuses it), and 2 (test_partially_fixed_month,
test_monthly_and_global_volume_limits) need product features -- historical/seeded
fixings, per-month rather than deal-wide volume limits -- this prototype does not
have, not just new tests. See docs/STATUS.md's own R-10 entry for the full mapping
of all 14 names against what actually covers them, and the design doc's own R-10
log entry for why this file holds only these two.
"""
import pandas as pd
import pytest

import reset_swing_exact as rse
import reset_swing_averaged as rsa
import reset_terms as rt
import storage_model as sm

NEAR_ZERO_VOL = 1e-7  # The one epsilon this file declares for both reset
                      # conventions -- R-10's own complaint was that no single
                      # convention existed; test_reset_swing_point.py and
                      # test_reset_swing_averaged.py each pick their own (1e-7
                      # and 1e-4 respectively) for reasons specific to what they
                      # individually need, which this fixture does not disturb.


def test_fixed_strike_equivalence():
    """sec.10.2's cross-engine reading of "fixed-strike equivalence": when the
    reset-swing engine's own strike converges to a known constant (near-zero vol
    collapses the model's conditional-expectation projection to a deterministic
    number), pricing the SAME delivery-month window through the INDEPENDENT,
    already-extensively-tested plain fixed-strike call-swing engine
    (storage_model.run_valuation) at that exact constant must reproduce the
    reset-swing engine's own PV. Not a hand computation checked against itself --
    two structurally different, independently implemented solvers agreeing on the
    same number. Reuses
    test_reset_swing_point.py::test_single_month_mandatory_volume_takes_the_best_margin_days's
    own fixture (independently hand-verified there to 5 * 1,000 * 5.0 = 25,000 EUR:
    5 mandatory clips, each capturing the best available 5 EUR/MWh margin) and
    reads the reset engine's own emergent strike via R-08's
    `value_point_reset_call_swing_detailed` rather than re-deriving or rounding
    it, so the check is against whatever number the reset engine actually
    reports, not a convenient round one.
    """
    terms = rt.ResetSwingTerms(
        val_date="2026-01-01", storage_start="2026-05-01", storage_end="2026-05-31",
        daily_max_mwh=1_000.0, v_step_mwh=1_000.0,
        global_min_mwh=5_000.0, global_max_mwh=5_000.0,
        vol=NEAR_ZERO_VOL, sMR=1.0, discount_rate=0.0, n_p=20)
    schedule = rt.build_reset_schedule(terms)

    curve = pd.Series(30.0, index=pd.date_range("2020-01-01", "2030-12-31", freq="D"))
    curve.loc["2026-05-01":"2026-05-15"] = 35.0
    curve.loc["2026-05-16":"2026-05-31"] = 30.0  # includes May 31 itself: K = 30.0

    result = rse.value_point_reset_call_swing_detailed(terms, schedule, daily_curve=curve)
    strike = result.reset_strikes[schedule.months[0].label]

    month = schedule.months[0]
    model, _ = sm.run_valuation(None, dict(
        product_type="call_swing", valDate=terms.val_date,
        storageStart=month.exercise_dates[0], storageEnd=month.exercise_dates[-1],
        capacity_mwh=5_000.0, daily_max=1_000.0, clips_per_day=1,
        vol=NEAR_ZERO_VOL, sMR=1.0, n_p_full=20, run_intrinsic=False, discount_rate=0.0,
        strike=strike, zero_penalty=False, daily_curve=curve))
    plain_pv = float(model.v[0, model.n_p, model.initial_state])

    assert result.pv == pytest.approx(plain_pv, abs=0.5)
    assert result.pv == pytest.approx(25_000.0, rel=1e-4)


def test_zero_volatility_reset():
    """sec.10.2's degenerate/hand-computable case, through the public
    ResetSwingTerms -> build_reset_schedule -> value_*_call_swing path end to
    end, for BOTH reset conventions at once -- the concrete gap R-10 named
    ("no single declared convention" for near-zero vol, scattered per-file
    epsilons instead). Literal vol=0.0 is refused by
    ResetSwingTerms.__post_init__ (a deliberate, separately-tested guard), so
    "zero" here means NEAR_ZERO_VOL, declared once at module level.

    Same single-month, mandatory-5-of-31-clips, two-block curve as
    test_fixed_strike_equivalence above (hand-computable to 25,000 EUR). Global
    volume here is MANDATORY (global_min_mwh == global_max_mwh), which
    tests/test_reconciliation.py::test_zero_rate_mandatory_strike_affinity
    already establishes makes V(K) exactly AFFINE in the strike -- no exercise-
    boundary kink for accumulate_step's linear interpolation to be biased by, so
    averaged-reset needs no generous n_r to converge tightly here (confirmed
    directly: n_r=10 through 100 all agree with point-reset to ~1e-11, unlike
    the genuinely kinked optional-volume case
    test_reset_swing_averaged.py::test_finer_r_grid_moves_averaged_reset_toward_point_reset_at_low_vol
    documents). Averaged-reset's own strike is fixed from APRIL (the calendar
    month immediately preceding May, per reset_terms.DeliveryMonth's own
    fixing_observation_dates), left at a flat, untouched default -- deliberately,
    since at near-zero vol the model's conditional expectation of a FUTURE
    month-end's price, viewed from any earlier date, converges to that future
    date's own forward-fitted curve value regardless of the earlier date's own
    curve level (confirmed directly against reset_swing_averaged.py's own
    near-zero-vol test, and the reason this fixture does not need April's curve
    to carry any particular shape at all).
    """
    terms = rt.ResetSwingTerms(
        val_date="2026-01-01", storage_start="2026-05-01", storage_end="2026-05-31",
        daily_max_mwh=1_000.0, v_step_mwh=1_000.0,
        global_min_mwh=5_000.0, global_max_mwh=5_000.0,
        vol=NEAR_ZERO_VOL, sMR=1.0, discount_rate=0.0, n_p=20)
    schedule = rt.build_reset_schedule(terms)

    curve = pd.Series(30.0, index=pd.date_range("2020-01-01", "2030-12-31", freq="D"))
    curve.loc["2026-05-01":"2026-05-15"] = 35.0
    curve.loc["2026-05-16":"2026-05-31"] = 30.0  # includes May 31 itself: K = 30.0

    point_pv = rse.value_point_reset_call_swing(terms, schedule, daily_curve=curve)
    averaged_pv = rsa.value_averaged_reset_call_swing(
        terms, schedule, curve, n_r=20, r_lo=29.0, r_hi=36.0)

    expected = 25_000.0  # 5 mandatory clips * 1,000 MWh * (35 - 30) EUR/MWh margin
    assert point_pv == pytest.approx(expected, rel=1e-4)
    assert averaged_pv == pytest.approx(expected, rel=1e-4)
    assert averaged_pv == pytest.approx(point_pv, abs=1e-3)
