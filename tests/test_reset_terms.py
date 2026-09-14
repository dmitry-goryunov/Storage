"""DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.10.1's contract/chronology tests,
against reset_terms.py's Phase 0 deliverable."""
import pandas as pd
import pytest

import reset_terms as rt


def _terms(**overrides):
    defaults = dict(
        val_date="2026-01-01", storage_start="2026-03-01", storage_end="2026-08-31",
        daily_max_mwh=10_000.0, v_step_mwh=5_000.0,
        global_min_mwh=0.0, global_max_mwh=300_000.0,
        vol=0.5, sMR=1.0, discount_rate=0.0, n_p=15)
    defaults.update(overrides)
    return rt.ResetSwingTerms(**defaults)


def test_schedule_generated_exactly_from_a_miniature_term_sheet():
    """A hand-checkable two-month deal: every fixing/exercise date named exactly."""
    terms = _terms(storage_start="2026-03-01", storage_end="2026-04-30")
    schedule = rt.build_reset_schedule(terms)

    assert len(schedule.months) == 2
    march, april = schedule.months
    assert march.fixing_date == pd.Timestamp("2026-02-28")
    assert march.exercise_dates[0] == pd.Timestamp("2026-03-01")
    assert march.exercise_dates[-1] == pd.Timestamp("2026-03-31")
    assert len(march.exercise_dates) == 31

    assert april.fixing_date == pd.Timestamp("2026-03-31")
    assert april.exercise_dates[0] == pd.Timestamp("2026-04-01")
    assert april.exercise_dates[-1] == pd.Timestamp("2026-04-30")
    assert len(april.exercise_dates) == 30

    assert schedule.month_end_dates == (pd.Timestamp("2026-03-31"), pd.Timestamp("2026-04-30"))


def test_a_partial_first_month_is_clipped_to_the_deal_window():
    """A mid-month storage_start must not manufacture exercise days before it --
    that month still settles at its own calendar end regardless."""
    terms = _terms(storage_start="2026-03-15", storage_end="2026-04-30")
    schedule = rt.build_reset_schedule(terms)
    assert len(schedule.months) == 2
    march, april = schedule.months
    assert march.exercise_dates[0] == pd.Timestamp("2026-03-15")
    assert march.exercise_dates[-1] == pd.Timestamp("2026-03-31")
    assert april.exercise_dates[0] == pd.Timestamp("2026-04-01")
    assert april.exercise_dates[-1] == pd.Timestamp("2026-04-30")


def test_fixing_observation_dates_is_the_full_calendar_month_not_exercise_dates():
    """2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-01/R-02: a delivery
    month's own `fixing_observation_dates` must be EVERY calendar day of the
    month before it, independent of `exercise_dates` -- the two coincide only
    when that preceding month is a COMPLETE delivery month in this same deal.
    March here is partial (15-31, clipped by storage_start), but April's own
    fixing_observation_dates must still be the FULL 1-31 March: the strike
    index is a contractual calendar, not the deal's own exercise window."""
    terms = _terms(storage_start="2026-03-15", storage_end="2026-04-30")
    schedule = rt.build_reset_schedule(terms)
    march, april = schedule.months

    assert march.fixing_observation_dates[0] == pd.Timestamp("2026-02-01")
    assert march.fixing_observation_dates[-1] == pd.Timestamp("2026-02-28")
    assert len(march.fixing_observation_dates) == 28

    assert april.fixing_observation_dates[0] == pd.Timestamp("2026-03-01")
    assert april.fixing_observation_dates[-1] == pd.Timestamp("2026-03-31")
    assert len(april.fixing_observation_dates) == 31
    # The point of the fix: April's window is wider than March's own exercise
    # window (15-31), reaching back to cover all of March, not just the days
    # this deal actually delivers on.
    assert april.fixing_observation_dates[0] < march.exercise_dates[0]
    assert set(march.exercise_dates) < set(april.fixing_observation_dates)


def test_fixing_observation_dates_matches_exercise_dates_for_a_complete_month():
    """For a month that is NOT the deal's (possibly partial) first one, its own
    exercise window already spans the whole calendar month, so the next
    month's fixing_observation_dates and this month's exercise_dates coincide
    exactly -- the degenerate case the original (buggy) implementation happened
    to get right, per R-02's own account of why the bug went unnoticed."""
    terms = _terms(storage_start="2026-03-01", storage_end="2026-05-31")
    schedule = rt.build_reset_schedule(terms)
    march, april, may = schedule.months
    assert april.fixing_observation_dates == march.exercise_dates
    assert may.fixing_observation_dates == april.exercise_dates


def test_valuation_inside_a_fixing_window_is_refused_not_silently_shortened():
    """2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-01 point 4: val_date
    must precede the START of every month's fixing window, not merely its end
    (the old check). storage_start leaves a full calendar month of lead time
    from storage_start's own month, but val_date itself lands inside that
    window (10 February, with the window being all of February) -- must be
    refused, not silently averaged over only the remaining Feb 10-28."""
    with pytest.raises(ValueError, match="must precede the START"):
        rt.build_reset_schedule(_terms(val_date="2026-02-10", storage_start="2026-03-01"))


def test_valuation_exactly_on_the_fixing_window_start_is_refused():
    """The boundary case: val_date lands exactly on the window's first day,
    not merely inside it -- must still be refused (strictly before, not
    before-or-equal), since that whole day's own observation is not yet
    available at valuation."""
    with pytest.raises(ValueError, match="must precede the START"):
        rt.build_reset_schedule(_terms(val_date="2026-02-01", storage_start="2026-03-01"))


def test_valuation_the_day_before_the_fixing_window_starts_is_accepted():
    terms = _terms(val_date="2026-01-31", storage_start="2026-03-01")
    schedule = rt.build_reset_schedule(terms)
    assert schedule.months[0].fixing_observation_dates[0] == pd.Timestamp("2026-02-01")


def test_a_partial_final_month_is_refused_not_guessed():
    """storage_end mid-month would need a strike fixed from a date after
    exercise has already stopped -- a real settlement question this generic
    prototype does not resolve, so it must refuse rather than silently clip."""
    with pytest.raises(ValueError, match="calendar month-end"):
        rt.build_reset_schedule(_terms(storage_start="2026-03-01", storage_end="2026-04-10"))


def test_every_fixing_strictly_precedes_its_own_months_exercise():
    """No fixing can influence an exercise decision that happens before it is
    published -- by construction here, since fixing = the day before the
    month it sets the strike for even starts."""
    terms = _terms(storage_start="2026-03-01", storage_end="2026-12-31")
    schedule = rt.build_reset_schedule(terms)
    for month in schedule.months:
        assert month.fixing_date < month.exercise_dates[0]


def test_global_volume_limits_are_carried_not_split_by_month():
    """This prototype's volume constraint is deal-wide, not per-month -- an
    explicit simplification, not a silent one."""
    terms = _terms(global_min_mwh=50_000.0, global_max_mwh=250_000.0)
    schedule = rt.build_reset_schedule(terms)
    assert schedule.terms.global_min_mwh == 50_000.0
    assert schedule.terms.global_max_mwh == 250_000.0


def test_a_fixing_before_valuation_date_is_refused_not_guessed():
    """This prototype carries no historical-fixing input (Release 1B scope). A
    mid-month storage_start still has a fixing date at the PRIOR month's end
    (28 Feb here) -- if val_date falls after that but still on or before
    storage_start (10 .. 15 March), the deal has not started yet by
    __post_init__'s check, but its first fixing already would have needed to
    happen in the past. Must be refused, not silently priced as if unfixed."""
    with pytest.raises(ValueError, match="historical-fixing"):
        rt.build_reset_schedule(_terms(val_date="2026-03-10", storage_start="2026-03-15"))


def test_valuation_after_storage_start_is_refused():
    with pytest.raises(ValueError, match="already started"):
        _terms(val_date="2026-04-01", storage_start="2026-03-01")


@pytest.mark.parametrize("field,value,match", [
    ("daily_max_mwh", 0.0, "daily_max_mwh"),
    ("v_step_mwh", -1.0, "v_step_mwh"),
    ("vol", 0.0, "vol"),
    ("sMR", -0.1, "sMR"),
    ("n_p", 0, "n_p"),
    ("n_p", 8.5, "n_p"),
])
def test_invalid_numerical_terms_are_refused(field, value, match):
    with pytest.raises(ValueError, match=match):
        _terms(**{field: value})


@pytest.mark.parametrize("field", [
    "daily_max_mwh", "v_step_mwh", "global_min_mwh", "global_max_mwh",
    "vol", "sMR", "discount_rate", "n_p",
])
@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_numerical_terms_are_refused(field, bad_value):
    """2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-09: +inf passes every
    one of the existing range checks (inf > 0 is True), and NaN, while it does
    already fail them (NaN comparisons are always False), gets a misleading
    message ("must be strictly positive") that names the wrong problem."""
    with pytest.raises(ValueError, match="finite"):
        _terms(**{field: bad_value})


def test_global_min_above_max_is_refused():
    with pytest.raises(ValueError, match="global_min_mwh"):
        _terms(global_min_mwh=200_000.0, global_max_mwh=100_000.0)


def test_a_mandatory_minimum_that_cannot_be_delivered_is_refused():
    """2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-09: global_min_mwh
    <= global_max_mwh alone is not sufficient -- a one-day deal can only ever
    deliver one day's worth, however generous global_max is. Checked in
    build_reset_schedule, not ResetSwingTerms.__post_init__, since it needs
    the actual exercise-day count. Without this, the DP would make every
    state infeasible and report a large, confusing negative number instead
    of refusing outright."""
    terms = _terms(storage_start="2026-03-30", storage_end="2026-03-31",  # two exercise days
                   daily_max_mwh=5_000.0, v_step_mwh=5_000.0,
                   global_min_mwh=15_000.0, global_max_mwh=20_000.0)  # min <= max, but > 2 days' worth
    with pytest.raises(ValueError, match="global_min_mwh"):
        rt.build_reset_schedule(terms)


def test_a_daily_rate_that_rounds_to_zero_clips_is_refused():
    """Found empirically while running the v_step convergence ladder:
    int(round(daily_max_mwh / v_step_mwh)) rounds 1000/2000=0.5 down to 0 under
    Python's banker's rounding, silently pricing a swing that can never
    exercise anything as if that were the requested contract -- the same
    defect class normalise_storage_contract exists to catch in the fixed-
    strike engine. Must refuse, not silently round the daily rate to zero."""
    with pytest.raises(ValueError, match="daily_max_mwh"):
        _terms(daily_max_mwh=1_000.0, v_step_mwh=2_000.0)


def test_a_global_volume_not_expressible_on_the_grid_is_refused():
    with pytest.raises(ValueError, match="global_max_mwh"):
        _terms(v_step_mwh=1_000.0, global_max_mwh=10_500.0)
    with pytest.raises(ValueError, match="global_min_mwh"):
        _terms(v_step_mwh=1_000.0, global_min_mwh=500.0, global_max_mwh=10_000.0)


def test_a_single_day_window_on_a_month_end_still_has_one_month():
    # storage_end before storage_start can't happen (constructor refuses it);
    # this is the smallest non-empty window the schedule builder accepts, to
    # confirm it doesn't need multiple days to produce a month. val_date is
    # 2025-12-31, not 2026-01-01: the fixing window is the full calendar month
    # of January, and val_date must precede its START, not land exactly on it.
    terms = _terms(val_date="2025-12-31", storage_start="2026-02-28", storage_end="2026-02-28")
    schedule = rt.build_reset_schedule(terms)
    assert len(schedule.months) == 1
    assert schedule.months[0].exercise_dates == (pd.Timestamp("2026-02-28"),)
