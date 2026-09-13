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
])
def test_invalid_numerical_terms_are_refused(field, value, match):
    with pytest.raises(ValueError, match=match):
        _terms(**{field: value})


def test_global_min_above_max_is_refused():
    with pytest.raises(ValueError, match="global_min_mwh"):
        _terms(global_min_mwh=200_000.0, global_max_mwh=100_000.0)


def test_a_single_day_window_on_a_month_end_still_has_one_month():
    # storage_end before storage_start can't happen (constructor refuses it);
    # this is the smallest non-empty window the schedule builder accepts, to
    # confirm it doesn't need multiple days to produce a month.
    terms = _terms(val_date="2026-01-01", storage_start="2026-02-28", storage_end="2026-02-28")
    schedule = rt.build_reset_schedule(terms)
    assert len(schedule.months) == 1
    assert schedule.months[0].exercise_dates == (pd.Timestamp("2026-02-28"),)
