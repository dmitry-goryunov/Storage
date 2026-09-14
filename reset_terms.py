"""Contract terms and event schedule for the monthly-reset swing prototype.

DESIGN-MONTHLY-RESET-SWING-2026-09-13.md Phase 0 / Release 1A (sec.11, sec.14.1-14.3).
No real term sheet exists for this feature -- the user asked for a generic model
where the strike resets from a month-ahead observation rather than a fixed one, not
for a specific deal. `ResetSwingTerms` therefore encodes Release 1A's PROTOTYPE
conventions explicitly, matching sec.2.1's own warning that "month-ahead" alone
does not determine a real contract's terms:

- call swing only (the sold-gas side; sec.14.1's first release scope);
- ONE fixing observation per delivery month -- the month-end POINT convention
  from sec.8/reset_forward.py, not a delivery-period average;
- the index is the model's OWN conditional projection of month-end spot, not an
  externally published one -- there is nothing else to index against without a
  real data source, and this must not be mistaken for a sourced market index;
- global (deal-wide), not per-month, minimum/maximum exercised volume;
- valuation always precedes the deal's first fixing window -- no partially or
  fully fixed months, no historical-fixing input. Release 1B's own scope.

Each simplification is named so a later release can drop it deliberately rather
than discover it was assumed.
"""
import math
from dataclasses import dataclass

import pandas as pd

import storage_model as sm


@dataclass(frozen=True)
class ResetSwingTerms:
    val_date: pd.Timestamp
    storage_start: pd.Timestamp
    storage_end: pd.Timestamp
    daily_max_mwh: float
    v_step_mwh: float
    global_min_mwh: float
    global_max_mwh: float
    vol: float
    sMR: float
    discount_rate: float
    n_p: int

    def __post_init__(self):
        val_date = pd.Timestamp(self.val_date)
        storage_start = pd.Timestamp(self.storage_start)
        storage_end = pd.Timestamp(self.storage_end)
        object.__setattr__(self, "val_date", val_date)
        object.__setattr__(self, "storage_start", storage_start)
        object.__setattr__(self, "storage_end", storage_end)

        if not storage_start <= storage_end:
            raise ValueError(
                f"storage_start {storage_start:%Y-%m-%d} must be on or before "
                f"storage_end {storage_end:%Y-%m-%d}.")
        if not val_date <= storage_start:
            raise ValueError(
                f"val_date {val_date:%Y-%m-%d} must be on or before "
                f"storage_start {storage_start:%Y-%m-%d} -- this prototype does not "
                f"support valuing a deal that has already started.")

        # 2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-09: every one of
        # the range checks below (`value > 0`, `0 <= lo <= hi`, `sMR < 0`) already
        # rejects NaN as a side effect of Python's NaN comparisons always being
        # False -- but +inf passes every one of them (inf > 0 is True), and the
        # error message for a NaN rejected this way ("must be strictly
        # positive") is misleading about what actually failed. Check finiteness
        # explicitly, and first, for a clear message either way.
        for name, value in (("daily_max_mwh", self.daily_max_mwh),
                           ("v_step_mwh", self.v_step_mwh),
                           ("global_min_mwh", self.global_min_mwh),
                           ("global_max_mwh", self.global_max_mwh),
                           ("vol", self.vol), ("sMR", self.sMR),
                           ("discount_rate", self.discount_rate),
                           ("n_p", self.n_p)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}.")

        for name, value in (("daily_max_mwh", self.daily_max_mwh),
                           ("v_step_mwh", self.v_step_mwh),
                           ("vol", self.vol)):
            if not value > 0:
                raise ValueError(f"{name} must be strictly positive, got {value!r}.")
        if not 0.0 <= self.global_min_mwh <= self.global_max_mwh:
            raise ValueError(
                f"Need 0 <= global_min_mwh <= global_max_mwh, got "
                f"{self.global_min_mwh!r} .. {self.global_max_mwh!r}.")

        # Refuse silent grid rounding, the same discipline
        # storage_model.normalise_storage_contract already enforces for the
        # fixed-strike engine -- found empirically here: v_step=2000 against
        # daily_max_mwh=1000 rounds int(round(1000/2000))=0 (banker's rounding
        # on an exact .5), silently pricing a swing that can never exercise
        # anything as if that were the requested contract, with no error.
        clips, achieved, ok = sm._grid_representable(self.daily_max_mwh, self.v_step_mwh)
        if not ok or clips < 1:
            raise ValueError(
                f"daily_max_mwh={self.daily_max_mwh:,.4f} is not expressible as a "
                f"whole number of v_step_mwh={self.v_step_mwh:,.4f} clips "
                f"(nearest: {clips} clip(s) = {achieved:,.4f} MWh/day). A clip size "
                f"at or below the daily rate is required; refine v_step_mwh rather "
                f"than silently round the daily rate to zero.")
        for name, value in (("global_min_mwh", self.global_min_mwh),
                           ("global_max_mwh", self.global_max_mwh)):
            clips, achieved, ok = sm._grid_representable(value, self.v_step_mwh)
            if not ok:
                raise ValueError(
                    f"{name}={value:,.4f} is not a whole number of "
                    f"v_step_mwh={self.v_step_mwh:,.4f} clips (nearest: {clips} "
                    f"clip(s) = {achieved:,.4f} MWh). Refine v_step_mwh or the "
                    f"requested volume rather than silently rounding it.")
        if self.sMR < 0:
            raise ValueError(f"sMR must be non-negative, got {self.sMR!r}.")
        if self.n_p <= 0 or self.n_p != int(self.n_p):
            raise ValueError(f"n_p must be a positive integer, got {self.n_p!r}.")


@dataclass(frozen=True)
class DeliveryMonth:
    label: pd.Period
    fixing_date: pd.Timestamp
    exercise_dates: tuple  # pd.Timestamp, ascending, within [storage_start, storage_end]
    fixing_observation_dates: tuple  # pd.Timestamp, ascending: EVERY calendar day of the
        # ONE calendar month immediately preceding this one -- independent of
        # exercise_dates, per DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.3's own
        # "reset observations and exercise dates are separate calendars." The two
        # coincide only when that preceding month is itself a COMPLETE delivery month
        # in this same deal; for the deal's first delivery month (or any month whose
        # predecessor falls partly or wholly before storage_start) they do not, and
        # conflating them was a real, independent-review-found defect (2026-09-14
        # INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-01/R-02): the first delivery
        # month's own average silently used every day since val_date instead of just
        # its one contractual preceding month, and a partial first delivery month's
        # own exercise days were silently substituted for the NEXT month's full
        # preceding-month window.


@dataclass(frozen=True)
class ResetSchedule:
    terms: ResetSwingTerms
    months: tuple  # DeliveryMonth, ascending by exercise window
    month_end_dates: tuple  # pd.Timestamp, one per month, for reset_forward.py


def build_reset_schedule(terms):
    """One term object maps to one deterministic ordered event schedule.

    The fixing date for delivery month M is the last calendar day of M-1 -- one
    month prior, per the user's own framing -- and every fixing strictly precedes
    every exercise date in the month it sets the strike for, by construction: no
    fixing-after-exercise case can arise from this schedule, so there is nothing
    to validate at exercise time for it.

    Each month's `fixing_observation_dates` is the FULL calendar month M-1, always
    -- regardless of `exercise_dates` (see `DeliveryMonth`'s own docstring for why
    these must be kept separate). `val_date` is required to precede the START of
    every month's own fixing window, not merely its end (`fixing_date`): the
    "wholly future" scope this prototype supports (sec.3.1's own three cases) means
    no part of any window may already be historical, and a val_date landing INSIDE
    a window used to pass the old end-only check while silently pricing as if the
    whole window were still ahead.
    """
    start, end = terms.storage_start, terms.storage_end
    if end != sm.month_end(end):
        raise ValueError(
            f"storage_end {end:%Y-%m-%d} does not land on a calendar month-end "
            f"({sm.month_end(end):%Y-%m-%d}). This prototype settles each delivery "
            f"month's index at that month's own calendar end -- a partial final "
            f"month would need a strike fixed from a date after exercise has "
            f"already stopped, which is a real settlement-convention question a "
            f"term sheet would have to answer, not one this generic prototype "
            f"resolves. A partial FIRST month is fine (storage_start mid-month "
            f"changes nothing about when that month settles).")

    months = []
    cursor = sm.month_start(start)
    while cursor <= end:
        month_period = pd.Period(cursor, freq="M")
        m_start, m_end = sm.month_start(cursor), sm.month_end(cursor)
        exercise_start = max(m_start, start)
        exercise_end = min(m_end, end)
        exercise_dates = tuple(pd.date_range(exercise_start, exercise_end, freq="D"))

        fixing_date = m_start - pd.Timedelta(days=1)
        fixing_window_start = sm.month_start(fixing_date)
        if fixing_window_start <= terms.val_date:
            raise ValueError(
                f"Delivery month {month_period}'s fixing observation window "
                f"({fixing_window_start:%Y-%m-%d} .. {fixing_date:%Y-%m-%d}) starts on "
                f"or before val_date {terms.val_date:%Y-%m-%d} -- this prototype has "
                f"no historical-fixing input, so val_date must precede the START of "
                f"every delivery month's own fixing window, not just its end "
                f"({fixing_date:%Y-%m-%d}). storage_start must leave at least one "
                f"full calendar month of lead time from val_date. Release 1B scope, "
                f"per DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.3.1.")
        fixing_observation_dates = tuple(pd.date_range(fixing_window_start, fixing_date, freq="D"))
        assert fixing_date < exercise_dates[0], "fixing must strictly precede exercise"

        months.append(DeliveryMonth(month_period, fixing_date, exercise_dates,
                                    fixing_observation_dates))
        cursor = m_end + pd.Timedelta(days=1)

    if not months:
        raise ValueError("No delivery month falls inside the storage window.")

    # 2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-09: a mandatory
    # global_min_mwh that exceeds what the deal could EVER deliver (every
    # exercise day, at the daily max) is not a hard constraint the DP can
    # satisfy -- every state would be infeasible, and value_averaged_reset_call_swing
    # / value_point_reset_call_swing would report a confusingly large negative
    # number (FORBIDDEN propagated to the root) rather than a clear refusal.
    # Checked here, not in ResetSwingTerms.__post_init__, because it needs the
    # actual exercise-day count, which only exists once the schedule is built.
    total_exercise_days = sum(len(m.exercise_dates) for m in months)
    max_deliverable_mwh = terms.daily_max_mwh * total_exercise_days
    if terms.global_min_mwh > max_deliverable_mwh:
        raise ValueError(
            f"global_min_mwh={terms.global_min_mwh:,.4f} exceeds the maximum this "
            f"deal could ever deliver: {total_exercise_days} exercise day(s) at "
            f"daily_max_mwh={terms.daily_max_mwh:,.4f} caps deliverable volume at "
            f"{max_deliverable_mwh:,.4f} MWh. This mandatory minimum can never be "
            f"satisfied -- widen the exercise window, raise daily_max_mwh, or "
            f"lower global_min_mwh.")

    month_end_dates = tuple(sm.month_end(m.label.to_timestamp()) for m in months)
    return ResetSchedule(terms=terms, months=tuple(months), month_end_dates=month_end_dates)
