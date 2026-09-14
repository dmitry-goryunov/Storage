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
        if self.n_p <= 0:
            raise ValueError(f"n_p must be a positive integer, got {self.n_p!r}.")


@dataclass(frozen=True)
class DeliveryMonth:
    label: pd.Period
    fixing_date: pd.Timestamp
    exercise_dates: tuple  # pd.Timestamp, ascending, within [storage_start, storage_end]


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
        if fixing_date < terms.val_date:
            raise ValueError(
                f"Delivery month {month_period}'s fixing date {fixing_date:%Y-%m-%d} "
                f"is before val_date {terms.val_date:%Y-%m-%d} -- this prototype has "
                f"no historical-fixing input, so storage_start must leave at least "
                f"one full calendar month of lead time from val_date. Release 1B "
                f"scope, per DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.3.1.")
        assert fixing_date < exercise_dates[0], "fixing must strictly precede exercise"

        months.append(DeliveryMonth(month_period, fixing_date, exercise_dates))
        cursor = m_end + pd.Timedelta(days=1)

    if not months:
        raise ValueError("No delivery month falls inside the storage window.")

    month_end_dates = tuple(sm.month_end(m.label.to_timestamp()) for m in months)
    return ResetSchedule(terms=terms, months=tuple(months), month_end_dates=month_end_dates)
