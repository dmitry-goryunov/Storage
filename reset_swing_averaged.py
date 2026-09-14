"""The averaged-reset exact benchmark: Release 1B (DESIGN-MONTHLY-RESET-SWING-
2026-09-13.md sec.6.2, sec.7.1, sec.11 Phase 2). Point-reset (reset_swing_exact.py)
fixes each month's strike from a SINGLE observation; this fixes it from the
EQUAL-WEIGHTED average of the model's month-end projection, observed on EVERY
calendar day of the preceding month -- "the strike resets month-ahead" as an
actual monthly average, not a point-reset stand-in for one.

State during delivery month M is genuinely (i, j, l, r): time, price node,
cumulative exercised volume, and the running average accumulating toward
M+1's strike. Unlike point-reset's j_fix (sec.6.4: an exact function of a
single lattice node, no grid needed), r is genuinely continuous -- different
paths reaching the SAME node can have accumulated DIFFERENT running
averages -- so it needs a real discretised grid with interpolation (sec.7.1's
explicit instruction, not avoidable here the way it was for point-reset).

Month-chaining structure (this took a real bug to get right -- see the r_grid
sizing test's docstring below for what went wrong first): each month's own
processing loops over every INCOMING accumulator bucket (fixing that month's
own strike) and, unless it is the deal's last month, tracks a FRESH outgoing
accumulator for the following month. Right when that outgoing tracking
starts (at the month's own fixing date, working backward), the axis has not
accumulated anything yet, so -- by the same boundary property already used
for the pre-deal window in `reset_swing_exact`'s point-reset code -- its
value must be independent of which bucket you look at. That lets the axis be
collapsed to a plain (width, n_l) array before it is handed to the PRECEDING
month as that month's own terminal condition, rather than carried forward as
an ever-growing stack of "one axis per remaining future month" (which would
cost O(n_r^(months-1)) and was the actual shape of the first, wrong,
attempt at this).

Convergence in n_r (tests/test_reset_swing_averaged.py has both the proof and
the regression test): the value-vs-K surface `accumulate_step` interpolates
across has a genuine KINK at the exercise boundary (from the max() in
`_exercise_step_3d`/`rse._exercise_step`), and linear interpolation across a
kink converges only at FIRST order in the grid spacing -- O(1/n_r), not
O(1/n_r^2) the way it would for a smooth function. Confirmed empirically: at
low vol, quadrupling n_r cuts the residual gap to point-reset by very close
to 4x, consistently. This is NOT the bug it first looks like (a coarse-n_r
run can sit far from the truth for a long time even though the direction is
always correct) -- three separate scenario-setup mistakes were chased and
ruled out as the actual explanation before this convergence order was
confirmed by literal brute-force enumeration matching to ~1e-12. Practical
consequence: pick n_r generously (results here use up to several hundred to
a few thousand) and bracket r_lo/r_hi tightly around the curve's own range --
a wide, sparse grid is the single biggest source of apparent-but-fake
disagreement with point-reset, not a sign of a logic error.
"""
import numpy as np
import pandas as pd

import reset_forward as rf
import reset_swing_exact as rse


def _exercise_step_3d(continuation, spot, strike, df_i, v_step, daily_max_clips):
    """continuation: (width, n_l, n_r). Exercise optimisation along l; spot and
    strike are fixed for the whole month, so this broadcasts over r untouched."""
    width, n_l, n_r = continuation.shape
    payoff_per_clip = df_i * v_step * (spot - strike)
    best = continuation.copy()
    for d in range(1, daily_max_clips + 1):
        if d >= n_l:
            break
        candidate = np.full_like(continuation, -np.inf)
        candidate[:, : n_l - d, :] = continuation[:, d:, :]
        candidate += (d * payoff_per_clip)[:, None, None]
        best = np.maximum(best, candidate)
    return best


def _propagate_price_step_3d(continuation, p_u_i, p_m_i, p_d_i):
    """continuation: (width, n_l, n_r). E_j'[continuation[j', :, :] | j]."""
    cont = p_m_i[:, None, None] * continuation
    cont[:-1] += p_u_i[:-1, None, None] * continuation[1:]
    cont[1:] += p_d_i[1:, None, None] * continuation[:-1]
    return cont


def accumulate_step(continuation, r_grid, quote_by_node, weight_so_far, day_weight=1.0):
    """continuation: (width, n_r) OR (width, n_l, n_r) -- value as a function of
    the running average BEFORE today's observation. Folds in today's own quote
    `quote_by_node[j]` (weight `day_weight`) into the running average, and
    interpolates the result back onto `r_grid` -- `np.interp`'s default clamps
    outside the grid rather than extrapolating (sec.7.1: never extrapolate
    silently). Exposed and tested standalone, not just as an internal step,
    since this is the one genuinely new numerical operation point-reset never
    needed.
    """
    width = continuation.shape[0]
    new_weight = weight_so_far + day_weight
    if continuation.ndim == 2:
        out = np.empty_like(continuation)
        for j in range(width):
            r_new = (r_grid * weight_so_far + day_weight * quote_by_node[j]) / new_weight
            out[j, :] = np.interp(r_new, r_grid, continuation[j, :])
        return out
    n_l = continuation.shape[1]
    out = np.empty_like(continuation)
    for j in range(width):
        r_new = (r_grid * weight_so_far + day_weight * quote_by_node[j]) / new_weight
        for l in range(n_l):
            out[j, l, :] = np.interp(r_new, r_grid, continuation[j, l, :])
    return out


def _run_accumulation_only(lattice, dates, r_grid, quote_by_date, terminal_value):
    """Track ONLY the running-average accumulator (j, r) backward through
    `dates` -- no exercise, no volume axis. `terminal_value`: (width, n_r),
    the value AFTER the last of `dates` has been folded in.
    `quote_by_date[date]`: (width,) array, that date's own H[month_end][.,:]
    row. Used for the pre-deal window (month 1's own fixing has no preceding
    delivery month to inherit a terminal condition from) and independently
    testable on its own, since the accumulator has an exact, hand-checkable
    answer regardless of any exercise decision.
    """
    date_span = lattice["date_span"]
    p_u, p_m, p_d = lattice["p_u"], lattice["p_m"], lattice["p_d"]
    v = terminal_value
    n_days = len(dates)
    for pos, date in enumerate(reversed(dates)):
        i = date_span.get_loc(date)
        weight_so_far = float(n_days - 1 - pos)
        v = accumulate_step(v, r_grid, quote_by_date[date], weight_so_far, 1.0)
        if i > 0:
            v = rse._propagate_one_step(v, p_u[i - 1, :], p_m[i - 1, :], p_d[i - 1, :])
    return v


def _run_month(lattice, month, r_grid, quotes_for_next_month, terminal_value,
              v_step, daily_max_clips):
    """One delivery month M, for every incoming accumulator bucket in r_grid
    (fixing K_M = r_grid[k], the identity reset formula -- sec.2's simple
    case) at once. `terminal_value`: a SINGLE (width, n_l) array -- the SAME
    for every k, because K_M does not affect anything after month M ends.
    `quotes_for_next_month`: (n_t, width) array, H[month_end(M+1)][i, :] for
    every date i -- this month's own days are M+1's accumulation window; None
    for the deal's last month (nothing left to accumulate for).

    Returns a list of n_r arrays, one per incoming bucket k: (width, n_l, n_r)
    if accumulating (the fresh, not-yet-collapsed outgoing axis -- the
    caller collapses it, see `_collapse_fresh_axis`), else (width, n_l).

    When accumulating, `terminal_value` is ALREADY (width, n_l, n_r) -- the
    stacked, collapsed result from whichever month follows this one -- not a
    2D array needing to be broadcast into one: that broadcast was the second
    bug this module's first version had, right after the diagonal-collapse
    one (see `_collapse_fresh_axis`'s own docstring).
    """
    date_span = lattice["date_span"]
    x, p_u, p_m, p_d, d_curve = (lattice["x"], lattice["p_u"], lattice["p_m"],
                                 lattice["p_d"], lattice["d_curve"])
    exercise_indices = [date_span.get_loc(d) for d in month.exercise_dates]
    fixing_idx = date_span.get_loc(month.fixing_date)
    n_r = len(r_grid)
    n_days = len(exercise_indices)
    accumulate = quotes_for_next_month is not None
    if accumulate:
        assert terminal_value.ndim == 3 and terminal_value.shape[2] == n_r, (
            f"expected an already-stacked (width, n_l, {n_r}) terminal when "
            f"accumulating, got shape {terminal_value.shape}")
    else:
        assert terminal_value.ndim == 2, (
            f"expected a plain (width, n_l) terminal for the last month, "
            f"got shape {terminal_value.shape}")

    results = []
    for k in range(n_r):
        strike = float(r_grid[k])
        v = terminal_value.copy()

        for pos, i in enumerate(reversed(exercise_indices)):
            spot = np.exp(x[i, :])
            if accumulate:
                weight_so_far = float(n_days - 1 - pos)
                v = accumulate_step(v, r_grid, quotes_for_next_month[i, :], weight_so_far, 1.0)
                v = _exercise_step_3d(v, spot, strike, d_curve[i], v_step, daily_max_clips)
            else:
                v = rse._exercise_step(v, spot, strike, d_curve[i], v_step, daily_max_clips)
            if i > 0:
                v = (_propagate_price_step_3d(v, p_u[i - 1, :], p_m[i - 1, :], p_d[i - 1, :])
                    if accumulate else
                    rse._propagate_one_step(v, p_u[i - 1, :], p_m[i - 1, :], p_d[i - 1, :]))

        first_i = exercise_indices[0]
        for i in range(first_i - 2, fixing_idx - 1, -1):
            v = (_propagate_price_step_3d(v, p_u[i, :], p_m[i, :], p_d[i, :]) if accumulate
                else rse._propagate_one_step(v, p_u[i, :], p_m[i, :], p_d[i, :]))
        results.append(v)
    return results


def _collapse_fresh_axis(results, label=""):
    """`results[k]`: (width, n_l, n_r), the OUTGOING accumulator for the month
    AFTER the one just processed, evaluated at that month's own fixing date --
    i.e. an axis that has not accumulated anything yet from this point's
    perspective. By the same boundary property `value_averaged_reset_call_swing`
    checks explicitly for the pre-deal window, its value must not depend on
    which bucket of that fresh axis you read: not doing this, and instead
    diagonal-collapsing it against the UNRELATED incoming-k axis, was the
    actual bug behind this module's first (wrong) version -- it does not
    raise, it just quietly prices something else, which is why every call
    site checks the spread rather than trusting the shape alone.

    Returns a list of n_r arrays, each (width, n_l) -- one per incoming k,
    with the fresh axis collapsed to a single representative slice.
    """
    collapsed = []
    for k, arr in enumerate(results):
        # Per-(j,l) spread across r, THEN the worst one -- not the global max
        # across every (j,l,r) minus the global min, which mixes genuinely
        # different (j,l) economic states together and is always huge
        # regardless of whether the r-axis itself is constant. Getting this
        # reduction backwards was a real bug here: it rejected an already-
        # correct computation, not the other way round.
        per_jl_spread = arr.max(axis=-1) - arr.min(axis=-1)  # (width, n_l)
        spread = float(per_jl_spread.max())
        # Absolute-EUR floor as well as relative: near a genuine zero the
        # relative test alone is meaningless.
        scale = max(1.0, float(np.abs(arr).max()))
        if spread > 1e-6 * scale:
            raise RuntimeError(
                f"{label} incoming bucket {k}: fresh outgoing-accumulator axis "
                f"should not yet depend on its own value, but varies by "
                f"{spread:.6g} (scale {scale:.6g}) -- widen r_lo/r_hi or "
                f"increase n_r before trusting this result.")
        collapsed.append(arr[:, :, 0])
    return collapsed


def value_averaged_reset_call_swing(terms, schedule, daily_curve, n_r, r_lo, r_hi):
    """The Release 1B root value: EUR PV at (val_date, centre price node, zero
    cumulative volume). Call swing, equal-weighted monthly average, identity
    reset formula (sec.2's simple case), same scope restrictions as
    `reset_terms.ResetSwingTerms` otherwise. `n_r`, `r_lo`, `r_hi`: the
    accumulator grid -- pick `r_lo`/`r_hi` to comfortably bracket the curve's
    own range; `_collapse_fresh_axis` and the pre-deal check below will raise
    rather than silently misprice if the grid turns out too narrow.
    """
    lattice = rf.build_lattice(terms.val_date, terms.storage_start, terms.storage_end,
                               vol=terms.vol, sMR=terms.sMR, n_p=terms.n_p,
                               daily_curve=daily_curve, discount_rate=terms.discount_rate)
    date_span = lattice["date_span"]
    v_step = terms.v_step_mwh
    n_l = int(round(terms.global_max_mwh / v_step)) + 1
    lo_clip = int(round(terms.global_min_mwh / v_step))
    hi_clip = n_l - 1
    daily_max_clips = int(round(terms.daily_max_mwh / v_step))
    width = 2 * terms.n_p + 1
    r_grid = np.linspace(r_lo, r_hi, n_r)

    all_h = rf.project_month_end_quotes(lattice, schedule.month_end_dates)

    l_grid = np.arange(n_l)
    admissible = (l_grid >= lo_clip) & (l_grid <= hi_clip)
    terminal_value = np.broadcast_to(
        np.where(admissible, 0.0, -np.inf), (width, n_l)).copy()

    months = schedule.months
    month_ends = schedule.month_end_dates
    for idx in range(len(months) - 1, -1, -1):
        month = months[idx]
        quotes_next = None
        if idx + 1 < len(months):
            u_next = date_span.get_loc(month_ends[idx + 1])
            quotes_next = all_h[u_next]
        results = _run_month(lattice, month, r_grid, quotes_next, terminal_value,
                             v_step, daily_max_clips)
        if quotes_next is None:
            # results[k] is already (width, n_l): no outgoing axis to collapse.
            collapsed = results
        else:
            collapsed = _collapse_fresh_axis(results, label=str(month.label))
        if idx == 0:
            # collapsed[k]: value at month 1's own fixing date given
            # K_1 = r_grid[k]. Needed unresolved (per r-bucket) for the
            # pre-deal accumulation step below, not stacked any further.
            first_month_terminal_by_r_in = collapsed
            break
        # collapsed[k]: value at THIS month's fixing date given K_this=r_grid[k].
        # That IS exactly the PRECEDING month's own outgoing-accumulator axis --
        # stack it into the (width, n_l, n_r) terminal the preceding month's
        # own accumulate_step calls will interpolate onto directly, with no
        # further broadcasting.
        terminal_value = np.stack(collapsed, axis=-1)
    # After the loop, `first_month_terminal_by_r_in[k]` is the value at month
    # 1's own fixing date given K_1 = r_grid[k]. K_1 is not a free choice --
    # it is whatever the pre-deal window (val_date .. month 1's fixing date)
    # actually accumulates, generally with real randomness. Reuse the exact
    # same accumulation machinery for that window, not a shortcut.
    u_first = date_span.get_loc(month_ends[0])
    pre_deal_quotes = all_h[u_first]
    pre_deal_dates = list(pd.date_range(terms.val_date, months[0].fixing_date, freq="D"))[1:]
    v_by_r_l0 = np.stack([arr[:, 0] for arr in first_month_terminal_by_r_in], axis=1)  # (width, n_r)
    if pre_deal_dates:
        quote_by_date = {d: pre_deal_quotes[date_span.get_loc(d), :] for d in pre_deal_dates}
        v_at_root_by_r = _run_accumulation_only(
            lattice, pre_deal_dates, r_grid, quote_by_date, v_by_r_l0)
    else:
        v_at_root_by_r = v_by_r_l0

    root_row = v_at_root_by_r[terms.n_p, :]
    spread = float(root_row.max() - root_row.min())
    scale = max(1.0, float(np.abs(root_row).max()))
    if spread > 1e-6 * scale:
        raise RuntimeError(
            f"root value should not depend on the (not-yet-accumulated) "
            f"starting r bucket, but varies by {spread:.6g} (scale {scale:.6g}) "
            f"-- widen r_lo/r_hi or increase n_r before trusting this result.")
    return float(root_row[0])
