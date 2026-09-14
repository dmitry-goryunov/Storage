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

Same-day ordering, an explicit named convention (2026-09-14
INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-03): on a day that is both a fixing-
observation day for M+1 and an exercise day for M, this module folds today's
quote into the running average FIRST, then makes today's exercise decision
against it -- "fixing-before-exercise". This is correct only if the fixing
observation is genuinely available before the nomination/exercise deadline;
if real settlement publishes AFTER nomination, the recursion would be giving
the exercise decision look-ahead it would not really have. No real term
sheet exists, so which order a real contract would specify is unknown --
what is fixed here is that the code consistently implements ONE named order
rather than an unstated, arbitrary one, and that the choice is genuinely
consequential, not a distinction without a difference: `accumulate_step`
(interpolation) and `_exercise_step_3d` (a pointwise max) do not commute in
general, so "fixing-after-exercise" is not merely an equivalent
reformulation -- confirmed directly, not just argued, in
`tests/test_reset_swing_averaged.py::test_same_day_ordering_is_fixing_before_exercise_and_the_choice_is_consequential`.
A real term sheet specifying the opposite order would need this module's own
per-day loop restructured (exercise before accumulate on the affected days),
not merely a parameter flip.

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

Release 2 (sec.13): "pick n_r generously" was not practical at first -- the
pure-Python accumulating-month recursion is O(n_r^2), and a realistic 6-month
deal at an n_r worth trusting extrapolated to on the order of an HOUR,
measured before `reset_swing_kernels.run_month_accumulate_core` (a Numba
kernel `_run_month` now delegates to for every non-last month) cut that to
minutes: the SAME 6-month deal reaches n_r=300 (residual gap to n_r=600 under
0.3% of PV) in about 3 minutes, n_r=30 -- the scale that used to take 54
seconds -- in under 2. Two numpy-vectorised attempts at just the interpolation
step were tried first and reverted (see `accumulate_step`'s own docstring):
both helped some deal shapes and hurt others, since the real cost was
Python/numpy per-call dispatch overhead, not the arithmetic -- only compiling
the WHOLE per-k recursion, removing that dispatch overhead categorically,
gave a shape-independent win.
"""
import math

import numpy as np
import pandas as pd

import reset_forward as rf
import reset_swing_exact as rse
import reset_swing_kernels as rsk


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

    Tried and reverted (Release 2's own "measure before choosing" question,
    sec.13): two numpy-vectorised rewrites of the `j`/`l` Python loops below
    (uniform-grid index arithmetic replacing `np.interp`, one via
    `np.take_along_axis`, one via a width-loop with flat-indexed gathers).
    Both were a genuine 2-5x win for a LARGE-n_l, small-to-moderate-n_r shape
    (e.g. a 60,000 MWh/500 MWh-clip deal), but 1.3-5x WORSE for a small-n_l,
    large-n_r shape (e.g. an 8,000 MWh/1,000 MWh-clip deal needing a fine
    r_grid) -- numpy per-call dispatch overhead dominating in one regime,
    `np.interp`'s own tight C loop winning in the other, with no shape-
    independent middle ground found. Kept the simple, uniformly-correct
    loop rather than ship a regression that depends on which deal you price.
    A real fix for the O(n_r^2)-per-accumulating-month cost this hot path
    sits in needs to eliminate per-call Python/numpy dispatch overhead
    categorically, which is what storage_kernels.py already uses Numba for
    elsewhere in this project -- not attempted here without checking in
    first, given the size of that undertaking and the re-verification it
    would need against every brute-force test in this module.
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


def _run_month(lattice, month, r_grid, quotes_for_next_month, fixing_observation_dates,
              terminal_value, v_step, daily_max_clips):
    """One delivery month M, for every incoming accumulator bucket in r_grid
    (fixing K_M = r_grid[k], the identity reset formula -- sec.2's simple
    case) at once. `terminal_value`: a SINGLE (width, n_l) array -- the SAME
    for every k, because K_M does not affect anything after month M ends.
    `quotes_for_next_month`: (n_t, width) array, H[month_end(M+1)][i, :] for
    every date i. `fixing_observation_dates`: M+1's OWN fixing-observation
    window (`schedule.months[idx+1].fixing_observation_dates`) -- the FULL
    calendar month M itself falls in, independent of M's own `exercise_dates`
    (see `reset_terms.DeliveryMonth`'s own docstring for why these must be
    kept separate: 2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-01/R-02
    found the two conflated, silently wrong whenever M is a partial delivery
    month). Both `quotes_for_next_month` and `fixing_observation_dates` are
    None together for the deal's last month (nothing left to accumulate for).

    Returns a list of n_r plain (width, n_l) arrays, one per incoming bucket
    k -- ALWAYS already collapsed now (R-04, 2026-09-14
    INDEPENDENT-REVIEW-MONTHLY-RESET-SWING: the kernel used to return the
    complete, un-collapsed (width, n_l, n_r) fresh outgoing-accumulator axis
    per k, 10.1 GiB at the notebook's own n_r=600, for the caller to collapse
    with `_collapse_fresh_axis`; the kernel now does that reduction itself,
    per k, before returning anything, and this function does the identical
    raise-with-message the caller used to do, using the kernel's own
    precomputed spread/scale -- so callers no longer call
    `_collapse_fresh_axis` on this function's own output. That function still
    exists, for `_run_month_accumulate_reference`'s own (deliberately
    un-collapsed, for testing) output.

    When accumulating, `terminal_value` is ALREADY (width, n_l, n_r) -- the
    stacked, collapsed result from whichever month follows this one -- not a
    2D array needing to be broadcast into one: that broadcast was the second
    bug this module's first version had, right after the diagonal-collapse
    one (see `_collapse_fresh_axis`'s own docstring).

    The accumulating case (every non-last month) delegates the actual
    per-k-bucket recursion to `reset_swing_kernels.run_month_accumulate_core`
    -- a Numba-compiled kernel computing EXACTLY what `_run_month_accumulate_reference`
    below computes in pure Python (see that function, `reset_swing_kernels.py`'s
    own docstring, and tests/test_reset_swing_kernels.py, which checks the two
    agree, collapse included). The deal's LAST month (`accumulate=False`)
    keeps the plain Python loop below directly: it has no r-axis
    interpolation or fresh axis of its own and was never the bottleneck
    (sec.13's Release 2 measurement).
    """
    date_span = lattice["date_span"]
    x, p_u, p_m, p_d, d_curve = (lattice["x"], lattice["p_u"], lattice["p_m"],
                                 lattice["p_d"], lattice["d_curve"])
    exercise_indices = [date_span.get_loc(d) for d in month.exercise_dates]
    fixing_idx = date_span.get_loc(month.fixing_date)
    n_r = len(r_grid)
    accumulate = quotes_for_next_month is not None
    if accumulate:
        assert terminal_value.ndim == 3 and terminal_value.shape[2] == n_r, (
            f"expected an already-stacked (width, n_l, {n_r}) terminal when "
            f"accumulating, got shape {terminal_value.shape}")
    else:
        assert terminal_value.ndim == 2, (
            f"expected a plain (width, n_l) terminal for the last month, "
            f"got shape {terminal_value.shape}")

    if accumulate:
        fixing_indices = [date_span.get_loc(d) for d in fixing_observation_dates]
        n_exercise_days = len(exercise_indices)
        assert fixing_indices[-n_exercise_days:] == exercise_indices, (
            f"{month.label}: fixing_observation_dates must end exactly at "
            f"this month's own exercise_dates -- they cover the same calendar "
            f"month by construction, so a mismatch here means the schedule "
            f"itself is wrong, not just this call's arguments.")
        fixing_indices_arr = np.asarray(fixing_indices, dtype=np.int64)
        collapsed, spread, scale = rsk.run_month_accumulate_core(
            np.ascontiguousarray(x), np.ascontiguousarray(p_u), np.ascontiguousarray(p_m),
            np.ascontiguousarray(p_d), np.ascontiguousarray(d_curve), fixing_indices_arr,
            n_exercise_days, np.ascontiguousarray(r_grid),
            np.ascontiguousarray(quotes_for_next_month),
            np.ascontiguousarray(terminal_value), float(v_step), int(daily_max_clips))
        # Same check `_collapse_fresh_axis` used to make on the (now never
        # materialised) full array -- the kernel precomputes spread/scale per
        # k itself since discarding the fresh axis loses the information
        # needed to check it here otherwise.
        for k in range(n_r):
            if spread[k] > 1e-6 * scale[k]:
                raise RuntimeError(
                    f"{month.label} incoming bucket {k}: fresh outgoing-accumulator "
                    f"axis should not yet depend on its own value, but varies by "
                    f"{spread[k]:.6g} (scale {scale[k]:.6g}) -- widen r_lo/r_hi or "
                    f"increase n_r before trusting this result.")
        return [collapsed[k] for k in range(n_r)]

    results = []
    for k in range(n_r):
        strike = float(r_grid[k])
        v = terminal_value.copy()

        for pos, i in enumerate(reversed(exercise_indices)):
            spot = np.exp(x[i, :])
            v = rse._exercise_step(v, spot, strike, d_curve[i], v_step, daily_max_clips)
            if i > 0:
                v = rse._propagate_one_step(v, p_u[i - 1, :], p_m[i - 1, :], p_d[i - 1, :])

        first_i = exercise_indices[0]
        for i in range(first_i - 2, fixing_idx - 1, -1):
            v = rse._propagate_one_step(v, p_u[i, :], p_m[i, :], p_d[i, :])
        results.append(v)
    return results


def _run_month_accumulate_reference(lattice, month, r_grid, quotes_for_next_month,
                                    fixing_observation_dates, terminal_value,
                                    v_step, daily_max_clips):
    """`_run_month`'s own accumulate=True path BEFORE it was wired to the
    Numba kernel -- the pure-Python `for k in range(n_r)` loop over
    `accumulate_step`/`_exercise_step_3d`/`_propagate_price_step_3d`, kept
    verbatim as an independent reference `tests/test_reset_swing_kernels.py`
    checks the kernel against directly. Not called by `_run_month` itself any
    more (that would make the comparison circular -- compiled kernel against
    itself); exists ONLY so a change to the kernel has something independent,
    slow-but-trusted to be checked against, the same role
    `_run_month_for_one_fixing_node` plays for the point-reset benchmark.

    Walks `fixing_observation_dates` (the FULL fixing window for the month
    after this one), not `month.exercise_dates`: every one of those days
    folds into the running average, but only its trailing suffix -- this
    month's own actual `exercise_dates` -- also gets an exercise decision.
    Because `fixing_observation_dates` always reaches back to this month's
    own fixing date already (it IS the calendar month this delivery month
    falls in), no separate gap-closing propagation is needed afterwards --
    unlike `_run_month`'s own `accumulate=False` branch above, which still
    needs one for a different reason (this month's OWN lead-in gap between
    its fixing date and its first exercise day, when this month itself is a
    partial one).
    """
    date_span = lattice["date_span"]
    x, p_u, p_m, p_d, d_curve = (lattice["x"], lattice["p_u"], lattice["p_m"],
                                 lattice["p_d"], lattice["d_curve"])
    exercise_indices = [date_span.get_loc(d) for d in month.exercise_dates]
    fixing_indices = [date_span.get_loc(d) for d in fixing_observation_dates]
    n_r = len(r_grid)
    n_fixing_days = len(fixing_indices)
    n_exercise_days = len(exercise_indices)
    assert fixing_indices[-n_exercise_days:] == exercise_indices, (
        f"{month.label}: fixing_observation_dates must end exactly at this "
        f"month's own exercise_dates.")

    results = []
    for k in range(n_r):
        strike = float(r_grid[k])
        v = terminal_value.copy()

        for pos, i in enumerate(reversed(fixing_indices)):
            weight_so_far = float(n_fixing_days - 1 - pos)
            v = accumulate_step(v, r_grid, quotes_for_next_month[i, :], weight_so_far, 1.0)
            if pos < n_exercise_days:
                spot = np.exp(x[i, :])
                v = _exercise_step_3d(v, spot, strike, d_curve[i], v_step, daily_max_clips)
            if i > 0:
                v = _propagate_price_step_3d(v, p_u[i - 1, :], p_m[i - 1, :], p_d[i - 1, :])
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


def value_averaged_reset_call_swing(terms, schedule, daily_curve, n_r, r_lo, r_hi, quotes=None):
    """The Release 1B root value: EUR PV at (val_date, centre price node, zero
    cumulative volume). Call swing, equal-weighted monthly average, identity
    reset formula (sec.2's simple case), same scope restrictions as
    `reset_terms.ResetSwingTerms` otherwise. `n_r`, `r_lo`, `r_hi`: the
    accumulator grid -- pick `r_lo`/`r_hi` to comfortably bracket EVERY
    month's own projected quote, not just the curve's spot level: a shaped
    curve gives each delivery month a genuinely different projected range
    (confirmed the hard way -- see tests/test_reset_swing_averaged.py's
    multi-month brute-force test, whose first attempt bracketed only the
    LAST month's range and silently clamped the first month's higher one to
    the grid edge, no exception, a confidently wrong answer). The check just
    below turns that into a raise instead.

    `quotes`: an optional precomputed `{month_end_index: H}` (from
    `reset_forward.project_month_end_quotes`) to use for the RESET STRIKES in
    place of the ones this call would otherwise derive from its own lattice --
    the same override `reset_swing_exact.value_point_reset_call_swing` takes,
    for the same reason: `compute_deltas` below freezes one lattice's strikes
    while bumping the other's spot.
    """
    # 2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-09: n_r, r_lo, r_hi
    # are caller-supplied numbers with no ResetSwingTerms.__post_init__ to
    # catch a mistake -- checked here, first, before building a lattice or
    # doing any other work. n_r=1 specifically would divide by zero inside
    # accumulate_step/the Numba kernel (dr = (r_hi - r_lo) / (n_r - 1)).
    if not (isinstance(n_r, int) or (hasattr(n_r, "is_integer") and n_r.is_integer())) or n_r < 2:
        raise ValueError(f"n_r must be an integer >= 2, got {n_r!r}.")
    if not (math.isfinite(r_lo) and math.isfinite(r_hi)):
        raise ValueError(f"r_lo and r_hi must be finite, got r_lo={r_lo!r}, r_hi={r_hi!r}.")
    if not r_lo < r_hi:
        raise ValueError(f"Need r_lo < r_hi, got r_lo={r_lo!r}, r_hi={r_hi!r}.")

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

    all_h = rf.project_month_end_quotes(lattice, schedule.month_end_dates) if quotes is None else quotes

    # Bracket check, weighted by lattice["q"] (unconditional probability of
    # (date i, node j) from val_date) rather than a plain min/max over every
    # node: a trinomial lattice truncated to `n_p` steps genuinely piles up
    # material probability at its own edge nodes once enough days have
    # elapsed relative to n_p (found empirically -- an n_p=8, ~4-month lattice
    # showed 3-4% probability sitting AT the edge, not a negligible tail), so
    # an unweighted check would demand an r_grid wide enough to cover states
    # that are edge-of-lattice-truncation artefacts, not economically
    # material ones -- defeating the purpose of a tight, accurate grid.
    # Cells below 1e-6 probability are excluded; worst case that admits at
    # most a few hundred dates' worth of genuinely negligible mass (~1e-4),
    # not a meaningfully mispriced result.
    q = lattice["q"]
    for month_end in schedule.month_end_dates:
        u = date_span.get_loc(month_end)
        H_rows, q_rows = all_h[u][: u + 1, :], q[: u + 1, :]
        material = q_rows >= 1e-6
        if not material.any():
            continue
        H_material = H_rows[material]
        if H_material.min() < r_lo or H_material.max() > r_hi:
            raise ValueError(
                f"r_lo/r_hi=[{r_lo}, {r_hi}] does not bracket the model's own "
                f"projected quote for {month_end:%Y-%m-%d} (observed range "
                f"[{H_material.min():.4f}, {H_material.max():.4f}] over lattice "
                f"states with >= 1e-6 probability of occurring). accumulate_step's "
                f"np.interp would silently CLAMP rather than extrapolate here -- "
                f"pricing against a strike that can never actually occur, with no "
                f"error. Widen r_lo/r_hi to bracket every delivery month's own "
                f"range, not just one of them.")

    l_grid = np.arange(n_l)
    admissible = (l_grid >= lo_clip) & (l_grid <= hi_clip)
    terminal_value = np.broadcast_to(
        np.where(admissible, 0.0, -np.inf), (width, n_l)).copy()

    months = schedule.months
    month_ends = schedule.month_end_dates
    for idx in range(len(months) - 1, -1, -1):
        month = months[idx]
        quotes_next = None
        fixing_obs_next = None
        if idx + 1 < len(months):
            u_next = date_span.get_loc(month_ends[idx + 1])
            quotes_next = all_h[u_next]
            fixing_obs_next = months[idx + 1].fixing_observation_dates
        # _run_month always returns already-collapsed (width, n_l) results now
        # (R-04): no separate _collapse_fresh_axis call needed here any more.
        collapsed = _run_month(lattice, month, r_grid, quotes_next, fixing_obs_next,
                               terminal_value, v_step, daily_max_clips)
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
    # it is whatever month 1's own fixing-observation window (the ONE calendar
    # month immediately before it -- NOT "every day since val_date", a real
    # bug 2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-01 found here)
    # actually accumulates, generally with real randomness. Reuse the exact
    # same accumulation machinery for that window, not a shortcut.
    u_first = date_span.get_loc(month_ends[0])
    pre_deal_quotes = all_h[u_first]
    fixing_window = months[0].fixing_observation_dates
    v_by_r_l0 = np.stack([arr[:, 0] for arr in first_month_terminal_by_r_in], axis=1)  # (width, n_r)
    quote_by_date = {d: pre_deal_quotes[date_span.get_loc(d), :] for d in fixing_window}
    v_at_window_start_by_r = _run_accumulation_only(
        lattice, fixing_window, r_grid, quote_by_date, v_by_r_l0)
    # `_run_accumulation_only` lands at (fixing_window[0]'s index - 1) -- the
    # calendar day right before the fixing window itself starts (NOT month 1's
    # own fixing_date, which is the window's LAST day, not the day before its
    # first). If val_date sits further back than that (the deal was valued
    # more than one month ahead of the first fixing window -- the schedule's
    # own validation only requires "before", not "immediately
    # before"), the value must still propagate the rest of the way back to
    # val_date with NO further accumulation: nothing is observed in that gap.
    window_start_idx = date_span.get_loc(fixing_window[0])
    v_at_root_by_r = v_at_window_start_by_r
    for i in range(window_start_idx - 2, -1, -1):
        v_at_root_by_r = rse._propagate_one_step(
            v_at_root_by_r, lattice["p_u"][i, :], lattice["p_m"][i, :], lattice["p_d"][i, :])

    root_row = v_at_root_by_r[terms.n_p, :]
    spread = float(root_row.max() - root_row.min())
    scale = max(1.0, float(np.abs(root_row).max()))
    if spread > 1e-6 * scale:
        raise RuntimeError(
            f"root value should not depend on the (not-yet-accumulated) "
            f"starting r bucket, but varies by {spread:.6g} (scale {scale:.6g}) "
            f"-- widen r_lo/r_hi or increase n_r before trusting this result.")
    pv = float(root_row[0])
    # 2026-09-14 INDEPENDENT-REVIEW-MONTHLY-RESET-SWING R-09: surface a
    # NaN/inf PV here, with the actual inputs in scope, rather than letting
    # the caller discover it several steps downstream (e.g. inside a delta's
    # own central difference) with no context left about which valuation
    # produced it.
    if not math.isfinite(pv):
        raise RuntimeError(
            f"Computed PV is not finite ({pv!r}) at n_p={terms.n_p}, n_r={n_r}, "
            f"r_lo={r_lo}, r_hi={r_hi} -- a genuine numerical failure, not a "
            f"valid price.")
    return pv


def compute_deltas(terms, schedule, daily_curve, n_r, r_lo, r_hi, bump_eur_mwh=0.10):
    """Same three central-finite-difference measures as
    `reset_swing_exact.compute_deltas` -- see there for the full rationale
    (total is authoritative; physical_leg/index_leg are a diagnostic
    attribution, not asserted to sum to total exactly). Repeated here rather
    than shared because the two `value_*_call_swing` signatures differ (this
    one also takes the accumulator grid); the bump/freeze pattern itself is
    identical.

    `r_lo`/`r_hi` are held fixed across every bumped valuation: a bump of
    `bump_eur_mwh` (0.10 EUR/MWh by default) moves any projected quote by at
    most that much, negligible next to any grid margin wide enough to pass
    `value_averaged_reset_call_swing`'s own bracket check in the first place.
    """
    def _bumped(sign):
        return daily_curve + sign * bump_eur_mwh

    def _quotes_for(curve_for_strikes):
        lattice = rf.build_lattice(terms.val_date, terms.storage_start, terms.storage_end,
                                   vol=terms.vol, sMR=terms.sMR, n_p=terms.n_p,
                                   daily_curve=curve_for_strikes, discount_rate=terms.discount_rate)
        return rf.project_month_end_quotes(lattice, schedule.month_end_dates)

    def _value(curve_for_spot, quotes_for_strikes):
        return value_averaged_reset_call_swing(
            terms, schedule, curve_for_spot, n_r, r_lo, r_hi, quotes=quotes_for_strikes)

    base_quotes = _quotes_for(daily_curve)

    pv_up = _value(_bumped(+1), None)
    pv_down = _value(_bumped(-1), None)
    total = (pv_up - pv_down) / (2.0 * bump_eur_mwh)

    phys_up = _value(_bumped(+1), base_quotes)
    phys_down = _value(_bumped(-1), base_quotes)
    physical_leg = (phys_up - phys_down) / (2.0 * bump_eur_mwh)

    idx_up = _value(daily_curve, _quotes_for(_bumped(+1)))
    idx_down = _value(daily_curve, _quotes_for(_bumped(-1)))
    index_leg = (idx_up - idx_down) / (2.0 * bump_eur_mwh)

    return dict(total=total, physical_leg=physical_leg, index_leg=index_leg,
               bump_eur_mwh=bump_eur_mwh)
