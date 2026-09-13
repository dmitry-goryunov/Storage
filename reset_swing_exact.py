"""The point-reset exact benchmark: a call swing whose strike resets every month
from the model's own month-ahead projection, not a number fixed at inception.

DESIGN-MONTHLY-RESET-SWING-2026-09-13.md Phase 1 / Release 1A (sec.7.1, sec.8,
sec.14.1). "Deliberately small... purpose is correctness, not production scale"
(sec.7.1) -- pure Python/NumPy, not the Numba kernel, because this state space
(price node x cumulative volume x which node the last fixing landed on) does not
exist in storage_kernels.py and sec.7.1 asks for a small reference implementation
before any production-scale one.

The state during delivery month M is (i, j, l, j_fix): time step, price node,
cumulative exercised volume, and the price node THE LAST FIXING landed on -- which
determines M's strike exactly (K_M = H[month_end(M)][fixing_date, j_fix], from
reset_forward.py), with no interpolation error, because a point reset's strike is
a deterministic function of a single lattice node, not a continuous quantity.
Months are solved from last to first; each month's value at its OWN fixing date,
collapsed onto j_fix == j (sec.6.2: "at the reset boundary... the old K_M can then
be discarded" -- symmetrically, the NEXT month's j_fix is just whatever node the
fixing date actually landed on), becomes the previous month's terminal condition.
"""
import numpy as np

import reset_forward as rf


def _propagate_one_step(v_next, p_u_i, p_m_i, p_d_i):
    """continuation[j, l] = E_j'[v_next[j', l] | j], for every l at once --
    the same shift-and-add pattern as _tree_core's forward q propagation and
    reset_forward's backward H recursion, applied to a value function instead
    of a probability or a price."""
    cont = p_m_i[:, None] * v_next
    cont[:-1, :] += p_u_i[:-1, None] * v_next[1:, :]
    cont[1:, :] += p_d_i[1:, None] * v_next[:-1, :]
    return cont


def _exercise_step(continuation, spot, strike, df_i, v_step, daily_max_clips):
    """V[j,l] = max over d in 0..daily_max_clips of
    { d*v_step*DF_i*(spot[j]-strike) + continuation[j, l+d] }, d capped wherever
    l+d would run off the volume grid."""
    width, n_l = continuation.shape
    payoff_per_clip = df_i * v_step * (spot - strike)
    best = continuation.copy()  # d = 0
    for d in range(1, daily_max_clips + 1):
        if d >= n_l:
            break
        candidate = np.full_like(continuation, -np.inf)
        candidate[:, : n_l - d] = continuation[:, d:]
        candidate += (d * payoff_per_clip)[:, None]
        best = np.maximum(best, candidate)
    return best


def _run_month_for_one_fixing_node(lattice, month, strike, terminal_value,
                                   v_step, daily_max_clips):
    """Backward DP over one delivery month's exercise days at a CONSTANT strike
    (the ordinary swing recursion -- sec.6.3 -- since within one month, given
    which node the fixing landed on, the strike is just a known number),
    starting from `terminal_value` (the continuation as seen FROM the month's
    own last exercise day), then propagated back to the fixing date itself.
    """
    date_span = lattice["date_span"]
    x, p_u, p_m, p_d, d_curve = (
        lattice["x"], lattice["p_u"], lattice["p_m"], lattice["p_d"], lattice["d_curve"])

    exercise_indices = [date_span.get_loc(d) for d in month.exercise_dates]
    v = terminal_value
    for i in reversed(exercise_indices):
        spot = np.exp(x[i, :])
        v = _exercise_step(v, spot, strike, d_curve[i], v_step, daily_max_clips)
        if i > 0:
            v = _propagate_one_step(v, p_u[i - 1, :], p_m[i - 1, :], p_d[i - 1, :])

    # Any gap between the fixing date and the month's own first exercise day
    # (only possible for a deal's partial first month) is plain propagation --
    # no exercise right can be used on a day before the deal itself starts.
    fixing_idx = date_span.get_loc(month.fixing_date)
    first_exercise_idx = exercise_indices[0]
    for i in range(first_exercise_idx - 2, fixing_idx - 1, -1):
        v = _propagate_one_step(v, p_u[i, :], p_m[i, :], p_d[i, :])
    return v


def value_point_reset_call_swing(terms, schedule, daily_curve=None, curve=None):
    """The exact benchmark's root value: EUR PV at (val_date, centre price node,
    zero cumulative volume). Call swing only (Release 1A scope).

    `daily_curve`/`curve` are the same two ways of supplying the forward curve
    `run_valuation`/`Storage` already accept -- exactly one must be given.
    """
    if (daily_curve is None) == (curve is None):
        raise ValueError("Give exactly one of daily_curve or curve.")

    lattice = rf.build_lattice(terms.val_date, terms.storage_start, terms.storage_end,
                               vol=terms.vol, sMR=terms.sMR, n_p=terms.n_p,
                               daily_curve=daily_curve, curve=curve,
                               discount_rate=terms.discount_rate)
    quotes = rf.project_month_end_quotes(lattice, schedule.month_end_dates)

    v_step = terms.v_step_mwh
    n_l = int(round(terms.global_max_mwh / v_step)) + 1
    global_min_clips = int(round(terms.global_min_mwh / v_step))
    global_max_clips = n_l - 1
    daily_max_clips = int(round(terms.daily_max_mwh / v_step))
    width = 2 * terms.n_p + 1

    l_grid = np.arange(n_l)
    admissible = (l_grid >= global_min_clips) & (l_grid <= global_max_clips)
    terminal_value = np.broadcast_to(
        np.where(admissible, 0.0, -np.inf), (width, n_l)).copy()

    date_span = lattice["date_span"]
    for month, month_end in zip(reversed(schedule.months), reversed(schedule.month_end_dates)):
        u = date_span.get_loc(month_end)
        H = quotes[u]
        collapsed = np.empty((width, n_l))
        for j_fix in range(width):
            strike = float(H[date_span.get_loc(month.fixing_date), j_fix])
            v = _run_month_for_one_fixing_node(
                lattice, month, strike, terminal_value, v_step, daily_max_clips)
            collapsed[j_fix, :] = v[j_fix, :]
        terminal_value = collapsed

    # Propagate from the first month's own fixing date back to val_date (i=0).
    v = terminal_value
    fixing_idx = date_span.get_loc(schedule.months[0].fixing_date)
    for i in range(fixing_idx - 1, -1, -1):
        v = _propagate_one_step(v, lattice["p_u"][i, :], lattice["p_m"][i, :], lattice["p_d"][i, :])

    return float(v[terms.n_p, 0])
