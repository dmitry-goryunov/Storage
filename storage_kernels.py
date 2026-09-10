"""Numba-compiled kernels for the storage/swing model.

Kept in a separate module from storage_model.py on purpose: Numba's disk
cache (cache=True) is invalidated whenever the source file containing the
jitted function changes. Isolating the kernels here means editing the
valuation/wrapper code in storage_model.py no longer triggers a 20-40s
recompile of all kernels.
"""

import numpy as np
from numba import jit, prange

# An inventory state a hard dated bound forbids. Large enough that a mixture of
# it with any real deal value is still unmistakably infeasible, small enough
# that propagating it across a few hundred timesteps cannot reach -inf.
FORBIDDEN = -1e30

# Anything at or below this is infeasibility propagated back to the reported
# value, not a very bad deal. Real deal values are millions.
INFEASIBLE_VALUE = -1e20

# Callers mark a disallowed terminal inventory with -1e9 in `t_p_curve`. It is
# read as a prohibition, not as a price: a penalty a large enough deal can pay is
# not a constraint, and it left the store buying its way out of the terminal
# condition and reporting a value of about -1e9 rather than a refusal. A genuine
# terminal valuation of leftover gas sits far above this.
TERMINAL_FORBIDDEN = -1e9


@jit(nopython=True, cache=True)
def _tree_core(x, p_u, p_m, p_d, fwd, vol_arr, mr_arr, n_t, n_p, dx, dt):
    # x initialisation
    for i in range(n_t):
        j_s = max(n_p - i, 0)
        j_e = min(n_p + i, 2*n_p) + 1
        for j in range(j_s, j_e):
            x[i, j] = (j - n_p) * dx

    if n_p > 0:
        # Phase 1: growing tree (i < n_p)
        for i in range(min(n_p, n_t)):
            j_s, j_e  = n_p - i, n_p + i + 1
            xi        = x[i, j_s:j_e]
            vol, mr   = vol_arr[i], mr_arr[i]
            mxi_dt    = mr * dt * xi
            mxi_dt_dx = mxi_dt / dx
            k_i = np.floor(-mxi_dt + 0.5)
            a   = (vol**2 * dt + mxi_dt**2) / dx**2 + k_i**2
            b   = -mxi_dt_dx * (1 - 2*k_i) - k_i
            c   =  mxi_dt_dx * (1 + 2*k_i) + k_i
            p_u[i, j_s:j_e] = 0.5 * (a + b)
            p_d[i, j_s:j_e] = 0.5 * (a + c)
            p_m[i, j_s:j_e] = 1.0 - p_u[i, j_s:j_e] - p_d[i, j_s:j_e]
        # Phase 2: full-width steps (i >= n_p)
        for i in range(n_p, n_t):
            xi        = x[i, :]
            vol, mr   = vol_arr[i], mr_arr[i]
            mxi_dt    = mr * dt * xi
            mxi_dt_dx = mxi_dt / dx
            k_i = np.floor(-mxi_dt + 0.5)
            a   = (vol**2 * dt + mxi_dt**2) / dx**2 + k_i**2
            b   = -mxi_dt_dx * (1 - 2*k_i) - k_i
            c   =  mxi_dt_dx * (1 + 2*k_i) + k_i
            p_u[i, :]     = 0.5 * (a + b)
            p_d[i, :]     = 0.5 * (a + c)
            p_m[i, :]     = 1.0 - p_u[i, :] - p_d[i, :]
            p_u[i, 0]     = 0.5 * (b[0] - c[0])
            p_d[i, 0]     = 0.0
            p_m[i, 0]     = 1.0 - p_u[i, 0]
            p_d[i, 2*n_p] = 0.5 * (c[-1] - b[-1])
            p_u[i, 2*n_p] = 0.0
            p_m[i, 2*n_p] = 1.0 - p_d[i, 2*n_p]
    else:
        p_m[:] = 1.0

    # q propagation
    q = np.zeros((n_t, 2*n_p+1))
    q[0, n_p] = 1.0
    for i in range(1, n_t):
        q_prev     = q[i-1]
        total      = p_m[i-1] * q_prev
        total[1:]  += p_u[i-1, :-1] * q_prev[:-1]
        total[:-1] += p_d[i-1, 1:]  * q_prev[1:]
        q[i] = total

    # Forward distortion
    for i in range(n_t):
        j_s = max(n_p - i, 0)
        j_e = min(n_p + i, 2*n_p) + 1
        expected = np.dot(q[i, j_s:j_e], np.exp(x[i, j_s:j_e]))
        x[i, j_s:j_e] += np.log(fwd[i] / expected)

    return q


@jit(nopython=True, parallel=True, cache=True)
def run_model(n_t, n_p, n_op, v_step, x, p_u, p_m, p_d,
              d_curve, i_curve, w_curve, i_cost, w_cost,
              t_p_curve, i_ratch, w_ratch, mintunnel, max_tunnel,
              inj_fuel_mult):
    """
    Backward induction. On each active day the controller may move ANY integer
    number of clips between 0 and the daily rate (further capped by remaining
    inventory / headroom and by the per-level ratchet), choosing the volume that
    maximises expected discounted value. strat stores the SIGNED clip count
    actually moved: negative = withdraw, positive = inject, 0 = idle.
    Price dimension (k) uses prange; volume (l) and clip-count (d) are scalar
    loops so the optimal partial-day volume can be searched per state.

    `mintunnel`/`max_tunnel` are HARD dated inventory bounds: an inventory state
    outside them on day i is inadmissible, and the infeasibility propagates back
    through every state that could reach it. They were a `1000 * v_step` per-clip
    penalty until 2026-09-10, which made a contractual breach purchasable -- a
    deal worth more than the penalty simply paid it, and at a high enough price
    level a fifth of paths opened a floored day empty. A finite number is not a
    constraint. Value FORBIDDEN (-1e30) marks an inadmissible state; callers
    detect an infeasible contract by testing the reported value against
    `INFEASIBLE_VALUE`, not by re-deriving the schedule.
    """
    n_k_max  = 2*n_p + 1

    v     = np.zeros((n_t, n_k_max, n_op), dtype=np.float64)
    strat = np.zeros((n_t, n_k_max, n_op), dtype=np.float64)

    l_arr = np.arange(n_op)
    l_f   = l_arr.astype(np.float64)

    # Per-timestep max clip counts (already capped at inventory / headroom by the
    # min() below), the hard bound mask and the discounted clip size.
    all_wdr_steps = np.empty((n_t, n_op), dtype=np.int64)
    all_inj_steps = np.empty((n_t, n_op), dtype=np.int64)
    all_barred    = np.empty((n_t, n_op), dtype=np.bool_)
    all_dc        = np.empty(n_t, dtype=np.float64)

    for i in range(n_t):
        wdr_raw = w_curve[i] * w_ratch
        inj_raw = i_curve[i] * i_ratch
        all_wdr_steps[i] = np.minimum(wdr_raw, l_f).astype(np.int64)
        all_inj_steps[i] = np.minimum(inj_raw, (n_op - 1) - l_f).astype(np.int64)
        all_barred[i] = ((l_f < float(mintunnel[i]))
                         | (l_f > float(max_tunnel[i])))
        all_dc[i] = d_curve[i] * v_step

    # The terminal slice carries its own bound: a floor on the last day is as
    # binding as one in the middle. A disallowed terminal inventory is forbidden
    # outright rather than priced at -1e9, so that a state which cannot reach a
    # permitted terminal inventory is infeasible rather than merely expensive.
    t_p = t_p_curve[:n_op]
    barred_last = all_barred[n_t-1]
    for ii in range(n_k_max):
        for l in range(n_op):
            if barred_last[l] or t_p[l] <= TERMINAL_FORBIDDEN:
                v[n_t-1, ii, l] = FORBIDDEN
            else:
                v[n_t-1, ii, l] = t_p[l]

    # Precompute exp(x) for all (i, k) once — removes exp() from the hot inner loop
    exp_x = np.exp(x)

    for i in range(n_t-2, -1, -1):
        wdr_max  = all_wdr_steps[i]
        inj_max  = all_inj_steps[i]
        barred   = all_barred[i]
        dc       = all_dc[i]
        w_cost_i = w_cost[i]
        i_cost_i = i_cost[i]

        k_lo = max(n_p - i, 0)
        k_hi = min(n_p + i, 2*n_p) + 1
        for k in prange(k_lo, k_hi):
            v_next_l = p_m[i, k] * v[i+1, k]
            if k > 0:     v_next_l = v_next_l + p_d[i, k] * v[i+1, k-1]
            if k < 2*n_p: v_next_l = v_next_l + p_u[i, k] * v[i+1, k+1]

            price_k = exp_x[i, k]
            pcw = dc * (price_k - w_cost_i)    # per-clip withdraw (sell) profit
            # Fuel: putting one clip INTO inventory means buying `inj_fuel_mult`
            # clips in the market, the excess being retained for compression.
            # It scales the price, not the cost, because the fuel is taken in kind.
            pci = dc * (-price_k * inj_fuel_mult - i_cost_i)

            for l in range(n_op):
                # Outside a hard dated bound: not a state the contract permits,
                # so there is nothing to optimise and nothing may route here.
                if barred[l]:
                    v[i, k, l]     = FORBIDDEN
                    strat[i, k, l] = 0
                    continue

                idle = v_next_l[l]
                best = idle
                step = 0

                # Withdraw d = 1..wdr_max[l] clips (all sold at today's price).
                wmax = wdr_max[l]
                d = 1
                while d <= wmax:
                    cand = v_next_l[l - d] + d * pcw
                    if cand > best:
                        best = cand
                        step = -d
                    d += 1

                # Inject d = 1..inj_max[l] clips (all bought at today's price).
                imax = inj_max[l]
                d = 1
                while d <= imax:
                    cand = v_next_l[l + d] + d * pci
                    if cand > best:
                        best = cand
                        step = d
                    d += 1

                # Snap to idle when the gain over doing nothing is negligible.
                if step != 0 and (best - idle) < 1e-6:
                    best = idle
                    step = 0

                v[i, k, l]     = best
                strat[i, k, l] = step

    return v, strat


@jit(nopython=True, parallel=True, cache=True)
def probabilities(n_t, n_p, n_op, q, strat, p_u, p_m, p_d, n_op_start):
    """Joint (price, volume) state probabilities under the optimal strategy.

    Takes no exercise curves or ratchets: `strat` already holds the exact signed
    clip move for every state, capped when it was chosen. The i_curve/w_curve,
    i_ratch/w_ratch, mintunnel and max_tunnel arguments this used to accept were
    never read.
    """
    prob = np.zeros((n_t, 2*n_p+1, n_op), dtype=np.float64)
    for i in range(2*n_p+1):
        prob[0, i, n_op_start] = q[0, i]

    for i in range(n_t - 1):
        j_lo = max(n_p - i, 0)
        j_hi = min(n_p + i, 2*n_p) + 1
        # 3-colour prange: j-values in the same pass are spaced by 3, so their
        # writes to rows j-1..j+1 never overlap → race-free parallel accumulation.
        for color in range(3):
            n_color = (j_hi - j_lo - color + 2) // 3
            if n_color > 0:
                for m in prange(n_color):
                    j = j_lo + color + m * 3
                    for k in range(n_op):
                        # strat already holds the exact signed clip move for this
                        # state (capped at inventory/headroom when it was chosen).
                        dk = int(round(strat[i, j, k]))
                        for dj in range(-1, 2):
                            nj = j + dj
                            if 0 <= nj <= 2*n_p:
                                if   dj ==  1: tp = p_u[i, j]
                                elif dj == -1: tp = p_d[i, j]
                                else:          tp = p_m[i, j]
                                prob[i+1, nj, k + dk] += tp * prob[i, j, k]
    return prob
