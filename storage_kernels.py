"""Numba-compiled kernels for the storage/swing model.

Kept in a separate module from storage_model.py on purpose: Numba's disk
cache (cache=True) is invalidated whenever the source file containing the
jitted function changes. Isolating the kernels here means editing the
valuation/wrapper code in storage_model.py no longer triggers a 20-40s
recompile of all kernels.
"""

import numpy as np
from numba import jit, prange


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
              t_p_curve, i_ratch, w_ratch, mintunnel, max_tunnel):
    """
    Backward induction: at each (time, price, volume) state choose the action
    maximising expected discounted future value.
    Volume dimension (l) is vectorised; price dimension (k) uses prange.
    """
    bigdummy = 1e10
    n_k_max  = 2*n_p + 1

    v     = np.zeros((n_t, n_k_max, n_op), dtype=np.float64)
    strat = np.zeros((n_t, n_k_max, n_op), dtype=np.float64)

    t_p = t_p_curve[:n_op]
    for ii in range(n_k_max):
        v[n_t-1, ii] = t_p

    l_arr = np.arange(n_op)
    l_f   = l_arr.astype(np.float64)

    # Precompute per-timestep arrays for all i upfront — avoids n_t repeated allocations
    all_wdr_steps   = np.empty((n_t, n_op), dtype=np.int64)
    all_inj_steps   = np.empty((n_t, n_op), dtype=np.int64)
    all_wdr_steps_f = np.empty((n_t, n_op), dtype=np.float64)
    all_inj_steps_f = np.empty((n_t, n_op), dtype=np.float64)
    all_wdr_valid   = np.empty((n_t, n_op), dtype=np.bool_)
    all_inj_valid   = np.empty((n_t, n_op), dtype=np.bool_)
    all_tunnel_pen  = np.empty((n_t, n_op), dtype=np.float64)
    all_dc          = np.empty(n_t, dtype=np.float64)

    for i in range(n_t):
        wdr_raw = w_curve[i] * w_ratch
        inj_raw = i_curve[i] * i_ratch
        ws  = np.minimum(wdr_raw, l_f).astype(np.int64)
        is_ = np.minimum(inj_raw, (n_op - 1) - l_f).astype(np.int64)
        all_wdr_steps[i]   = ws
        all_inj_steps[i]   = is_
        all_wdr_steps_f[i] = ws.astype(np.float64)
        all_inj_steps_f[i] = is_.astype(np.float64)
        all_wdr_valid[i]   = (wdr_raw > 0.0) & (l_arr > 0)
        all_inj_valid[i]   = (inj_raw > 0.0) & (l_arr < n_op - 1)
        all_tunnel_pen[i]  = 1000.0 * v_step * (
            np.maximum(float(mintunnel[i]) - l_f, 0.0) +
            np.maximum(l_f - float(max_tunnel[i]), 0.0)
        )
        all_dc[i] = d_curve[i] * v_step

    # Precompute exp(x) for all (i, k) once — removes exp() from the hot inner loop
    exp_x = np.exp(x)

    for i in range(n_t-2, -1, -1):
        wdr_steps  = all_wdr_steps[i]
        inj_steps  = all_inj_steps[i]
        wdr_valid  = all_wdr_valid[i]
        inj_valid  = all_inj_valid[i]
        tunnel_pen = all_tunnel_pen[i]
        dc         = all_dc[i]
        w_cost_i   = w_cost[i]
        i_cost_i   = i_cost[i]

        # Hoist k-independent expressions out of prange
        gather_wdr = l_arr - wdr_steps          # same for every k
        gather_inj = l_arr + inj_steps
        wdr_base   = all_wdr_steps_f[i] * dc   # price-independent profit factor
        inj_base   = all_inj_steps_f[i] * dc

        k_lo = max(n_p - i, 0)
        k_hi = min(n_p + i, 2*n_p) + 1
        for k in prange(k_lo, k_hi):
            v_next_l = p_m[i, k] * v[i+1, k]
            if k > 0:     v_next_l = v_next_l + p_d[i, k] * v[i+1, k-1]
            if k < 2*n_p: v_next_l = v_next_l + p_u[i, k] * v[i+1, k+1]

            price_k    = exp_x[i, k]
            ev_wdr     = v_next_l[gather_wdr]
            ev_inj     = v_next_l[gather_inj]
            wdr_profit = wdr_base * (price_k - w_cost_i)
            inj_profit = inj_base * (-price_k - i_cost_i)

            wdr_val = np.where(wdr_valid, ev_wdr + wdr_profit, -bigdummy)
            inj_val = np.where(inj_valid, ev_inj + inj_profit, -bigdummy)

            best     = np.maximum(np.maximum(inj_val, wdr_val), v_next_l)
            is_no_ex = (best - v_next_l) < 1e-6
            is_wdr   = ((best - v_next_l) >= 1e-6) & (best == wdr_val)
            strat[i, k] = np.where(is_no_ex, 0.0, np.where(is_wdr, -1.0, 1.0))
            v[i, k]     = best - tunnel_pen

    return v, strat


@jit(nopython=True, parallel=True, cache=True)
def probabilities(n_t, n_p, n_op, q, strat, p_u, p_m, p_d,
                  i_curve, w_curve, i_ratch, w_ratch,
                  n_op_start, mintunnel, max_tunnel):
    """Joint (price, volume) state probabilities under the optimal strategy."""
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
                        wdr_step = w_curve[i] * w_ratch[k]
                        inj_step = i_curve[i] * i_ratch[k]
                        dki = strat[i, j, k]
                        if   dki == -1: dk = -int(round(min(wdr_step, k)))
                        elif dki ==  1: dk =  int(round(min(inj_step, n_op - 1 - k)))
                        else:           dk = 0
                        for dj in range(-1, 2):
                            nj = j + dj
                            if 0 <= nj <= 2*n_p:
                                if   dj ==  1: tp = p_u[i, j]
                                elif dj == -1: tp = p_d[i, j]
                                else:          tp = p_m[i, j]
                                prob[i+1, nj, k + dk] += tp * prob[i, j, k]
    return prob


@jit(nopython=True, cache=True)
def get_exercise(i, n_p, n_op, prob, strat, i_ratch, w_ratch, v_step, w_curve, i_curve):
    result = 0.
    for j in range(2*n_p+1):
        for k in range(n_op):
            wdr_step = w_curve[i] * w_ratch[k]
            inj_step = i_curve[i] * i_ratch[k]
            if   strat[i, j, k] == -1: action = -min(wdr_step, k) * v_step
            elif strat[i, j, k] ==  1: action =  min(inj_step, n_op - 1 - k) * v_step
            else:                       action = 0.
            result += action * prob[i, j, k]
    return -round(result, 3)


@jit(nopython=True, cache=True)
def get_delta(i, n_p, n_op, prob, strat, i_ratch, w_ratch, v_step, w_curve, i_curve, x, fwd):
    result = 0.
    for j in range(2*n_p+1):
        for k in range(n_op):
            wdr_step = w_curve[i] * w_ratch[k]
            inj_step = i_curve[i] * i_ratch[k]
            if   strat[i, j, k] == -1: action = -min(wdr_step, k) * v_step
            elif strat[i, j, k] ==  1: action =  min(inj_step, n_op - 1 - k) * v_step
            else:                       action = 0.
            result += action * prob[i, j, k] * np.exp(x[i, j])
    return -round(result / fwd[i], 3)
