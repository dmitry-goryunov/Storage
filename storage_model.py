import numpy as np
from math import sqrt
import re
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
import pandas as pd

# Numba kernels live in storage_kernels.py so that edits to this file do not
# invalidate their disk cache (which would trigger a 20-40s recompile).
# Re-exported here for backward compatibility.
from storage_kernels import (
    _tree_core, run_model, probabilities, get_exercise, get_delta,
)


# ── Curve utilities ───────────────────────────────────────────────────────────

def map_curve_to_dates(date_span, df):
    """Vectorised forward-curve lookup: map each date to its contract value."""
    result = pd.Series(index=date_span, dtype=float)
    for _, row in df.iterrows():
        mask = (date_span >= row['contractStart']) & (date_span <= row['contractEnd'])
        result[mask] = row['value']
    return result


def smoothen_curve(coarse_curve, alpha=1.2):
    """
    Smooth a stepped forward curve to daily resolution through monthly midpoints.

    alpha=1.0 reproduces standard PCHIP slopes. alpha>1 relaxes PCHIP slope
    limiting by scaling the node derivatives before rebuilding the curve as a
    cubic Hermite spline.

    A one-shot additive correction is applied per month so that the smoothed
    daily averages exactly reproduce the input monthly averages (keeps the
    daily curve arbitrage-consistent with the contract prices).
    """
    monthly = coarse_curve.resample('ME').mean()

    t0      = coarse_curve.index[0]
    ms      = coarse_curve.resample('MS').first().index          # month starts
    mid     = ms + (monthly.index - ms) / 2                     # midpoint of each month
    x_knots = np.array([(d - t0).days for d in mid],            dtype=float)
    x_all   = np.array([(d - t0).days for d in coarse_curve.index], dtype=float)

    pchip = PchipInterpolator(x_knots, monthly.values, extrapolate=True)
    slopes = alpha * pchip.derivative()(x_knots)
    vals = CubicHermiteSpline(
        x_knots, monthly.values, slopes, extrapolate=True
    )(x_all)

    smoothed = pd.Series(vals, index=coarse_curve.index)

    # One-shot additive correction (see docstring).
    periods    = smoothed.index.to_period('M')
    correction = monthly.values - smoothed.groupby(periods).mean().values
    smoothed   = smoothed + pd.Series(
        correction, index=monthly.index.to_period('M')
    ).reindex(periods).values

    return smoothed


def check_curve(coarse_curve, smoothed_curve):
    print(pd.DataFrame({
        'Stepped_Monthly_Avg':  coarse_curve.resample('ME').mean(),
        'Smoothed_Monthly_Avg': smoothed_curve.resample('ME').mean(),
    }))


# ── Storage facility ──────────────────────────────────────────────────────────

class Storage:
    """
    Encapsulates the date arithmetic and array setup for a storage/swing contract.

    Parameters
    ----------
    valDate, storageStart, storageEnd : date-like
    v_step  : volume per inventory state (MWh)
    sVol    : daily spot volatility (fraction)
    sMR     : mean-reversion speed
    """

    def __init__(self, valDate, storageStart, storageEnd,
                 curve, n_p=0, v_step=1000, sVol=0.9, sMR=1.0):
        self.valDate      = pd.Timestamp(valDate)
        self.storageStart = pd.Timestamp(storageStart)
        self.storageEnd   = pd.Timestamp(storageEnd)
        self.v_step       = v_step
        self.n_p          = n_p

        # Date grid
        self.backStop  = (self.storageEnd + pd.DateOffset(months=1)).normalize() + pd.offsets.MonthEnd(0)
        self.Dt        = (self.storageStart - self.valDate).days
        self.n_t       = (self.backStop     - self.valDate).days
        self.date_span = pd.date_range(self.valDate, self.backStop, freq='D')
        self._active   = len(pd.date_range(self.valDate, self.storageEnd, freq='D'))

        # Forward / price curve
        self.price_curve = smoothen_curve(
            map_curve_to_dates(self.date_span, curve).to_frame(name='value')['value']
        )

        # Price-tree vol / mean-reversion profiles
        self.sVol = [sVol] * self.n_t
        self.sMR  = [sMR]  * self.n_t

        # Discount curve (flat 1)
        self.d_curve = np.ones(self.n_t)

        # Exercise curves: no injection (swing = sell-only), withdraw during active window
        n = len(self.date_span)
        self.i_curve = np.zeros(n)
        self.w_curve = np.ones(n)
        self.w_curve[self._active:] = 0.
        self.w_curve[:self.Dt]      = 0.

        # Cost curves (zero by default)
        self.i_cost = np.zeros(n)
        self.w_cost = np.zeros(n)

        # Inventory floor
        self.mintunnel = np.zeros(n, dtype=int)

        # Volume states — call set_volume_states() to override
        self.n_op_start = (self.storageEnd - self.storageStart).days + 1
        self._init_volume_arrays()

    def _init_volume_arrays(self):
        """(Re-)build arrays that depend on n_op_start / n_op."""
        self.n_op      = self.n_op_start + 1
        self.i_ratch   = np.ones(self.n_op)
        self.w_ratch   = np.ones(self.n_op)
        self.max_tunnel = np.full(len(self.date_span), self.n_op)
        self.t_p_curve  = np.full(self.n_op + 2, -1e9)
        self.t_p_curve[0] = 0.

    def set_volume_states(self, n_op_start):
        """Change the starting inventory level and rebuild dependent arrays."""
        self.n_op_start = n_op_start
        self._init_volume_arrays()

    def build(self):
        """Build price tree, run DP model, compute probabilities and metrics.
        Results stored as: self.fwd, self.x, self.v, self.strat,
                           self.prob, self.exp_ex, self.delta
        """
        self.fwd, self.x, q, p_u, p_m, p_d = build_tree(
            self.price_curve, self.n_t, self.n_p, self.sVol, self.sMR)

        self.v, self.strat = run_model(
            self.n_t, self.n_p, self.n_op, self.v_step, self.x, p_u, p_m, p_d,
            self.d_curve, self.i_curve, self.w_curve, self.i_cost, self.w_cost,
            self.t_p_curve, self.i_ratch, self.w_ratch, self.mintunnel, self.max_tunnel)

        self.prob = probabilities(
            self.n_t, self.n_p, self.n_op, q, self.strat, p_u, p_m, p_d,
            self.i_curve, self.w_curve, self.i_ratch, self.w_ratch,
            self.n_op_start, self.mintunnel, self.max_tunnel)

        self.exp_ex, self.delta = compute_all_metrics(
            self.n_t, self.n_p, self.n_op, self.prob, self.strat,
            self.i_ratch, self.w_ratch, self.v_step,
            self.w_curve, self.i_curve, self.x, self.fwd)

        return self

    def flat(self):
        return self.v[0, 0, self.n_op_start] / np.sum(self.delta)

    def profiled(self):
        ACQ = np.sum(self.delta)
        return self.v[0, 0, self.n_op_start] / ACQ


def month_start(ts):
    ts = pd.Timestamp(ts)
    return pd.Timestamp(ts.year, ts.month, 1)


def month_end(ts):
    return month_start(ts) + pd.offsets.MonthEnd(0)


def front_month_start(quote_date):
    return month_start(quote_date) + pd.DateOffset(months=1)


def monthly_curve_from_quote(row, contract_columns):
    front_month = front_month_start(row["quote_date"])
    rows = []
    for col in contract_columns:
        value = row[col]
        if pd.isna(value):
            continue
        contract_number = int(re.search(r"\d+", str(col)).group())
        contract_start = front_month + pd.DateOffset(months=contract_number - 1)
        rows.append({"contract": col, "contractStart": contract_start, "value": float(value)})
    curve_df = pd.DataFrame(rows)
    if curve_df.empty:
        return pd.Series(dtype=float), curve_df
    curve_df["contractEnd"] = curve_df["contractStart"].map(month_end)
    curve_df = curve_df[["contract", "contractStart", "contractEnd", "value"]]
    monthly = curve_df.set_index("contractStart")["value"].sort_index().rename("value")
    return monthly, curve_df


def quote_row_for_fd_date(quotes, contract_columns, fd_date, exact=False):
    fd_date = pd.Timestamp(fd_date)
    if exact:
        eligible = quotes.loc[quotes["quote_date"].eq(fd_date)]
    else:
        eligible = quotes.loc[quotes["quote_date"] <= fd_date]
    eligible = eligible.loc[eligible[contract_columns].notna().any(axis=1)]
    if eligible.empty:
        raise ValueError(f"No forward quote available for FDDate {fd_date:%Y-%m-%d}")
    return eligible.iloc[-1]


def curve_df_for_storage(row, contract_columns, curve_start=None, include_da=True):
    monthly, curve_df = monthly_curve_from_quote(row, contract_columns)
    curve_df = curve_df.copy()

    if not monthly.empty:
        full_months = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
        monthly = monthly.reindex(full_months).interpolate(method="time").ffill().bfill()
        curve_df = pd.DataFrame({
            "contract": [f"TTFc{i + 1}" for i in range(len(monthly))],
            "contractStart": monthly.index,
            "value": monthly.values,
        })
        curve_df["contractEnd"] = curve_df["contractStart"].map(month_end)
        curve_df = curve_df[["contract", "contractStart", "contractEnd", "value"]]

    quote_date = pd.Timestamp(row["quote_date"])
    curve_start = pd.Timestamp(curve_start) if curve_start is not None else quote_date

    if include_da and pd.notna(row.get("DA", np.nan)):
        front_month = front_month_start(quote_date)
        da_start = min(curve_start, quote_date)
        da_end = front_month - pd.Timedelta(days=1)
        da_row = pd.DataFrame([{
            "contract": "DA",
            "contractStart": da_start,
            "contractEnd": da_end,
            "value": float(row["DA"]),
        }])
        curve_df = pd.concat([da_row, curve_df], ignore_index=True)
    elif curve_start < curve_df["contractStart"].min():
        first_value = float(curve_df.sort_values("contractStart").iloc[0]["value"])
        first_start = curve_df["contractStart"].min()
        stub_row = pd.DataFrame([{
            "contract": "FRONT_STUB",
            "contractStart": curve_start,
            "contractEnd": first_start - pd.Timedelta(days=1),
            "value": first_value,
        }])
        curve_df = pd.concat([stub_row, curve_df], ignore_index=True)

    result = curve_df[["contractStart", "contractEnd", "value"]].sort_values("contractStart").reset_index(drop=True)
    if result["value"].isna().any():
        raise ValueError("Curve contains missing values after interpolation/stub fill.")
    return result


def active_masks(model):
    n = len(model.date_span)
    active = np.ones(n)
    active[model._active:] = 0.0
    active[:model.Dt] = 0.0
    return n, active


def daily_arithmetic_flat_metric(model):
    exercise_dates = model.date_span[model.Dt:model._active]
    return float(pd.Series(model.price_curve, index=model.date_span).loc[exercise_dates].mean())


def warm_numba_kernels():
    curve = pd.DataFrame({
        "contractStart": [pd.Timestamp("2026-01-01")],
        "contractEnd": [pd.Timestamp("2026-03-31")],
        "value": [25.0],
    })
    model = Storage(
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
        curve=curve,
        n_p=1,
        v_step=1000,
        sVol=0.6,
    )
    model.set_volume_states(1)
    model.build()
    return True


def value_put_swing(curve, params):
    s = Storage(params["valDate"], params["storageStart"], params["storageEnd"], curve=curve, n_p=0, v_step=params["v_step"], sVol=params["vol"])
    n, active = active_masks(s)
    s.i_curve = active.copy()
    s.w_curve = np.zeros(n)

    flat_metric = daily_arithmetic_flat_metric(s)

    init_inv = params["initial_inv_clips"] if params.get("initial_inv_clips") is not None else 0
    term_inv = params["terminal_inv_clips"] if params.get("terminal_inv_clips") is not None else params["days"]
    s.set_volume_states(params["days"])
    s.n_op_start = init_inv
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    s.t_p_curve[term_inv] = 0.0
    if params["run_intrinsic"]:
        s.build()
        profiled_eur = s.v[0, 0, init_inv]
        acq = -np.sum(s.delta)
        profiled_metric = s.profiled()
        intrinsic = flat_metric - profiled_metric
        intrinsic_profile_raw = -np.array(s.exp_ex)
    else:
        profiled_eur = np.nan
        acq = np.nan
        profiled_metric = np.nan
        intrinsic = np.nan
        intrinsic_profile_raw = np.zeros(len(s.date_span))

    s.n_p = params["n_p_full"]
    s.build()
    full_eur = s.v[0, s.n_p, init_inv]
    stochastic_metric = full_eur / np.sum(s.delta)
    extrinsic = (full_eur - profiled_eur) / acq if params["run_intrinsic"] else np.nan
    extrinsic_profile_raw = -np.array(s.exp_ex)

    return s, {
        "flat_metric": flat_metric,
        "profiled_metric": profiled_metric,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "total": intrinsic + extrinsic if params["run_intrinsic"] else stochastic_metric,
        "stochastic_metric": stochastic_metric,
        "profile_label": "Expected buy offtake (MWh/day)",
        "title_prefix": "Put swing",
        "intrinsic_profile_raw": intrinsic_profile_raw,
        "extrinsic_profile_raw": extrinsic_profile_raw,
    }


def value_call_swing(curve, params):
    s = Storage(params["valDate"], params["storageStart"], params["storageEnd"], curve=curve, n_p=0, v_step=params["v_step"], sVol=params["vol"])
    flat_metric = daily_arithmetic_flat_metric(s)

    init_inv     = params["initial_inv_clips"]  if params.get("initial_inv_clips")  is not None else params["days"]
    term_inv     = params["terminal_inv_clips"] if params.get("terminal_inv_clips") is not None else 0
    strike       = params.get("strike", 0.0)
    zero_penalty = params.get("zero_penalty", False)

    if strike:
        s.w_cost[s.Dt:s._active] = strike

    s.set_volume_states(params["days"])
    s.n_op_start = init_inv
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    if zero_penalty:
        s.t_p_curve[:s.n_op + 1] = 0.0   # any residual inventory is fine
    else:
        s.t_p_curve[term_inv] = 0.0

    if params["run_intrinsic"]:
        s.build()
        profiled_eur = s.v[0, 0, init_inv]
        acq = np.sum(s.delta)
        profiled_metric = profiled_eur / acq if acq else np.nan
        intrinsic = profiled_metric - flat_metric if acq else np.nan
        intrinsic_profile_raw = np.array(s.exp_ex)
    else:
        profiled_eur = np.nan
        acq = np.nan
        profiled_metric = np.nan
        intrinsic = np.nan
        intrinsic_profile_raw = np.zeros(len(s.date_span))

    s.n_p = params["n_p_full"]
    s.build()
    full_eur = s.v[0, s.n_p, init_inv]
    full_acq = np.sum(s.delta)
    stochastic_metric = full_eur / full_acq if full_acq else np.nan
    extrinsic = (full_eur - profiled_eur) / acq if (params["run_intrinsic"] and acq) else np.nan
    extrinsic_profile_raw = np.array(s.exp_ex)

    return s, {
        "flat_metric": flat_metric,
        "profiled_metric": profiled_metric,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "total": intrinsic + extrinsic if params["run_intrinsic"] else stochastic_metric,
        "stochastic_metric": stochastic_metric,
        "profile_label": "Expected sell offtake (MWh/day)",
        "title_prefix": "Call swing",
        "intrinsic_profile_raw": intrinsic_profile_raw,
        "extrinsic_profile_raw": extrinsic_profile_raw,
    }


def value_storage(curve, params):
    s = Storage(params["valDate"], params["storageStart"], params["storageEnd"], curve=curve, n_p=0, v_step=params["v_step"], sVol=params["vol"])
    n, active = active_masks(s)
    s.i_curve = active.copy()
    s.w_curve = active.copy()
    s.i_cost[:] = params["inj_cost"]
    s.w_cost[:] = params["wdr_cost"]
    init_inv = params["initial_inv_clips"] if params.get("initial_inv_clips") is not None else 0
    term_inv = params["terminal_inv_clips"] if params.get("terminal_inv_clips") is not None else 0
    s.set_volume_states(params["inj_days"])
    s.n_op_start = init_inv
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    s.t_p_curve[term_inv] = 0.0

    max_vol = params["inj_days"] * s.v_step
    if params["run_intrinsic"]:
        s.build()
        intrinsic_eur = s.v[0, 0, init_inv]
        intrinsic_profile_raw = np.array(s.exp_ex)
    else:
        intrinsic_eur = np.nan
        intrinsic_profile_raw = np.zeros(len(s.date_span))

    s.n_p = params["n_p_full"]
    s.build()
    total_eur = s.v[0, s.n_p, init_inv]
    extrinsic_eur = total_eur - intrinsic_eur if params["run_intrinsic"] else np.nan
    extrinsic_profile_raw = np.array(s.exp_ex)

    return s, {
        "flat_metric": np.nan,
        "profiled_metric": intrinsic_eur / max_vol if params["run_intrinsic"] else np.nan,
        "intrinsic": intrinsic_eur / max_vol if params["run_intrinsic"] else np.nan,
        "extrinsic": extrinsic_eur / max_vol if params["run_intrinsic"] else np.nan,
        "total": total_eur / max_vol,
        "stochastic_metric": total_eur / max_vol,
        "intrinsic_eur": intrinsic_eur,
        "extrinsic_eur": extrinsic_eur,
        "total_eur": total_eur,
        "profile_label": "Expected exercise: sell + / buy - (MWh/day)",
        "title_prefix": "Storage",
        "intrinsic_profile_raw": intrinsic_profile_raw,
        "extrinsic_profile_raw": extrinsic_profile_raw,
    }


def run_valuation(curve, params):
    if params["product_type"] == "put_swing":
        return value_put_swing(curve, params)
    if params["product_type"] == "call_swing":
        return value_call_swing(curve, params)
    if params["product_type"] == "storage":
        return value_storage(curve, params)
    raise ValueError("Unknown product_type")


# ── Price tree ────────────────────────────────────────────────────────────────

def build_tree(price_curve, n_t, n_p, vol_curve, mr_curve):
    dt      = 1. / 365.25
    dx      = vol_curve[0] * sqrt(3 * dt)
    vol_arr = np.asarray(vol_curve, dtype=np.float64)
    mr_arr  = np.asarray(mr_curve,  dtype=np.float64)
    fwd     = np.asarray(price_curve, dtype=np.float64)[:n_t]

    x   = np.zeros((n_t, 2*n_p+1))
    p_u = np.zeros((n_t, 2*n_p+1))
    p_d = np.zeros((n_t, 2*n_p+1))
    p_m = np.zeros((n_t, 2*n_p+1))

    q = _tree_core(x, p_u, p_m, p_d, fwd, vol_arr, mr_arr, n_t, n_p, dx, dt)

    return fwd, x, q, p_u, p_m, p_d


# ── Valuation helpers / metrics ───────────────────────────────────────────────

def valuation(n_p, v, q, n_op_start):
    """Expected value at contract start, averaging over price states."""
    result = 0
    for i in range(max(n_p, 0), min(n_p, 2*n_p) + 1):
        result += v[0, i, n_op_start] * q[0, i]
    return result


def compute_all_metrics(n_t, n_p, n_op, prob, strat, i_ratch, w_ratch, v_step, w_curve, i_curve, x, fwd):
    l_arr  = np.arange(n_op, dtype=float)
    w_step = (w_curve[:n_t, None] * w_ratch[None, :])[:, None, :]
    i_step = (i_curve[:n_t, None] * i_ratch[None, :])[:, None, :]
    l_bc   = l_arr[None, None, :]

    wdr_amt = np.minimum(w_step, l_bc) * v_step
    inj_amt = np.minimum(i_step, (n_op - 1) - l_bc) * v_step
    action  = np.where(strat == -1, -wdr_amt,
                       np.where(strat == 1, inj_amt, 0.0))

    pa     = prob * action
    exp_ex = list(-np.round(pa.sum(axis=(1, 2)), 3)) + [0.0]

    exp_x  = np.exp(x)[:, :, None]
    delta  = list(-np.round((pa * exp_x).sum(axis=(1, 2)) / fwd[:n_t], 3)) + [0.0]

    return exp_ex, delta
