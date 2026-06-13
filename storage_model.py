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
    sVol    : annualised spot volatility (fraction); the tree uses
              dx = sVol * sqrt(3*dt) with dt = 1/365.25
    sMR     : mean-reversion speed
    clips_per_day : max number of v_step clips injected/withdrawn per active day
                    (the daily injection/withdrawal rate). Default 3.
    """

    def __init__(self, valDate, storageStart, storageEnd,
                 curve=None, n_p=0, v_step=1000, sVol=0.9, sMR=1.0, clips_per_day=3,
                 daily_curve=None):
        self.valDate      = pd.Timestamp(valDate)
        self.storageStart = pd.Timestamp(storageStart)
        self.storageEnd   = pd.Timestamp(storageEnd)
        self.v_step       = v_step
        self.n_p          = n_p
        self.clips_per_day = clips_per_day

        # Date grid
        self.backStop  = (self.storageEnd + pd.DateOffset(months=1)).normalize() + pd.offsets.MonthEnd(0)
        self.Dt        = (self.storageStart - self.valDate).days
        self.n_t       = (self.backStop     - self.valDate).days
        self.date_span = pd.date_range(self.valDate, self.backStop, freq='D')
        self._active   = len(pd.date_range(self.valDate, self.storageEnd, freq='D'))

        # Forward / price curve. A precomputed daily_curve (a date-indexed Series)
        # is used verbatim — reindexed onto the date grid, edges filled — so the
        # SAME daily curve can feed any deal regardless of its dates/product. With
        # no daily_curve, the stepped contract curve is smoothed to daily here.
        if daily_curve is not None:
            self.price_curve = pd.Series(daily_curve).reindex(self.date_span).ffill().bfill()
        else:
            if curve is None:
                raise ValueError("Storage requires either `curve` or `daily_curve`.")
            self.price_curve = smoothen_curve(
                map_curve_to_dates(self.date_span, curve).to_frame(name='value')['value']
            )

        # Price-tree vol / mean-reversion profiles
        self.sVol = [sVol] * self.n_t
        self.sMR  = [sMR]  * self.n_t

        # Discount curve (flat 1)
        self.d_curve = np.ones(self.n_t)

        # Exercise curves: no injection (swing = sell-only), withdraw during active
        # window. The withdraw rate per active day is clips_per_day clips.
        n = len(self.date_span)
        self.i_curve = np.zeros(n)
        self.w_curve = np.full(n, float(self.clips_per_day))
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

    def apply_ratchets(self, fullness, inj_mult, wdr_mult):
        """Set per-inventory-level injection/withdrawal rate multipliers from a
        ratchet table (fullness -> multiplier). The daily clip rate at level l
        becomes clips_per_day * multiplier(l / (n_op-1)). Call AFTER
        set_volume_states, which resets the ratchets to 1."""
        self.i_ratch, self.w_ratch = ratchet_arrays(self.n_op, fullness, inj_mult, wdr_mult)
        return self

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
        """Unweighted average forward price over the exercise window — the
        zero-optionality baseline. Independent of n_p and the optimisation."""
        exercise_dates = self.date_span[self.Dt:self._active]
        return float(pd.Series(self.price_curve, index=self.date_span).loc[exercise_dates].mean())

    def profiled(self):
        """Volume-weighted achieved price per MWh under the optimal strategy.
        Reads the central forward node v[0, n_p, .], so it is correct for both
        the intrinsic (n_p=0) and the full (n_p>0) build."""
        ACQ = np.sum(self.delta)
        return self.v[0, self.n_p, self.n_op_start] / ACQ


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


def resolve_grid(params, states_key):
    """Map physical inputs to the model's (clip size, #states, clips/day).

    Two ways to size the volume grid:

    * Physical   — ``daily_max`` (MWh injected/withdrawn per active day) together
      with ``clips_per_day`` fixes the clip size ``v_step = daily_max/clips_per_day``;
      ``capacity_mwh`` then fixes the number of inventory states as
      ``round(capacity_mwh / v_step)`` (total working volume stays put when the
      clip is refined).
    * Legacy     — fall back to ``params['v_step']`` and ``params[states_key]``
      (clip count) when the physical inputs are absent.

    Returns (v_step, n_states, clips_per_day).
    """
    cpd = int(params.get("clips_per_day", 3))

    daily_max = params.get("daily_max")
    v_step = float(daily_max) / cpd if daily_max is not None else float(params["v_step"])

    capacity = params.get("capacity_mwh")
    n_states = int(round(float(capacity) / v_step)) if capacity is not None else int(params[states_key])

    return v_step, n_states, cpd


def load_ratchets(source):
    """Load a ratchet table with columns: fullness, injection, withdrawal.

    ``fullness`` is fraction full in [0, 1] (percent values >1.5 are divided by
    100); ``injection``/``withdrawal`` are rate multipliers applied to the base
    daily clip rate at that inventory fullness. Returns sorted numpy arrays
    (fullness, inj_mult, wdr_mult). ``source`` may be a path or a DataFrame.
    """
    df = source if isinstance(source, pd.DataFrame) else pd.read_excel(source)
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    missing = {"fullness", "injection", "withdrawal"} - set(df.columns)
    if missing:
        raise ValueError(f"Ratchet table missing columns: {', '.join(sorted(missing))}")
    df = df[["fullness", "injection", "withdrawal"]].dropna().sort_values("fullness")
    f = df["fullness"].to_numpy(dtype=float)
    if f.size and f.max() > 1.5:        # given in percent -> fraction
        f = f / 100.0
    return f, df["injection"].to_numpy(dtype=float), df["withdrawal"].to_numpy(dtype=float)


def ratchet_arrays(n_op, fullness, inj_mult, wdr_mult):
    """Interpolate a ratchet table onto the n_op inventory states.

    Inventory state l has fullness l/(n_op-1); returns (i_ratch, w_ratch) of
    length n_op (rate multipliers per state).
    """
    if n_op <= 1:
        return np.ones(n_op), np.ones(n_op)
    states_full = np.arange(n_op, dtype=float) / (n_op - 1)
    i_ratch = np.interp(states_full, fullness, inj_mult)
    w_ratch = np.interp(states_full, fullness, wdr_mult)
    return i_ratch, w_ratch


def apply_ratchets_from_params(model, params):
    """Apply ratchets to a model if params['ratchets'] is set (path, DataFrame,
    or a pre-loaded (fullness, inj, wdr) tuple). Call after set_volume_states."""
    ratch = params.get("ratchets")
    if ratch is None:
        return
    f, inj, wdr = ratch if isinstance(ratch, tuple) else load_ratchets(ratch)
    model.apply_ratchets(f, inj, wdr)


def assert_cycle_feasible(model, clips_needed, clips_per_day, what):
    """Guard against an infeasible forced terminal inventory.

    A put/call swing must move ``clips_needed`` clips (buy to full / sell to empty)
    within the active window. If that exceeds ``active_days * clips_per_day`` the DP
    cannot satisfy its terminal constraint and silently returns the -1e9 penalty,
    surfacing as a wildly negative "intrinsic". Fail loudly instead.
    """
    _, active = active_masks(model)
    active_days = int(active.sum())
    max_clips = active_days * clips_per_day
    if clips_needed > max_clips:
        need_mwh = clips_needed * model.v_step
        feasible_mwh = max_clips * model.v_step
        raise ValueError(
            f"{what}: needs to move {need_mwh:,.0f} MWh but the window allows at most "
            f"{active_days} active days x {clips_per_day} clips/day = {max_clips} clips "
            f"({feasible_mwh:,.0f} MWh). "
            f"Lower capacity_mwh, extend the window, or raise daily_max/clips_per_day."
        )


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
    v_step, n_states, cpd = resolve_grid(params, "days")
    s = Storage(params["valDate"], params["storageStart"], params["storageEnd"], curve=curve, n_p=0, v_step=v_step, sVol=params["vol"], sMR=params.get("sMR", 1.0), clips_per_day=cpd, daily_curve=params.get("daily_curve"))
    n, active = active_masks(s)
    s.i_curve = cpd * active
    s.w_curve = np.zeros(n)

    strike = params.get("strike", 0.0)
    if strike:
        s.i_cost[s.Dt:s._active] = -strike   # per-clip inject profit = strike - price

    flat_metric = daily_arithmetic_flat_metric(s)

    init_inv = params["initial_inv_clips"] if params.get("initial_inv_clips") is not None else 0
    term_inv = params["terminal_inv_clips"] if params.get("terminal_inv_clips") is not None else n_states
    s.set_volume_states(n_states)
    apply_ratchets_from_params(s, params)
    assert_cycle_feasible(s, abs(term_inv - init_inv), cpd, "Put swing")
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
    v_step, n_states, cpd = resolve_grid(params, "days")
    s = Storage(params["valDate"], params["storageStart"], params["storageEnd"], curve=curve, n_p=0, v_step=v_step, sVol=params["vol"], sMR=params.get("sMR", 1.0), clips_per_day=cpd, daily_curve=params.get("daily_curve"))
    flat_metric = daily_arithmetic_flat_metric(s)

    init_inv     = params["initial_inv_clips"]  if params.get("initial_inv_clips")  is not None else n_states
    term_inv     = params["terminal_inv_clips"] if params.get("terminal_inv_clips") is not None else 0
    strike       = params.get("strike", 0.0)
    zero_penalty = params.get("zero_penalty", False)

    if strike:
        s.w_cost[s.Dt:s._active] = strike

    s.set_volume_states(n_states)
    apply_ratchets_from_params(s, params)
    if not zero_penalty:
        assert_cycle_feasible(s, abs(term_inv - init_inv), cpd, "Call swing")
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
    v_step, n_states, cpd = resolve_grid(params, "inj_days")
    s = Storage(params["valDate"], params["storageStart"], params["storageEnd"], curve=curve, daily_curve=params.get("daily_curve"), n_p=0, v_step=v_step, sVol=params["vol"], sMR=params.get("sMR", 1.0), clips_per_day=cpd)
    n, active = active_masks(s)
    s.i_curve = cpd * active
    s.w_curve = cpd * active
    s.i_cost[:] = params["inj_cost"]
    s.w_cost[:] = params["wdr_cost"]
    init_inv = params["initial_inv_clips"] if params.get("initial_inv_clips") is not None else 0
    term_inv = params["terminal_inv_clips"] if params.get("terminal_inv_clips") is not None else 0
    s.set_volume_states(n_states)
    apply_ratchets_from_params(s, params)
    s.n_op_start = init_inv
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    s.t_p_curve[term_inv] = 0.0

    max_vol = n_states * v_step
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


# ── Valuation helpers ─────────────────────────────────────────────────────────

def valuation(n_p, v, q, n_op_start):
    """Expected value at contract start, averaging over price states."""
    result = 0
    for i in range(max(n_p, 0), min(n_p, 2*n_p) + 1):
        result += v[0, i, n_op_start] * q[0, i]
    return result


def compute_all_metrics(n_t, n_p, n_op, prob, strat, i_ratch, w_ratch, v_step, w_curve, i_curve, x, fwd):
    # strat holds the signed clip count moved per state (neg=withdraw, pos=inject).
    action = strat[:n_t] * v_step               # MWh moved per (time, price, vol)

    pa     = prob * action
    exp_ex = list(-np.round(pa.sum(axis=(1, 2)), 3)) + [0.0]

    exp_x  = np.exp(x)[:, :, None]
    delta  = list(-np.round((pa * exp_x).sum(axis=(1, 2)) / fwd[:n_t], 3)) + [0.0]

    return exp_ex, delta


# ── Product parameter workbook ────────────────────────────────────────────────

def list_products(path="products.xlsx"):
    """Product names available in the workbook's 'products' sheet."""
    df = pd.read_excel(path, sheet_name="products")
    return df["product"].astype(str).tolist()


def load_product_params(path="products.xlsx", product=None):
    """Load one product's parameters from the workbook.

    Workbook layout:
      * sheet 'products' — one row per product. Columns: product, product_type,
        FDDate, valDate, storageStart, storageEnd, vol, n_p_full, run_intrinsic,
        capacity_mwh, initial_storage_mwh, terminal_storage_mwh, inj_days,
        wdr_days, n_states, inj_cost, wdr_cost, ratchet_profile, notes.
      * sheet 'ratchets' — named rate-multiplier profiles. Columns: profile,
        fullness, injection, withdrawal. A blank ratchet_profile disables
        ratchets for that product.

    Returns a dict of typed primary inputs; 'ratchets' is None or a
    (fullness, inj_mult, wdr_mult) tuple ready for apply_ratchets.
    Derived grid quantities (clip size, rates) are intentionally NOT computed
    here — that stays in the notebook so the derivation is visible.
    """
    df = pd.read_excel(path, sheet_name="products")
    if "product" not in df.columns:
        raise ValueError(f"{path}: 'products' sheet needs a 'product' column")
    df["product"] = df["product"].astype(str)

    if product is None:
        if len(df) != 1:
            raise ValueError(
                f"{path} has {len(df)} products: {', '.join(df['product'])}. "
                "Pass product=<name>.")
        row = df.iloc[0]
    else:
        match = df.loc[df["product"] == str(product)]
        if match.empty:
            raise ValueError(
                f"Product {product!r} not in {path}. "
                f"Available: {', '.join(df['product'])}")
        if len(match) > 1:
            raise ValueError(f"Product {product!r} appears {len(match)} times in {path}")
        row = match.iloc[0]

    def _req(name):
        if name not in row.index or pd.isna(row[name]):
            raise ValueError(f"Product {row['product']!r}: missing required field {name!r}")
        return row[name]

    def _bool(x):
        if isinstance(x, str):
            return x.strip().lower() in ("true", "yes", "y", "1")
        return bool(x)

    ptype = str(_req("product_type")).strip()
    if ptype not in ("put_swing", "call_swing", "storage"):
        raise ValueError(f"product_type must be put_swing/call_swing/storage, got {ptype!r}")

    params = {
        "product":              str(row["product"]),
        "product_type":         ptype,
        "FDDate":               pd.Timestamp(_req("FDDate")),
        "valDate":              pd.Timestamp(_req("valDate")),
        "storageStart":         pd.Timestamp(_req("storageStart")),
        "storageEnd":           pd.Timestamp(_req("storageEnd")),
        "vol":                  float(_req("vol")),
        "n_p_full":             int(_req("n_p_full")),
        "run_intrinsic":        _bool(_req("run_intrinsic")),
        "capacity_mwh":         float(_req("capacity_mwh")),
        "initial_storage_mwh":  float(row["initial_storage_mwh"]) if pd.notna(row.get("initial_storage_mwh")) else 0.0,
        "terminal_storage_mwh": float(row["terminal_storage_mwh"]) if pd.notna(row.get("terminal_storage_mwh")) else 0.0,
        "inj_days":             int(_req("inj_days")),
        "wdr_days":             int(_req("wdr_days")),
        "n_states":             int(_req("n_states")),
        "inj_cost":             float(_req("inj_cost")),
        "wdr_cost":             float(_req("wdr_cost")),
        "notes":                "" if pd.isna(row.get("notes")) else str(row.get("notes")),
    }

    if params["storageEnd"] < params["storageStart"]:
        raise ValueError(f"Product {params['product']!r}: storageEnd before storageStart")

    profile = row.get("ratchet_profile")
    if pd.isna(profile) or str(profile).strip() == "":
        params["ratchets"] = None
        params["ratchet_profile"] = None
    else:
        profile = str(profile).strip()
        rdf = pd.read_excel(path, sheet_name="ratchets")
        rdf = rdf.rename(columns={c: str(c).strip().lower() for c in rdf.columns})
        sub = rdf.loc[rdf["profile"].astype(str).str.strip() == profile]
        if sub.empty:
            avail = ", ".join(sorted(rdf["profile"].astype(str).str.strip().unique()))
            raise ValueError(f"Ratchet profile {profile!r} not in {path} (available: {avail})")
        params["ratchets"] = load_ratchets(sub[["fullness", "injection", "withdrawal"]])
        params["ratchet_profile"] = profile

    return params


def params_for_run_valuation(prm):
    """Augment a ``load_product_params()`` dict with the derived grid fields that
    ``run_valuation()`` / ``value_*`` need, so the two documented entry points can
    be chained directly:

        params = params_for_run_valuation(load_product_params("products.xlsx", name))
        s, result = run_valuation(curve, params)

    Without this step ``run_valuation`` raises ``KeyError: 'v_step'`` because
    ``load_product_params`` deliberately leaves the grid derivation out (it stays
    visible in the notebook). The clip size is ``v_step = capacity_mwh / n_states``
    and the daily rate is the base inject rate ``round(n_states / inj_days)``.

    NOTE: this uses the SYMMETRIC library grid — a single daily clip rate shared
    by injection and withdrawal. Asymmetric rates (``inj_days != wdr_days``) are
    not supported by ``value_storage`` yet; use ``forward.ipynb`` for those.
    """
    p = dict(prm)
    n_states = int(p["n_states"])
    capacity = float(p["capacity_mwh"])
    v_step = capacity / n_states
    p["v_step"] = v_step
    p["daily_max"] = None                       # resolve_grid then uses v_step + capacity_mwh
    p["clips_per_day"] = max(1, int(round(n_states / int(p["inj_days"]))))
    # Initial/terminal inventory map from the storage_mwh fields ONLY for the
    # storage product. For put/call swing the forced-cycle start and end states
    # are fixed by the product (a call swing starts full, a put swing ends full),
    # so leave them unset and let value_call_swing / value_put_swing apply their
    # own defaults — forcing initial_inv_clips=0 on a call swing would start it
    # empty, leaving nothing to sell and yielding 0/0 = NaN.
    if p.get("product_type") == "storage":
        if p.get("initial_inv_clips") is None:
            p["initial_inv_clips"] = int(round(float(p.get("initial_storage_mwh", 0.0)) / v_step))
        if p.get("terminal_inv_clips") is None:
            p["terminal_inv_clips"] = int(round(float(p.get("terminal_storage_mwh", 0.0)) / v_step))
    return p
