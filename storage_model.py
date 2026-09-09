"""Gas storage / swing valuation: forward-curve utilities, the `Storage` class,
and product valuation wrappers over a trinomial price tree + dynamic program.

Symbol glossary (used throughout this module and the kernels)
------------------------------------------------------------
    n_t        number of daily time steps (valDate .. backStop)
    n_p        price-tree half-width; the tree has 2*n_p+1 price states
               (n_p = 0 means a single price path = no optionality)
    n_states   inventory GRID SIZE: the number of clip levels
    n_op       number of inventory states, = n_states + 1
    initial_state  where inventory STARTS, in 0..n_states
    n_op_start deprecated alias for initial_state. It used to mean the grid size
               to set_volume_states() and the initial state to build(), so every
               caller had to set it twice; the two now have their own names
    Dt         offset (days) from valDate to storageStart (first active day)
    v_step     MWh per inventory state (the "clip" size)
    clips_per_day  max clips injected/withdrawn per active day (the daily rate)
    x          log-price deviation from the forward at each (time, price) node
    strat      signed clip count moved per state (neg=withdraw, pos=inject, 0=idle)
    exp_ex     expected daily exercise volume (MWh), length n_t+1
    delta      undiscounted hedge volume: the forward MWh to trade for each day
               (E[S*Q]/F), length n_t+1. Right against an OTC forward settling
               with the deal, where the discount factor cancels
    delta_pv   the same tailed by d_curve: right against margined futures, and
               the PV risk number. Equals delta when the rate is zero. See
               docs/MODEL-CONVENTIONS.md
    t_p_curve  terminal inventory payoff/penalty by state (-1e9 forbids a state)
    i_curve/w_curve   per-day injection/withdrawal permission (clips/day)
    i_cost/w_cost     per-MWh injection/withdrawal cost (a strike enters here)
    i_ratch/w_ratch   per-inventory-level rate multipliers (ratchets)
    mintunnel/max_tunnel  per-day inventory floor/ceiling

Kernel penalty constants (storage_kernels.py): a forbidden terminal inventory
carries -1e9; violating a tunnel costs 1000*v_step per clip out of bounds; an
exercise whose gain over idling is < 1e-6 is snapped to idle (treated as no-trade).
"""
import numpy as np
from math import sqrt
import re
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
import pandas as pd

# Numba kernels live in storage_kernels.py so that edits to this file do not
# invalidate their disk cache (which would trigger a 20-40s recompile).
# Re-exported here for backward compatibility.
from storage_kernels import _tree_core, run_model, probabilities


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
                 daily_curve=None, discount_rate=0.0):
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
            # Check coverage BEFORE smoothing: smoothen_curve fits a spline through
            # the monthly means, and SciPy rejects NaN knots with "`y` must contain
            # only finite values" -- which says nothing about the real problem. The
            # guard below never fired on this path until the check moved up here.
            mapped = map_curve_to_dates(self.date_span, curve).to_frame(name='value')['value']
            self._require_full_coverage(mapped, "curve")
            self.price_curve = smoothen_curve(mapped)

        # Fail loudly on curve gaps. Otherwise NaNs propagate silently through the
        # tree and the valuation returns garbage with no error (a real foot-gun on
        # the contract-curve path, where days outside any contract stay NaN).
        self._require_full_coverage(self.price_curve, "curve/daily_curve")

        # Price-tree vol / mean-reversion profiles
        self.sVol = [sVol] * self.n_t
        self.sMR  = [sMR]  * self.n_t

        # Discount curve. Cash from an earlier withdrawal can be redeployed, so a
        # euro on day i is worth exp(-r*i/365.25) today. The DP multiplies every
        # day's cash flow by this, so a positive rate makes it prefer earlier
        # exercise -- with rate 0 (the default) it is all ones and nothing changes.
        # Assign self.d_curve directly for a real, non-flat discount curve.
        self.discount_rate = float(discount_rate)
        self.d_curve = discount_factors(self.n_t, self.discount_rate)

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
        # Two separate things: how many clip levels the grid holds, and where
        # inventory starts. Both used to be `n_op_start`. The default start is a
        # full grid, as it has always been.
        self.n_states = (self.storageEnd - self.storageStart).days + 1
        self.initial_state = self.n_states
        self._init_volume_arrays()

    def _require_full_coverage(self, series, what):
        """Raise unless `series` covers every day of the valuation grid."""
        if series.isna().any():
            n_missing = int(series.isna().sum())
            gap = series.index[series.isna()]
            raise ValueError(
                f"Price curve has {n_missing} missing day(s) over the valuation grid "
                f"{self.date_span[0]:%Y-%m-%d}..{self.date_span[-1]:%Y-%m-%d} "
                f"(first {gap[0]:%Y-%m-%d}, last {gap[-1]:%Y-%m-%d}): the supplied "
                f"{what} does not cover the full storage period, including the "
                f"month-end backstop at {self.backStop:%Y-%m-%d}. Extend the curve.")

    def _init_volume_arrays(self):
        """(Re-)build arrays that depend on the grid size."""
        self.n_op      = self.n_states + 1
        self.i_ratch   = np.ones(self.n_op)
        self.w_ratch   = np.ones(self.n_op)
        self.max_tunnel = np.full(len(self.date_span), self.n_op)
        self.t_p_curve  = np.full(self.n_op + 2, -1e9)
        self.t_p_curve[0] = 0.

    def set_volume_states(self, n_states, initial_state=None):
        """Size the inventory grid, and optionally set where inventory starts.

        `n_states` is the number of clip levels; the grid then holds states
        0..n_states (`n_op = n_states + 1`). `initial_state` is where inventory
        begins; it defaults to a full grid, which is what this did before.

        Both used to be `n_op_start`, which meant the grid size here and the
        initial state to build(), so every caller set it twice. `n_op_start`
        still works as an alias for `initial_state`.

        Note this resets the ratchets, tunnels and t_p_curve, so call
        apply_ratchets() and set t_p_curve after it, not before.
        """
        self.n_states = n_states
        self.initial_state = n_states if initial_state is None else initial_state
        self._init_volume_arrays()

    @property
    def n_op_start(self):
        """Deprecated alias for `initial_state`, kept for existing callers."""
        return self.initial_state

    @n_op_start.setter
    def n_op_start(self, value):
        self.initial_state = value

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
        # Nothing checked this before: an out-of-range start indexed past the
        # value array, in a Numba kernel where that is undefined rather than an
        # IndexError.
        if not (0 <= self.initial_state <= self.n_states):
            raise ValueError(
                f"initial_state={self.initial_state} is outside the inventory grid "
                f"0..{self.n_states} (n_op={self.n_op}). Set it with "
                f"set_volume_states(n_states, initial_state=...).")

        self.fwd, self.x, q, p_u, p_m, p_d = build_tree(
            self.price_curve, self.n_t, self.n_p, self.sVol, self.sMR)

        self.v, self.strat = run_model(
            self.n_t, self.n_p, self.n_op, self.v_step, self.x, p_u, p_m, p_d,
            self.d_curve, self.i_curve, self.w_curve, self.i_cost, self.w_cost,
            self.t_p_curve, self.i_ratch, self.w_ratch, self.mintunnel, self.max_tunnel)

        self.prob = probabilities(
            self.n_t, self.n_p, self.n_op, q, self.strat, p_u, p_m, p_d,
            self.initial_state)

        self._assert_terminal_inventory_reached()

        # Two hedge ratios, because there are two hedge instruments. `delta` is the
        # physical forward volume -- right against an OTC forward settling with the
        # deal, where DF cancels. `delta_pv` is that tailed by the discount factor:
        # right against margined futures, whose variation margin moves today while
        # the gas settles later, and the right number for PV risk. They coincide
        # when the rate is zero.
        self.exp_ex, self.delta = compute_all_metrics(
            self.n_t, self.n_p, self.n_op, self.prob, self.strat,
            self.i_ratch, self.w_ratch, self.v_step,
            self.w_curve, self.i_curve, self.d_curve, self.x, self.fwd)
        self.delta_pv = list(np.asarray(self.delta[:self.n_t]) * self.d_curve) + [0.0]

        return self

    def _assert_terminal_inventory_reached(self, tolerance=1e-9):
        """Fail when the optimal policy cannot satisfy the terminal constraint.

        Unlike the inexpensive pre-build swing guard, this check observes the
        actual policy and therefore accounts for ratchets, date masks, tunnels,
        asymmetric rates and arbitrary initial/terminal inventory states.
        """
        allowed = np.asarray(self.t_p_curve[:self.n_op]) > -1e9
        if not allowed.any():
            raise ValueError("No permitted terminal inventory state is configured.")
        terminal_mass = float(self.prob[-1, :, allowed].sum())
        if not np.isfinite(terminal_mass) or terminal_mass < 1.0 - tolerance:
            raise ValueError(
                "The terminal inventory constraint is infeasible under the configured "
                f"rates, ratchets, date masks and tunnels: only {terminal_mass:.6%} of "
                "model probability reaches a permitted terminal state."
            )

    def flat(self):
        """Unweighted average forward price over the exercise window — the
        zero-optionality baseline. Independent of n_p and the optimisation."""
        exercise_dates = self.date_span[self.Dt:self._active]
        return float(pd.Series(self.price_curve, index=self.date_span).loc[exercise_dates].mean())

    def profiled(self):
        """Contract value per expected net exercised MWh.

        This legacy method name is retained for compatibility. The denominator
        is the physical expected exercise schedule, not the hedge delta. It is
        therefore meaningful for one-directional swing contracts and raises for
        a zero-net-volume strategy such as a cycling storage contract.
        """
        volume = float(np.sum(self.exp_ex))
        gross_volume = float(np.sum(np.abs(self.exp_ex)))
        if abs(volume) <= 1e-12 * max(gross_volume, 1.0):
            raise ValueError(
                "Value per net exercised MWh is undefined for a zero-net-volume strategy."
            )
        return self.v[0, self.n_p, self.initial_state] / volume


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
    if curve_start < quote_date:
        raise ValueError(
            f"curve_start {curve_start:%Y-%m-%d} is before the quote it is built from "
            f"({quote_date:%Y-%m-%d}). That values a date using a curve observed after it, "
            f"and the {(quote_date - curve_start).days}-day gap would be back-filled with "
            f"this quote's day-ahead price. Move the valuation date to the quote date or "
            f"later, or pick an earlier quote."
        )

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


def discount_factors(n_t, rate):
    """Daily discount factors to the valuation date, continuously compounded."""
    return np.exp(-float(rate) * np.arange(n_t) / 365.25)


def daily_arithmetic_flat_metric(model, strike=0.0):
    """The zero-optionality benchmark: one MWh spread evenly over the window.

    Present-valued, and net of any strike, so it is on the same footing as the
    `profiled_metric` it is subtracted from. Both legs must be PV'd or the
    difference measures the discount factor instead of the day-selection spread --
    the same mistake as benchmarking a strike-net value against a raw average.

    With rate 0 and no strike this is exactly the old unweighted mean forward.
    """
    win = slice(model.Dt, model._active)
    fwd = np.asarray(model.price_curve, dtype=float)[win]
    return float(np.mean(model.d_curve[win] * (fwd - float(strike))))


def intrinsic_components(model, strike=0.0):
    r"""Split `intrinsic` into what the schedule saves on price and on timing.

    `model` is the **deterministic** run -- the `n_p = 0` build whose schedule
    produces `profiled_metric`. Returns `(day_selection, financing)`, which sum
    to `intrinsic` exactly.

    Intrinsic is the gain from choosing days rather than spreading evenly over
    the window, and with a discount rate that is two different gains. A flat
    forward curve is not flat once discounted: the curve the optimiser actually
    sees is `DF * F`, and at 10 % a flat 40 slopes from 36.84 on 1 Jan to 33.34
    on 31 Dec. There is then something to choose with no price shape at all.
    Writing `flat = Fbar * df_bench` and `profiled = P * df_paid`, where `P` is
    the undiscounted price the schedule pays:

        intrinsic = df_bench * (P - Fbar)  +  P * (df_paid - df_bench)
                    \___ day selection __/    \_____ financing _____/

    The first term is what the schedule saves on price, held at the benchmark's
    own discount factor. The second is what it saves by settling on different
    days. **On a flat curve the first is exactly zero and every euro of
    intrinsic is financing** -- a hurdle-rate gain on deferred cash, not a
    commodity gain, and realised by actually deferring rather than by trading.

    Raises for a two-sided strategy, where `profiled_metric` is undefined.
    """
    n = model.n_t
    win = slice(model.Dt, model._active)
    net = np.asarray(model.price_curve, dtype=float)[:n] - float(strike)
    ex = np.asarray(model.exp_ex[:n], dtype=float)
    volume = float(ex.sum())
    if abs(volume) <= 1e-12 * max(float(np.abs(ex).sum()), 1.0):
        raise ValueError(
            "The intrinsic split is undefined for a zero-net-volume strategy."
        )
    net_bar = float(np.mean(net[win]))
    paid = float(np.dot(ex, net) / volume)
    if abs(net_bar) < 1e-12 or abs(paid) < 1e-12:
        raise ValueError(
            "The intrinsic split is undefined when the strike sits on the curve: "
            f"mean forward net of strike is {net_bar:.3e} and the schedule pays "
            f"{paid:.3e}, so the discount factors it divides out are not recoverable."
        )
    df_bench = float(np.mean(model.d_curve[win] * net[win])) / net_bar
    df_paid = model.profiled() / paid
    sign = 1.0 if volume > 0 else -1.0
    # The trailing `+ 0.0` normalises a signed zero. On a flat curve the first
    # term is -1.0 * 0.0 = -0.0, which renders as "-0.000" in a table and reads
    # like a defect; adding zero is exact for every other value.
    return (sign * df_bench * (paid - net_bar) + 0.0,
            sign * paid * (df_paid - df_bench) + 0.0)


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
    s = Storage(params["valDate"], params["storageStart"], params["storageEnd"], curve=curve, n_p=0, v_step=v_step, sVol=params["vol"], sMR=params.get("sMR", 1.0), clips_per_day=cpd, daily_curve=params.get("daily_curve"), discount_rate=params.get("discount_rate", 0.0))
    n, active = active_masks(s)
    s.i_curve = cpd * active
    s.w_curve = np.zeros(n)

    strike = params.get("strike", 0.0)
    if strike:
        s.i_cost[s.Dt:s._active] = -strike   # per-clip inject profit = strike - price

    # Net of the strike, for the same reason as the call swing below: `profiled_metric`
    # is the effective price after the strike leg, so the zero-optionality benchmark
    # must be too. Buying on average days nets mean(F) - K per MWh. Without this the
    # put's intrinsic shifted by +K -- 0.157, 10.157 and 20.157 EUR/MWh for K = 0, 10
    # and 20 on one deal -- the mirror of the call's -K.
    flat_metric = daily_arithmetic_flat_metric(s, strike)

    init_inv = params["initial_inv_clips"] if params.get("initial_inv_clips") is not None else 0
    term_inv = params["terminal_inv_clips"] if params.get("terminal_inv_clips") is not None else n_states
    s.set_volume_states(n_states, initial_state=init_inv)
    apply_ratchets_from_params(s, params)
    assert_cycle_feasible(s, abs(term_inv - init_inv), cpd, "Put swing")
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    s.t_p_curve[term_inv] = 0.0
    if params["run_intrinsic"]:
        s.build()
        profiled_eur = s.v[0, 0, init_inv]
        acq = -np.sum(s.exp_ex)
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
    stochastic_metric = s.profiled()
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
    s = Storage(params["valDate"], params["storageStart"], params["storageEnd"], curve=curve, n_p=0, v_step=v_step, sVol=params["vol"], sMR=params.get("sMR", 1.0), clips_per_day=cpd, daily_curve=params.get("daily_curve"), discount_rate=params.get("discount_rate", 0.0))
    init_inv     = params["initial_inv_clips"]  if params.get("initial_inv_clips")  is not None else n_states
    term_inv     = params["terminal_inv_clips"] if params.get("terminal_inv_clips") is not None else 0
    strike       = params.get("strike", 0.0)

    # The zero-optionality benchmark must be net of the strike, because the value it
    # is compared against is. Selling every day at the average forward earns
    # mean(F) - K per MWh, not mean(F). Benchmarking a strike-net value against a raw
    # forward average made `intrinsic` (and `total`) shift by about -K: on one deal,
    # K = 0, 10 and 28 gave 0.314, -9.686 and -27.686 EUR/MWh for what is the same
    # spread. A constant per-MWh strike cannot reorder the days, so the intrinsic
    # spread a mandatory swing captures does not depend on it.
    flat_metric = daily_arithmetic_flat_metric(s, strike)
    zero_penalty = params.get("zero_penalty", False)

    if strike:
        s.w_cost[s.Dt:s._active] = strike

    s.set_volume_states(n_states, initial_state=init_inv)
    apply_ratchets_from_params(s, params)
    if not zero_penalty:
        assert_cycle_feasible(s, abs(term_inv - init_inv), cpd, "Call swing")
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    if zero_penalty:
        s.t_p_curve[:s.n_op + 1] = 0.0   # any residual inventory is fine
    else:
        s.t_p_curve[term_inv] = 0.0

    if params["run_intrinsic"]:
        s.build()
        profiled_eur = s.v[0, 0, init_inv]
        acq = np.sum(s.exp_ex)
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
    stochastic_metric = s.profiled()
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
    # Optional asymmetric daily clip rates (clips/day): set params["inj_rate"] /
    # params["wdr_rate"] for "30 in, 45 out" style storage. Both default to the
    # symmetric resolve_grid rate `cpd`, preserving prior behaviour.
    # Capacity expressed in days is NOT read here — only inj_rate/wdr_rate are.
    # Accepting it silently gave the caller the default symmetric rate instead of
    # the deal they described: wdr_days of 30, 45, 90 and 365 all priced alike
    # while the rate itself moves the value by ~2.6 % across 1..10 clips/day.
    # params_for_run_valuation converts days -> rate; callers building params by
    # hand must do the same rather than have the input dropped.
    # NB `inj_days` is read — resolve_grid uses it as the inventory-state count on
    # the legacy path — but `wdr_days` is read by nothing here, so accepting it
    # silently gave the caller the default symmetric rate instead of the deal they
    # described: 30, 45, 90 and 365 all priced alike, while the rate itself moves
    # the value ~2.6 % across 1..10 clips/day.
    if params.get("wdr_days") is not None and params.get("wdr_rate") is None:
        raise ValueError(
            f"value_storage reads `wdr_rate` (clips per active day), not `wdr_days`, so "
            f"`wdr_days={params['wdr_days']}` would be ignored and the deal priced at the "
            f"symmetric default of {cpd} clip(s)/day. Pass `wdr_rate`, or build the params "
            f"with params_for_run_valuation(), which derives it as "
            f"max(1, round(n_states / wdr_days)). Two cautions: the rate is a whole number "
            f"of clips, so anything slower than 1 clip/day needs a smaller v_step; and on "
            f"this path `inj_days` means the inventory-state count, not days to fill.")

    inj_rate = int(params["inj_rate"]) if params.get("inj_rate") is not None else cpd
    wdr_rate = int(params["wdr_rate"]) if params.get("wdr_rate") is not None else cpd
    clips_per_day = max(inj_rate, wdr_rate)
    s = Storage(params["valDate"], params["storageStart"], params["storageEnd"], curve=curve, daily_curve=params.get("daily_curve"), n_p=0, v_step=v_step, sVol=params["vol"], sMR=params.get("sMR", 1.0), clips_per_day=clips_per_day, discount_rate=params.get("discount_rate", 0.0))
    n, active = active_masks(s)
    s.i_curve = inj_rate * active
    s.w_curve = wdr_rate * active
    s.i_cost[:] = params["inj_cost"]
    s.w_cost[:] = params["wdr_cost"]
    init_inv = params["initial_inv_clips"] if params.get("initial_inv_clips") is not None else 0
    term_inv = params["terminal_inv_clips"] if params.get("terminal_inv_clips") is not None else 0
    # The forward pass must start where the reported value is read (init_inv), or
    # exp_ex/delta describe a different deal from the one priced.
    s.set_volume_states(n_states, initial_state=init_inv)
    apply_ratchets_from_params(s, params)
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
    """Value a product from a parameter dict; returns (Storage, result_dict).

    `curve` is a contract DataFrame [contractStart, contractEnd, value], or None
    when `params["daily_curve"]` supplies a precomputed daily curve.

    Required `params` keys (all products):
        product_type   "put_swing" | "call_swing" | "storage"
        valDate, storageStart, storageEnd   date-like
        vol            annualised spot volatility (-> sVol)
        n_p_full       price-tree half-width for the full (stochastic) run
        run_intrinsic  also run the n_p=0 intrinsic pass (bool)
        grid sizing — EITHER daily_max (MWh/day) + capacity_mwh, OR v_step +
                      the legacy clip-count key ("days" for swings, "inj_days"
                      for storage). See resolve_grid().
    Optional keys:
        clips_per_day (default 3), sMR (default 1.0), daily_curve (default None),
        strike (default 0.0), initial_inv_clips / terminal_inv_clips,
        ratchets, zero_penalty (call swing). storage additionally needs
        inj_cost / wdr_cost.

    A missing required key raises KeyError. When loading from products.xlsx via
    load_product_params(), pass the dict through params_for_run_valuation() first
    (it derives the grid keys), otherwise this raises KeyError: 'v_step'.
    """
    if params["product_type"] == "put_swing":
        return value_put_swing(curve, params)
    if params["product_type"] == "call_swing":
        return value_call_swing(curve, params)
    if params["product_type"] == "storage":
        return value_storage(curve, params)
    raise ValueError("Unknown product_type")


# ── Price tree ────────────────────────────────────────────────────────────────

def build_tree(price_curve, n_t, n_p, vol_curve, mr_curve):
    if n_t <= 0:
        raise ValueError("Price tree requires at least one time step.")
    if n_p < 0:
        raise ValueError("Price-tree half-width n_p must be non-negative.")

    vol_arr = np.asarray(vol_curve, dtype=np.float64)
    mr_arr  = np.asarray(mr_curve,  dtype=np.float64)
    fwd     = np.asarray(price_curve, dtype=np.float64)[:n_t]
    if len(fwd) != n_t or len(vol_arr) < n_t or len(mr_arr) < n_t:
        raise ValueError("Forward, volatility and mean-reversion curves must cover every time step.")
    if not np.isfinite(fwd).all() or np.any(fwd <= 0.0):
        raise ValueError("Forward prices must be finite and strictly positive.")
    if not np.isfinite(vol_arr[:n_t]).all() or np.any(vol_arr[:n_t] < 0.0):
        raise ValueError("Volatility values must be finite and non-negative.")
    if not np.isfinite(mr_arr[:n_t]).all():
        raise ValueError("Mean-reversion values must be finite.")

    dt = 1. / 365.25
    max_vol = float(np.max(vol_arr[:n_t]))
    if n_p > 0 and max_vol == 0.0:
        raise ValueError("A stochastic tree (n_p > 0) requires positive volatility.")
    dx = max_vol * sqrt(3 * dt)

    x   = np.zeros((n_t, 2*n_p+1))
    p_u = np.zeros((n_t, 2*n_p+1))
    p_d = np.zeros((n_t, 2*n_p+1))
    p_m = np.zeros((n_t, 2*n_p+1))

    q = _tree_core(x, p_u, p_m, p_d, fwd, vol_arr, mr_arr, n_t, n_p, dx, dt)

    tolerance = 1e-12
    for i in range(n_t):
        j_s = max(n_p - i, 0)
        j_e = min(n_p + i, 2*n_p) + 1
        transitions = np.column_stack((p_u[i, j_s:j_e], p_m[i, j_s:j_e], p_d[i, j_s:j_e]))
        if (not np.isfinite(transitions).all()
                or np.min(transitions) < -tolerance
                or np.max(transitions) > 1.0 + tolerance
                or not np.allclose(transitions.sum(axis=1), 1.0, rtol=0.0, atol=tolerance)):
            raise ValueError(
                f"Invalid transition probabilities at time step {i}; "
                "check the volatility, mean-reversion and tree-width inputs."
            )

    if (not np.isfinite(q).all()
            or np.min(q) < -tolerance
            or not np.allclose(q.sum(axis=1), 1.0, rtol=0.0, atol=tolerance)):
        raise ValueError("Invalid propagated price-state probabilities.")

    return fwd, x, q, p_u, p_m, p_d


# ── Valuation helpers ─────────────────────────────────────────────────────────

def compute_all_metrics(n_t, n_p, n_op, prob, strat, i_ratch, w_ratch, v_step,
                        w_curve, i_curve, d_curve, x, fwd):
    # strat holds the signed clip count moved per state (neg=withdraw, pos=inject).
    action = strat[:n_t] * v_step               # MWh moved per (time, price, vol)

    pa     = prob * action
    exp_ex = list(-pa.sum(axis=(1, 2))) + [0.0]

    # `delta` is an UNDISCOUNTED physical hedge volume: the forward MWh to trade
    # (decision D-O2). Hedging day i with h forwards gives PV = h*DF_i*F_i*eps
    # against dV/deps = DF_i*E[S_i*Q_i], so the discount factor cancels and
    # h = E[S_i*Q_i]/F_i. Discounting it here would make the reported number move
    # with the yield curve while the gas did not, and under-hedge by DF (4.6 % at
    # 3 %, 7.7 % at 5 %). The discount weights belong in the repricing identity
    # instead: sum_i d_curve[i] * delta[i] * fwd[i] == v[0, n_p, initial_state].
    exp_x    = np.exp(x)[:, :, None]
    delta    = list(-(pa * exp_x).sum(axis=(1, 2)) / fwd[:n_t]) + [0.0]

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
    # storage product, which also gets asymmetric inject/withdraw rates derived
    # from inj_days/wdr_days. For put/call swing the forced-cycle start and end
    # states are fixed by the product (a call swing starts full, a put swing ends
    # full), so leave them unset and let value_call_swing / value_put_swing apply
    # their own defaults — forcing initial_inv_clips=0 on a call swing would start
    # it empty, leaving nothing to sell and yielding 0/0 = NaN.
    if p.get("product_type") == "storage":
        p["inj_rate"] = max(1, int(round(n_states / int(p["inj_days"]))))
        p["wdr_rate"] = max(1, int(round(n_states / int(p["wdr_days"]))))
        if p.get("initial_inv_clips") is None:
            p["initial_inv_clips"] = int(round(float(p.get("initial_storage_mwh", 0.0)) / v_step))
        if p.get("terminal_inv_clips") is None:
            p["terminal_inv_clips"] = int(round(float(p.get("terminal_storage_mwh", 0.0)) / v_step))
    return p
