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
    mintunnel/max_tunnel  per-day inventory floor/ceiling — HARD constraints

Kernel constants (storage_kernels.py): a forbidden terminal inventory carries
-1e9; an inventory state outside a dated bound is FORBIDDEN (-1e30) and cannot
be entered at any price, with `INFEASIBLE_VALUE` marking that propagated back to
the reported value; an exercise whose gain over idling is < 1e-6 is snapped to
idle (treated as no-trade). The tunnel was a 1000*v_step per-clip penalty until
2026-09-10 — see `apply_inventory_bounds`.
"""
import numpy as np
from math import sqrt, isfinite, lcm
import re
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
import pandas as pd

# Numba kernels live in storage_kernels.py so that edits to this file do not
# invalidate their disk cache (which would trigger a 20-40s recompile).
# Re-exported here for backward compatibility.
import storage_kernels as sk
from storage_kernels import _tree_core, run_model, probabilities

# Floating-point slack when a requested inventory fraction should land exactly on
# a grid state (0.7 * 10 == 7.000000000000001). Scaled by the state count in
# `apply_inventory_bounds`. Deliberately far smaller than a clip.
_GRID_TOLERANCE = 1e-9


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
                 daily_curve=None, discount_rate=0.0, fuel_loss=0.0):
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
        # Fuel retained on injection, as a fraction of the gas bought. Putting one
        # clip into inventory therefore takes 1/(1 - fuel_loss) clips out of the
        # market. Real storage retains 1-2 %; the default of 0 keeps every existing
        # valuation unchanged.
        self.fuel_loss = float(fuel_loss)
        if not 0.0 <= self.fuel_loss < 1.0:
            raise ValueError(
                f"fuel_loss is a fraction of injected gas retained and must be in [0, 1); "
                f"got {self.fuel_loss}. For 1.5 % pass 0.015.")
        self.inj_fuel_mult = 1.0 / (1.0 - self.fuel_loss)
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
            self.t_p_curve, self.i_ratch, self.w_ratch, self.mintunnel, self.max_tunnel,
            self.inj_fuel_mult)

        # Before the forward pass: with hard dated bounds an infeasible contract
        # leaves `strat` meaningless, so `prob` would flow into forbidden states
        # and the terminal check below would blame the terminal condition for a
        # floor that was never reachable.
        assert_contract_feasible(self)

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
            self.w_curve, self.i_curve, self.d_curve, self.x, self.fwd,
            self.inj_fuel_mult)
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
    rate = float(rate)
    if not np.isfinite(rate):
        raise ValueError("Discount and funding rates must be finite numbers.")
    return np.exp(-rate * np.arange(n_t) / 365.25)


def normalise_rate_parameters(params):
    """Validate and return the selected valuation-rate configuration.

    The model supports either one market/valuation ``discount_rate`` or a pair
    of treasury scenario rates with an explicit ``funding_direction``. Presence
    is tested explicitly, so ``discount_rate=0.0`` remains a supplied rate and
    cannot silently coexist with the treasury scenario fields.

    An empty configuration is valid and leaves the model's default zero rate in
    place. Negative finite rates are also valid.
    """
    supplied = {
        key: params.get(key)
        for key in ("discount_rate", "borrow_rate", "invest_rate", "funding_direction")
        if key in params and params.get(key) is not None
    }
    if not supplied:
        return {}

    has_discount = "discount_rate" in supplied
    has_treasury = any(
        key in supplied for key in ("borrow_rate", "invest_rate", "funding_direction")
    )
    if has_discount and has_treasury:
        raise ValueError(
            "Set either `discount_rate` or the `borrow_rate`/`invest_rate` pair, not "
            "both; otherwise which one prices the deal is silent."
        )

    if has_discount:
        rate = float(supplied["discount_rate"])
        if not np.isfinite(rate):
            raise ValueError("`discount_rate` must be a finite number.")
        return {"discount_rate": rate}

    if "borrow_rate" not in supplied or "invest_rate" not in supplied:
        raise ValueError(
            "`borrow_rate` and `invest_rate` must be given together; one alone leaves "
            "the other direction undefined."
        )
    direction = supplied.get("funding_direction")
    if direction not in ("borrow", "invest"):
        raise ValueError(
            "A `borrow_rate`/`invest_rate` pair requires an explicit "
            "`funding_direction` of 'borrow' or 'invest'. Direction cannot be "
            "inferred safely from product type or mean forward net of strike."
        )
    borrow = float(supplied["borrow_rate"])
    invest = float(supplied["invest_rate"])
    if not np.isfinite(borrow) or not np.isfinite(invest):
        raise ValueError("`borrow_rate` and `invest_rate` must be finite numbers.")
    return {
        "borrow_rate": borrow,
        "invest_rate": invest,
        "funding_direction": direction,
    }


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


def apply_funding_rate(model, params, side, strike=0.0):
    r"""Apply an explicitly selected borrow/invest *scenario* rate.

    This is not a market FVA model.  It replaces the valuation discount curve with
    one treasury scenario rate for the whole deal.  Give `borrow_rate` and
    `invest_rate` together and select one explicitly with
    `funding_direction="borrow"` or `"invest"`.

    Direction must not be inferred from the mean forward net of strike.  Optional
    exercise can select receipts even when the window mean is a payment (and vice
    versa), while stochastic prices can cross the strike.  The old inference was
    therefore wrong for otherwise ordinary swings.  `side="both"` identifies
    storage, whose injections and withdrawals have no single funding direction;
    it remains unsupported here.

    A genuine asymmetric-funding valuation would track the cash balance (or solve
    an equivalent nonlinear recursion) and is outside this helper.  Returns the
    scenario rate put in force.
    """
    has_discount = "discount_rate" in params and params.get("discount_rate") is not None
    has_borrow = "borrow_rate" in params and params.get("borrow_rate") is not None
    has_invest = "invest_rate" in params and params.get("invest_rate") is not None
    if side == "both" and not has_discount and has_borrow and has_invest:
        raise ValueError(
            "A storage deal pays on injection and receives on withdrawal, so it has no "
            "single funding direction. Use a market `discount_rate`/`d_curve`, or "
            "model the cash balance explicitly."
        )

    rates = normalise_rate_parameters(params)
    if not rates:
        return model.discount_rate
    if "discount_rate" in rates:
        rate = rates["discount_rate"]
        model.discount_rate = rate
        model.d_curve = discount_factors(model.n_t, rate)
        return rate
    if side == "both":
        raise ValueError(
            "A storage deal pays on injection and receives on withdrawal, so it has no "
            "single funding direction. Use a market `discount_rate`/`d_curve`, or "
            "model the cash balance explicitly.")

    direction = rates["funding_direction"]
    rate = rates["borrow_rate"] if direction == "borrow" else rates["invest_rate"]
    model.discount_rate = rate
    model.d_curve = discount_factors(model.n_t, rate)
    return rate


def intrinsic_attribution(model, strike=0.0):
    r"""Return a transparent three-way attribution of deterministic intrinsic.

    Price selection and settlement timing interact.  There is therefore no unique
    two-way split into "shape" and "financing" once both the forward curve and
    discount factors vary.  This function reports the chosen reference-ordering
    explicitly:

        day_selection       = sign * DF_bench * (P - Fbar)
        settlement_timing   = sign * Fbar * (DF_paid - DF_bench)
        interaction         = sign * (P - Fbar) * (DF_paid - DF_bench)

    The three terms sum exactly to intrinsic.  Calling the second term
    "financing" does not establish that the business can borrow or invest at the
    discount rate; that requires a separate treasury assumption.
    """
    n = model.n_t
    win = slice(model.Dt, model._active)
    net = np.asarray(model.price_curve, dtype=float)[:n] - float(strike)
    ex = np.asarray(model.exp_ex[:n], dtype=float)
    volume = float(ex.sum())
    if abs(volume) <= 1e-12 * max(float(np.abs(ex).sum()), 1.0):
        raise ValueError(
            "The intrinsic attribution is undefined for a zero-net-volume strategy."
        )
    net_bar = float(np.mean(net[win]))
    paid = float(np.dot(ex, net) / volume)
    if abs(net_bar) < 1e-12 or abs(paid) < 1e-12:
        raise ValueError(
            "The intrinsic attribution is undefined when the strike sits on the curve: "
            f"mean forward net of strike is {net_bar:.3e} and the schedule pays "
            f"{paid:.3e}, so the discount factors it divides out are not recoverable."
        )
    df_bench = float(np.mean(model.d_curve[win] * net[win])) / net_bar
    df_paid = model.profiled() / paid
    sign = 1.0 if volume > 0 else -1.0
    selection = sign * df_bench * (paid - net_bar) + 0.0
    timing = sign * net_bar * (df_paid - df_bench) + 0.0
    interaction = sign * (paid - net_bar) * (df_paid - df_bench) + 0.0
    return {
        "day_selection": selection,
        "settlement_timing": timing,
        "interaction": interaction,
    }


def intrinsic_components(model, strike=0.0):
    r"""Return the legacy two-way attribution of deterministic intrinsic.

    `model` is the **deterministic** run -- the `n_p = 0` build whose schedule
    produces `profiled_metric`. Returns `(day_selection, financing)`, which sum
    to `intrinsic` exactly.

    Intrinsic is the gain from choosing days rather than spreading evenly over
    the window.  With a varying curve and a non-zero rate, price selection and
    settlement timing have an interaction, so a two-way split is not unique. A flat
    forward curve is not flat once discounted: the curve the optimiser actually
    sees is `DF * F`, and at 10 % a flat 40 slopes from 36.84 on 1 Jan to 33.34
    on 31 Dec. There is then something to choose with no price shape at all.
    Writing `flat = Fbar * df_bench` and `profiled = P * df_paid`, where `P` is
    the undiscounted price the schedule pays:

        intrinsic = df_bench * (P - Fbar)  +  P * (df_paid - df_bench)
                    \___ day selection __/    \_____ financing _____/

    This legacy function allocates the entire interaction to the second term. Use
    `intrinsic_attribution()` to see day selection, settlement timing and their
    interaction separately. **On a flat curve the interaction is zero and the
    first term is exactly zero.** The remaining timing attribution becomes an
    actual financing benefit only if the relevant balance can genuinely earn or
    avoid the selected rate.

    Raises for a two-sided strategy, where `profiled_metric` is undefined.
    """
    attribution = intrinsic_attribution(model, strike=strike)
    # Preserve the published two-number API: the old "financing" term is the
    # reference-price timing effect plus the interaction.  The additions of zero
    # in intrinsic_attribution normalise signed zeros for display.
    return (attribution["day_selection"],
            attribution["settlement_timing"] + attribution["interaction"] + 0.0)


def apply_inventory_bounds(model, params):
    """Set dated inventory floors and ceilings from `min_inventory`/`max_inventory`.

    Both are mappings of date -> fraction of working volume, so a term sheet
    reading "1 October inventory at least 70 %" becomes
    ``{"2027-10-01": 0.70}``. A fraction is used rather than MWh because it is
    what the contract says and it survives a change of clip size.

    **The bound applies to the balance the day OPENS with**, before that day's
    injection or withdrawal. That is the model's own convention -- the constraint
    attaches to the state at time `i` -- and it is the reading most storage
    contracts intend, but it is not the only one: a report of the closing
    balance will show the day's move already applied and can look a clip short.
    Roadmap P1.1 covers making the choice explicit rather than implied.

    **Rounding is conservative, not nearest.** A floor rounds UP to the next grid
    state and a ceiling rounds DOWN, so the enforced contract is never weaker
    than the one asked for. `round()` gave the opposite: on a ten-clip grid a
    71 % floor became 70 % and a 29 % ceiling became 30 %, each relaxed past what
    the term sheet says, and the post-check then validated the relaxed bound and
    saw nothing. Rounding a bound is a change of contract, so it is reported.

    Call after `set_volume_states`, which resets the tunnel arrays. The bound is
    a hard constraint in the DP; `assert_inventory_bounds` independently re-reads
    the built policy's state distribution rather than trusting it.

    Returns a list of `(index, kind, stamp, fraction, clips)`, where `clips` is
    the effective grid bound actually enforced.
    """
    n_states = model.n_op - 1
    # Grid states are exact integers scaled by a fraction, so a requested bound
    # that lands on a state can miss it by an ulp (0.7 * 10 = 7.000000000000001).
    # Absorb that much and no more: the tolerance is for floating point, not for
    # granting a clip.
    tol = _GRID_TOLERANCE * max(1.0, float(n_states))
    bounds = []
    for key, kind in (("min_inventory", "min"), ("max_inventory", "max")):
        spec = params.get(key)
        if not spec:
            continue
        for when, fraction in dict(spec).items():
            stamp = pd.Timestamp(when)
            fraction = float(fraction)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"{key}[{when}] is {fraction}, but it is a fraction of working volume "
                    f"and must lie in [0, 1]. For 70 % pass 0.70.")
            index = (stamp - model.valDate).days
            if not 0 <= index < model.n_t:
                raise ValueError(
                    f"{key}[{when}] falls outside the model's grid, which runs "
                    f"{model.valDate:%Y-%m-%d} to {model.backStop:%Y-%m-%d}. A bound the "
                    f"model cannot see would be silently ignored.")
            raw = fraction * n_states
            clips = (int(np.ceil(raw - tol)) if kind == "min"
                     else int(np.floor(raw + tol)))
            bounds.append((index, kind, stamp, fraction, clips))

    for index, kind, stamp, fraction, clips in bounds:
        if kind == "min":
            model.mintunnel[index] = clips
        else:
            model.max_tunnel[index] = clips
    for index, _, stamp, _, _ in bounds:
        floor, ceiling = int(model.mintunnel[index]), int(model.max_tunnel[index])
        if floor > min(ceiling, n_states):
            raise ValueError(
                f"inventory bounds leave no admissible state on {stamp:%Y-%m-%d}: the grid "
                f"holds 0..{n_states} clips, the floor rounds up to {floor} and the ceiling "
                f"down to {min(ceiling, n_states)}. Conservative rounding cannot satisfy both "
                f"on a {n_states}-clip grid -- refine the grid so the bounds land on states, "
                f"or restate them.")
    return bounds


def describe_inventory_bounds(model, bounds):
    """What was asked for against what the grid can actually express.

    Rounding a bound to a grid state changes the contract, so both numbers are
    reported rather than only the enforced one.
    """
    n_states = model.n_op - 1
    rows = []
    for index, kind, stamp, fraction, clips in bounds:
        effective = clips / n_states if n_states else float("nan")
        rows.append({"date": stamp, "bound": kind, "requested": fraction,
                     "effective": effective, "clips": clips,
                     "MWh": clips * model.v_step,
                     "moved by rounding": effective - fraction})
    return pd.DataFrame(rows)


def assert_inventory_bounds(model, bounds, mass_tolerance=1e-9):
    """Verify dated inventory bounds on the built policy's state distribution.

    Read directly from `model.prob[index, :, l]`, which IS the law of opening
    inventory on day `index`. Two earlier versions of this check were wrong in
    ways that a distribution cannot be:

    * It compared an **expectation** against the bound. An average above a floor
      says nothing about the states beneath it, and while the tunnel was a finite
      penalty the optimiser would buy its way below one -- 19.5 % of paths opened
      a floored day empty at a high enough price level, and this returned
      quietly.
    * It rebuilt the balance as `cumsum(net moves)` and added `initial_state` to
      the first day only, so every later day was short by the opening inventory.
      A full store that correctly held everything was rejected against a 100 %
      floor, reported as "0.00 clips against 10".

    `mass_tolerance` absorbs accumulated floating-point error in the forward
    probability pass. It is not an allowance for a breach: the DP makes an
    out-of-bounds state inadmissible, so any mass found there is a numerical
    fault or a bug, and either way not something to price through.
    """
    if not bounds:
        return
    n_states = model.n_op - 1
    for index, kind, stamp, fraction, clips in bounds:
        dist = model.prob[index].sum(axis=0)
        outside = dist[:clips].sum() if kind == "min" else dist[clips + 1:].sum()
        if outside <= mass_tolerance:
            continue
        held = float((np.arange(model.n_op) * dist).sum())
        word, side = (("floor", "below"), ("ceiling", "above"))[kind == "max"]
        raise ValueError(
            f"the {fraction:.0%} {word} on {stamp:%Y-%m-%d} does not hold: {outside:.6%} of "
            f"probability opens the day {side} the {clips}-clip bound "
            f"({clips * model.v_step:,.0f} MWh of {n_states * model.v_step:,.0f}), against a "
            f"tolerance of {mass_tolerance:.0e}. Expected opening inventory is {held:.2f} "
            f"clips, which is why an average is not a check. The bound is hard in the DP, so "
            f"this is a numerical fault rather than an economic choice.")


def assert_contract_feasible(model, label="contract"):
    """Fail loudly when no admissible policy exists, rather than reporting -1e30.

    Dated inventory bounds and the terminal inventory requirement are both hard,
    and they can be jointly unsatisfiable with the rates, ratchets and date masks.
    The DP marks those states FORBIDDEN and the infeasibility propagates back to
    the reported value; without this the caller would divide -1e30 by a volume and
    print it.

    Both used to be finite penalties, which a large enough deal simply paid.
    """
    value = float(model.v[0, model.n_p, model.initial_state])
    if value > sk.INFEASIBLE_VALUE:
        return value
    raise ValueError(
        f"no admissible policy exists for this {label}: from an opening inventory of "
        f"{model.initial_state} clips, every schedule either breaches a dated inventory "
        f"bound or cannot reach a permitted terminal inventory under the configured rates, "
        f"ratchets and date masks. **A bound refused on a coarse grid may be reachable on a "
        f"finer one**, because the rates are whole clips and `int(rate x multiplier)` "
        f"understates a ratcheted rate -- on the reference store the 70 % October floor is "
        f"refused at 240 clips and met at 480. Refine the clip before concluding that the "
        f"contract is physically impossible (roadmap P1.4).")


# Engineering tolerance for "is this physical quantity an exact whole number of
# clips at this grid" -- an absolute floor plus a relative term, in MWh (or
# MWh/day, which is the same unit family). IMPLEMENTATION-GUIDE-2026-09-11.md
# §4.2. This is floating-point slack, not a commercial allowance: a genuine
# half-clip mismatch is many orders of magnitude larger than this and must
# still be refused.
GRID_TOLERANCE_MWH = 1e-7
GRID_TOLERANCE_RELATIVE = 1e-10


def _grid_representable(requested_mwh, v_step):
    """Is `requested_mwh` a whole number of `v_step`-sized clips, at tolerance?

    Returns `(clips, achieved_mwh, ok)`. `clips` is the NEAREST integer clip
    count regardless of `ok` -- round only after checking representability, per
    the guide; callers must not use `clips` when `ok` is False.
    """
    clips = int(round(requested_mwh / v_step))
    achieved = clips * v_step
    tolerance = GRID_TOLERANCE_MWH + GRID_TOLERANCE_RELATIVE * abs(requested_mwh)
    return clips, achieved, abs(achieved - requested_mwh) <= tolerance


def _compatible_n_states_hint(n_states, inj_days, wdr_days):
    """The smallest multiple of `lcm(inj_days, wdr_days)` at or above
    `n_states`, when both day counts are (nearly) whole numbers. `None` when no
    simple suggestion applies -- e.g. a fractional day count, where the
    "smallest compatible grid" question does not have a clean answer.
    """
    def _as_int(value):
        rounded = round(value)
        return rounded if abs(value - rounded) < 1e-9 else None

    di, dw = _as_int(inj_days), _as_int(wdr_days)
    if di is None or dw is None or di <= 0 or dw <= 0:
        return None
    base = lcm(di, dw)
    multiple = base * max(1, -(-n_states // base))          # ceil(n_states / base)
    return base, multiple


def resolve_grid(params, states_key):
    """Map physical inputs to the model's (clip size, #states, clips/day).

    Two ways to size the volume grid:

    * Physical   — ``daily_max`` (MWh injected/withdrawn per active day) together
      with ``clips_per_day`` fixes the clip size ``v_step = daily_max/clips_per_day``;
      ``capacity_mwh`` then fixes the number of inventory states as
      ``round(capacity_mwh / v_step)`` (total working volume stays put when the
      clip is refined).
    * Explicit   — ``v_step`` with ``n_states``, the inventory-state count.
    * Legacy     — ``v_step`` with ``params[states_key]``, where ``states_key``
      doubles as the state count.

    **``inj_days`` means two different things and this is where they meet.** On
    the legacy storage path it is the number of inventory STATES; in
    ``params_for_run_valuation`` and every notebook it is DAYS TO FILL, from
    which the rate is derived as ``round(n_states / inj_days)``. A 30/60 store
    written with ``inj_days=30`` on the legacy path therefore silently gets a
    30-state grid rather than a 30-day fill. Pass ``n_states`` instead; the
    legacy reading still works alone, but not beside a key that only makes sense
    under the other meaning. Roadmap P1.4.

    **When ``capacity_mwh`` is given, it must be an exact whole number of
    clips, and an explicit ``n_states`` must agree with it.** Both used to be
    silently overridden: a ``v_step`` that does not divide ``capacity_mwh``
    evenly rounded away the remainder with no error, and an explicit
    ``n_states`` alongside ``capacity_mwh`` was discarded outright rather than
    checked. IMPLEMENTATION-GUIDE-2026-09-11.md §4.2/§4.3.

    Returns (v_step, n_states, clips_per_day).
    """
    cpd = int(params.get("clips_per_day", 3))

    daily_max = params.get("daily_max")
    v_step = float(daily_max) / cpd if daily_max is not None else float(params["v_step"])

    capacity = params.get("capacity_mwh")
    if capacity is not None:
        capacity = float(capacity)
        n_states, achieved, ok = _grid_representable(capacity, v_step)
        if not ok:
            raise ValueError(
                f"capacity_mwh={capacity:,.4f} is not a whole number of clips at "
                f"v_step={v_step:,.6f} (from clips_per_day={cpd} on daily_max, or "
                f"v_step directly): the nearest grid gives {achieved:,.4f} MWh at "
                f"{n_states} states, {achieved - capacity:+,.4f} MWh away. Choose a "
                f"v_step/clips_per_day that divides the capacity evenly, or state "
                f"n_states directly and let it define v_step instead.")
        explicit_n = params.get("n_states")
        if explicit_n is not None and int(explicit_n) != n_states:
            raise ValueError(
                f"capacity_mwh={capacity:,.0f} and v_step={v_step:,.4f} imply "
                f"{n_states} states, but n_states={int(explicit_n)} was also given and "
                f"disagrees ({int(explicit_n)} states would need "
                f"{int(explicit_n) * v_step:,.0f} MWh of capacity, not {capacity:,.0f}). "
                f"Supply capacity_mwh with daily_max/v_step, OR v_step with n_states -- "
                f"not a mismatched mix of all three.")
        return v_step, n_states, cpd

    if params.get("n_states") is not None:
        return v_step, int(params["n_states"]), cpd

    if params.get(states_key) is None:
        raise KeyError(
            f"resolve_grid needs the inventory grid: either capacity_mwh with daily_max, "
            f"or v_step with n_states. ({states_key!r} is accepted as a legacy alias for "
            f"n_states, but it also means days-to-fill elsewhere, so prefer n_states.)")

    # The legacy reading, and the one contradiction worth refusing here: an
    # explicit clip rate only makes sense when `inj_days` means days, so its
    # presence says the caller does not mean a state count. `wdr_days` alone is
    # left to `value_storage`, which has a more specific message for it.
    conflicting = [k for k in ("inj_rate", "wdr_rate") if params.get(k) is not None]
    if states_key == "inj_days" and conflicting:
        raise ValueError(
            f"`inj_days={params[states_key]}` would be read as the number of inventory "
            f"states, but {', '.join(conflicting)} is also set, which only makes sense if "
            f"`inj_days` means days to fill. The key means both things in this codebase and "
            f"cannot mean both at once here. Pass `n_states` for the grid size, or supply "
            f"`capacity_mwh` with `daily_max` and let the physical inputs size it.")
    return v_step, int(params[states_key]), cpd


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


def assert_ratchets_expressible(model):
    """Fail when a ratchet multiplier truncates a daily rate to zero clips.

    The DP moves whole clips, so the kernel takes ``int(rate * multiplier)``.
    A multiplier that is positive but small enough to floor to zero therefore
    means "cannot move at all" where the caller meant "move slowly" -- and it
    does so silently: the store freezes and the deal prices at zero with no
    error. A perfectly ordinary profile does it. Withdrawal at 1 clip/day with a
    0.30 multiplier near empty gives 0.30 clips/day, floors to 0, and the store
    can never take out its first clip.

    A multiplier of exactly zero is left alone: that is the legitimate way to say
    the rate is shut off at that fullness.

    The fix when this fires is a finer clip: the smallest non-zero multiplier `m`
    needs a base rate of at least `ceil(1/m)` clips per day, which means dividing
    `v_step` by the same factor and multiplying `n_states` by it.
    """
    i_ratch = np.asarray(model.i_ratch, dtype=float)
    w_ratch = np.asarray(model.w_ratch, dtype=float)
    i_curve = np.asarray(model.i_curve, dtype=float)
    w_curve = np.asarray(model.w_curve, dtype=float)

    for label, curve, ratch in (("injection", i_curve, i_ratch),
                                ("withdrawal", w_curve, w_ratch)):
        rate = float(curve.max()) if curve.size else 0.0
        if rate <= 0.0:
            continue
        effective = rate * ratch
        lost = np.flatnonzero((effective > 0.0) & (effective < 1.0))
        if lost.size == 0:
            continue
        worst = int(lost[np.argmin(effective[lost])])
        multiplier = float(ratch[worst])
        fullness = worst / max(model.n_op - 1, 1)
        needed = int(np.ceil(1.0 / multiplier))
        raise ValueError(
            f"the {label} ratchet truncates to zero at {fullness:.0%} full: a multiplier of "
            f"{multiplier:.3f} on {rate:g} clip(s)/day gives {effective[worst]:.3f} clips, "
            f"which the DP floors to 0 -- the store would be unable to move there at all and "
            f"the deal would price at zero with no error. {lost.size} of {model.n_op} "
            f"inventory states are affected. Use a base rate of at least {needed} clip(s)/day "
            f"(divide v_step by {needed}, multiply n_states by {needed}), or set the "
            f"multiplier to exactly 0 if the rate really is shut off there."
        )


def describe_ratchet_rates(model):
    """The contract's MWh/day against what the inventory grid can express.

    The DP moves whole clips, so the kernel takes ``int(rate * multiplier)`` and
    the remainder is thrown away. `assert_ratchets_expressible` catches only the
    case where that reaches zero; everything short of zero passed silently, and
    a silently slower store fills less, so the deal was under-valued with no
    indication. On the reference 240-clip store a contractual 3,800 MWh/day at
    10 % full is delivered as 2,500 -- a **34 % shortfall** that the guard,
    the tests and the notebook all cleared.

    Returns one row per inventory state: fullness, the rate the contract grants
    at that fullness, the rate the grid actually gives, and the loss in both
    MWh/day and per cent. Levels where the move is physically impossible anyway
    (withdrawing from empty, injecting into a full store) are reported with a
    null loss rather than a spurious 100 %.

    Roadmap P1.4. Read `worst_ratchet_rate_loss()` for the single number.
    """
    n_states = model.n_op - 1
    levels = np.arange(model.n_op)
    fullness = levels / max(n_states, 1)
    out = {"clips": levels, "fullness": fullness}

    for side, curve, ratch in (("injection", model.i_curve, model.i_ratch),
                               ("withdrawal", model.w_curve, model.w_ratch)):
        curve = np.asarray(curve, dtype=float)
        base = float(curve.max()) if curve.size else 0.0        # the active-day rate
        mult = np.asarray(ratch, dtype=float)
        contract = base * mult * model.v_step
        grid = np.floor(base * mult) * model.v_step
        # Headroom, not discretisation: a full store cannot inject and an empty
        # one cannot withdraw however fine the clip is.
        possible = (levels < n_states) if side == "injection" else (levels > 0)
        loss = np.where(possible & (contract > 0.0), contract - grid, np.nan)
        relative = np.where(possible & (contract > 0.0), 1.0 - grid / np.where(
            contract > 0.0, contract, 1.0), np.nan)
        out[f"{side} contract MWh/day"] = contract
        out[f"{side} grid MWh/day"] = grid
        out[f"{side} loss MWh/day"] = loss
        out[f"{side} loss"] = relative

    return pd.DataFrame(out)


def worst_ratchet_rate_loss(model):
    """Largest relative rate loss the grid imposes, per side. 0.0 when exact."""
    table = describe_ratchet_rates(model)
    return {side: float(np.nanmax(np.concatenate([
                table[f"{side} loss"].to_numpy(dtype=float), [0.0]])))
            for side in ("injection", "withdrawal")}


#: How far a ratcheted rate may fall short of the contract's before
#: `value_storage` refuses the grid. Chosen 2026-09-10 to sit between the
#: reference store's 33.1 % at 240 clips (refused) and its 5.8 % at 1,920
#: (accepted), so an obviously wrong grid fails and a defensible one does not.
#: It is an engineering threshold, not a commercial tolerance. Override per deal
#: with `max_ratchet_rate_loss`; pass 1.0 to study a coarse grid deliberately.
DEFAULT_MAX_RATCHET_RATE_LOSS = 0.10


def assert_ratchet_rates_expressible(model, max_relative_loss=0.0, tolerance=1e-9):
    """Gate the discretisation loss above.

    `value_storage` applies this at `DEFAULT_MAX_RATCHET_RATE_LOSS` whenever
    ratchets are set. It is a *rate* gate and a leading indicator, not a
    statement about value: on the reference store 11 % of rate loss costs about
    0.5 % of value. The value question is convergence, which `benchmarks.py`
    answers separately, and the two do not agree on a grid -- the rate gate
    clears 1,920 clips and the 0.5 % value gate wants 3,840. Say which one a
    number was accepted under.
    """
    worst = worst_ratchet_rate_loss(model)
    for side, loss in worst.items():
        if loss > max_relative_loss + tolerance:
            table = describe_ratchet_rates(model)
            at = int(table[f"{side} loss"].idxmax())
            raise ValueError(
                f"the {side} rate is understated by {loss:.1%} at {table['fullness'][at]:.0%} "
                f"full: the contract grants {table[f'{side} contract MWh/day'][at]:,.0f} "
                f"MWh/day and a {model.n_op - 1}-clip grid delivers "
                f"{table[f'{side} grid MWh/day'][at]:,.0f}, against a limit of "
                f"{max_relative_loss:.1%}. The rates are whole clips, so refine the clip: "
                f"halving v_step halves this loss. See describe_ratchet_rates().")
    return worst


def apply_ratchets_from_params(model, params):
    """Apply ratchets to a model if params['ratchets'] is set (path, DataFrame,
    or a pre-loaded (fullness, inj, wdr) tuple). Call after set_volume_states."""
    ratch = params.get("ratchets")
    if ratch is None:
        return
    f, inj, wdr = ratch if isinstance(ratch, tuple) else load_ratchets(ratch)
    model.apply_ratchets(f, inj, wdr)
    # The kernel floors rate * multiplier to whole clips, so check the profile is
    # expressible on this grid rather than letting it silently freeze the store.
    assert_ratchets_expressible(model)


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
    apply_funding_rate(s, params, "payer", params.get("strike", 0.0))
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
    apply_funding_rate(s, params, "receiver", params.get("strike", 0.0))
    init_inv     = params["initial_inv_clips"]  if params.get("initial_inv_clips")  is not None else n_states
    term_inv     = params["terminal_inv_clips"] if params.get("terminal_inv_clips") is not None else 0
    strike       = params.get("strike", 0.0)

    # The zero-optionality benchmark must be net of the strike, because the value it
    # is compared against is. Selling every day at the average forward earns
    # mean(F) - K per MWh, not mean(F). Benchmarking a strike-net value against a raw
    # forward average made `intrinsic` (and `total`) shift by about -K: on one deal,
    # K = 0, 10 and 28 gave 0.314, -9.686 and -27.686 EUR/MWh for what is the same
    # spread. At a zero discount rate a constant per-MWh strike cannot reorder the
    # days. At a non-zero rate its settlement timing can, so intrinsic and
    # extrinsic may then depend on the strike.
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
    s = Storage(params["valDate"], params["storageStart"], params["storageEnd"], curve=curve, daily_curve=params.get("daily_curve"), n_p=0, v_step=v_step, sVol=params["vol"], sMR=params.get("sMR", 1.0), clips_per_day=clips_per_day, discount_rate=params.get("discount_rate", 0.0), fuel_loss=params.get("fuel_loss", 0.0))
    apply_funding_rate(s, params, "both", 0.0)
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
    if params.get("ratchets") is not None:
        # The zero-rate guard only catches int(rate * multiplier) == 0. Anything
        # short of zero was rounded down in silence, and a silently slower store
        # fills less, so the deal came out under-valued with nothing to see.
        assert_ratchet_rates_expressible(
            s, params.get("max_ratchet_rate_loss", DEFAULT_MAX_RATCHET_RATE_LOSS))
    # After set_volume_states, which resets the tunnel arrays.
    inventory_bounds = apply_inventory_bounds(s, params)
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    s.t_p_curve[term_inv] = 0.0

    max_vol = n_states * v_step
    if params["run_intrinsic"]:
        s.build()
        # Both passes price the same contract, so both are checked against it.
        # Only the stochastic one was, which left the intrinsic schedule free to
        # breach a bound the reported extrinsic split was then measured against.
        # (`build` has already established that an admissible policy exists.)
        assert_inventory_bounds(s, inventory_bounds)
        intrinsic_eur = s.v[0, 0, init_inv]
        intrinsic_profile_raw = np.array(s.exp_ex)
    else:
        intrinsic_eur = np.nan
        intrinsic_profile_raw = np.zeros(len(s.date_span))

    s.n_p = params["n_p_full"]
    s.build()
    # The bounds are hard in the DP; this re-reads the state distribution rather
    # than trusting the kernel to have applied them.
    assert_inventory_bounds(s, inventory_bounds)
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
                        w_curve, i_curve, d_curve, x, fwd, inj_fuel_mult=1.0):
    # strat holds the signed clip count moved per state (neg=withdraw, pos=inject).
    action = strat[:n_t] * v_step               # MWh moved per (time, price, vol)

    pa     = prob * action
    exp_ex = list(-pa.sum(axis=(1, 2))) + [0.0]

    # With fuel loss the gas bought exceeds the gas stored, so the market volume
    # and the inventory volume part company. `exp_ex` stays physical -- it is what
    # the store holds -- while `delta` is scaled on the injection leg, because the
    # hedge is what you trade, not what you keep. Decision D-O3, 2026-09-10: with
    # `fuel_loss = 0` the two coincide and nothing changes.
    traded = np.where(action > 0.0, action * inj_fuel_mult, action)
    pa_traded = prob * traded

    # `delta` is an UNDISCOUNTED physical hedge volume: the forward MWh to trade
    # (decision D-O2). Hedging day i with h forwards gives PV = h*DF_i*F_i*eps
    # against dV/deps = DF_i*E[S_i*Q_i], so the discount factor cancels and
    # h = E[S_i*Q_i]/F_i. Discounting it here would make the reported number move
    # with the yield curve while the gas did not, and under-hedge by DF (4.6 % at
    # 3 %, 7.7 % at 5 %). The discount weights belong in the repricing identity
    # instead: sum_i d_curve[i] * delta[i] * fwd[i] == v[0, n_p, initial_state].
    exp_x    = np.exp(x)[:, :, None]
    delta    = list(-(pa_traded * exp_x).sum(axis=(1, 2)) / fwd[:n_t]) + [0.0]

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
        Optional rate columns are discount_rate, or borrow_rate, invest_rate and
        funding_direction together.
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

    # Optional rate fields use the same validation as the app, notebook and core
    # valuation entry point. Blank workbook cells are absent, while an explicit
    # zero remains a supplied discount rate.
    rate_inputs = {}
    for name in ("discount_rate", "borrow_rate", "invest_rate"):
        value = row.get(name)
        if pd.notna(value):
            rate_inputs[name] = float(value)
    direction = row.get("funding_direction")
    if pd.notna(direction) and str(direction).strip():
        rate_inputs["funding_direction"] = str(direction).strip()
    params.update(normalise_rate_parameters(rate_inputs))

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


def normalise_storage_contract(params):
    """Convert a storage term sheet (capacity, days, opening/terminal MWh) into
    exact grid quantities, or refuse the request.

    IMPLEMENTATION-GUIDE-2026-09-11.md §4.2. Until 2026-09-11 this step used
    ``max(1, round(n_states / days))`` with no check: 30/90 at ``n_states=30``
    silently rounded withdrawal to the FULL injection rate -- 20,000 MWh/day
    against a requested 6,666.67 -- and priced EUR 131,369 (5.51 %) above the
    contract asked for, with no error anywhere. A 5,000 MWh opening inventory
    on a 10,000 MWh clip rounded to zero the same way: "start half full of a
    clip" became "start empty".

    A pure function -- no Numba, no I/O, no plotting -- so a bad contract fails
    before any grid is allocated. Required keys: ``capacity_mwh`` (finite,
    positive), ``n_states`` (positive integer -- the ONLY numerical knob;
    nothing here chooses it for you), ``inj_days``/``wdr_days`` (finite,
    positive days to fill/empty; a zero or infinite day count cannot express a
    rate through this field, so it is refused rather than read as "closed").
    Optional: ``initial_storage_mwh``/``terminal_storage_mwh`` (default 0.0),
    and ``initial_inv_clips``/``terminal_inv_clips`` -- if either clip field is
    ALSO given, it must agree with its MWh field to within the grid tolerance,
    or the request is refused rather than one field silently winning.

    Returns ``{"v_step", "inj_rate", "wdr_rate", "initial_inv_clips",
    "terminal_inv_clips"}``, each an exact reproduction of what was requested.
    Raises ``ValueError`` naming the requested quantity, what the grid actually
    gives, and -- when ``inj_days``/``wdr_days`` are whole numbers -- the
    smallest compatible ``n_states`` to ask for instead.
    """
    capacity = params.get("capacity_mwh")
    if capacity is None or not isfinite(float(capacity)) or float(capacity) <= 0.0:
        raise ValueError(f"capacity_mwh must be a finite positive number, got {capacity!r}.")
    capacity = float(capacity)

    if params.get("n_states") is None:
        raise ValueError(
            "normalise_storage_contract needs n_states, the inventory-grid resolution "
            "-- it is the one numerical input this function does not derive.")
    n_states = int(params["n_states"])
    if n_states <= 0:
        raise ValueError(f"n_states must be a positive integer, got {n_states}.")
    v_step = capacity / n_states

    def _days(key):
        value = params.get(key)
        if value is None:
            raise ValueError(f"normalise_storage_contract needs {key} (days to fill/empty).")
        value = float(value)
        if not isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{key} must be a finite positive number of days, got {value!r}. A "
                f"deliberately closed rate belongs in a direct rate schedule, not a "
                f"zero or infinite day count here.")
        return value

    inj_days, wdr_days = _days("inj_days"), _days("wdr_days")
    hint = _compatible_n_states_hint(n_states, inj_days, wdr_days)
    suggestion = (f" The smallest grid expressing both rates exactly is "
                  f"n_states={hint[1]} (a multiple of lcm({inj_days:g}, {wdr_days:g}) "
                  f"= {hint[0]})." if hint else "")

    rates = {}
    for label, key, days in (("injection", "inj_rate", inj_days),
                             ("withdrawal", "wdr_rate", wdr_days)):
        requested_mwh_day = capacity / days
        clips, achieved, ok = _grid_representable(requested_mwh_day, v_step)
        if not ok:
            raise ValueError(
                f"the {n_states}-state grid (clip {v_step:,.4f} MWh) cannot express "
                f"{capacity:,.0f} MWh over {days:g} {label} days: requested rate "
                f"{requested_mwh_day:,.4f} MWh/day, nearest whole-clip rate "
                f"{achieved:,.4f} MWh/day ({clips} clip(s)/day), a "
                f"{(achieved - requested_mwh_day) / requested_mwh_day:+.2%} difference."
                f"{suggestion}")
        rates[key] = clips

    boundary = {}
    for label, mwh_key, clip_key in (("opening", "initial_storage_mwh", "initial_inv_clips"),
                                     ("terminal", "terminal_storage_mwh", "terminal_inv_clips")):
        requested_mwh = float(params.get(mwh_key, 0.0) or 0.0)
        if not 0.0 <= requested_mwh <= capacity + GRID_TOLERANCE_MWH:
            raise ValueError(
                f"{mwh_key}={requested_mwh:,.4f} is outside the working volume "
                f"0..{capacity:,.0f} MWh.")
        explicit_clips = params.get(clip_key)
        if explicit_clips is not None:
            explicit_clips = int(explicit_clips)
            achieved = explicit_clips * v_step
            if abs(achieved - requested_mwh) > GRID_TOLERANCE_MWH + (
                    GRID_TOLERANCE_RELATIVE * abs(requested_mwh)):
                raise ValueError(
                    f"{mwh_key}={requested_mwh:,.4f} and {clip_key}={explicit_clips} "
                    f"(= {achieved:,.4f} MWh at this grid) disagree. Supply one, or "
                    f"make them agree.")
            boundary[clip_key] = explicit_clips
        else:
            clips, achieved, ok = _grid_representable(requested_mwh, v_step)
            if not ok:
                raise ValueError(
                    f"{mwh_key}={requested_mwh:,.4f} is not a whole number of clips "
                    f"at v_step={v_step:,.4f}: the nearest grid gives "
                    f"{achieved:,.4f} MWh ({clips} clip(s)), "
                    f"{achieved - requested_mwh:+,.4f} MWh away.{suggestion}")
            boundary[clip_key] = clips

    return {"v_step": v_step, "inj_rate": rates["inj_rate"], "wdr_rate": rates["wdr_rate"],
            "initial_inv_clips": boundary["initial_inv_clips"],
            "terminal_inv_clips": boundary["terminal_inv_clips"]}


def params_for_run_valuation(prm):
    """Augment a ``load_product_params()`` dict with the derived grid fields that
    ``run_valuation()`` / ``value_*`` need, so the two documented entry points can
    be chained directly:

        params = params_for_run_valuation(load_product_params("products.xlsx", name))
        s, result = run_valuation(curve, params)

    Without this step ``run_valuation`` raises ``KeyError: 'v_step'`` because
    ``load_product_params`` deliberately leaves the grid derivation out (it stays
    visible in the notebook). The clip size is ``v_step = capacity_mwh / n_states``.

    For storage, asymmetric rates and boundary inventory are resolved by
    ``normalise_storage_contract()``: it preserves the requested MWh/day and MWh
    exactly, or raises rather than silently pricing a different contract. A
    "30 in, 60 out" deal on a 60-state grid comes out as ``inj_rate=2,
    wdr_rate=1``; on a grid that cannot express both exactly, this raises
    instead of guessing. See that function for what "cannot express" means and
    what the error names.

    ``clips_per_day`` here stays a swing-only symmetric rate (``round(n_states /
    inj_days)``, unchanged) -- storage's own asymmetric rates come from
    ``normalise_storage_contract`` and overwrite it below. Migrating swings onto
    the same strict check is IMPLEMENTATION-GUIDE-2026-09-11.md §5.1, item 6,
    not this step.
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
        contract = normalise_storage_contract(p)
        p["inj_rate"] = contract["inj_rate"]
        p["wdr_rate"] = contract["wdr_rate"]
        p["initial_inv_clips"] = contract["initial_inv_clips"]
        p["terminal_inv_clips"] = contract["terminal_inv_clips"]
    return p
