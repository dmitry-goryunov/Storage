"""Named, executable benchmark configurations.

Every headline number this project quotes should be reconstructible from inputs
recorded somewhere runnable. Several were not: the "4.2 % extrinsic" baseline and
the "nine times too little" spread comparison in the September design note match
no configuration that was written down, and an independent review could not
reproduce either. Prose is not a fixture.

Each case here is a full parameter set -- capacity, actual MWh/day rates,
ratchets, dates, opening and terminal inventory, bounds, curve, fuel, fees,
discounting, price parameters and both grids. `describe_environment()` records
the source revision and package versions beside any result.

Run it directly for the table:

    python benchmarks.py

The cases are the ones the review turned on, so a change of answer here is a
change of answer to the review.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import quote_data as qd
import storage_model as sm

_ROOT = os.path.dirname(os.path.abspath(__file__))

CAPACITY = 600_000.0
INJ_MWH_DAY = 20_000.0          # 30 days to fill
WDR_MWH_DAY = 10_000.0          # 60 days to empty

# The 30/60 notebook's own curve: flat within each month, repeating annually.
# Nearly flat by design -- a 5 EUR summer/winter spread -- which is why the
# optimiser flips between months it cannot tell apart and the hedge is unstable
# from April to August.
NOTEBOOK_MONTHLY = {1: 30.0, 2: 30.0, 3: 24.9, 4: 25.0, 5: 25.0, 6: 25.0,
                    7: 25.0, 8: 25.0, 9: 25.0, 10: 30.0, 11: 30.0, 12: 30.0}

# The notebook's softened withdrawal ratchet, chosen so the ratchets and the 70 %
# floor can both be on.
SOFT_RATCHETS = {"fullness": [0.0, 0.5, 0.8, 1.0],
                 "injection": [1.0, 1.0, 0.6, 0.3],
                 "withdrawal": [0.5, 0.8, 1.0, 1.0]}

# The harsher profile, which is what produced the 52.5 % peak that was recorded
# as physical and is in fact a property of a 240-clip grid.
HARSH_RATCHETS = {"fullness": [0.0, 0.5, 0.8, 1.0],
                  "injection": [1.0, 1.0, 0.6, 0.3],
                  "withdrawal": [0.3, 0.7, 1.0, 1.0]}


def monthly_curve(monthly=None, start="2026-01-01", end="2029-06-30"):
    """Flat-within-month daily curve, repeating each year."""
    monthly = NOTEBOOK_MONTHLY if monthly is None else monthly
    span = pd.date_range(start, end, freq="D")
    return pd.Series([float(monthly[d.month]) for d in span], index=span)


def storage_params(n_states, ratchets=None, curve=None, n_p=25, rate=0.10,
                   run_intrinsic=True, max_ratchet_rate_loss=1.0, **extra):
    """A 30/60 store at a chosen inventory clip, physical deal held fixed.

    `n_states` changes only the resolution: capacity and both MWh/day rates are
    preserved EXACTLY via `sm.normalise_storage_contract` -- the same check
    IMPLEMENTATION-GUIDE-2026-09-11.md §4.2 put into `params_for_run_valuation`.
    Until 2026-09-11 this rounded independently (`int(round(INJ_MWH_DAY /
    v_step))`, no check), which is the identical defect fixed there: an
    `n_states` that cannot express 30/60 exactly would have been silently
    repriced rather than refused. Every value this module actually uses is a
    multiple of `lcm(30, 60) = 60` -- the ladders all start from 60 and double
    -- so this changes nothing for any of them; it only means a grid that
    genuinely cannot express the rate now fails loudly instead of quietly
    studying a different deal.
    """
    contract = sm.normalise_storage_contract(dict(
        capacity_mwh=CAPACITY, n_states=n_states,
        inj_days=CAPACITY / INJ_MWH_DAY, wdr_days=CAPACITY / WDR_MWH_DAY,
        initial_storage_mwh=0.0, terminal_storage_mwh=0.0))
    v_step = contract["v_step"]
    inj_rate, wdr_rate = contract["inj_rate"], contract["wdr_rate"]
    params = dict(
        product_type="storage", valDate="2026-06-01",
        storageStart="2027-01-01", storageEnd="2027-12-31",
        capacity_mwh=CAPACITY, daily_max=inj_rate * v_step,
        clips_per_day=inj_rate, inj_rate=inj_rate, wdr_rate=wdr_rate,
        initial_inv_clips=contract["initial_inv_clips"],
        terminal_inv_clips=contract["terminal_inv_clips"],
        inj_cost=0.0, wdr_cost=0.0, fuel_loss=0.0,
        vol=0.50, sMR=1.0, n_p_full=n_p, run_intrinsic=run_intrinsic,
        discount_rate=rate,
        # These fixtures exist to STUDY coarse grids, so the library's rate
        # gate is lifted here and the loss is reported instead.
        max_ratchet_rate_loss=max_ratchet_rate_loss,
        daily_curve=monthly_curve() if curve is None else curve)
    if ratchets is not None:
        params["ratchets"] = pd.DataFrame(ratchets)
    params.update(extra)
    return params


# ── The cases the review turned on ────────────────────────────────────────────

CASES = {
    "shipped-30-60": lambda: storage_params(
        240, SOFT_RATCHETS, min_inventory={"2027-10-01": 0.70}),
    "shipped-30-60-x2": lambda: storage_params(
        480, SOFT_RATCHETS, min_inventory={"2027-10-01": 0.70}),
    "shipped-30-60-x4": lambda: storage_params(
        960, SOFT_RATCHETS, min_inventory={"2027-10-01": 0.70}),
    "unratcheted-control": lambda: storage_params(240),
    "unratcheted-control-x2": lambda: storage_params(480),
    "harsh-ratchet-240": lambda: storage_params(240, HARSH_RATCHETS, n_p=0),
    "harsh-ratchet-480": lambda: storage_params(480, HARSH_RATCHETS, n_p=0),
    "harsh-ratchet-3840": lambda: storage_params(3840, HARSH_RATCHETS, n_p=0),
    "harsh-ratchet-240-with-70pc-floor": lambda: storage_params(
        240, HARSH_RATCHETS, n_p=0, min_inventory={"2027-10-01": 0.70}),
    "harsh-ratchet-480-with-70pc-floor": lambda: storage_params(
        480, HARSH_RATCHETS, n_p=0, min_inventory={"2027-10-01": 0.70}),
    "fuel-free": lambda: storage_params(60, n_p=15, run_intrinsic=False),
    "fuel-1.5pc": lambda: storage_params(60, n_p=15, run_intrinsic=False,
                                         fuel_loss=0.015),
}


def bound_counterexamples():
    """The three inventory-bound defects, as configurations rather than prose.

    Each raised no error and returned a wrong answer before 2026-09-10.
    """
    flat = pd.Series(30.0, index=pd.date_range("2026-01-01", "2026-12-31", freq="D"))
    base = dict(product_type="storage", valDate="2026-01-01",
                storageStart="2026-01-01", storageEnd="2026-12-31",
                v_step=10.0, inj_days=10, clips_per_day=1,
                inj_cost=0.0, wdr_cost=0.0, vol=0.5, sMR=1.0, n_p_full=0,
                run_intrinsic=False, discount_rate=0.0, daily_curve=flat)
    return {
        # Nearest rounding relaxed a floor from 71 % to 70 %, and a ceiling from
        # 29 % to 30 %. Both are contracts the caller did not ask for.
        "rounding-floor-71pc": dict(base, initial_inv_clips=0, terminal_inv_clips=0,
                                    min_inventory={"2026-06-01": 0.71}),
        "rounding-ceiling-29pc": dict(base, initial_inv_clips=0, terminal_inv_clips=0,
                                      max_inventory={"2026-06-01": 0.29}),
        # A full store that holds everything was rejected against a 100 % floor,
        # because the balance was rebuilt without its opening inventory.
        "full-store-100pc-floor": dict(base, initial_inv_clips=10, terminal_inv_clips=10,
                                       min_inventory={"2026-01-03": 1.0}),
    }


def adverse_price_floor(level, n_p=30):
    """A floor the optimiser had an economic reason to breach.

    Ten 1 MWh clips, one cheap injection day and two withdrawal days, with a
    2-clip floor the day after the first sale. While the bound was a
    `1000 * v_step` penalty, raising `level` bought a breach: 0.014398 % of paths
    opened the day empty at EUR 300 and 19.520537 % at EUR 10,000, and the
    expectation-based checker accepted all of them.
    """
    val, end = "2026-01-01", "2027-12-31"
    inj_day = pd.Timestamp("2027-01-01")
    sell_days = [pd.Timestamp("2027-06-01"), pd.Timestamp("2027-12-01")]
    floor_day, n_states, floor_clips = pd.Timestamp("2027-06-02"), 10, 2

    prices = pd.Series(float(level), index=pd.date_range(val, end, freq="D"))
    prices.loc[inj_day] = 1.0
    s = sm.Storage(val, val, end, curve=None, daily_curve=prices, n_p=0,
                   v_step=1.0, sVol=0.8, sMR=1.0, clips_per_day=n_states)
    s.set_volume_states(n_states, initial_state=0)
    s.i_curve = np.zeros(len(s.date_span), dtype=np.int64)
    s.w_curve = np.zeros(len(s.date_span), dtype=np.int64)
    s.i_curve[(inj_day - s.valDate).days] = n_states
    for d in sell_days:
        s.w_curve[(d - s.valDate).days] = n_states
    s.i_cost[:] = 0.0
    s.w_cost[:] = 0.0
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    s.t_p_curve[0] = 0.0
    s.mintunnel[(floor_day - s.valDate).days] = floor_clips
    s.n_p = n_p
    return s, floor_day, floor_clips


# ── Provenance ────────────────────────────────────────────────────────────────

def describe_environment():
    """Source revision and package versions, to record beside any result."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True,
            text=True, check=True).stdout.strip())
    except Exception:
        revision, dirty = "unknown", False
    import numba
    import scipy
    return {
        "revision": revision + ("+dirty" if dirty else ""),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
    }


def run_case(name):
    """Value one named case; returns the result dict with its peak inventory."""
    model, res = sm.run_valuation(None, CASES[name]())
    return model, _summarise(model, res)


def _summarise(model, res):
    n = model.n_t
    levels = np.arange(model.n_op) * model.v_step
    opening = (model.prob[:n].sum(axis=1) * levels).sum(axis=1)
    out = dict(res)
    out["peak_opening_fraction"] = float(opening.max()) / CAPACITY
    # Equal-value policy switches move the hedge without moving the value, so
    # the hedge is reported beside it rather than inferred from it.
    out["net_hedge_mwh"] = float(np.sum(model.delta))
    out["gross_hedge_mwh"] = float(np.sum(np.abs(model.delta)))
    out["worst_rate_loss"] = sm.worst_ratchet_rate_loss(model)
    return out


# ── Convergence ladders ───────────────────────────────────────────────────────

# Proposed engineering thresholds, not achieved results and not commercial
# tolerances: total AND intrinsic within this over each of two successive grid
# doublings, on a benchmark whose value is material. IMPLEMENTATION-GUIDE-
# 2026-09-11.md §7.2's own proposed numbers: a relative test for anything
# material, and a SEPARATE absolute allowance that applies ONLY when both
# endpoints of the step are themselves near zero -- not whenever the MOVE
# happens to be small in absolute terms. Conflating those two was the defect:
# 10,000 -> 9,500 -> 9,000 (5 % steps, both endpoints plainly material) used
# to pass because each 500 EUR move was under an unconditional 1,000 EUR
# floor, regardless of what the values themselves were.
CONVERGENCE_TOLERANCE = 0.005
CONVERGENCE_NEAR_ZERO_EUR = 100.0    # "material" cutoff: both endpoints must be at or below this
CONVERGENCE_ABS_EUR = 1.0            # ...to use this absolute tolerance instead of the relative one

#: What `convergence_verdict` can return, in place of a bare bool.
#: IMPLEMENTATION-GUIDE-2026-09-11.md §7.2 item 6: "do not present all
#: non-successes as an infeasible physical contract" -- a table with too few
#: eligible steps, one with a non-finite value in it, and a table that
#: genuinely has not settled are three different findings, not one "False".
CONVERGENCE_STATUSES = ("within_declared_tolerance", "outside_tolerance",
                        "insufficient", "invalid")


def inventory_grid_ladder(ladder=(240, 480, 960, 1920, 3840), **case):
    """Refine the inventory clip with the PHYSICAL deal held fixed.

    Capacity and both MWh/day rates are preserved; only the resolution changes.
    Reports value, intrinsic, peak inventory, the hedge and the runtime, because
    the answer to "is this converged" is not only about the value.
    """
    rows = []
    for n_states in ladder:
        started = time.perf_counter()
        try:
            model, res = sm.run_valuation(None, storage_params(n_states, **case))
        except ValueError as exc:
            rows.append({"n_states": n_states, "v_step": CAPACITY / n_states,
                         "refused": str(exc)[:60]})
            continue
        summary = _summarise(model, res)
        rows.append({
            "n_states": n_states, "v_step": CAPACITY / n_states,
            "inj clips/day": int(np.max(model.i_curve)),
            "wdr clips/day": int(np.max(model.w_curve)),
            "total_eur": summary["total_eur"],
            "intrinsic_eur": summary["intrinsic_eur"],
            "extrinsic_eur": summary["extrinsic_eur"],
            "peak": summary["peak_opening_fraction"],
            "net hedge MWh": summary["net_hedge_mwh"],
            "worst wdr rate loss": summary["worst_rate_loss"]["withdrawal"],
            "seconds": time.perf_counter() - started,
            "refused": None,
        })
    return pd.DataFrame(rows)


def price_grid_ladder(ladder=(10, 15, 20, 25, 30), n_states=240, **case):
    """Refine the PRICE TREE's half-width `n_p`, at a fixed inventory grid.

    Its own check, so that inventory refinement cannot conceal price error --
    or be blamed for it.

    **This is a price-BOUNDARY-WIDTH check, not a full price-discretisation
    study.** `n_p` controls how many price states the tree carries
    (`2*n_p+1`), i.e. how far the price can range before truncating against
    the tree's edge; the underlying TIME STEP is fixed at one calendar day
    regardless of `n_p` (`storage_model.build_tree`'s `dt = 1/365.25`
    throughout). Refining `n_p` therefore tests whether the tree is wide
    enough to hold the relevant price range without truncation error -- it
    does NOT test daily-step-size / transition-spacing accuracy, a genuinely
    separate numerical dimension this project does not yet have a way to
    refine independently. IMPLEMENTATION-GUIDE-2026-09-11.md §7.4.
    """
    rows = []
    for n_p in ladder:
        started = time.perf_counter()
        model, res = sm.run_valuation(None, storage_params(n_states, n_p=n_p, **case))
        rows.append({"n_p": n_p, "total_eur": res["total_eur"],
                     "extrinsic_eur": res["extrinsic_eur"],
                     "seconds": time.perf_counter() - started})
    return pd.DataFrame(rows)


def convergence_verdict(table, tolerance=CONVERGENCE_TOLERANCE,
                        abs_eur=CONVERGENCE_ABS_EUR, near_zero_eur=CONVERGENCE_NEAR_ZERO_EUR,
                        columns=("total_eur", "intrinsic_eur")):
    """Did the last two eligible refinements each move every tracked column by
    less than the declared tolerance?

    Returns `(status, steps, message)`. `status` is one of
    `CONVERGENCE_STATUSES`: `"within_declared_tolerance"` /
    `"outside_tolerance"` are verdicts; `"insufficient"` (too few eligible
    steps to judge) and `"invalid"` (bad input) are refusals to judge at all,
    not a quiet pass. `message` names the reason for a non-tolerance status;
    `steps` is a DataFrame of every ELIGIBLE consecutive step, each carrying
    its own per-column relative move, absolute move (EUR) and
    within-tolerance flag, plus a `status` for that individual step.

    Until 2026-09-11 this returned a bare bool and had seven confirmed ways to
    fail open, reproduced independently and cross-checked against the
    guide's own acceptance-pack oracle:

    * NaN or infinite values compared as `nan > tolerance`, which is `False`
      in Python -- three non-finite rows in a row silently PASSED.
    * `n_states` was never checked for being sorted or for actually doubling:
      three copies of the same grid, or a descending/non-doubling sequence,
      all "converged" trivially or by accident.
    * A refused row was filtered OUT entirely (`table[refused.isna()]`) and
      the steps either side of the gap it left were treated as adjacent --
      240 -> REFUSED -> 960 -> 1920 silently became "240 -> 960 -> 1920".
    * The `abs_eur` absolute fallback was an unconditional OR on the SIZE OF
      THE MOVE: 10,000 -> 9,500 -> 9,000 (5 % steps, plainly material values)
      passed anyway, because each step's absolute move (500 EUR) happened to
      be under the 1,000 EUR floor, regardless of what the underlying values
      were. `abs_eur` now applies only when both endpoints' own MAGNITUDE are
      at or below `near_zero_eur` -- a relative comparison is meaningless
      near zero (0.01 -> 0.02 is a "100 % move" of nothing), but it is not a
      substitute for the relative test on a material value just because one
      particular move happened to be small in EUR terms.

    A step is ELIGIBLE only between two ADJACENT rows of `table`, neither
    refused, where the later `n_states` is exactly double the earlier -- the
    project's own declared rule ("two successive doublings"), enforced
    literally rather than approximated by "any two points that ended up next
    to each other".
    """
    required = {"n_states", *columns}
    missing = required - set(table.columns)
    if missing:
        return "invalid", pd.DataFrame(), f"missing required column(s): {sorted(missing)}"
    if len(table) == 0:
        return "invalid", pd.DataFrame(), "empty table"

    n_states = table["n_states"].to_numpy()
    if not np.all(np.diff(n_states.astype(float)) > 0):
        return "invalid", pd.DataFrame(), "n_states is not strictly increasing"

    refused = (table["refused"].to_numpy() if "refused" in table
              else np.full(len(table), None, dtype=object))

    steps = []
    for i in range(len(table) - 1):
        lo_n, hi_n = int(n_states[i]), int(n_states[i + 1])
        if refused[i] is not None or refused[i + 1] is not None:
            continue                      # a refused row breaks the sequence, not just its own row
        if hi_n != 2 * lo_n:
            continue                      # only a genuine doubling is an eligible refinement step

        row = {"from": lo_n, "to": hi_n}
        row_status = "within_declared_tolerance"
        for col in columns:
            lo, hi = float(table[col].iloc[i]), float(table[col].iloc[i + 1])
            if not (np.isfinite(lo) and np.isfinite(hi)):
                row_status = "invalid"
                row[col + "_relative"] = np.nan
                row[col + "_eur"] = np.nan
                row[col + "_within_tolerance"] = False
                continue
            moved = abs(hi - lo)
            magnitude = max(abs(lo), abs(hi))
            relative = moved / magnitude if magnitude > 0 else 0.0
            # The absolute allowance applies ONLY when the VALUES themselves
            # are near zero -- a relative comparison is meaningless there
            # (0.01 -> 0.02 is a "100 % move" of nothing). It is not a
            # substitute for the relative test whenever a move happens to be
            # small in EUR terms: a material value must pass on relative
            # terms alone.
            within = (relative <= tolerance if magnitude > near_zero_eur
                     else moved <= abs_eur)
            row[col + "_relative"] = relative
            row[col + "_eur"] = moved
            row[col + "_within_tolerance"] = within
            if not within and row_status != "invalid":
                row_status = "outside_tolerance"
        row["status"] = row_status
        steps.append(row)

    frame = pd.DataFrame(steps)
    if len(frame) < 2:
        return "insufficient", frame, (
            f"{len(frame)} eligible successive-doubling step(s); at least 2 are needed "
            f"to judge convergence (a refused row, a gap, or too short a ladder can all "
            f"cause this)")

    final_two = frame["status"].iloc[-2:]
    if (final_two == "invalid").any():
        return "invalid", frame, "a non-finite value appears in the final two eligible steps"
    if (final_two == "within_declared_tolerance").all():
        return "within_declared_tolerance", frame, None
    return "outside_tolerance", frame, (
        f"the final step moved {[c for c in columns if not frame[c + '_within_tolerance'].iloc[-1]]} "
        f"by more than {tolerance:.1%} (and more than {abs_eur:,.0f} EUR)")


# ── Market statistics: the spread comparison, made executable ─────────────────

# The comparison the "nine times too little" claim rested on, with every choice
# stated. It was reconstructible from no recorded calculation, which is the
# reason this section exists.
#
#   return definition   daily differences of log mid prices, positive prices only
#   spread definition   d(log F_i) - d(log F_j): the volatility of changes in the
#                       log price RATIO. NOT the log of a monetary spread (which
#                       can cross zero) and NOT the EUR/MWh volatility of F_j - F_i
#   observation horizon one trading day
#   annualisation       sqrt(252)
#   estimation window   2015-01-01 onward, stated per call
#   delivery periods    APPROXIMATED by point maturities tau_i = i/12 years. A
#                       real contract delivers over a month, and CORRECTED
#                       2026-09-10 (evening): that does NOT necessarily lower
#                       the comparison -- the point-maturity figure is not a
#                       reliable upper bound on the delivery-averaged one. For
#                       equal one-month periods ending 0.5y/1y at kappa=1,
#                       sigma=0.5, the flat-forward delivery-average log-ratio
#                       vol is 0.12443854, ABOVE the month-end point value
#                       0.11932561 -- a direct counterexample. The ratio below
#                       is a point-maturity illustration only, not a
#                       calibration result either way; see
#                       IMPLEMENTATION-GUIDE-2026-09-11.md item 20 for the
#                       actual delivery-weighted observation function
#   rolls               continuous rank c6 refers to a different delivery month
#                       after a roll, so month-change observations are optionally
#                       excluded
# The tracked workbook, NOT `ttf q.parquet` -- that is a gitignored local cache
# derived from it, so pointing here at the cache made these numbers reproducible
# on one machine and nowhere else. CI caught it on the first push.
WORKBOOK = os.path.join(_ROOT, "ttf q.xlsx")
SPREAD_PAIRS = ((1, 3), (1, 6), (6, 12), (1, 12), (12, 24), (1, 24))
TRADING_DAYS = 252.0


def load_quote_matrix(path=None, cache_dir=None, use_cache=True):
    """The TTF quote matrix, via `quote_data.load_quote_matrix` -- the shared
    cache-identity policy also used by `portfolio_app.py`.

    Until 2026-09-11 this checked a SINGLE fixed parquet cache path regardless
    of what `path` was actually requested, validated only by modification
    time: an explicit request for a different workbook could silently return
    the repository's cached default, and a same-path edit with its mtime
    restored stayed stale. Content-addressed caching in `quote_data` makes
    both impossible by construction -- see its docstring. `cache_dir` and
    `use_cache` pass straight through, so an isolated cache location (a
    temporary directory in a test, for instance) or a no-cache reproducible
    read are both available without reaching into module internals.

    Returns the DataFrame only, dropping the provenance record this module's
    own callers do not (yet) consume; call `quote_data.load_quote_matrix`
    directly for that.
    """
    path = WORKBOOK if path is None else path
    quotes, _provenance = qd.load_quote_matrix(path, cache_dir=cache_dir, use_cache=use_cache)
    return quotes


def forward_panel(path=None, since="2015-01-01"):
    df = load_quote_matrix(path).sort_values("quote_date").reset_index(drop=True)
    return df[df["quote_date"] >= pd.Timestamp(since)]


def spread_statistics(pairs=SPREAD_PAIRS, since="2015-01-01", exclude_rolls=False,
                      path=None):
    """Realised correlation and annualised log-ratio volatility, by maturity pair."""
    panel = forward_panel(path, since)
    rows = []
    for a, b in pairs:
        ca, cb = f"TTFc{a}", f"TTFc{b}"
        s = panel[["quote_date", ca, cb]].dropna()
        s = s[(s[ca] > 0) & (s[cb] > 0)].reset_index(drop=True)
        ra = np.diff(np.log(s[ca].to_numpy()))
        rb = np.diff(np.log(s[cb].to_numpy()))
        keep = np.ones(ra.shape, dtype=bool)
        if exclude_rolls:
            months = s["quote_date"].dt.to_period("M").to_numpy()
            keep = months[1:] == months[:-1]
        ra, rb = ra[keep], rb[keep]
        rows.append({"pair": f"c{a}/c{b}", "observations": int(ra.size),
                     "correlation": float(np.corrcoef(ra, rb)[0, 1]),
                     "log_ratio_vol": float(np.std(ra - rb, ddof=0) * np.sqrt(TRADING_DAYS))})
    return pd.DataFrame(rows)


def model_log_ratio_vol(sigma, kappa, tau_1, tau_2):
    """The one-factor model's annualised volatility of daily log-ratio changes.

    Under `d log F(t,T) = drift dt + sigma * exp(-kappa*(T-t)) dW`, the log ratio
    of two forwards has diffusion `sigma * |a_1 - a_2|` with `a_i = exp(-kappa*tau_i)`.

    This is the quantity comparable with `spread_statistics()`. The September
    design note instead used `sigma * |a_1 - a_2| / sqrt(2*kappa)`, which is the
    stationary standard deviation of the LEVEL of the log ratio -- a different
    object with different units of time. Note the result is NOT monotone in
    kappa: both loadings tend to 1 as kappa -> 0 and to 0 as kappa -> infinity,
    so it peaks in between (near kappa = 1.385 at these maturities). A proposed
    acceptance gate requiring the sign to flip with mean reversion would fail a
    correct model.
    """
    a_1, a_2 = np.exp(-kappa * tau_1), np.exp(-kappa * tau_2)
    return float(sigma * abs(a_1 - a_2))


def spread_comparison(sigma=0.50, kappa=1.0, pair=(6, 12), since="2015-01-01",
                      path=None):
    """Model against realised for one pair, with the maturity approximation stated."""
    tau_1, tau_2 = pair[0] / 12.0, pair[1] / 12.0
    model = model_log_ratio_vol(sigma, kappa, tau_1, tau_2)
    stats = spread_statistics((pair,), since=since, path=path)
    no_roll = spread_statistics((pair,), since=since, exclude_rolls=True, path=path)
    realised = float(stats["log_ratio_vol"][0])
    realised_no_roll = float(no_roll["log_ratio_vol"][0])
    return {
        "pair": f"c{pair[0]}/c{pair[1]}", "sigma": sigma, "kappa": kappa,
        "tau_years": (tau_1, tau_2), "model_log_ratio_vol": model,
        "realised_raw": realised, "realised_excluding_rolls": realised_no_roll,
        "ratio_raw": realised / model,
        "ratio_excluding_rolls": realised_no_roll / model,
        "caveat": ("point maturities, one pair, one window -- illustrative, not a "
                   "calibrated shortfall. One spread observation is one equation in "
                   "two unknowns and pins a curve of (sigma, kappa), not a point."),
    }


def _rule(title):
    print()
    print(title)
    print("-" * max(len(title), 60))


def main():
    env = describe_environment()
    print(" ".join(f"{k}={v}" for k, v in env.items()))

    _rule("Named cases")
    print(f"{'case':>34} {'total EUR':>13} {'intrinsic':>13} {'extrinsic':>12} "
          f"{'share':>7} {'peak':>7}")
    for name in CASES:
        try:
            _, res = run_case(name)
        except ValueError as exc:
            print(f"{name:>34}  REFUSED: {str(exc)[:44]}")
            continue
        share = (res["extrinsic_eur"] / res["total_eur"]
                 if np.isfinite(res["extrinsic_eur"]) else float("nan"))
        print(f"{name:>34} {res['total_eur']:>13,.0f} {res['intrinsic_eur']:>13,.0f} "
              f"{res['extrinsic_eur']:>12,.0f} {share:>6.2%} "
              f"{res['peak_opening_fraction']:>6.2%}")

    _rule("Deliverability the grid actually gives, shipped 30/60 at 240 clips")
    model, _ = sm.run_valuation(None, CASES["shipped-30-60"]())
    rates = sm.describe_ratchet_rates(model)
    show = rates.nlargest(6, "withdrawal loss")
    print(show[["fullness", "withdrawal contract MWh/day", "withdrawal grid MWh/day",
                "withdrawal loss MWh/day", "withdrawal loss"]].to_string(
        index=False, float_format=lambda v: f"{v:,.3f}"))
    worst = sm.worst_ratchet_rate_loss(model)
    print(f"worst loss  injection {worst['injection']:.1%}   "
          f"withdrawal {worst['withdrawal']:.1%}")
    print("The loss is a sawtooth: it is zero wherever rate x multiplier lands on an")
    print("integer and worst just below one. A mild ratchet is not a safe ratchet.")

    _rule("Inventory-grid convergence, shipped 30/60 "
          f"(gate: < {CONVERGENCE_TOLERANCE:.1%} over each of the last two doublings)")
    ladder = inventory_grid_ladder(
        ratchets=SOFT_RATCHETS, min_inventory={"2027-10-01": 0.70})
    print(ladder.drop(columns=["refused"]).to_string(
        index=False, float_format=lambda v: f"{v:,.4f}"))
    status, steps, message = convergence_verdict(ladder)
    if len(steps):
        print(steps.to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    print(f"VERDICT: {status}" + (f" -- {message}" if message else ""))

    _rule("Price-tree BOUNDARY WIDTH at a fixed inventory grid (n_p; NOT a "
          "daily-step/transition-spacing study -- see price_grid_ladder's docstring)")
    print(price_grid_ladder(ratchets=SOFT_RATCHETS,
                            min_inventory={"2027-10-01": 0.70}).to_string(
        index=False, float_format=lambda v: f"{v:,.4f}"))

    _rule("Realised forward statistics, 2015 onward")
    print(spread_statistics().to_string(index=False,
                                        float_format=lambda v: f"{v:,.6f}"))

    _rule("The spread comparison, with its assumptions stated")
    for key, value in spread_comparison().items():
        print(f"  {key:>24}: {value}")


if __name__ == "__main__":
    main()
