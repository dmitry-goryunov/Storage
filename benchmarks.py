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
import subprocess
import sys

import numpy as np
import pandas as pd

import storage_model as sm

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
                   run_intrinsic=True, **extra):
    """A 30/60 store at a chosen inventory clip, physical deal held fixed.

    `n_states` changes only the resolution: capacity and both MWh/day rates are
    preserved by scaling the clip rates with the clip size. That is the property
    a refinement ladder needs and the one the original "refinement is free" check
    silently relied on the unratcheted case for.
    """
    v_step = CAPACITY / n_states
    inj_rate = int(round(INJ_MWH_DAY / v_step))
    wdr_rate = int(round(WDR_MWH_DAY / v_step))
    params = dict(
        product_type="storage", valDate="2026-06-01",
        storageStart="2027-01-01", storageEnd="2027-12-31",
        capacity_mwh=CAPACITY, daily_max=inj_rate * v_step,
        clips_per_day=inj_rate, inj_rate=inj_rate, wdr_rate=wdr_rate,
        initial_inv_clips=0, terminal_inv_clips=0,
        inj_cost=0.0, wdr_cost=0.0, fuel_loss=0.0,
        vol=0.50, sMR=1.0, n_p_full=n_p, run_intrinsic=run_intrinsic,
        discount_rate=rate,
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
    n = model.n_t
    levels = np.arange(model.n_op) * model.v_step
    opening = (model.prob[:n].sum(axis=1) * levels).sum(axis=1)
    out = dict(res)
    out["peak_opening_fraction"] = float(opening.max()) / CAPACITY
    return model, out


def main():
    env = describe_environment()
    print(" ".join(f"{k}={v}" for k, v in env.items()))
    print()
    print(f"{'case':>34} {'total EUR':>13} {'intrinsic':>13} {'extrinsic':>12} "
          f"{'share':>7} {'peak':>7}")
    print("-" * 92)
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


if __name__ == "__main__":
    main()
