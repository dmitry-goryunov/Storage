"""Regression anchors for the storage / swing model.

These pin numbers and invariants that were independently verified (brute-force
day selection, hand-checked MtM) so future refactors can't silently change them.

Run with pytest:                python -m pytest tests
or as a plain script (no deps): python tests/test_regression.py

Tests that need the bundled data files (ttf q.xlsx, quotes_2.csv) skip cleanly
when those files are absent.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import storage_model as sm  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────────

def _curve_csv():
    path = os.path.join(ROOT, "curve.csv")
    curve = pd.read_csv(path)
    curve["contractStart"] = pd.to_datetime(curve["contractStart"], format="mixed")
    curve["contractEnd"] = pd.to_datetime(curve["contractEnd"], format="mixed")
    return curve


def _num(x):
    s = str(x).strip().replace(",", "").replace(" ", "")
    return 0.0 if s in ("", "-", "nan") else float(s)


# ── invariant tests (no data files beyond curve.csv) ───────────────────────────

def test_flat_is_window_mean_and_below_profiled():
    """flat() = arithmetic window average; for a sell-only default profiled >= flat."""
    curve = _curve_csv()
    s = sm.Storage("2026-01-01", "2026-04-01", "2026-09-30", curve=curve,
                   n_p=0, v_step=1000, sVol=0.6, sMR=1.0)
    s.build()
    win = s.date_span[s.Dt:s._active]
    expected = float(pd.Series(s.price_curve, index=s.date_span).loc[win].mean())
    assert abs(s.flat() - expected) < 1e-6, (s.flat(), expected)
    assert s.profiled() >= s.flat() - 1e-9   # picking the best days beats the average


def test_profiled_central_node_nonzero_at_np30():
    """profiled() must read the central node v[0, n_p, .], not the 0-valued boundary."""
    curve = _curve_csv()
    s = sm.Storage("2026-01-01", "2026-04-01", "2026-09-30", curve=curve,
                   n_p=30, v_step=1000, sVol=0.6, sMR=1.0)
    s.build()
    assert s.profiled() > 0.0
    assert s.v[0, 0, s.n_op_start] == 0.0   # the old (buggy) boundary read


def test_readme_example_decomposition_nonnegative():
    """The README Usage Example: intrinsic and extrinsic must both be >= 0."""
    curve = _curve_csv()
    common = dict(curve=curve, v_step=1000, sVol=0.6, sMR=1.0)
    s_flat = sm.Storage("2026-01-01", "2026-04-01", "2026-09-30", n_p=0, **common)
    s_flat.build()
    s_full = sm.Storage("2026-01-01", "2026-04-01", "2026-09-30", n_p=30, **common)
    s_full.build()
    intrinsic = s_flat.profiled() - s_flat.flat()
    acq = np.sum(s_flat.delta)
    extrinsic = (s_full.v[0, s_full.n_p, s_full.n_op_start]
                 - s_flat.v[0, 0, s_flat.n_op_start]) / acq
    assert intrinsic >= -1e-9, intrinsic
    assert extrinsic >= -1e-9, extrinsic


def test_curve_gap_raises():
    """A contract curve that does not cover the storage period must fail loudly
    (the smoothing step or the explicit guard raises) — never silent NaN garbage."""
    short = pd.DataFrame({
        "contractStart": [pd.Timestamp("2026-04-01")],
        "contractEnd": [pd.Timestamp("2026-04-30")],
        "value": [25.0],
    })
    try:
        sm.Storage("2026-01-01", "2026-04-01", "2026-12-31", curve=short,
                   n_p=0, v_step=1000)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on an uncovered contract curve")


def test_nan_daily_curve_guard():
    """The explicit NaN guard: an all-NaN daily curve can't be filled and must raise."""
    days = pd.date_range("2026-01-01", "2027-02-28", freq="D")
    nan_curve = pd.Series(np.nan, index=days)
    try:
        sm.Storage("2026-01-01", "2026-04-01", "2026-12-31", daily_curve=nan_curve,
                   n_p=0, v_step=1000)
    except ValueError as exc:
        assert "missing" in str(exc).lower(), str(exc)
    else:
        raise AssertionError("expected ValueError on an all-NaN daily curve")


def test_np0_call_swing_equals_greedy():
    """With n_p=0 the DP equals greedily picking the N best days (to the cent)."""
    days = pd.date_range("2026-04-01", "2026-09-30", freq="D")
    rng_vals = 20 + 10 * np.sin(np.arange(len(days)) / 7.0)   # deterministic wiggle
    daily = pd.Series(rng_vals, index=days)
    n_days = 30
    strike = 20.0
    daily_vol = 1000.0
    params = {
        "product_type": "call_swing", "valDate": "2026-03-15",
        "storageStart": "2026-04-01", "storageEnd": "2026-09-30",
        "vol": 0.2, "sMR": 1.0, "n_p_full": 0, "run_intrinsic": True,
        "daily_max": daily_vol, "clips_per_day": 1,
        "capacity_mwh": n_days * daily_vol, "strike": strike, "daily_curve": daily,
    }
    s, _ = sm.run_valuation(None, params)
    model_eur = float(s.v[0, 0, s.n_op_start])
    # Greedy: sell the n_days highest-price days, payoff (price - strike) * volume.
    window = s.price_curve.loc[pd.Timestamp("2026-04-01"):pd.Timestamp("2026-09-30")]
    best = np.sort(window.values)[-n_days:]
    greedy_eur = float(np.sum(best - strike) * daily_vol)
    assert abs(model_eur - greedy_eur) < 1.0, (model_eur, greedy_eur)


def test_storage_asymmetric_rates():
    """value_storage honours separate inj_rate/wdr_rate; symmetric default is
    unchanged when they are absent."""
    days = pd.date_range("2026-04-01", "2027-03-31", freq="D")
    daily = pd.Series(22 + 6 * np.sin(np.arange(len(days)) / 30.0), index=days)
    base = {
        "product_type": "storage", "valDate": "2026-03-15",
        "storageStart": "2026-04-01", "storageEnd": "2027-03-30",
        "vol": 0.2, "sMR": 1.0, "n_p_full": 0, "run_intrinsic": False,
        "daily_max": 3000.0, "clips_per_day": 3, "capacity_mwh": 90000.0,
        "inj_cost": 0.5, "wdr_cost": 0.5, "daily_curve": daily,
    }
    _, sym = sm.run_valuation(None, dict(base))
    _, asym = sm.run_valuation(None, dict(base, inj_rate=3, wdr_rate=2))
    assert np.isfinite(sym["total"]) and np.isfinite(asym["total"])
    # A tighter withdraw rate constrains the schedule -> generally a different value.
    assert sym["total"] != asym["total"]


# ── data-dependent tests (skip if files absent) ────────────────────────────────

def _have(*names):
    return all(os.path.exists(os.path.join(ROOT, n)) for n in names)


def test_products_bridge_runs_finite():
    """load_product_params -> params_for_run_valuation -> run_valuation must not
    KeyError or return NaN for any product in the workbook."""
    if not _have("products.xlsx", "ttf q.xlsx"):
        print("SKIP test_products_bridge_runs_finite (missing data)")
        return
    quotes = pd.read_excel(os.path.join(ROOT, "ttf q.xlsx"))
    quotes = quotes.rename(columns={quotes.columns[0]: "quote_date"}).dropna(subset=["quote_date"])
    quotes["quote_date"] = pd.to_datetime(quotes["quote_date"], format="mixed")
    quotes = quotes.sort_values("quote_date").reset_index(drop=True)
    cc = sorted([c for c in quotes.columns if re.fullmatch(r"TTFc\d+", str(c))],
                key=lambda c: int(re.search(r"\d+", c).group()))
    for c in cc:
        quotes[c] = pd.to_numeric(quotes[c], errors="coerce")
    for name in sm.list_products(os.path.join(ROOT, "products.xlsx")):
        prm = sm.load_product_params(os.path.join(ROOT, "products.xlsx"), product=name)
        params = sm.params_for_run_valuation(prm)
        row = sm.quote_row_for_fd_date(quotes, cc, prm["FDDate"])
        curve = sm.curve_df_for_storage(row, cc, curve_start=prm["valDate"])
        _, res = sm.run_valuation(curve, params)
        assert np.isfinite(res["total"]), f"{name}: total is not finite"


def test_portfolio_total_mtm():
    """End-to-end portfolio MtM anchor: TOTAL = -56,901 EUR at VAL_DATE 2010-08-19."""
    if not _have("ttf q.xlsx", "quotes_2.csv"):
        print("SKIP test_portfolio_total_mtm (missing data)")
        return
    val_date = pd.Timestamp("2010-08-19")
    quotes = pd.read_excel(os.path.join(ROOT, "ttf q.xlsx"))
    quotes = quotes.rename(columns={quotes.columns[0]: "quote_date"}).dropna(subset=["quote_date"])
    quotes["quote_date"] = pd.to_datetime(quotes["quote_date"], format="mixed")
    quotes = quotes.sort_values("quote_date").reset_index(drop=True)
    cc = sorted([c for c in quotes.columns if re.fullmatch(r"TTFc\d+", str(c))],
                key=lambda c: int(re.search(r"\d+", c).group()))
    for c in cc:
        quotes[c] = pd.to_numeric(quotes[c], errors="coerce")

    strip = quotes[cc].notna().sum(axis=1)
    eligible = quotes.loc[strip >= 43].reset_index(drop=True)
    fd = sm.quote_row_for_fd_date(eligible, cc, val_date)
    quote_date = pd.Timestamp(fd["quote_date"])
    monthly, _ = sm.monthly_curve_from_quote(fd, cc)
    day_index = pd.date_range(monthly.index.min(), sm.month_end(monthly.index.max()), freq="D")
    stepped = monthly.reindex(day_index, method="ffill")
    da = fd.get("DA", np.nan)
    if pd.notna(da):
        stepped = stepped.reindex(pd.date_range(quote_date, day_index.max(), freq="D"))
        stepped.loc[quote_date] = float(da)
        stepped = stepped.ffill()
    daily_curve = sm.smoothen_curve(stepped).rename("value")

    raw = pd.read_csv(os.path.join(ROOT, "quotes_2.csv"))
    raw.columns = [str(c).strip() for c in raw.columns]
    total_mtm = 0.0
    for _, r in raw.iterrows():
        ptype = str(r["Product"]).strip().lower().replace(" ", "_")
        daily_vol = _num(r["daily volume, Mwh"])
        direction = int(np.sign(daily_vol))
        abs_vol = abs(daily_vol)
        n_days = int(_num(r["N_days"]))
        capacity = n_days * abs_vol
        premium = _num(r["executed price, Eur/Mwh"]) * capacity
        params = {
            "product_type": ptype, "valDate": val_date,
            "storageStart": pd.to_datetime(str(r["Start"]).strip(), format="%d-%b-%y"),
            "storageEnd": pd.to_datetime(str(r["End"]).strip(), format="%d-%b-%y"),
            "vol": _num(r["vol"]), "sMR": _num(r["MR"]), "n_p_full": 30,
            "run_intrinsic": True, "daily_max": abs_vol, "clips_per_day": 1,
            "capacity_mwh": capacity, "strike": _num(r["strike price, Eur/Mwh"]),
            "daily_curve": daily_curve,
        }
        s, _ = sm.run_valuation(None, params)
        value_eur = float(s.v[0, s.n_p, s.n_op_start])
        total_mtm += direction * (value_eur - premium)
    assert abs(total_mtm - (-56901)) < 1.0, f"TOTAL MtM = {total_mtm:,.0f}, expected -56,901"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
