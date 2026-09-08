==============================================================================
# Swing_new.ipynb — source cells as they stood in the Drive review copy
# Outputs and metadata omitted. Preserved under section 5 of the
# reconciliation procedure (old-only analytical cells).
==============================================================================


# ---- Swing_new.ipynb cell 0 ----
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import storage_model
importlib.reload(storage_model)
from storage_model import Storage


# ---- Swing_new.ipynb cell 2 ----
# Read Forward Curve
curve = pd.DataFrame(pd.read_csv("curve.csv"))
curve['contractStart'] = pd.to_datetime(curve['contractStart'], format='mixed')
curve['contractEnd'] = pd.to_datetime(curve['contractEnd'], format='mixed')


# ---- Swing_new.ipynb cell 3 ----
quotes = pd.read_csv("quotes.csv")
quotes.columns = quotes.columns.str.strip()
quotes['valDate'] = pd.to_datetime(quotes['valDate'], format='mixed', dayfirst=True)
quotes['Start'] = pd.to_datetime(quotes['Start'], format='mixed', dayfirst=True)
quotes['End'] = pd.to_datetime(quotes['End'], format='mixed', dayfirst=True)
quotes


# ---- Swing_new.ipynb cell 4 ----
quotes["model"] = 0.0
quotes["intrinsic"] = 0.0

for row in quotes.itertuples():

    valDate      = pd.Timestamp(row.valDate)
    storageStart = pd.Timestamp(row.Start)
    storageEnd   = pd.Timestamp(row.End)
    days         = int(row.N_days)

    s = Storage(valDate, storageStart, storageEnd, curve=curve, n_p=0, v_step=1000)
    s.build()
    flat_int = s.flat()
    print(f"flat = {s.flat():.2f}")


    # Intrinsic — start fully loaded
    s.set_volume_states(days)
    s.build()

    ProfiledEuro = s.v[0, 0, s.n_op_start]
    ACQ = np.sum(s.delta)

    intrinsic    = s.profiled() - flat_int
    print(f"Profiled price = {s.flat():.2f}")
    print(f"Intrinsic Value  = {intrinsic:.2f} on top of flat price")


    # Extrinsic — full price tree
    s.n_p = 30
    s.build()

    FullValueEuro = s.v[0, s.n_p, s.n_op_start]
    print(f"Extrinsic price = {FullValueEuro / ACQ:.2f}")
    ExtrValue = (FullValueEuro - ProfiledEuro) / ACQ
    print(f"Extrinsic Value = {ExtrValue:.2f}")
    print("")

    quotes.loc[row.Index, "intrinsic"] = float(intrinsic)
    quotes.loc[row.Index, "model"]     = float(ExtrValue)


# ---- Swing_new.ipynb cell 5 ----
plt.scatter(quotes.intrinsic,quotes.bid,label='bid')
plt.scatter(quotes.intrinsic,quotes.ask,label='ask')
plt.scatter(quotes.intrinsic,quotes.model,label='model')
plt.legend()


# ---- Swing_new.ipynb cell 6 ----
plt.plot(quotes.bid,'o-',label='bid')
plt.plot(quotes.ask,'o-',label='ask')
plt.plot(quotes.model,'o-',label='model')
plt.legend()
plt.xticks(ticks=range(len(quotes['Product'])), labels=quotes['Product'])
plt.ylabel("EU/MWh")
plt.xlabel("Product")
plt.title("Modelled Swing Premium vs Market Bids/Asks")


# ---- Swing_new.ipynb cell 7 ----

# Single swing: Jan-Dec 2025, 120 exercise days, 30% vol
valDate      = pd.Timestamp("2026-01-01")
storageStart = pd.Timestamp("2027-01-01")
storageEnd   = pd.Timestamp("2027-12-31")
days         = 120
vol          = 0.30

# Flat (no optionality baseline)
s = Storage(valDate, storageStart, storageEnd, curve=curve, n_p=0, v_step=1000, sVol=vol)
s.build()
flat_int = s.flat()
print(f"Flat price       = {flat_int:.2f}")

# Intrinsic — profiled, fully loaded at start
s.set_volume_states(days)
s.build()
ProfiledEuro = s.v[0, 0, s.n_op_start]
ACQ = np.sum(s.delta)
intrinsic = s.profiled() - flat_int
print(f"Profiled price   = {s.profiled():.2f}")
print(f"Intrinsic value  = {intrinsic:.2f}")

# Extrinsic — full price tree with 30% vol
s.n_p = 30
s.build()
FullValueEuro = s.v[0, s.n_p, s.n_op_start]
ExtrValue = (FullValueEuro - ProfiledEuro) / ACQ
print(f"Extrinsic price  = {FullValueEuro / ACQ:.2f}")
print(f"Extrinsic value  = {ExtrValue:.2f}")
print(f"Total swing value= {intrinsic + ExtrValue:.2f}")



# ---- Swing_new.ipynb cell 8 ----

# Put swing: Jan-Dec 2025, 120 exercise days, 30% vol
# Right to inject (BUY) gas — profits from choosing the cheapest days/scenarios
valDate      = pd.Timestamp("2026-01-01")
storageStart = pd.Timestamp("2027-01-01")
storageEnd   = pd.Timestamp("2027-12-31")
days         = 120
vol          = 0.30

# ── Flat baseline: must buy all year days, no timing choice ─────────────────
s = Storage(valDate, storageStart, storageEnd, curve=curve, n_p=0, v_step=1000, sVol=vol)
n = len(s.date_span)
s.i_curve = np.ones(n);  s.i_curve[s._active:] = 0.;  s.i_curve[:s.Dt] = 0.
s.w_curve = np.zeros(n)
full_cap     = s.n_op_start            # full-year exercise capacity
s.n_op_start = 0                       # start empty
s.t_p_curve  = np.full(s.n_op + 2, -1e9)
s.t_p_curve[full_cap] = 0.            # must buy all full_cap units by end
s.build()
flat_cost = s.flat()                   # average forward price over full year (EUR/MWh)
print(f"Flat cost        = {flat_cost:.2f}")

# ── Intrinsic: choose cheapest 'days' out of full year, no vol ──────────────
s.set_volume_states(days)              # n_op = days+1, states 0..days
s.n_op_start = 0                       # start empty
s.t_p_curve  = np.full(s.n_op + 2, -1e9)
s.t_p_curve[days] = 0.                # must buy exactly 'days' units
s.build()

ProfiledEuro  = s.v[0, 0, 0]
ACQ           = -np.sum(s.delta)      # positive: total volume bought
profiled_cost = s.profiled()          # average cost of cheapest 'days' forward prices
intrinsic     = flat_cost - profiled_cost
print(f"Profiled cost    = {profiled_cost:.2f}")
print(f"Intrinsic saving = {intrinsic:.2f}")

# ── Extrinsic: vol=30% lets you buy even cheaper on low-price realisations ──
s.n_p = 30
s.build()
FullValueEuro = s.v[0, s.n_p, 0]
ExtrValue     = (FullValueEuro - ProfiledEuro) / ACQ
print(f"Extrinsic saving = {ExtrValue:.2f}")
print(f"Total put swing  = {intrinsic + ExtrValue:.2f}")



# ---- Swing_new.ipynb cell 9 ----

# Storage: 50 inj days / 60 wdr days, 0→0 volume, Jan-Dec 2025, 30% vol
# Buy low (inject), sell high (withdraw), start and end empty
valDate      = pd.Timestamp("2026-01-01")
storageStart = pd.Timestamp("2026-04-01")
storageEnd   = pd.Timestamp("2027-03-30")
inj_days     = 30
wdr_days     = 30
vol          = 0.30

s = Storage(valDate, storageStart, storageEnd, curve=curve, n_p=0, v_step=1000, sVol=vol)
n = len(s.date_span)

s.i_curve = np.ones(n);  s.i_curve[s._active:] = 0.;  s.i_curve[:s.Dt] = 0.
s.w_curve = np.ones(n);  s.w_curve[s._active:] = 0.;  s.w_curve[:s.Dt] = 0.
s.i_cost[:] = 0.5
s.w_cost[:] = 0.5

s.set_volume_states(inj_days)
s.n_op_start = 0
s.t_p_curve  = np.full(s.n_op + 2, -1e9)
s.t_p_curve[0] = 0.

max_vol = inj_days * s.v_step

# ── Intrinsic ────────────────────────────────────────────────────────────────
s.build()
intrinsic_eur = s.v[0, 0, 0]
intr_exp_ex   = np.array(s.exp_ex)

# ── Extrinsic ────────────────────────────────────────────────────────────────
s.n_p = 30
s.build()
total_eur = s.v[0, s.n_p, 0]
extr_eur  = total_eur - intrinsic_eur

print(f"Intrinsic value  = {intrinsic_eur:,.0f} EUR  ({intrinsic_eur/max_vol:.2f} EUR/MWh)")
print(f"Extrinsic value  = {extr_eur:,.0f} EUR  ({extr_eur/max_vol:.2f} EUR/MWh)")
print(f"Total value      = {total_eur:,.0f} EUR  ({total_eur/max_vol:.2f} EUR/MWh)")



# ---- Swing_new.ipynb cell 10 ----

dates  = s.date_span[s.Dt:s._active]
prices = s.price_curve[s.Dt:s._active]
ex     = intr_exp_ex[s.Dt:s._active]

fig, ax1 = plt.subplots(figsize=(14, 4))
ax2 = ax1.twinx()

ax1.plot(dates, prices, 'k-', lw=1.5, label='Forward price')
ax2.bar(dates, np.where(ex > 0,  ex, 0), width=1, color='green', alpha=0.7, label='Withdraw (sell)')
ax2.bar(dates, np.where(ex < 0,  ex, 0), width=1, color='red',   alpha=0.7, label='Inject (buy)')

ax1.set_ylabel('EUR/MWh')
ax2.set_ylabel('MWh/day')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Storage intrinsic profile — forward price vs exercise')
plt.tight_layout()
plt.show()


==============================================================================
# forward.ipynb — source cells as they stood in the Drive review copy
# Outputs and metadata omitted. Preserved under section 5 of the
# reconciliation procedure (old-only analytical cells).
==============================================================================


# ---- forward.ipynb cell 0 ----



# ---- forward.ipynb cell 1 ----
import importlib
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import clear_output, display

import storage_model
importlib.reload(storage_model)
from storage_model import Storage
from pathlib import Path



# ---- forward.ipynb cell 3 ----
# Read TTF forward curve matrix
QUOTE_FILE = "ttf q.xlsx"
quote_path = Path.cwd() / QUOTE_FILE
if not quote_path.exists():
    matches = sorted(Path.cwd().glob("*ttf*q*.xlsx"))
    if matches:
        quote_path = matches[0]
    else:
        raise FileNotFoundError(f"Could not find {QUOTE_FILE!r} in {Path.cwd()}")

quotes = pd.read_excel(str(quote_path))
quotes = quotes.rename(columns={quotes.columns[0]: "quote_date"})
quotes = quotes.dropna(subset=["quote_date"]).copy()
quotes["quote_date"] = pd.to_datetime(quotes["quote_date"], format="mixed")
quotes = quotes.sort_values("quote_date").reset_index(drop=True)

contract_columns = sorted(
    [c for c in quotes.columns if re.fullmatch(r"TTFc\d+", str(c))],
    key=lambda c: int(re.search(r"\d+", str(c)).group()),
)

print(
    f"Loaded {len(quotes):,} quote dates and {len(contract_columns)} monthly forward contracts "
    f"from {quote_path}: {quotes['quote_date'].min():%Y-%m-%d} to {quotes['quote_date'].max():%Y-%m-%d}."
)
display(quotes.tail())



# ---- forward.ipynb cell 4 ----
# Build monthly and daily curves from the selected quote date
# The new file structure is quote_date + DA + TTFc1..TTFc55.
# TTFc1 is treated as the front-month contract, i.e. the next calendar month after quote_date.
# TTFc2, TTFc3, ... are consecutive monthly contracts.
# DA is included as the first daily anchor at quote_date before smoothing.
# Daily curves are smoothed with storage_model.smoothen_curve, the same PCHIP algorithm used by Storage.

def _month_start(ts):
    ts = pd.Timestamp(ts)
    return pd.Timestamp(ts.year, ts.month, 1)


def _month_end(ts):
    return _month_start(ts) + pd.offsets.MonthEnd(0)


def _front_month_start(quote_date):
    return _month_start(quote_date) + pd.DateOffset(months=1)


def selected_quote_row(quote_date):
    quote_date = pd.Timestamp(quote_date)
    return quotes.loc[quotes["quote_date"].eq(quote_date)].iloc[0]


def monthly_curve_from_quote(row):
    front_month = _front_month_start(row["quote_date"])
    data = []

    for col in contract_columns:
        value = row[col]
        if pd.isna(value):
            continue
        contract_number = int(re.search(r"\d+", col).group())
        contract_start = front_month + pd.DateOffset(months=contract_number - 1)
        data.append((contract_start, float(value), col))

    if not data:
        return pd.Series(dtype=float, name="value"), pd.DataFrame(columns=["contract", "contractStart", "contractEnd", "value"])

    curve_df = pd.DataFrame(data, columns=["contractStart", "value", "contract"])
    curve_df["contractEnd"] = curve_df["contractStart"].map(_month_end)
    curve_df = curve_df[["contract", "contractStart", "contractEnd", "value"]]

    monthly = curve_df.set_index("contractStart")["value"].sort_index().rename("value")
    return monthly, curve_df


def stepped_daily_curve_from_monthly(monthly):
    if monthly.empty:
        return pd.Series(dtype=float, name="value")
    day_index = pd.date_range(monthly.index.min(), _month_end(monthly.index.max()), freq="D")
    daily = pd.Series(index=day_index, dtype=float, name="value")
    for month_start, value in monthly.items():
        mask = (daily.index >= month_start) & (daily.index <= _month_end(month_start))
        daily.loc[mask] = value
    return daily


def daily_curve_from_monthly(monthly, quote_date=None, da_value=np.nan):
    stepped_daily = stepped_daily_curve_from_monthly(monthly)

    if pd.notna(da_value) and quote_date is not None and not stepped_daily.empty:
        quote_date = pd.Timestamp(quote_date)
        daily_index = pd.date_range(quote_date, stepped_daily.index.max(), freq="D")
        anchored = pd.Series(index=daily_index, dtype=float, name="value")
        anchored.loc[quote_date] = float(da_value)
        anchored.loc[stepped_daily.index] = stepped_daily.values
        stepped_daily = anchored.ffill()

    if stepped_daily.empty:
        return stepped_daily
    return storage_model.smoothen_curve(stepped_daily).rename("value")


def plot_forward_curve(quote_date, curve_view):
    plt.close("all")
    row = selected_quote_row(quote_date)
    monthly, curve_df = monthly_curve_from_quote(row)
    spot = row.get("DA", np.nan)
    daily = daily_curve_from_monthly(monthly, row["quote_date"], spot)

    if monthly.empty:
        print("No usable TTFc monthly quotes for this date.")
        return

    spot_label = f"DA: {spot:.3f}" if pd.notna(spot) else "DA: n/a"

    if curve_view == "Monthly":
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(monthly.index, monthly.values, marker="o", linewidth=2, label="Monthly forwards")
        if pd.notna(spot):
            ax.scatter([pd.Timestamp(quote_date)], [spot], color="tab:red", zorder=3, label=spot_label)
        ax.set_title(f"TTF monthly forward curve | {pd.Timestamp(quote_date):%Y-%m-%d}")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()
        display(curve_df.head(24))
        return

    if curve_view == "Daily":
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(daily.index, daily.values, linewidth=2, label="Smoothed daily curve incl. DA")
        if pd.notna(spot):
            ax.scatter([pd.Timestamp(quote_date)], [spot], color="tab:red", zorder=3, label=spot_label)
        ax.set_title(f"TTF smoothed daily forward curve | {pd.Timestamp(quote_date):%Y-%m-%d}")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    axes[0].plot(monthly.index, monthly.values, marker="o", linewidth=2, label="Monthly forwards")
    if pd.notna(spot):
        axes[0].scatter([pd.Timestamp(quote_date)], [spot], color="tab:red", zorder=3, label=spot_label)
    axes[0].set_title(f"TTF monthly forward curve | {pd.Timestamp(quote_date):%Y-%m-%d}")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(daily.index, daily.values, linewidth=2, label="Smoothed daily curve incl. DA")
    if pd.notna(spot):
        axes[1].scatter([pd.Timestamp(quote_date)], [spot], color="tab:red", zorder=3, label=spot_label)
    axes[1].set_title("Smoothed daily curve")
    axes[1].set_ylabel("Price")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# Set these and rerun this cell to draw exactly one forward-curve view.
forward_curve_date = quotes["quote_date"].iloc[-1]
curve_view = "Both"  # "Both", "Monthly", or "Daily"

clear_output(wait=True)
plt.close("all")
plot_forward_curve(forward_curve_date, curve_view)



# ---- forward.ipynb cell 5 ----
# Value put swing, call swing, or storage from a changeable historical forward curve date.
# Set product_type to one of: "put_swing", "call_swing", "storage".

product_type = "call_swing"

FDDate       = pd.Timestamp("2026-01-10")
valDate      = pd.Timestamp("2026-01-01")
storageStart = pd.Timestamp("2026-04-01")
storageEnd   = pd.Timestamp("2027-03-30")
days         = 30       # used by put_swing and call_swing
vol          = 0.60
n_p_full     = 30
run_intrinsic = True   # Set False to skip the profiled intrinsic build/decomposition.
v_step       = 1000

# Used only when product_type == "storage"
inj_days     = 30
wdr_days     = 30
inj_cost     = 0.5
wdr_cost     = 0.5

def quote_row_for_fd_date(fd_date, exact=False):
    fd_date = pd.Timestamp(fd_date)
    if exact:
        eligible = quotes.loc[quotes["quote_date"].eq(fd_date)]
    else:
        eligible = quotes.loc[quotes["quote_date"] <= fd_date]
    eligible = eligible.loc[eligible[contract_columns].notna().any(axis=1)]
    if eligible.empty:
        raise ValueError(f"No forward quote available for FDDate {fd_date:%Y-%m-%d}")
    return eligible.iloc[-1]


def curve_df_for_storage(row, curve_start=None, include_da=True):
    monthly, curve_df = monthly_curve_from_quote(row)
    curve_df = curve_df.copy()

    if not monthly.empty:
        full_months = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
        monthly = monthly.reindex(full_months).interpolate(method="time").ffill().bfill()
        curve_df = pd.DataFrame({
            "contract": [f"TTFc{i + 1}" for i in range(len(monthly))],
            "contractStart": monthly.index,
            "value": monthly.values,
        })
        curve_df["contractEnd"] = curve_df["contractStart"].map(_month_end)
        curve_df = curve_df[["contract", "contractStart", "contractEnd", "value"]]

    quote_date = pd.Timestamp(row["quote_date"])
    curve_start = pd.Timestamp(curve_start) if curve_start is not None else quote_date

    # Use DA as the front stub so Storage has a finite curve before TTFc1.
    if include_da and pd.notna(row.get("DA", np.nan)):
        front_month = _front_month_start(quote_date)
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
        raise ValueError("Storage curve contains missing values after interpolation/stub fill.")
    return result


def _active_masks(model):
    n = len(model.date_span)
    active = np.ones(n)
    active[model._active:] = 0.0
    active[:model.Dt] = 0.0
    return n, active


def _daily_arithmetic_flat_metric(model):
    exercise_dates = model.date_span[model.Dt:model._active]
    return float(pd.Series(model.price_curve, index=model.date_span).loc[exercise_dates].mean())


def value_put_swing(curve):
    # BUY/inject right: value is saving from choosing low-price days/scenarios.
    s = Storage(valDate, storageStart, storageEnd, curve=curve, n_p=0, v_step=v_step, sVol=vol)
    n, active = _active_masks(s)
    s.i_curve = active.copy()
    s.w_curve = np.zeros(n)

    flat_metric = _daily_arithmetic_flat_metric(s)

    s.set_volume_states(days)
    s.n_op_start = 0
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    s.t_p_curve[days] = 0.0
    if run_intrinsic:
        s.build()
        profiled_eur = s.v[0, 0, 0]
        acq = -np.sum(s.delta)
        profiled_metric = s.profiled()
        intrinsic = flat_metric - profiled_metric
        intrinsic_profile_raw = -np.array(s.exp_ex)  # positive buy volume
    else:
        profiled_eur = np.nan
        acq = np.nan
        profiled_metric = np.nan
        intrinsic = np.nan
        intrinsic_profile_raw = np.zeros(len(s.date_span))

    s.n_p = n_p_full
    s.build()
    full_eur = s.v[0, s.n_p, 0]
    full_acq = np.sum(s.delta)
    stochastic_metric = full_eur / full_acq
    extrinsic = (full_eur - profiled_eur) / acq if run_intrinsic else np.nan
    extrinsic_profile_raw = -np.array(s.exp_ex)

    return s, {
        "flat_metric": flat_metric,
        "profiled_metric": profiled_metric,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "total": intrinsic + extrinsic if run_intrinsic else stochastic_metric,
        "stochastic_metric": stochastic_metric,
        "profile_label": "Expected buy offtake (MWh/day)",
        "title_prefix": "Put swing",
        "intrinsic_profile_raw": intrinsic_profile_raw,
        "extrinsic_profile_raw": extrinsic_profile_raw,
    }


def value_call_swing(curve):
    # SELL/withdraw right: value is uplift from choosing high-price days/scenarios.
    s = Storage(valDate, storageStart, storageEnd, curve=curve, n_p=0, v_step=v_step, sVol=vol)
    flat_metric = _daily_arithmetic_flat_metric(s)

    s.set_volume_states(days)  # start with exactly `days` sell clips available; terminal state 0 is enforced.
    if run_intrinsic:
        s.build()
        profiled_eur = s.v[0, 0, s.n_op_start]
        acq = np.sum(s.delta)
        profiled_metric = s.profiled()
        intrinsic = profiled_metric - flat_metric
        intrinsic_profile_raw = np.array(s.exp_ex)  # positive sell volume
    else:
        profiled_eur = np.nan
        acq = np.nan
        profiled_metric = np.nan
        intrinsic = np.nan
        intrinsic_profile_raw = np.zeros(len(s.date_span))

    s.n_p = n_p_full
    s.build()
    full_eur = s.v[0, s.n_p, s.n_op_start]
    full_acq = np.sum(s.delta)
    stochastic_metric = full_eur / full_acq
    extrinsic = (full_eur - profiled_eur) / acq if run_intrinsic else np.nan
    extrinsic_profile_raw = np.array(s.exp_ex)

    return s, {
        "flat_metric": flat_metric,
        "profiled_metric": profiled_metric,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "total": intrinsic + extrinsic if run_intrinsic else stochastic_metric,
        "stochastic_metric": stochastic_metric,
        "profile_label": "Expected sell offtake (MWh/day)",
        "title_prefix": "Call swing",
        "intrinsic_profile_raw": intrinsic_profile_raw,
        "extrinsic_profile_raw": extrinsic_profile_raw,
    }


def value_storage(curve):
    # Storage: buy low, sell high, start and end empty.
    if inj_days != wdr_days:
        print("Note: storage uses a one-clip daily injection/withdrawal grid; wdr_days is informational unless it equals inj_days.", flush=True)
    s = Storage(valDate, storageStart, storageEnd, curve=curve, n_p=0, v_step=v_step, sVol=vol)
    n, active = _active_masks(s)
    s.i_curve = active.copy()
    s.w_curve = active.copy()
    s.i_cost[:] = inj_cost
    s.w_cost[:] = wdr_cost
    s.set_volume_states(inj_days)
    s.n_op_start = 0
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    s.t_p_curve[0] = 0.0

    max_vol = inj_days * s.v_step
    if run_intrinsic:
        print("Building storage intrinsic case (first run may spend ~20-40s compiling Numba kernels)...", flush=True)
        t0 = time.perf_counter()
        s.build()
        print(f"Storage intrinsic build complete in {time.perf_counter() - t0:.1f}s", flush=True)
        intrinsic_eur = s.v[0, 0, 0]
        intrinsic_profile_raw = np.array(s.exp_ex)  # positive sell, negative buy
    else:
        intrinsic_eur = np.nan
        intrinsic_profile_raw = np.zeros(len(s.date_span))

    s.n_p = n_p_full
    print(f"Building storage full tree with n_p={n_p_full}...", flush=True)
    t0 = time.perf_counter()
    s.build()
    print(f"Storage full-tree build complete in {time.perf_counter() - t0:.1f}s", flush=True)
    total_eur = s.v[0, s.n_p, 0]
    extrinsic_eur = total_eur - intrinsic_eur if run_intrinsic else np.nan
    extrinsic_profile_raw = np.array(s.exp_ex)

    return s, {
        "flat_metric": np.nan,
        "profiled_metric": intrinsic_eur / max_vol if run_intrinsic else np.nan,
        "intrinsic": intrinsic_eur / max_vol if run_intrinsic else np.nan,
        "extrinsic": extrinsic_eur / max_vol if run_intrinsic else np.nan,
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


def full_value_eur(curve_input):
    if product_type == "put_swing":
        sb = Storage(valDate, storageStart, storageEnd, curve=curve_input, n_p=n_p_full, v_step=v_step, sVol=vol)
        n, active = _active_masks(sb)
        sb.i_curve = active.copy()
        sb.w_curve = np.zeros(n)
        sb.set_volume_states(days)
        sb.n_op_start = 0
        sb.t_p_curve = np.full(sb.n_op + 2, -1e9)
        sb.t_p_curve[days] = 0.0
        sb.build()
        return sb.v[0, sb.n_p, 0]

    if product_type == "call_swing":
        sb = Storage(valDate, storageStart, storageEnd, curve=curve_input, n_p=n_p_full, v_step=v_step, sVol=vol)
        sb.set_volume_states(days)
        sb.build()
        return sb.v[0, sb.n_p, sb.n_op_start]

    if product_type == "storage":
        sb = Storage(valDate, storageStart, storageEnd, curve=curve_input, n_p=n_p_full, v_step=v_step, sVol=vol)
        n, active = _active_masks(sb)
        sb.i_curve = active.copy()
        sb.w_curve = active.copy()
        sb.i_cost[:] = inj_cost
        sb.w_cost[:] = wdr_cost
        sb.set_volume_states(inj_days)
        sb.n_op_start = 0
        sb.t_p_curve = np.full(sb.n_op + 2, -1e9)
        sb.t_p_curve[0] = 0.0
        sb.build()
        return sb.v[0, sb.n_p, 0]

    raise ValueError('product_type must be "put_swing", "call_swing", or "storage"')


fd_quote = quote_row_for_fd_date(FDDate, exact=False)
curve = curve_df_for_storage(fd_quote, curve_start=valDate, include_da=True)

print(f"Product type:                 {product_type}")
print(f"Forward curve date requested: {FDDate:%Y-%m-%d}")
print(f"Forward curve quote used:     {fd_quote['quote_date']:%Y-%m-%d}")
print(f"Curve covers:                 {curve['contractStart'].min():%Y-%m-%d} to {curve['contractEnd'].max():%Y-%m-%d}")

if product_type == "put_swing":
    s, result = value_put_swing(curve)
elif product_type == "call_swing":
    s, result = value_call_swing(curve)
elif product_type == "storage":
    s, result = value_storage(curve)
else:
    raise ValueError('product_type must be "put_swing", "call_swing", or "storage"')

print(f"Flat metric       = {result['flat_metric']:.4f}" if pd.notna(result["flat_metric"]) else "Flat metric       = n/a")
print(f"Profiled metric   = {result['profiled_metric']:.4f}")
print(f"Intrinsic value   = {result['intrinsic']:.4f}")
print(f"Extrinsic value   = {result['extrinsic']:.4f}")
print(f"Total value       = {result['total']:.4f}" if run_intrinsic else f"Stochastic metric = {result['stochastic_metric']:.4f}")
if product_type == "storage":
    print(f"Intrinsic EUR     = {result['intrinsic_eur']:,.0f}")
    print(f"Extrinsic EUR     = {result['extrinsic_eur']:,.0f}")
    print(f"Total EUR         = {result['total_eur']:,.0f}")

exercise_dates = s.date_span[s.Dt:s._active]
plot_prices = pd.Series(s.price_curve, index=s.date_span)
arithmetic_flat_metric = (
    float(plot_prices.loc[exercise_dates].mean())
    if product_type in ("put_swing", "call_swing")
    else np.nan
)
flat_metric_difference = result["flat_metric"] - arithmetic_flat_metric if pd.notna(result["flat_metric"]) else np.nan

swing_summary = pd.DataFrame({
    "metric": [
        "product_type",
        "FDDate_requested",
        "forward_quote_used",
        "flat_metric",
        "profiled_metric",
        "arithmetic_flat_metric",
        "flat_minus_arithmetic",
        "intrinsic_value",
        "extrinsic_value",
        "total_value",
    ],
    "value": [
        product_type,
        FDDate,
        fd_quote["quote_date"],
        result["flat_metric"],
        result["profiled_metric"],
        arithmetic_flat_metric,
        flat_metric_difference,
        result["intrinsic"],
        result["extrinsic"],
        result["total"],
    ],
})
display(swing_summary)

# Expected exercise/offtake profiles vs forward curve
exercise_dates = s.date_span[s.Dt:s._active]
delta_dates = s.date_span[:s.n_t]
plot_prices = pd.Series(s.price_curve, index=s.date_span)
intrinsic_profile = pd.Series(result["intrinsic_profile_raw"][:len(s.date_span)], index=s.date_span).loc[exercise_dates]
extrinsic_profile = pd.Series(result["extrinsic_profile_raw"][:len(s.date_span)], index=s.date_span).loc[exercise_dates]
extrinsic_delta_profile = pd.Series(np.array(s.delta[:s.n_t], dtype=float), index=delta_dates)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax, profile, title, color in [
    (axes[0], intrinsic_profile, f"{result['title_prefix']} intrinsic expected exercise vs forward curve", "tab:red"),
    (axes[1], extrinsic_delta_profile, f"{result['title_prefix']} extrinsic delta vs forward curve", "tab:purple"),
]:
    dates = exercise_dates if ax is axes[0] else delta_dates
    ax_price = ax.twinx()
    if product_type == "storage":
        ax.bar(dates, np.where(profile.values > 0, profile.values, 0), width=1.0, color="tab:green", alpha=0.65, label="Expected sell" if ax is axes[0] else "Positive delta")
        ax.bar(dates, np.where(profile.values < 0, profile.values, 0), width=1.0, color="tab:red", alpha=0.65, label="Expected buy" if ax is axes[0] else "Negative delta")
    else:
        ax.bar(dates, profile.values, width=1.0, color=color, alpha=0.65, label="Expected offtake" if ax is axes[0] else "Delta")
    ax_price.plot(dates, plot_prices.loc[dates].values, color="black", linewidth=1.6, label="Forward curve")

    ax.set_title(title)
    ax.set_ylabel(result["profile_label"] if ax is axes[0] else "Delta (MWh/day)")
    ax_price.set_ylabel("Forward price")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(storageStart, storageEnd)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_price.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

plt.tight_layout()
plt.show()
print("Run finished. If VS Code says outputs are collapsed, expand the cell output to see the table and charts.", flush=True)



# ---- forward.ipynb cell 6 ----
# Monthly extrinsic deltas
monthly_delta = extrinsic_delta_profile.loc[storageStart:storageEnd]
monthly_delta_by_period = monthly_delta.resample("MS").sum()
monthly_delta_table = pd.DataFrame({
    "period": monthly_delta_by_period.index.strftime("%b-%y"),
    "delta": monthly_delta_by_period.values,
})
monthly_delta_table = pd.concat([
    monthly_delta_table,
    pd.DataFrame([{"period": "Sum", "delta": monthly_delta_table["delta"].sum()}]),
], ignore_index=True)

display_monthly_delta_table = monthly_delta_table.copy()
display_monthly_delta_table["delta"] = display_monthly_delta_table["delta"].map("{:,.2f}".format)
display(display_monthly_delta_table)


==============================================================================
# pricing.ipynb — source cells as they stood in the Drive review copy
# Outputs and metadata omitted. Preserved under section 5 of the
# reconciliation procedure (old-only analytical cells).
==============================================================================


# ---- pricing.ipynb cell 0 ----
import importlib
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import storage_model
importlib.reload(storage_model)
from storage_model import (
    Storage,
    run_valuation,
    curve_df_for_storage,
    quote_row_for_fd_date,
    active_masks,
    daily_arithmetic_flat_metric,
    smoothen_curve,
    monthly_curve_from_quote,
)


# ---- pricing.ipynb cell 2 ----
# ── CURVE SOURCE ──────────────────────────────────────────────────────────────
# "ttf_excel" : reads from a TTF-style Excel file (quote_date + DA + TTFc1..N)
# "csv"       : reads from a CSV with columns contractStart, contractEnd, value
# "manual"    : define the curve inline as a list of dicts below
curve_source = "ttf_excel"

# -- ttf_excel settings -------------------------------------------------------
ttf_file = "ttf q.xlsx"          # path relative to this notebook
fd_date  = pd.Timestamp("2026-01-10")   # use the last available quote on or before this date

# -- csv settings -------------------------------------------------------------
csv_file = "curve.csv"           # must have columns: contractStart, contractEnd, value

# -- manual settings ----------------------------------------------------------
# Each row: {"contractStart": "YYYY-MM-DD", "contractEnd": "YYYY-MM-DD", "value": <float>}
manual_curve_rows = []


# ---- pricing.ipynb cell 3 ----
# ── PRODUCT PARAMETERS ───────────────────────────────────────────────────────
product_type  = "call_swing"   # "put_swing" | "call_swing" | "storage"

val_date      = pd.Timestamp("2026-01-01")
storage_start = pd.Timestamp("2026-04-01")
storage_end   = pd.Timestamp("2027-03-30")

days          = 30     # exercise days / max capacity in clips (put_swing / call_swing)
vol           = 0.60   # daily spot vol
n_p_full      = 30     # price tree half-width
run_intrinsic = True   # set False to skip intrinsic build (faster, no decomposition)
v_step        = 1000   # MWh per clip

# Initial and terminal inventory in clips (1 clip = v_step MWh).
# None = product default: put_swing(init=0, term=days), call_swing(init=days, term=0), storage(init=0, term=0)
initial_inv_clips  = None
terminal_inv_clips = None

# call_swing only:
strike       = 0.0    # fixed exercise price (0 = exercise at forward; e.g. 40.0 for K=40 call)
zero_penalty = False  # True = unused clips carry no penalty at expiry (right, not obligation)

# storage-only
inj_days = 30
wdr_days = 30
inj_cost = 0.5
wdr_cost = 0.5

params = dict(
    product_type       = product_type,
    valDate            = val_date,
    storageStart       = storage_start,
    storageEnd         = storage_end,
    days               = days,
    vol                = vol,
    n_p_full           = n_p_full,
    run_intrinsic      = run_intrinsic,
    v_step             = v_step,
    initial_inv_clips  = initial_inv_clips,
    terminal_inv_clips = terminal_inv_clips,
    strike             = strike,
    zero_penalty       = zero_penalty,
    inj_days           = inj_days,
    wdr_days           = wdr_days,
    inj_cost           = inj_cost,
    wdr_cost           = wdr_cost,
)


# ---- pricing.ipynb cell 5 ----
def _load_ttf_excel(path, fd_date, val_date):
    """Load TTF-style Excel (quote_date + DA + TTFc1..N) and return a storage curve."""
    path = Path(path)
    if not path.exists():
        matches = sorted(Path.cwd().glob("*ttf*q*.xlsx"))
        if not matches:
            raise FileNotFoundError(f"Could not find {path}")
        path = matches[0]

    quotes = pd.read_excel(str(path))
    quotes = quotes.rename(columns={quotes.columns[0]: "quote_date"})
    quotes = quotes.dropna(subset=["quote_date"]).copy()
    quotes["quote_date"] = pd.to_datetime(quotes["quote_date"], format="mixed")
    quotes = quotes.sort_values("quote_date").reset_index(drop=True)

    contract_cols = sorted(
        [c for c in quotes.columns if re.fullmatch(r"TTFc\d+", str(c))],
        key=lambda c: int(re.search(r"\d+", str(c)).group()),
    )
    print(f"Loaded {len(quotes):,} quote dates, {len(contract_cols)} contracts from {path.name}")
    print(f"  Range: {quotes['quote_date'].min():%Y-%m-%d} to {quotes['quote_date'].max():%Y-%m-%d}")

    row = quote_row_for_fd_date(quotes, contract_cols, fd_date, exact=False)
    print(f"  FDDate requested: {pd.Timestamp(fd_date):%Y-%m-%d}  |  quote used: {row['quote_date']:%Y-%m-%d}")

    curve = curve_df_for_storage(row, contract_cols, curve_start=val_date, include_da=True)
    return curve, row


def _load_csv(path):
    """Load a CSV with columns contractStart, contractEnd, value."""
    df = pd.read_csv(path)
    df["contractStart"] = pd.to_datetime(df["contractStart"], format="mixed")
    df["contractEnd"]   = pd.to_datetime(df["contractEnd"],   format="mixed")
    df["value"]         = df["value"].astype(float)
    return df[["contractStart", "contractEnd", "value"]]


def _load_manual(rows):
    """Build a curve from a list of dicts with contractStart, contractEnd, value."""
    df = pd.DataFrame(rows)
    df["contractStart"] = pd.to_datetime(df["contractStart"])
    df["contractEnd"]   = pd.to_datetime(df["contractEnd"])
    df["value"]         = df["value"].astype(float)
    return df[["contractStart", "contractEnd", "value"]]


# ── load ──────────────────────────────────────────────────────────────────────
fd_quote_row = None

if curve_source == "ttf_excel":
    curve, fd_quote_row = _load_ttf_excel(ttf_file, fd_date, val_date)
elif curve_source == "csv":
    curve = _load_csv(csv_file)
elif curve_source == "manual":
    curve = _load_manual(manual_curve_rows)
else:
    raise ValueError(f"Unknown curve_source {curve_source!r}")

print(f"\nCurve covers: {curve['contractStart'].min():%Y-%m-%d} to {curve['contractEnd'].max():%Y-%m-%d}")
display(curve.head(6))


# ---- pricing.ipynb cell 6 ----
# ── Plot forward curve ────────────────────────────────────────────────────────
tmp = Storage(val_date, storage_start, storage_end, curve=curve, n_p=0, v_step=v_step, sVol=vol)
plot_curve = pd.Series(tmp.price_curve, index=tmp.date_span)

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(plot_curve.index, plot_curve.values, linewidth=1.8, label="Smoothed daily curve")
ax.axvspan(storage_start, storage_end, alpha=0.08, color="tab:blue", label="Exercise window")
ax.set_title(f"Forward curve | {curve_source}" + (f" | quote {fd_quote_row['quote_date']:%Y-%m-%d}" if fd_quote_row is not None else ""))
ax.set_ylabel("Price")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()


# ---- pricing.ipynb cell 8 ----
print(f"Running {product_type} valuation...")
t0 = time.perf_counter()

s, result = run_valuation(curve, params)

print(f"Done in {time.perf_counter() - t0:.1f}s")


# ---- pricing.ipynb cell 10 ----
exercise_dates = s.date_span[s.Dt:s._active]
price_series   = pd.Series(s.price_curve, index=s.date_span)
arith_flat     = float(price_series.loc[exercise_dates].mean()) if product_type in ("put_swing", "call_swing") else np.nan

summary = pd.DataFrame([
    ("product_type",        product_type),
    ("val_date",            val_date.date()),
    ("storage_start",       storage_start.date()),
    ("storage_end",         storage_end.date()),
    ("flat_metric",         result["flat_metric"]),
    ("arith_flat_metric",   arith_flat),
    ("profiled_metric",     result["profiled_metric"]),
    ("intrinsic_value",     result["intrinsic"]),
    ("extrinsic_value",     result["extrinsic"]),
    ("total_value",         result["total"]),
], columns=["metric", "value"])

if product_type == "storage":
    summary = pd.concat([
        summary,
        pd.DataFrame([
            ("intrinsic_EUR", result.get("intrinsic_eur", np.nan)),
            ("extrinsic_EUR", result.get("extrinsic_eur", np.nan)),
            ("total_EUR",     result.get("total_eur",     np.nan)),
        ], columns=["metric", "value"]),
    ], ignore_index=True)

display(summary)


# ---- pricing.ipynb cell 12 ----
delta_dates = s.date_span[:s.n_t]
intrinsic_profile  = pd.Series(result["intrinsic_profile_raw"][:len(s.date_span)],  index=s.date_span).loc[exercise_dates]
extrinsic_delta    = pd.Series(np.array(s.delta[:s.n_t], dtype=float), index=delta_dates)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)

for ax, profile, dates, title, color in [
    (axes[0], intrinsic_profile, exercise_dates,
     f"{result['title_prefix']} — intrinsic expected exercise vs forward curve", "tab:red"),
    (axes[1], extrinsic_delta,   delta_dates,
     f"{result['title_prefix']} — extrinsic delta vs forward curve", "tab:purple"),
]:
    ax_price = ax.twinx()
    if product_type == "storage":
        ax.bar(dates, np.where(profile.values > 0, profile.values, 0),
               width=1.0, color="tab:green", alpha=0.65, label="Sell / +delta")
        ax.bar(dates, np.where(profile.values < 0, profile.values, 0),
               width=1.0, color="tab:red",   alpha=0.65, label="Buy / -delta")
    else:
        ax.bar(dates, profile.values, width=1.0, color=color, alpha=0.65, label="Expected offtake")

    ax_price.plot(dates, price_series.loc[dates].values,
                  color="black", linewidth=1.6, label="Forward curve")
    ax.set_title(title)
    ax.set_ylabel(result["profile_label"] if ax is axes[0] else "Delta (MWh/day)")
    ax_price.set_ylabel("Forward price")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(storage_start, storage_end)

    lines  = ax.get_legend_handles_labels()
    lines2 = ax_price.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper right")

plt.tight_layout()
plt.show()


# ---- pricing.ipynb cell 14 ----
monthly_delta = extrinsic_delta.loc[storage_start:storage_end].resample("MS").sum()
monthly_table = pd.DataFrame({
    "period": monthly_delta.index.strftime("%b-%y"),
    "delta":  monthly_delta.values,
})
monthly_table = pd.concat([
    monthly_table,
    pd.DataFrame([{"period": "Sum", "delta": monthly_table["delta"].sum()}]),
], ignore_index=True)

disp = monthly_table.copy()
disp["delta"] = disp["delta"].map("{:,.2f}".format)
display(disp)
