from pathlib import Path
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from storage_model import Storage


st.set_page_config(page_title="Swing / Storage Valuation", layout="wide")


def month_start(ts):
    ts = pd.Timestamp(ts)
    return pd.Timestamp(ts.year, ts.month, 1)


def month_end(ts):
    return month_start(ts) + pd.offsets.MonthEnd(0)


def front_month_start(quote_date):
    return month_start(quote_date) + pd.DateOffset(months=1)


@st.cache_data(show_spinner=False)
def load_quote_matrix(source):
    if source is None:
        path = Path("ttf q.xlsx")
        if not path.exists():
            raise FileNotFoundError("Could not find local 'ttf q.xlsx'. Upload a forward curve file instead.")
        quotes = pd.read_excel(path)
    else:
        name = source.name.lower()
        if name.endswith(".csv"):
            quotes = pd.read_csv(source)
        else:
            quotes = pd.read_excel(source)

    quotes = quotes.rename(columns={quotes.columns[0]: "quote_date"})
    quotes = quotes.dropna(subset=["quote_date"]).copy()
    quotes["quote_date"] = pd.to_datetime(quotes["quote_date"], format="mixed")
    quotes = quotes.sort_values("quote_date").reset_index(drop=True)
    contract_columns = sorted(
        [c for c in quotes.columns if re.fullmatch(r"TTFc\d+", str(c))],
        key=lambda c: int(re.search(r"\d+", str(c)).group()),
    )
    return quotes, contract_columns


@st.cache_data(show_spinner=False)
def load_direct_curve(source):
    if source is None:
        path = Path("curve.csv")
        if not path.exists():
            raise FileNotFoundError("Could not find local 'curve.csv'. Upload a curve file instead.")
        curve = pd.read_csv(path)
    else:
        name = source.name.lower()
        if name.endswith(".csv"):
            curve = pd.read_csv(source)
        else:
            curve = pd.read_excel(source)

    required = {"contractStart", "contractEnd", "value"}
    missing = required.difference(curve.columns)
    if missing:
        raise ValueError(f"Direct curve file is missing columns: {', '.join(sorted(missing))}")
    curve = curve[["contractStart", "contractEnd", "value"]].copy()
    curve["contractStart"] = pd.to_datetime(curve["contractStart"], format="mixed")
    curve["contractEnd"] = pd.to_datetime(curve["contractEnd"], format="mixed")
    curve["value"] = pd.to_numeric(curve["value"])
    return curve.sort_values("contractStart").reset_index(drop=True)


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


def value_put_swing(curve, params):
    s = Storage(params["valDate"], params["storageStart"], params["storageEnd"], curve=curve, n_p=0, v_step=params["v_step"], sVol=params["vol"])
    n, active = active_masks(s)
    s.i_curve = active.copy()
    s.w_curve = np.zeros(n)

    full_cap = s.n_op_start
    s.n_op_start = 0
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    s.t_p_curve[full_cap] = 0.0
    s.build()
    flat_metric = s.flat()

    s.set_volume_states(params["days"])
    s.n_op_start = 0
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    s.t_p_curve[params["days"]] = 0.0
    s.build()
    profiled_eur = s.v[0, 0, 0]
    acq = -np.sum(s.delta)
    profiled_metric = s.profiled()
    intrinsic = flat_metric - profiled_metric
    intrinsic_profile_raw = -np.array(s.exp_ex)

    s.n_p = params["n_p_full"]
    s.build()
    full_eur = s.v[0, s.n_p, 0]
    extrinsic = (full_eur - profiled_eur) / acq
    extrinsic_profile_raw = -np.array(s.exp_ex)

    return s, {
        "flat_metric": flat_metric,
        "profiled_metric": profiled_metric,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "total": intrinsic + extrinsic,
        "profile_label": "Expected buy offtake (MWh/day)",
        "title_prefix": "Put swing",
        "intrinsic_profile_raw": intrinsic_profile_raw,
        "extrinsic_profile_raw": extrinsic_profile_raw,
    }


def value_call_swing(curve, params):
    s = Storage(params["valDate"], params["storageStart"], params["storageEnd"], curve=curve, n_p=0, v_step=params["v_step"], sVol=params["vol"])
    s.build()
    flat_metric = s.flat()

    s.set_volume_states(params["days"])
    s.build()
    profiled_eur = s.v[0, 0, s.n_op_start]
    acq = np.sum(s.delta)
    profiled_metric = s.profiled()
    intrinsic = profiled_metric - flat_metric
    intrinsic_profile_raw = np.array(s.exp_ex)

    s.n_p = params["n_p_full"]
    s.build()
    full_eur = s.v[0, s.n_p, s.n_op_start]
    extrinsic = (full_eur - profiled_eur) / acq
    extrinsic_profile_raw = np.array(s.exp_ex)

    return s, {
        "flat_metric": flat_metric,
        "profiled_metric": profiled_metric,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "total": intrinsic + extrinsic,
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
    s.set_volume_states(params["inj_days"])
    s.n_op_start = 0
    s.t_p_curve = np.full(s.n_op + 2, -1e9)
    s.t_p_curve[0] = 0.0

    max_vol = params["inj_days"] * s.v_step
    s.build()
    intrinsic_eur = s.v[0, 0, 0]
    intrinsic_profile_raw = np.array(s.exp_ex)

    s.n_p = params["n_p_full"]
    s.build()
    total_eur = s.v[0, s.n_p, 0]
    extrinsic_eur = total_eur - intrinsic_eur
    extrinsic_profile_raw = np.array(s.exp_ex)

    return s, {
        "flat_metric": np.nan,
        "profiled_metric": intrinsic_eur / max_vol,
        "intrinsic": intrinsic_eur / max_vol,
        "extrinsic": extrinsic_eur / max_vol,
        "total": total_eur / max_vol,
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


def format_number(x):
    if pd.isna(x):
        return "n/a"
    return f"{x:,.4f}"


st.title("Swing / Storage Valuation")

with st.sidebar:
    st.header("Inputs")
    product_type = st.selectbox("Product type", ["put_swing", "call_swing", "storage"], index=0)

    FDDate = pd.Timestamp(st.date_input("FDDate", pd.Timestamp("2026-01-05")))
    valDate = pd.Timestamp(st.date_input("valDate", pd.Timestamp("2026-01-01")))
    storageStart = pd.Timestamp(st.date_input("storageStart", pd.Timestamp("2026-04-01")))
    storageEnd = pd.Timestamp(st.date_input("storageEnd", pd.Timestamp("2027-03-30")))

    days = st.number_input("days", min_value=1, max_value=3660, value=30, step=1)
    vol = st.number_input("vol", min_value=0.0, max_value=5.0, value=0.60, step=0.01, format="%.2f")
    n_p_full = st.number_input("n_p_full", min_value=0, max_value=100, value=30, step=1)
    v_step = st.number_input("v_step", min_value=1, max_value=1_000_000, value=1000, step=100)

    st.header("Storage inputs")
    inj_days = st.number_input("inj_days", min_value=1, max_value=3660, value=30, step=1)
    wdr_days = st.number_input("wdr_days", min_value=1, max_value=3660, value=30, step=1)
    inj_cost = st.number_input("inj_cost", value=0.5, step=0.1, format="%.2f")
    wdr_cost = st.number_input("wdr_cost", value=0.5, step=0.1, format="%.2f")

    st.header("Forward curve")
    curve_mode = st.radio("Curve source", ["TTF quote matrix", "Direct curve"], index=0)
    uploaded_curve = st.file_uploader("Upload xlsx/csv", type=["xlsx", "xls", "csv"])
    include_da = st.checkbox("Include DA front stub", value=True, disabled=(curve_mode != "TTF quote matrix"))

    run = st.button("Run valuation", type="primary")

if storageEnd < storageStart:
    st.error("storageEnd must be on or after storageStart.")
    st.stop()

params = {
    "product_type": product_type,
    "FDDate": FDDate,
    "valDate": valDate,
    "storageStart": storageStart,
    "storageEnd": storageEnd,
    "days": int(days),
    "vol": float(vol),
    "n_p_full": int(n_p_full),
    "v_step": int(v_step),
    "inj_days": int(inj_days),
    "wdr_days": int(wdr_days),
    "inj_cost": float(inj_cost),
    "wdr_cost": float(wdr_cost),
}

if not run:
    st.info("Set inputs in the sidebar, then run valuation.")
    st.stop()

try:
    if curve_mode == "TTF quote matrix":
        quotes, contract_columns = load_quote_matrix(uploaded_curve)
        fd_quote = quote_row_for_fd_date(quotes, contract_columns, FDDate, exact=False)
        curve = curve_df_for_storage(fd_quote, contract_columns, curve_start=valDate, include_da=include_da)
        curve_note = f"Quote used: {fd_quote['quote_date']:%Y-%m-%d}"
    else:
        curve = load_direct_curve(uploaded_curve)
        curve_note = "Direct curve file"
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"{curve_note}. Curve covers {curve['contractStart'].min():%Y-%m-%d} to {curve['contractEnd'].max():%Y-%m-%d}.")

with st.spinner("Running valuation. First run may compile Numba kernels..."):
    t0 = time.perf_counter()
    try:
        s, result = run_valuation(curve, params)
    except Exception as exc:
        st.exception(exc)
        st.stop()
    elapsed = time.perf_counter() - t0

st.success(f"Valuation complete in {elapsed:.1f}s")

summary = pd.DataFrame({
    "metric": ["flat_metric", "profiled_metric", "intrinsic_value", "extrinsic_value", "total_value"],
    "value": [
        result["flat_metric"],
        result["profiled_metric"],
        result["intrinsic"],
        result["extrinsic"],
        result["total"],
    ],
})
summary["value"] = summary["value"].map(format_number)

cols = st.columns(5)
for col, metric, value in zip(cols, summary["metric"], summary["value"]):
    col.metric(metric, value)

if product_type == "storage":
    st.dataframe(
        pd.DataFrame({
            "metric": ["intrinsic_eur", "extrinsic_eur", "total_eur"],
            "value": [result["intrinsic_eur"], result["extrinsic_eur"], result["total_eur"]],
        }).assign(value=lambda x: x["value"].map(lambda v: f"{v:,.0f}")),
        hide_index=True,
        use_container_width=True,
    )

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
st.pyplot(fig)

st.subheader("Monthly Deltas")
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
monthly_delta_display = monthly_delta_table.copy()
monthly_delta_display["delta"] = monthly_delta_display["delta"].map("{:,.2f}".format)
st.dataframe(monthly_delta_display, hide_index=True, use_container_width=True)

with st.expander("Forward curve used"):
    curve_display = curve.copy()
    curve_display["contractStart"] = curve_display["contractStart"].dt.strftime("%Y-%m-%d")
    curve_display["contractEnd"] = curve_display["contractEnd"].dt.strftime("%Y-%m-%d")
    st.dataframe(curve_display, hide_index=True, use_container_width=True)
