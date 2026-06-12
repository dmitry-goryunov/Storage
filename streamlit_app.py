from pathlib import Path
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from storage_model import (
    curve_df_for_storage,
    quote_row_for_fd_date,
    run_valuation,
    warm_numba_kernels as model_warm_numba_kernels,
)


st.set_page_config(page_title="Swing / Storage Valuation", layout="wide")


@st.cache_resource(show_spinner="Preparing Numba kernels...")
def warm_numba_kernels():
    return model_warm_numba_kernels()


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


@st.cache_data(show_spinner=False, max_entries=20)
def cached_run_valuation(curve, params):
    """Run the valuation but cache only what the UI needs.

    Returning the full Storage object would make st.cache_data pickle the
    v/strat/prob state cubes (8-130+ MB per parameter set) on every store and
    copy them on every retrieval. The payload below is < 1 MB.
    """
    s, result = run_valuation(curve, params)
    payload = {
        "date_span": s.date_span,
        "Dt": s.Dt,
        "active": s._active,
        "n_t": s.n_t,
        "price_curve": np.asarray(s.price_curve, dtype=float),
        "delta": np.asarray(s.delta, dtype=float),
    }
    return payload, result


def format_number(x):
    if pd.isna(x):
        return "n/a"
    return f"{x:,.4f}"


st.title("Swing / Storage Valuation")

with st.sidebar:
    with st.form("valuation_inputs"):
        st.header("Inputs")
        product_type = st.selectbox("Product type", ["put_swing", "call_swing", "storage"], index=0)

        FDDate = pd.Timestamp(st.date_input("FDDate", pd.Timestamp("2026-01-05")))
        valDate = pd.Timestamp(st.date_input("valDate", pd.Timestamp("2026-01-01")))
        storageStart = pd.Timestamp(st.date_input("storageStart", pd.Timestamp("2026-04-01")))
        storageEnd = pd.Timestamp(st.date_input("storageEnd", pd.Timestamp("2027-03-30")))

        days = st.number_input("days", min_value=1, max_value=3660, value=30, step=1)
        vol = st.number_input("vol", min_value=0.0, max_value=5.0, value=0.60, step=0.01, format="%.2f")
        n_p_full = st.number_input("n_p_full", min_value=0, max_value=100, value=10, step=1)
        run_intrinsic = st.checkbox("Run intrinsic decomposition", value=True)
        clips_per_day = st.number_input("clips_per_day", min_value=1, max_value=1000, value=3, step=1,
                                        help="Daily granularity / max clips moved per active day. With daily_max set, clip size = daily_max / clips_per_day.")
        daily_max = st.number_input("daily_max (MWh/day, 0 = use v_step)", min_value=0, max_value=10_000_000, value=0, step=100,
                                    help="Max MWh injected/withdrawn per active day. If > 0, clip size v_step is derived as daily_max / clips_per_day.")
        capacity_mwh = st.number_input("capacity_mwh (0 = use days/inj_days)", min_value=0, max_value=100_000_000, value=0, step=1000,
                                       help="Total working volume in MWh. If > 0, the number of inventory states is capacity_mwh / clip size.")
        v_step = st.number_input("v_step (used only when daily_max = 0)", min_value=1, max_value=1_000_000, value=1000, step=100)

        st.header("Storage inputs")
        inj_days = st.number_input("inj_days", min_value=1, max_value=3660, value=30, step=1)
        wdr_days = st.number_input("wdr_days", min_value=1, max_value=3660, value=30, step=1)
        inj_cost = st.number_input("inj_cost", value=0.5, step=0.1, format="%.2f")
        wdr_cost = st.number_input("wdr_cost", value=0.5, step=0.1, format="%.2f")

        st.header("Forward curve")
        use_direct_curve = st.toggle("Use direct curve", value=True)
        curve_mode = "Direct curve" if use_direct_curve else "TTF quote matrix"
        uploaded_curve = st.file_uploader("Upload xlsx/csv", type=["xlsx", "xls", "csv"])
        include_da = st.checkbox("Include DA front stub", value=True, disabled=(curve_mode != "TTF quote matrix"))

        run = st.form_submit_button("Run valuation", type="primary")

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
    "run_intrinsic": bool(run_intrinsic),
    "v_step": int(v_step),
    "clips_per_day": int(clips_per_day),
    "daily_max": int(daily_max) if daily_max > 0 else None,
    "capacity_mwh": int(capacity_mwh) if capacity_mwh > 0 else None,
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
        warm_numba_kernels()
        payload, result = cached_run_valuation(curve, params)
    except Exception as exc:
        st.exception(exc)
        st.stop()
    elapsed = time.perf_counter() - t0

st.success(f"Valuation complete in {elapsed:.1f}s")

if run_intrinsic:
    summary_metrics = ["flat_metric", "profiled_metric", "intrinsic_value", "extrinsic_value", "total_value"]
else:
    summary_metrics = ["flat_metric", "profiled_metric", "intrinsic_value", "extrinsic_value", "stochastic_metric"]

summary = pd.DataFrame({
    "metric": summary_metrics,
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
        }).assign(value=lambda x: x["value"].map(lambda v: "n/a" if pd.isna(v) else f"{v:,.0f}")),
        hide_index=True,
        use_container_width=True,
    )

date_span = payload["date_span"]
exercise_dates = date_span[payload["Dt"]:payload["active"]]
delta_dates = date_span[:payload["n_t"]]
plot_prices = pd.Series(payload["price_curve"], index=date_span)
intrinsic_profile = pd.Series(result["intrinsic_profile_raw"][:len(date_span)], index=date_span).loc[exercise_dates]
extrinsic_profile = pd.Series(result["extrinsic_profile_raw"][:len(date_span)], index=date_span).loc[exercise_dates]
extrinsic_delta_profile = pd.Series(payload["delta"][:payload["n_t"]], index=delta_dates)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, profile, title, color in [
    (axes[0], intrinsic_profile, f"{result['title_prefix']} intrinsic expected exercise vs forward curve", "tab:red"),
    (axes[1], extrinsic_delta_profile, f"{result['title_prefix']} native extrinsic delta vs forward curve", "tab:purple"),
]:
    dates = exercise_dates if ax is axes[0] else delta_dates
    ax_price = ax.twinx()
    if product_type == "storage":
        ax.bar(dates, np.where(profile.values > 0, profile.values, 0), width=1.0, color="tab:green", alpha=0.65, label="Expected sell" if ax is axes[0] else "Positive native delta")
        ax.bar(dates, np.where(profile.values < 0, profile.values, 0), width=1.0, color="tab:red", alpha=0.65, label="Expected buy" if ax is axes[0] else "Negative native delta")
    else:
        ax.bar(dates, profile.values, width=1.0, color=color, alpha=0.65, label="Expected offtake" if ax is axes[0] else "Native delta")
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

st.subheader("Monthly Native Deltas")
monthly_delta = extrinsic_delta_profile.loc[storageStart:storageEnd]
monthly_delta_by_period = monthly_delta.resample("MS").sum()
monthly_delta_table = pd.DataFrame({
    "period": monthly_delta_by_period.index.strftime("%b-%y"),
    "native_delta": monthly_delta_by_period.values,
})
monthly_delta_table = pd.concat([
    monthly_delta_table,
    pd.DataFrame([{"period": "Sum", "native_delta": monthly_delta_table["native_delta"].sum()}]),
], ignore_index=True)
monthly_delta_display = monthly_delta_table.copy()
monthly_delta_display["native_delta"] = monthly_delta_display["native_delta"].map("{:,.2f}".format)
st.dataframe(monthly_delta_display, hide_index=True, use_container_width=True)

with st.expander("Forward curve used"):
    curve_display = curve.copy()
    curve_display["contractStart"] = curve_display["contractStart"].dt.strftime("%Y-%m-%d")
    curve_display["contractEnd"] = curve_display["contractEnd"].dt.strftime("%Y-%m-%d")
    st.dataframe(curve_display, hide_index=True, use_container_width=True)
