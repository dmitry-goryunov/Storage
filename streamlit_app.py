"""Unified Streamlit interface for swing and gas-storage valuation."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pricing_app_core import (
    build_output_tables,
    build_pricing_params,
    parse_inventory_bounds,
    parse_ratchet_table,
    prepare_direct_curve,
    prepare_quote_matrix,
    read_uploaded_table,
)
from storage_model import (
    apply_inventory_bounds,
    curve_df_for_storage,
    describe_inventory_bounds,
    describe_ratchet_rates,
    quote_row_for_fd_date,
    run_valuation,
    warm_numba_kernels as model_warm_numba_kernels,
)


ROOT = Path(__file__).resolve().parent
DIRECT_CURVE_PATH = ROOT / "curve.csv"
QUOTE_MATRIX_PATH = ROOT / "ttf q.xlsx"

st.set_page_config(page_title="Swing & Storage Pricer", page_icon="⚡", layout="wide")


@st.cache_resource(show_spinner="Preparing valuation kernels...")
def warm_numba_kernels():
    return model_warm_numba_kernels()


@st.cache_data(show_spinner=False)
def load_uploaded_direct_curve(raw_bytes, file_name):
    return prepare_direct_curve(read_uploaded_table(raw_bytes, file_name))


@st.cache_data(show_spinner=False)
def load_uploaded_quote_matrix(raw_bytes, file_name):
    return prepare_quote_matrix(raw_bytes, file_name)


@st.cache_data(show_spinner=False, max_entries=20)
def cached_run_valuation(curve, params):
    """Run the model while caching only compact, user-facing outputs."""
    model, result = run_valuation(curve, params)
    allowed_terminal = np.flatnonzero(
        np.asarray(model.t_p_curve[:model.n_op]) > -1e9
    )
    payload = {
        "date_span": model.date_span,
        "Dt": model.Dt,
        "active": model._active,
        "n_t": model.n_t,
        "price_curve": np.asarray(model.price_curve, dtype=float),
        "physical_delta": np.asarray(model.delta, dtype=float),
        "pv_tailed_delta": np.asarray(model.delta_pv, dtype=float),
        "discount_rate": float(model.discount_rate),
        "contract_pv_eur": float(model.v[0, model.n_p, model.initial_state]),
        "initial_inventory_clips": int(model.initial_state),
        "terminal_inventory_clips": (
            int(allowed_terminal[0]) if len(allowed_terminal) == 1 else None
        ),
    }
    if params["product_type"] == "storage":
        bounds = apply_inventory_bounds(model, params)
        payload["bound_diagnostics"] = describe_inventory_bounds(model, bounds)
        payload["ratchet_diagnostics"] = (
            describe_ratchet_rates(model)
            if params.get("ratchets") is not None
            else pd.DataFrame()
        )
    else:
        payload["bound_diagnostics"] = pd.DataFrame()
        payload["ratchet_diagnostics"] = pd.DataFrame()
    return payload, result


def display_number(value, decimals=2):
    return "n/a" if pd.isna(value) else f"{float(value):,.{decimals}f}"


def serialisable(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return serialisable(value.item())
    if isinstance(value, tuple):
        return [serialisable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialisable(item) for key, item in value.items()}
    if isinstance(value, (list, set)):
        return [serialisable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


st.title("Swing & Storage Pricer")
st.caption(
    "Price one physical put swing, call swing, or gas-storage contract from a "
    "complete term sheet. Inputs are converted to the model grid exactly or refused."
)
st.warning(
    "Model-risk note: volatility and mean reversion are explicit scenario inputs. "
    "They are not claimed to be market-calibrated parameters; the S7 evidence is a "
    "methodology and convergence result, not a completed quote-history calibration.",
    icon="⚠️",
)
with st.expander("Product and valuation conventions"):
    st.markdown(
        """
- **Put swing:** buy gas on selected days at the strike; full maximum quantity
  is required by expiry.
- **Call swing:** sell gas on selected days at the strike; quantity can be
  mandatory or optional up to the maximum.
- **Storage:** inject, hold and withdraw gas between exact opening and terminal
  inventory levels. Injection fuel is consumed in addition to injected inventory.

Intrinsic and extrinsic value are shown when **Run intrinsic decomposition** is
enabled. Rates are annual continuously compounded. Dated bounds constrain opening
inventory on the stated date.
        """
    )

with st.sidebar:
    st.header("Valuation setup")
    product_type = st.selectbox(
        "Product type",
        ["put_swing", "call_swing", "storage"],
        format_func=lambda value: {
            "put_swing": "Put swing",
            "call_swing": "Call swing",
            "storage": "Storage",
        }[value],
    )
    curve_source = st.selectbox(
        "Forward-curve source",
        [
            "Bundled direct curve",
            "Upload direct curve",
            "Bundled TTF quote matrix",
            "Upload TTF quote matrix",
        ],
        help=(
            "Direct curves need contractStart, contractEnd and value. Quote matrices "
            "need a quote-date first column and TTFc1, TTFc2, ... columns."
        ),
    )
    uses_quote_matrix = "quote matrix" in curve_source.lower()
    if product_type == "storage":
        rate_mode = "Market / valuation discount rate"
        st.caption(
            "Storage uses one valuation rate because its strategy has payments and receipts."
        )
    else:
        rate_mode = st.selectbox(
            "Rate mode",
            ["Market / valuation discount rate", "Treasury scenario"],
            help="Treasury scenarios require an explicit borrow or invest direction.",
        )

    with st.form("valuation_inputs"):
        st.subheader("Dates")
        date_a, date_b = st.columns(2)
        val_date = pd.Timestamp(
            date_a.date_input("Valuation date", pd.Timestamp("2026-01-05"))
        )
        if uses_quote_matrix:
            fd_date = pd.Timestamp(
                date_b.date_input("Forward-curve date", pd.Timestamp("2026-01-05"))
            )
        else:
            fd_date = val_date
            date_b.date_input("Forward-curve date", val_date, disabled=True)
        date_c, date_d = st.columns(2)
        contract_start = pd.Timestamp(
            date_c.date_input("Contract start", pd.Timestamp("2026-04-01"))
        )
        contract_end = pd.Timestamp(
            date_d.date_input("Contract end", pd.Timestamp("2027-03-30"))
        )

        st.subheader("Commercial terms")
        if product_type in ("put_swing", "call_swing"):
            strike = st.number_input(
                "Strike (EUR/MWh)", value=30.0, step=0.25, format="%.4f"
            )
            swing_a, swing_b = st.columns(2)
            capacity_mwh = swing_a.number_input(
                "Maximum total quantity (MWh)",
                min_value=0.001, value=600_000.0, step=10_000.0, format="%.3f",
            )
            daily_max = swing_b.number_input(
                "Maximum daily quantity (MWh/day)",
                min_value=0.001, value=20_000.0, step=1_000.0, format="%.3f",
            )
            clips_per_day = st.number_input(
                "Clips per daily maximum",
                min_value=1, max_value=1_000, value=2, step=1,
                help="The clip is maximum daily quantity divided by this number.",
            )
            if product_type == "call_swing":
                volume_style = st.radio(
                    "Volume obligation",
                    ["Mandatory full quantity", "Optional up to maximum"],
                )
                zero_penalty = volume_style == "Optional up to maximum"
            else:
                zero_penalty = False
            n_states = inj_days = wdr_days = None
            initial_storage_mwh = terminal_storage_mwh = None
            inj_cost = wdr_cost = fuel_loss_pct = 0.0
        else:
            store_a, store_b = st.columns(2)
            capacity_mwh = store_a.number_input(
                "Working capacity (MWh)",
                min_value=0.001, value=600_000.0, step=10_000.0, format="%.3f",
            )
            n_states = store_b.number_input(
                "Inventory grid states",
                min_value=1, max_value=20_000, value=60, step=1,
                help="Capacity, rates and inventory must land exactly on this grid.",
            )
            rate_a, rate_b = st.columns(2)
            inj_days = rate_a.number_input(
                "Days to fill", min_value=0.001, value=30.0, step=1.0, format="%.3f"
            )
            wdr_days = rate_b.number_input(
                "Days to empty", min_value=0.001, value=60.0, step=1.0, format="%.3f"
            )
            inv_a, inv_b = st.columns(2)
            initial_storage_mwh = inv_a.number_input(
                "Opening inventory (MWh)",
                min_value=0.0, value=0.0, step=10_000.0, format="%.3f",
            )
            terminal_storage_mwh = inv_b.number_input(
                "Terminal inventory (MWh)",
                min_value=0.0, value=0.0, step=10_000.0, format="%.3f",
            )
            cost_a, cost_b = st.columns(2)
            inj_cost = cost_a.number_input(
                "Injection variable cost (EUR/MWh)",
                value=0.50, step=0.05, format="%.4f",
            )
            wdr_cost = cost_b.number_input(
                "Withdrawal variable cost (EUR/MWh)",
                value=0.50, step=0.05, format="%.4f",
            )
            fuel_loss_pct = st.number_input(
                "Injection fuel loss (%)",
                min_value=0.0, max_value=99.999, value=0.0, step=0.1, format="%.3f",
            )
            strike = 0.0
            daily_max = clips_per_day = None
            zero_penalty = False

        st.subheader("Model")
        model_a, model_b = st.columns(2)
        vol = model_a.number_input(
            "Volatility (annualised)",
            min_value=0.0, max_value=5.0, value=0.60, step=0.01, format="%.4f",
        )
        mean_reversion = model_b.number_input(
            "Mean reversion",
            min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.4f",
        )
        n_p_full = st.number_input(
            "n_p_full", min_value=0, max_value=100, value=30, step=1,
            help="Price-tree half-width; the full tree has 2 × n_p_full + 1 states.",
        )
        run_intrinsic = st.checkbox("Run intrinsic decomposition", value=True)

        st.subheader("Discounting")
        if rate_mode == "Market / valuation discount rate":
            discount_rate = st.number_input(
                "discount_rate (annual, continuous)",
                value=0.0, step=0.01, format="%.4f",
            )
            rate_inputs = {"discount_rate": float(discount_rate)}
        else:
            treasury_a, treasury_b = st.columns(2)
            borrow_rate = treasury_a.number_input(
                "borrow_rate (annual, continuous)",
                value=0.05, step=0.01, format="%.4f",
            )
            invest_rate = treasury_b.number_input(
                "invest_rate (annual, continuous)",
                value=0.03, step=0.01, format="%.4f",
            )
            funding_direction = st.selectbox(
                "funding_direction", ["borrow", "invest"],
                help="The model does not infer direction from product or moneyness.",
            )
            rate_inputs = {
                "borrow_rate": float(borrow_rate),
                "invest_rate": float(invest_rate),
                "funding_direction": funding_direction,
            }

        ratchet_frame = None
        bound_frame = None
        max_ratchet_rate_loss_pct = 10.0
        if product_type == "storage":
            st.subheader("Storage constraints")
            use_ratchets = st.checkbox(
                "Use inventory ratchets",
                value=False,
                help="Rate multipliers are linearly interpolated by inventory fullness.",
            )
            if use_ratchets:
                ratchet_frame = st.data_editor(
                    pd.DataFrame(
                        {
                            "fullness": [0.0, 0.5, 1.0],
                            "injection": [1.0, 1.0, 1.0],
                            "withdrawal": [1.0, 1.0, 1.0],
                        }
                    ),
                    num_rows="dynamic",
                    hide_index=True,
                    width="stretch",
                    key="ratchet_editor",
                )
                max_ratchet_rate_loss_pct = st.number_input(
                    "Maximum permitted ratchet rate loss (%)",
                    min_value=0.0, max_value=100.0, value=10.0, step=0.5,
                )
            use_bounds = st.checkbox(
                "Use dated inventory bounds",
                value=False,
                help="Fractions apply to opening inventory on each stated date.",
            )
            if use_bounds:
                bound_frame = st.data_editor(
                    pd.DataFrame(
                        {
                            "date": pd.Series(dtype="datetime64[ns]"),
                            "minimum": pd.Series(dtype=float),
                            "maximum": pd.Series(dtype=float),
                        }
                    ),
                    num_rows="dynamic",
                    hide_index=True,
                    width="stretch",
                    key="bounds_editor",
                )
                st.caption("Enter bounds as fractions: 0.70 means 70% of capacity.")

        uploaded_curve = None
        include_da = True
        if curve_source.startswith("Upload"):
            st.subheader("Forward curve")
            uploaded_curve = st.file_uploader(
                "Curve file", type=["xlsx", "xls", "csv"]
            )
        if uses_quote_matrix:
            include_da = st.checkbox("Include day-ahead front stub", value=True)

        run = st.form_submit_button("Run valuation", type="primary")


if not run:
    st.info("Complete the term sheet in the sidebar and select Run valuation.")
    st.stop()

try:
    ratchets = parse_ratchet_table(ratchet_frame) if ratchet_frame is not None else None
    inventory_bounds = (
        parse_inventory_bounds(bound_frame) if bound_frame is not None else None
    )
    values = {
        "product_type": product_type,
        "FDDate": fd_date,
        "valDate": val_date,
        "storageStart": contract_start,
        "storageEnd": contract_end,
        "uses_quote_matrix": uses_quote_matrix,
        "vol": vol,
        "sMR": mean_reversion,
        "n_p_full": n_p_full,
        "run_intrinsic": run_intrinsic,
        "capacity_mwh": capacity_mwh,
        "strike": strike,
        "daily_max": daily_max,
        "clips_per_day": clips_per_day,
        "zero_penalty": zero_penalty,
        "n_states": n_states,
        "inj_days": inj_days,
        "wdr_days": wdr_days,
        "initial_storage_mwh": initial_storage_mwh,
        "terminal_storage_mwh": terminal_storage_mwh,
        "inj_cost": inj_cost,
        "wdr_cost": wdr_cost,
        "fuel_loss": fuel_loss_pct / 100.0,
        "max_ratchet_rate_loss": max_ratchet_rate_loss_pct / 100.0,
        **rate_inputs,
    }
    params, effective = build_pricing_params(
        values, ratchets=ratchets, inventory_bounds=inventory_bounds
    )
except Exception as exc:
    st.error(f"Term sheet is not valid: {exc}")
    st.stop()

try:
    source_metadata = {"source_name": curve_source}
    if curve_source == "Bundled direct curve":
        source_bytes = DIRECT_CURVE_PATH.read_bytes()
        curve = load_uploaded_direct_curve(source_bytes, DIRECT_CURVE_PATH.name)
        source_metadata["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    elif curve_source == "Upload direct curve":
        if uploaded_curve is None:
            raise ValueError("Upload a direct curve file before running the valuation.")
        source_bytes = uploaded_curve.getvalue()
        curve = load_uploaded_direct_curve(source_bytes, uploaded_curve.name)
        source_metadata.update(
            {
                "file_name": uploaded_curve.name,
                "file_size": uploaded_curve.size,
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        )
    elif curve_source == "Bundled TTF quote matrix":
        quotes, contract_cols, quote_stats = load_uploaded_quote_matrix(
            QUOTE_MATRIX_PATH.read_bytes(), QUOTE_MATRIX_PATH.name
        )
        quote = quote_row_for_fd_date(quotes, contract_cols, fd_date, exact=False)
        curve = curve_df_for_storage(
            quote, contract_cols, curve_start=val_date, include_da=include_da
        )
        source_metadata.update(
            quote_stats | {"quote_used": pd.Timestamp(quote["quote_date"]).isoformat()}
        )
    else:
        if uploaded_curve is None:
            raise ValueError("Upload a TTF quote-matrix file before running.")
        quotes, contract_cols, quote_stats = load_uploaded_quote_matrix(
            uploaded_curve.getvalue(), uploaded_curve.name
        )
        quote = quote_row_for_fd_date(quotes, contract_cols, fd_date, exact=False)
        curve = curve_df_for_storage(
            quote, contract_cols, curve_start=val_date, include_da=include_da
        )
        source_metadata.update(
            quote_stats | {"quote_used": pd.Timestamp(quote["quote_date"]).isoformat()}
        )
except Exception as exc:
    st.error(f"Forward curve could not be prepared: {exc}")
    st.stop()

st.caption(
    f"Curve covers {curve['contractStart'].min():%Y-%m-%d} to "
    f"{curve['contractEnd'].max():%Y-%m-%d}."
)
with st.spinner("Running valuation. The first run may compile numerical kernels..."):
    started = time.perf_counter()
    try:
        warm_numba_kernels()
        payload, result = cached_run_valuation(curve, params)
    except Exception as exc:
        st.error(f"Valuation was refused: {exc}")
        st.stop()
    elapsed = time.perf_counter() - started

daily, monthly = build_output_tables(
    payload, result, product_type, contract_start, contract_end
)

st.success(f"Valuation complete in {elapsed:.1f}s.")
st.caption(
    "Applied annual continuously compounded rate: "
    f"{payload['discount_rate']:.4%}."
)

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
metric_1.metric("Contract PV (EUR)", display_number(payload["contract_pv_eur"], 0))
metric_2.metric("Total (EUR/MWh)", display_number(result["total"], 4))
metric_3.metric("Intrinsic (EUR/MWh)", display_number(result["intrinsic"], 4))
metric_4.metric("Extrinsic (EUR/MWh)", display_number(result["extrinsic"], 4))
metric_5.metric("Discount rate", f"{payload['discount_rate']:.4%}")

results_tab, schedule_tab, terms_tab, downloads_tab = st.tabs(
    ["Results", "Schedule & hedge", "Effective contract", "Curve & downloads"]
)

with results_tab:
    summary = pd.DataFrame(
        {
            "metric": [
                "Flat benchmark (EUR/MWh)",
                "Profiled intrinsic metric (EUR/MWh)",
                "Intrinsic (EUR/MWh)",
                "Extrinsic (EUR/MWh)",
                "Total / stochastic (EUR/MWh)",
                "Contract PV (EUR)",
            ],
            "value": [
                result["flat_metric"],
                result["profiled_metric"],
                result["intrinsic"],
                result["extrinsic"],
                result["total"],
                payload["contract_pv_eur"],
            ],
        }
    )
    st.dataframe(summary, hide_index=True, width="stretch")
    if product_type == "storage":
        st.dataframe(
            pd.DataFrame(
                {
                    "component": ["Intrinsic", "Extrinsic", "Total"],
                    "EUR": [
                        result["intrinsic_eur"],
                        result["extrinsic_eur"],
                        result["total_eur"],
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "EUR/MWh storage values use working capacity as denominator. Swing metrics "
        "follow the model's exercised-volume convention; Contract PV is unscaled."
    )

with schedule_tab:
    chart_data = daily.set_index("date")[
        [
            "intrinsic_expected_volume_mwh",
            "stochastic_expected_volume_mwh",
            "physical_forward_delta_mwh",
        ]
    ]
    st.subheader("Expected exercise and physical forward delta")
    st.line_chart(chart_data)
    st.dataframe(daily, hide_index=True, width="stretch")
    st.subheader("Monthly hedge aggregation")
    st.caption(
        "Physical delta is for matching-settlement OTC forwards. PV-tailed delta "
        "is the corresponding exposure against daily-margined futures."
    )
    st.dataframe(monthly, hide_index=True, width="stretch")

with terms_tab:
    st.subheader("Effective numerical contract")
    effective_display = {
        key: display_number(value, 6)
        if isinstance(value, (int, float, np.integer, np.floating))
        else str(value)
        for key, value in effective.items()
    }
    st.dataframe(
        pd.DataFrame(
            {
                "quantity": list(effective_display),
                "effective value": list(effective_display.values()),
            }
        ),
        hide_index=True,
        width="stretch",
    )
    if product_type == "storage" and not payload["bound_diagnostics"].empty:
        st.subheader("Inventory-bound rounding")
        st.dataframe(
            payload["bound_diagnostics"], hide_index=True, width="stretch"
        )
    if product_type == "storage" and not payload["ratchet_diagnostics"].empty:
        st.subheader("Ratchet rate representation")
        st.caption(
            "Contract rates are headroom-capped before whole-clip flooring. Loss "
            "reports the grid effect, not the physical headroom limit."
        )
        st.dataframe(
            payload["ratchet_diagnostics"], hide_index=True, width="stretch"
        )

with downloads_tab:
    st.subheader("Forward curve used")
    st.dataframe(curve, hide_index=True, width="stretch")
    report = {
        "product_type": product_type,
        "parameters": params,
        "effective_contract": effective,
        "curve_source": source_metadata,
        "results": {
            key: value
            for key, value in result.items()
            if key not in ("intrinsic_profile_raw", "extrinsic_profile_raw")
        },
        "contract_pv_eur": payload["contract_pv_eur"],
        "discount_rate": payload["discount_rate"],
    }
    download_a, download_b = st.columns(2)
    download_a.download_button(
        "Download valuation JSON",
        json.dumps(serialisable(report), indent=2, allow_nan=False),
        file_name=f"{product_type}_valuation.json",
        mime="application/json",
    )
    download_b.download_button(
        "Download curve CSV",
        curve.to_csv(index=False),
        file_name=f"{product_type}_curve.csv",
        mime="text/csv",
    )
    download_c, download_d = st.columns(2)
    download_c.download_button(
        "Download daily schedule CSV",
        daily.to_csv(index=False),
        file_name=f"{product_type}_daily_schedule.csv",
        mime="text/csv",
    )
    download_d.download_button(
        "Download monthly hedge CSV",
        monthly.to_csv(index=False),
        file_name=f"{product_type}_monthly_hedge.csv",
        mime="text/csv",
    )
