"""Portfolio Mark-to-Market Streamlit app.

Mirrors the 6-step flow of portfolio.ipynb:
  1. Sidebar form — valuation inputs + optional file uploads
  2. Quote matrix loading → daily curve (parquet cache for local file)
  3. Portfolio load/clean/validate
  4. Per-deal valuation (cached small payload)
  5. MtM table + TOTAL metric
  6. Monthly exposure table + chart
"""
from pathlib import Path
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import storage_model

st.set_page_config(page_title="Portfolio Mark-to-Market", layout="wide")


# ── Numba warm-up ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Preparing Numba kernels...")
def warm_numba_kernels():
    return storage_model.warm_numba_kernels()


# ── Data loading helpers ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_quote_matrix_local(xlsx_path: str, parquet_path: str):
    """Load the TTF quote matrix from xlsx, caching as parquet when fresher."""
    p_xlsx = Path(xlsx_path)
    p_parq = Path(parquet_path)
    if p_parq.exists() and p_parq.stat().st_mtime >= p_xlsx.stat().st_mtime:
        quotes = pd.read_parquet(p_parq)
    else:
        quotes = pd.read_excel(p_xlsx)
        quotes = quotes.rename(columns={quotes.columns[0]: "quote_date"})
        quotes = quotes.dropna(subset=["quote_date"]).copy()
        quotes["quote_date"] = pd.to_datetime(quotes["quote_date"], format="mixed")
        # Coerce junk strings (e.g. "Retrieving...") to NaN.
        for c in quotes.columns[1:]:
            quotes[c] = pd.to_numeric(quotes[c], errors="coerce")
        quotes = quotes.sort_values("quote_date").reset_index(drop=True)
        quotes.to_parquet(p_parq)
    return quotes


@st.cache_data(show_spinner=False)
def load_quote_matrix_upload(uploaded_bytes: bytes, file_name: str):
    """Parse an uploaded quote matrix file."""
    import io
    buf = io.BytesIO(uploaded_bytes)
    if file_name.lower().endswith(".csv"):
        quotes = pd.read_csv(buf)
    else:
        quotes = pd.read_excel(buf)
    quotes = quotes.rename(columns={quotes.columns[0]: "quote_date"})
    quotes = quotes.dropna(subset=["quote_date"]).copy()
    quotes["quote_date"] = pd.to_datetime(quotes["quote_date"], format="mixed")
    for c in quotes.columns[1:]:
        quotes[c] = pd.to_numeric(quotes[c], errors="coerce")
    quotes = quotes.sort_values("quote_date").reset_index(drop=True)
    return quotes


def _contract_columns(quotes):
    return sorted(
        [c for c in quotes.columns if re.fullmatch(r"TTFc\d+", str(c))],
        key=lambda c: int(re.search(r"\d+", str(c)).group()),
    )


@st.cache_data(show_spinner=False)
def build_daily_curve(quotes_key, val_date: str, min_strip_months: int):
    """Derive ONE smoothed daily curve from the nearest eligible quote.

    quotes_key is the quotes DataFrame (cache key includes its content via hash).
    """
    quotes = quotes_key
    contract_cols = _contract_columns(quotes)
    strip_len = quotes[contract_cols].notna().sum(axis=1)
    eligible = quotes.loc[strip_len >= min_strip_months].reset_index(drop=True)
    if eligible.empty:
        raise ValueError(
            f"No quote row has >= {min_strip_months} populated TTFc contracts."
        )

    val_ts = pd.Timestamp(val_date)
    fd_quote = storage_model.quote_row_for_fd_date(eligible, contract_cols, val_ts)
    quote_date = pd.Timestamp(fd_quote["quote_date"])

    monthly, _ = storage_model.monthly_curve_from_quote(fd_quote, contract_cols)
    day_index = pd.date_range(
        monthly.index.min(),
        storage_model.month_end(monthly.index.max()),
        freq="D",
    )
    stepped = monthly.reindex(day_index, method="ffill")

    da_value = fd_quote.get("DA", np.nan)
    if pd.notna(da_value):
        stepped = stepped.reindex(
            pd.date_range(quote_date, day_index.max(), freq="D")
        )
        stepped.loc[quote_date] = float(da_value)
        stepped = stepped.ffill()

    daily_curve = storage_model.smoothen_curve(stepped).rename("value")
    return daily_curve, quote_date


@st.cache_data(show_spinner=False)
def load_portfolio_local(csv_path: str):
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def load_portfolio_upload(uploaded_bytes: bytes):
    import io
    return pd.read_csv(io.BytesIO(uploaded_bytes))


def _num(x):
    """Parse dirty numbers: ' 2,400 ' → 2400.0, ' -   ' → 0.0."""
    s = str(x).strip().replace(",", "").replace(" ", "")
    if s in ("", "-", "nan"):
        return 0.0
    return float(s)


def clean_portfolio(raw: pd.DataFrame):
    """Clean and validate the raw portfolio CSV (mirrors notebook cell 4)."""
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]

    port = pd.DataFrame({
        "product_type": raw["Product"].str.strip().str.lower().str.replace(
            r"\s+", "_", regex=True
        ),
        "market": raw["market"].str.strip(),
        "Start": pd.to_datetime(raw["Start"].str.strip(), format="%d-%b-%y"),
        "End": pd.to_datetime(raw["End"].str.strip(), format="%d-%b-%y"),
        "stock_days": raw["current stock, days"].map(_num),
        "daily_vol": raw["daily volume, Mwh"].map(_num),
        "n_days": raw["N_days"].map(_num).astype(int),
        "executed": raw["executed price, Eur/Mwh"].map(_num),
        "vol": raw["vol"].map(_num),
        "sMR": raw["MR"].map(_num),
        "strike": raw["strike price, Eur/Mwh"].map(_num),
    })
    port.index = [f"deal_{i + 1}" for i in range(len(port))]

    port["direction"] = np.sign(port["daily_vol"]).astype(int)
    port["abs_daily_vol"] = port["daily_vol"].abs()
    port["capacity_mwh"] = port["n_days"] * port["abs_daily_vol"]
    port["stock_mwh"] = port["stock_days"] * port["abs_daily_vol"]
    port["premium_eur"] = port["executed"] * port["capacity_mwh"]

    return port


def validate_portfolio(port: pd.DataFrame, daily_curve: pd.Series, val_date: pd.Timestamp):
    """Run per-deal sanity checks (mirrors notebook cell 4). Returns error list."""
    errors = []
    for name, d in port.iterrows():
        window = (d["End"] - d["Start"]).days + 1
        if window < d["n_days"]:
            errors.append(f"{name}: exercise window {window}d < N_days {d['n_days']}")
        backstop = (
            (d["End"] + pd.DateOffset(months=1)).normalize()
            + pd.offsets.MonthEnd(0)
        )
        if daily_curve.index.min() > val_date or daily_curve.index.max() < backstop:
            errors.append(
                f"{name}: daily curve does not cover "
                f"{val_date:%Y-%m-%d}..{backstop:%Y-%m-%d}"
            )
        if d["product_type"] == "put_swing" and d["stock_days"] < d["n_days"]:
            errors.append(
                f"{name}: put swing stock {d['stock_days']:.0f}d < N_days {d['n_days']}"
            )
    return errors


@st.cache_data(show_spinner=False, max_entries=50)
def cached_deal_valuation(
    daily_curve_series: pd.Series,
    product_type: str,
    val_date: str,
    storage_start: str,
    storage_end: str,
    vol: float,
    s_mr: float,
    n_p_full: int,
    daily_max: float,
    clips_per_day: int,
    capacity_mwh: float,
    strike: float,
):
    """Value one deal; return a SMALL payload (no Storage cubes).

    The Storage object's v/strat/prob arrays can be tens of MB — never cache them.
    Cache key covers all numerical inputs via positional args.
    """
    params = {
        "product_type": product_type,
        "valDate": pd.Timestamp(val_date),
        "storageStart": pd.Timestamp(storage_start),
        "storageEnd": pd.Timestamp(storage_end),
        "vol": vol,
        "sMR": s_mr,
        "n_p_full": n_p_full,
        "run_intrinsic": True,
        "daily_max": daily_max,
        "clips_per_day": clips_per_day,
        "capacity_mwh": capacity_mwh,
        "strike": strike,
        "daily_curve": daily_curve_series,
    }
    s, res = storage_model.run_valuation(None, params)

    init_inv = s.n_op_start
    value_eur = float(s.v[0, s.n_p, init_inv])

    # Direction-neutral delta series (caller multiplies by direction)
    delta_arr = np.asarray(s.delta[: s.n_t], dtype=float)
    delta_index = list(s.date_span[: s.n_t].astype(str))  # str for pickling

    return {
        "value_eur": value_eur,
        "n_t": s.n_t,
        "delta_arr": delta_arr,
        "delta_index": delta_index,
        "intrinsic": res["intrinsic"],
        "extrinsic": res["extrinsic"],
    }


# ── Sidebar form ───────────────────────────────────────────────────────────────

st.title("Portfolio Mark-to-Market")
st.caption("Valuing a single deal instead? Use the **Swing / Storage Valuation** app (`streamlit run streamlit_app.py`).")
warm_numba_kernels()

with st.expander("Portfolio CSV format"):
    st.markdown(
        "Upload a CSV with one row per trade, or leave blank to use the bundled `quotes_2.csv`. "
        "Numbers may carry spaces/commas (`\" 2,400 \"`); `\" -   \"` means 0. Columns:\n\n"
        "| Column | Meaning |\n"
        "|---|---|\n"
        "| `Product` | `call swing` (sell at strike on best days) or `Put Swing` (buy at strike on cheapest days) |\n"
        "| `Start` / `End` | Exercise window, `DD-MMM-YY` (e.g. `01-Jan-11`) |\n"
        "| `current stock, days` | Initial stock in days of daily volume (a put must hold ≥ `N_days`) |\n"
        "| `daily volume, Mwh` | Volume per exercised day; **negative = sold/short** (value & exposures flip) |\n"
        "| `N_days` | Days that must be exercised within the window (forced full exercise) |\n"
        "| `executed price, Eur/Mwh` | Premium paid per MWh of total obligation volume |\n"
        "| `vol` / `MR` | Per-deal annualised volatility and mean-reversion speed |\n"
        "| `strike price, Eur/Mwh` | Strike at which gas is bought (put) / sold (call) |\n\n"
        "MtM = direction × (model value − premium), where premium = executed × N_days × |daily volume|."
    )

with st.sidebar:
    with st.form("portfolio_inputs"):
        st.header("Valuation inputs")

        val_date_input = st.date_input(
            "VAL_DATE (valuation date)", pd.Timestamp("2010-08-19"),
            help="The book is marked as of this date; the nearest quote on/before it with a full "
                 "contract strip builds the curve.",
        )
        n_p_full = st.number_input(
            "n_p_full", min_value=0, max_value=100, value=30, step=1,
            help="Price-tree half-width (2*n_p_full+1 price states). ~30 is converged.",
        )
        clips_per_day = st.number_input(
            "clips_per_day", min_value=1, max_value=24, value=1, step=1,
            help="Grid resolution: clips moved per active day (1 clip = one day's volume). For these "
                 "linear-payoff deals a finer grid does not change the value.",
        )
        min_strip_months = st.number_input(
            "MIN_STRIP_MONTHS", min_value=1, value=43, step=1,
            help="Skip quote rows with fewer than this many populated monthly contracts. The portfolio "
                 "runs to 2013, which needs ~43 months of strip from a 2010 quote.",
        )

        st.subheader("Quote matrix (xlsx/csv)")
        uploaded_quotes = st.file_uploader(
            "Upload quote matrix", type=["xlsx", "xls", "csv"],
            key="quotes_upload", label_visibility="collapsed"
        )

        st.subheader("Portfolio (csv)")
        uploaded_portfolio = st.file_uploader(
            "Upload portfolio CSV", type=["csv"],
            key="portfolio_upload", label_visibility="collapsed"
        )

        run = st.form_submit_button("Run portfolio valuation", type="primary")

if not run:
    st.info("Configure inputs in the sidebar, then click **Run portfolio valuation**.")
    st.stop()

val_date = pd.Timestamp(val_date_input)
n_p_full = int(n_p_full)
clips_per_day = int(clips_per_day)
min_strip_months = int(min_strip_months)

# ── Step 2: Quote matrix → daily curve ────────────────────────────────────────

try:
    if uploaded_quotes is not None:
        quotes = load_quote_matrix_upload(
            uploaded_quotes.getvalue(), uploaded_quotes.name
        )
    else:
        xlsx_path = str(Path("ttf q.xlsx").resolve())
        parquet_path = str(Path("ttf q.parquet").resolve())
        quotes = load_quote_matrix_local(xlsx_path, parquet_path)

    daily_curve, quote_date = build_daily_curve(
        quotes, str(val_date.date()), min_strip_months
    )
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.caption(
    f"Quote date used: **{quote_date:%Y-%m-%d}** (VAL_DATE {val_date:%Y-%m-%d}). "
    f"Curve coverage: {daily_curve.index.min():%Y-%m-%d} .. {daily_curve.index.max():%Y-%m-%d} "
    f"({daily_curve.iloc[0]:.2f} .. {daily_curve.iloc[-1]:.2f} EUR/MWh)."
)

# ── Step 3: Portfolio load, clean, validate ────────────────────────────────────

try:
    if uploaded_portfolio is not None:
        raw_port = load_portfolio_upload(uploaded_portfolio.getvalue())
    else:
        csv_path = str(Path("quotes_2.csv").resolve())
        raw_port = load_portfolio_local(csv_path)

    port = clean_portfolio(raw_port)
except Exception as exc:
    st.error(f"Portfolio loading error: {exc}")
    st.stop()

errors = validate_portfolio(port, daily_curve, val_date)
if errors:
    for e in errors:
        st.error(e)
    st.stop()

with st.expander("Cleaned portfolio"):
    display_cols = [
        "product_type", "Start", "End", "daily_vol", "n_days",
        "strike", "executed", "vol", "sMR", "premium_eur",
    ]
    fmt_port = port[display_cols].copy()
    for c in ["Start", "End"]:
        fmt_port[c] = fmt_port[c].dt.strftime("%Y-%m-%d")
    for c in ["daily_vol", "n_days", "premium_eur"]:
        fmt_port[c] = fmt_port[c].map(lambda v: f"{v:,.0f}")
    st.dataframe(fmt_port, width="stretch")

# Curve plot with deal windows (expander)
with st.expander("Daily curve with deal windows"):
    fig0, ax0 = plt.subplots(figsize=(12, 4))
    ax0.plot(daily_curve.index, daily_curve.values, color="black", lw=1.2,
             label="daily forward curve")
    for name, d in port.iterrows():
        ax0.axvspan(
            d["Start"], d["End"], alpha=0.08,
            color="tab:blue" if d["product_type"] == "call_swing" else "tab:red",
        )
    ax0.set_title(
        f"Daily curve ({quote_date:%Y-%m-%d}) with deal windows "
        "(blue = call swing, red = put swing)"
    )
    ax0.set_ylabel("EUR/MWh")
    ax0.legend()
    st.pyplot(fig0)

# ── Step 4: Valuation loop ─────────────────────────────────────────────────────

st.subheader("Valuation")

deal_values = {}      # name → direction-adjusted value_eur
delta_profiles = {}   # name → direction-adjusted daily delta Series

progress_bar = st.progress(0, text="Running valuations…")
timing_lines = []

for i, (name, d) in enumerate(port.iterrows()):
    t0 = time.perf_counter()
    with st.spinner(f"Valuing {name} ({d['product_type']})…"):
        try:
            payload = cached_deal_valuation(
                daily_curve_series=daily_curve,
                product_type=d["product_type"],
                val_date=str(val_date.date()),
                storage_start=str(d["Start"].date()),
                storage_end=str(d["End"].date()),
                vol=float(d["vol"]),
                s_mr=float(d["sMR"]),
                n_p_full=n_p_full,
                daily_max=float(d["abs_daily_vol"]),
                clips_per_day=clips_per_day,
                capacity_mwh=float(d["capacity_mwh"]),
                strike=float(d["strike"]),
            )
        except Exception as exc:
            st.error(f"{name}: {exc}")
            st.stop()

    elapsed = time.perf_counter() - t0
    raw_value = payload["value_eur"]
    deal_values[name] = int(d["direction"]) * raw_value

    idx = pd.DatetimeIndex(payload["delta_index"])
    delta_profiles[name] = int(d["direction"]) * pd.Series(
        payload["delta_arr"], index=idx
    )

    timing_lines.append(
        f"**{name}** ({d['product_type']}): value = {raw_value:,.0f} EUR  "
        f"| intrinsic = {payload['intrinsic']:.3f} EUR/MWh  "
        f"| extrinsic = {payload['extrinsic']:.3f} EUR/MWh  "
        f"| {elapsed:.2f}s"
    )
    progress_bar.progress((i + 1) / len(port), text=f"Done {name}")

for line in timing_lines:
    st.markdown(line)

# ── Step 5: MtM table ─────────────────────────────────────────────────────────

st.subheader("Mark-to-Market")

mtm = port[["product_type", "Start", "End", "daily_vol", "n_days", "strike", "executed"]].copy()
mtm["premium_eur"] = port["premium_eur"]
mtm["value_eur"] = pd.Series(deal_values)
# mtm = direction × (raw_model_value − premium_eur)
# Since deal_values[n] = direction × raw, this equals: deal_values[n] − direction × premium.
mtm["mtm_eur"] = [
    deal_values[n] - int(port.loc[n, "direction"]) * port.loc[n, "premium_eur"]
    for n in port.index
]

total_row = pd.Series(
    {
        "product_type": "TOTAL",
        "premium_eur": mtm["premium_eur"].sum(),
        "value_eur": mtm["value_eur"].sum(),
        "mtm_eur": mtm["mtm_eur"].sum(),
    },
    name="TOTAL",
)
mtm_table = pd.concat([mtm, total_row.to_frame().T])

total_mtm = float(mtm["mtm_eur"].sum())
st.metric("Total MtM (EUR)", f"{total_mtm:,.0f}")

fmt_mtm = mtm_table.copy()
for c in ["Start", "End"]:
    fmt_mtm[c] = fmt_mtm[c].map(lambda v: "" if pd.isna(v) else f"{v:%Y-%m-%d}")
for c in ["daily_vol", "n_days", "premium_eur", "value_eur", "mtm_eur"]:
    fmt_mtm[c] = fmt_mtm[c].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
# Convert all remaining numeric columns to str so Arrow sees a homogenous object dtype.
for c in ["strike", "executed"]:
    fmt_mtm[c] = fmt_mtm[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
fmt_mtm = fmt_mtm.fillna("").astype(str)
st.dataframe(fmt_mtm, width="stretch")

# ── Step 6: Monthly exposure ───────────────────────────────────────────────────

st.subheader("Monthly Forward Exposure")

monthly_exp = pd.DataFrame(
    {n: p.resample("MS").sum() for n, p in delta_profiles.items()}
).fillna(0.0)
monthly_exp = monthly_exp.loc[monthly_exp.abs().sum(axis=1) > 1e-9]
monthly_exp["PORTFOLIO"] = monthly_exp.sum(axis=1)

# Consistency check: monthly sums must equal daily sums.
chk = pd.DataFrame(
    {
        "daily_sum": {n: p.sum() for n, p in delta_profiles.items()},
        "monthly_sum": monthly_exp.drop(columns="PORTFOLIO").sum(),
    }
)
if not np.allclose(chk["daily_sum"], chk["monthly_sum"], atol=1e-6):
    st.error(f"Monthly/daily delta mismatch:\n{chk}")

fmt_exp = monthly_exp.copy()
fmt_exp.index = fmt_exp.index.strftime("%Y-%m")
fmt_exp = fmt_exp.map(lambda v: f"{v:,.0f}")
st.dataframe(fmt_exp, width="stretch")

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(
    monthly_exp.index, monthly_exp["PORTFOLIO"],
    width=20, color="tab:blue", alpha=0.7, label="portfolio exposure (MWh/month)",
)
ax.axhline(0, color="grey", lw=0.8)
ax.set_ylabel("MWh per month")
ax2 = ax.twinx()
ax2.plot(daily_curve.index, daily_curve.values, color="black", lw=1.0,
         label="forward curve")
ax2.set_ylabel("EUR/MWh")
ax.set_title("Portfolio monthly forward exposure vs forward curve")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="best")
st.pyplot(fig)
