# Gas Storage / Swing Option Pricing Model

portfolio app https://storage-ksfhunyzfmkff3xptay3et.streamlit.app/

A quantitative library for valuing natural gas storage and swing contracts on the TTF market. Built around a trinomial price tree with Ornstein-Uhlenbeck mean reversion and a dynamic programming solver over a joint (time × price × volume) state space. The inner DP loop is JIT-compiled and parallelised with Numba for performance.

---

## Quick Start

Tested on **Python 3.12** (Numba is sensitive to the interpreter version; 3.9–3.12 supported).

```bash
# 1. Create a fresh environment (do NOT reuse the .venv shipped in this repo —
#    it is incomplete and will fail with "No module named 'pandas'").
python -m venv .venv-new
.venv-new\Scripts\activate          # Windows;  source .venv-new/bin/activate on macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch an interactive valuation app
streamlit run portfolio_app.py      # portfolio Mark-to-Market of the deals in quotes_2.csv
# or
streamlit run streamlit_app.py      # value a single swing / storage deal
```

> **First run compiles the Numba kernels (~20–40 s).** This happens once; the
> compiled kernels are cached to disk, so later runs start in a second or two.
> The Streamlit apps show a "Preparing Numba kernels…" spinner while this runs.

Prefer a notebook? See [Which tool should I use?](#which-tool-should-i-use) below.

---

## Repository Structure

| File | Description |
|---|---|
| `storage_model.py` | Core library — curve utilities, `Storage` class, valuation wrappers, and metric computation |
| `storage_kernels.py` | Numba-compiled kernels (tree core, DP solver, probabilities). Kept separate so edits to `storage_model.py` do not invalidate the Numba disk cache (avoids 20-40s recompiles) |
| `streamlit_app.py` | Streamlit app — value a single swing/storage deal interactively |
| `portfolio_app.py` | Streamlit app — portfolio Mark-to-Market of the deals in `quotes_2.csv` (MtM table, monthly exposures, charts) |
| `forward.ipynb` | **Primary notebook** — builds the daily forward curve from `ttf q.xlsx` and values a deal; the most feature-complete path (per-deal `sMR`, deal-independent daily curve, asymmetric inject/withdraw rates) |
| `portfolio.ipynb` | Portfolio Mark-to-Market notebook — the scriptable version of `portfolio_app.py` |
| `Swing_new.ipynb` | *Legacy* driver — original 6-product intrinsic/extrinsic loop over `curve.csv` + `quotes.csv` |
| `pricing.ipynb` | *Legacy* `run_valuation` driver (no `sMR` / deal-independent-curve support) |
| `finding.md` | Developer research note — the put-swing delta investigation (January step-down, December amplification); not required reading |
| `curve.csv` | Monthly TTF forward curve (48 contracts) |
| `quotes.csv` | Market bid/ask quotes for 6 swing products |
| `quotes_2.csv` | Trade portfolio for `portfolio.ipynb` — 7 executed swing deals (product, window, daily volume, N_days, executed price, vol, MR, strike) |
| `products.xlsx`, `ratchets.xlsx` | Product parameter workbook and ratchet (rate-multiplier vs fullness) profiles |
| `products_cell_snippet.py` | Optional snippet to drive `forward.ipynb` from `products.xlsx` instead of its hard-coded inputs cell |
| `ttf q.xlsx` | Historical TTF quote matrix (used by `streamlit_app.py`, `portfolio_app.py`, `forward.ipynb`, `portfolio.ipynb`) |
| `quotes.xlsx`, `curve_work.xlsx` | Reference data, not read by code |
| `requirements.txt` | Python dependencies |

---

## Which tool should I use?

| I want to… | Use | Notes |
|---|---|---|
| Value a single swing/storage deal, no coding | `streamlit run streamlit_app.py` | Sidebar inputs → MtM + exercise/delta charts |
| Mark a whole portfolio of trades to market | `streamlit run portfolio_app.py` | Reads `quotes_2.csv` → per-deal MtM + monthly exposures |
| Do the same portfolio work in editable code | `portfolio.ipynb` | The notebook `portfolio_app.py` is built from |
| Build/inspect a daily forward curve, or value storage with **asymmetric** inject/withdraw rates | `forward.ipynb` | The most feature-complete notebook |
| Call the model from your own Python | `import storage_model` | Start with `run_valuation()` or the `Storage` class — see [API Reference](#api-reference) |
| Value products defined in a workbook | `products.xlsx` → `load_product_params()` → `params_for_run_valuation()` → `run_valuation()` | See [Valuing a product from `products.xlsx`](#valuing-a-product-from-productsxlsx) |

**Legacy notebooks** (kept for reference, not the recommended path): `Swing_new.ipynb` (the original 6-product driver over `curve.csv` + `quotes.csv`) and `pricing.ipynb` (an earlier `run_valuation` driver without `sMR` or the deal-independent daily curve).

---

## Model Overview

### 1. Forward Curve (`map_curve_to_dates` → `smoothen_curve`)

Raw monthly contract prices are vectorised into a daily series, then smoothed via a cubic Hermite spline through monthly midpoints (PCHIP-derived slopes scaled by `alpha=1.2`, which relaxes PCHIP's monotonicity limiting). A one-shot additive correction is applied to each month so that smoothed daily averages exactly reproduce the original contract prices.

### 2. Price Tree (`build_tree`)

A trinomial tree discretises log-normal price dynamics with Ornstein-Uhlenbeck mean reversion:

- Time step `dt = 1 / 365.25` (daily)
- Grid spacing `dx = σ · √(3 · dt)` for numerical stability
- **Growing phase** (steps 0 → `n_p`): tree width expands from 1 to `2·n_p+1` nodes
- **Full-width phase** (steps `n_p` → `n_t`): constant-width grid; up/middle/down transition probabilities recalculated each step
- **Forward-fitting**: at every step the central node is shifted so the tree reproduces the input forward curve exactly

Returns `fwd`, `x` (log-price deviations), `q` (state probabilities), and `p_u / p_m / p_d` transition arrays.

### 3. Dynamic Programming Solver (`run_model`)

Backward induction from `n_t − 1` to `0` over the full `(time, price, volume)` state space:

- At each node, evaluates **every integer clip count from −`clips_per_day` to +`clips_per_day`** (negative = withdraw, positive = inject, 0 = hold) — subject to:
  - Daily ratchet limits (`i_ratch`, `w_ratch`) by volume state
  - Exercise permission masks (`i_curve`, `w_curve`)
  - Per-unit transaction costs (`i_cost`, `w_cost`)
  - Inventory floor/ceiling tunnels (`mintunnel`, `max_tunnel`)
- Selects the clip count maximising expected discounted continuation value
- The price dimension (`k`) is parallelised via `prange`; the volume (`l`) and clip-count (`d`) loops are scalar
- Infeasible terminal inventories are penalised with a large negative value (`−1e9`)

Returns `v` (optimal values) and `strat` (**signed clip count moved** per state) over the full grid.

### 4. Post-Processing

| Function | Purpose |
|---|---|
| `probabilities` | Forward simulation of joint (price, volume) state probabilities under the optimal strategy |
| `get_exercise` / `compute_all_metrics` | Expected daily exercise volume (MWh) and delta hedge ratios at each time step |
| `valuation` | Contract value at start, averaged over price states |

### 5. Value Decomposition

| Component | How computed |
|---|---|
| **Flat price** | Unweighted average forward price over the exercise window — the zero-optionality baseline |
| **Intrinsic value** | Profiled price (best deterministic schedule, `n_p = 0`) minus flat price; the seasonal spread |
| **Extrinsic value** | Full-model profiled (`n_p = 30`) minus intrinsic profiled; the price-optionality premium |

---

## API Reference

### Glossary

| Symbol | Meaning |
|---|---|
| `n_t` | number of daily time steps (valDate → backStop) |
| `n_p` | price-tree half-width; tree has `2·n_p+1` price states (`0` = single path, no optionality) |
| `n_op` | number of inventory states (clip levels) = grid size + 1 |
| `n_op_start` | dual role: inventory **grid size** (into `set_volume_states`) vs **initial inventory state** (read by `build`) |
| `Dt` | days from valDate to storageStart (first active day) |
| `v_step` | MWh per inventory state (the "clip" size) |
| `clips_per_day` | max clips injected/withdrawn per active day (the daily rate) |
| `strat` | signed clip count moved per state (neg = withdraw, pos = inject, 0 = idle) |
| `exp_ex` / `delta` | expected daily exercise volume / forward-equivalent delta (MWh) |
| `t_p_curve` | terminal inventory payoff/penalty by state (`-1e9` forbids a state) |
| `i_ratch` / `w_ratch` | per-inventory-level inject/withdraw rate multipliers (ratchets) |

The same glossary heads [`storage_model.py`](storage_model.py).

### `Storage` class

```python
Storage(
    valDate,        # str or date — contract valuation date
    storageStart,   # str or date — first day storage can operate
    storageEnd,     # str or date — last day storage can operate
    curve,          # pd.DataFrame with columns: contractStart, contractEnd, value
    n_p          = 0,    # int   — price tree half-width (0 = single path, no optionality)
    v_step       = 1000, # float — MWh per inventory state (clip size)
    sVol         = 0.9,  # float — annualised spot volatility (default 90 %)
    sMR          = 1.0,  # float — mean-reversion speed (Ornstein-Uhlenbeck)
    clips_per_day= 3,    # int   — max clips injected/withdrawn per active day (daily rate)
    daily_curve  = None, # pd.Series — precomputed daily curve used verbatim instead of `curve`
)
```

#### Methods

| Method | Returns | Description |
|---|---|---|
| `build()` | `self` | Runs the full valuation pipeline and populates all result attributes (returns `self`, so calls can be chained) |
| `flat()` | `float` | Unweighted average forward price over the exercise window (zero-optionality baseline) |
| `profiled()` | `float` | Volume-weighted achieved price per MWh under the optimal strategy (valid for any `n_p`) |
| `set_volume_states(n_op_start)` | `None` | Set the inventory **grid size** (number of states) and rebuild dependent arrays |

#### Result attributes (available after `build()`)

| Attribute | Shape | Description |
|---|---|---|
| `fwd` | `(n_t,)` | Daily forward prices |
| `x` | `(n_t, 2·n_p+1)` | Log-price deviations from forward |
| `v` | `(n_t, 2·n_p+1, n_op)` | Optimal contract value at each state |
| `strat` | `(n_t, 2·n_p+1, n_op)` | Signed clip count moved: negative = withdraw, positive = inject, 0 = hold (magnitude up to `clips_per_day`) |
| `prob` | `(n_t, 2·n_p+1, n_op)` | Joint state probabilities under optimal strategy |
| `exp_ex` | `list[float]` (n_t+1) | Expected daily exercise volume (MWh) |
| `delta` | `list[float]` (n_t+1) | Delta hedge ratios (volume-weighted, price-normalised) |

#### Customisable attributes (set before `build()`)

| Attribute | Default | Description |
|---|---|---|
| `d_curve` | all `1.0` | Daily discount factors |
| `i_curve` / `w_curve` | masks | Injection / withdrawal permission by day |
| `i_cost` / `w_cost` | all `0.0` | Per-unit transaction costs |
| `i_ratch` / `w_ratch` | arrays | Daily injection / withdrawal limits by volume state |
| `mintunnel` / `max_tunnel` | arrays | Inventory floor and ceiling by day |

---

## Usage Example

```python
import numpy as np
import pandas as pd
from storage_model import Storage

# Load forward curve
curve = pd.read_csv("curve.csv")
curve['contractStart'] = pd.to_datetime(curve['contractStart'], format='mixed')
curve['contractEnd']   = pd.to_datetime(curve['contractEnd'],   format='mixed')

# --- Intrinsic valuation (single price path, n_p = 0) ---
s_flat = Storage(
    valDate      = "2026-01-01",
    storageStart = "2026-04-01",
    storageEnd   = "2026-09-30",
    curve        = curve,
    n_p          = 0,       # single path — no price optionality
    v_step       = 1000,
    sVol         = 0.6,     # annualised spot volatility (60 %)
    sMR          = 1.0,
)
s_flat.build()
flat = s_flat.flat()                       # average forward price over the window (baseline)
print("Flat price :", flat,                     "€/MWh")
print("Profiled   :", s_flat.profiled(),        "€/MWh")
print("Intrinsic  :", s_flat.profiled() - flat, "€/MWh")

# --- Full valuation (trinomial tree, n_p = 30) ---
s_full = Storage(
    valDate      = "2026-01-01",
    storageStart = "2026-04-01",
    storageEnd   = "2026-09-30",
    curve        = curve,
    n_p          = 30,      # 61-node price tree
    v_step       = 1000,
    sVol         = 0.6,     # annualised spot volatility (60 %)
    sMR          = 1.0,
)
s_full.build()
# Extrinsic = optionality premium. Both EUR values are divided by the SAME
# (intrinsic) acquired volume — the run_valuation convention — so it is >= 0.
# (Don't subtract two profiled() calls: their delta denominators differ between
# the n_p=0 and n_p=30 builds, which can make the difference go negative.)
acq           = np.sum(s_flat.delta)
intrinsic_eur = s_flat.v[0, 0, s_flat.n_op_start]
full_eur      = s_full.v[0, s_full.n_p, s_full.n_op_start]
extrinsic     = (full_eur - intrinsic_eur) / acq
print("Extrinsic  :", extrinsic,                             "€/MWh")
print("Total      :", s_flat.profiled() - flat + extrinsic,  "€/MWh")
```

### Valuing a product from `products.xlsx`

`load_product_params()` returns the *primary* inputs only — it deliberately
leaves the grid derivation (clip size, daily rate, inventory clips) out so it
stays visible. Passing its dict straight to `run_valuation()` therefore raises
`KeyError: 'v_step'`. Bridge the two with `params_for_run_valuation()`:

```python
from storage_model import (
    load_product_params, params_for_run_valuation, run_valuation,
    curve_df_for_storage, quote_row_for_fd_date,
)

prm    = load_product_params("products.xlsx", product="call_swing_2010")
params = params_for_run_valuation(prm)          # adds v_step, clips_per_day, inv clips
s, result = run_valuation(curve, params)
print(result["total"])
```

> Uses the symmetric library grid (one daily rate for inject and withdraw).
> Asymmetric rates (`inj_days != wdr_days`) require `forward.ipynb`.

---

## Notebook Workflow (`Swing_new.ipynb`)

1. **Import & reload** `storage_model` (supports live development)
2. **Load `curve.csv`** — 48-contract TTF monthly forward curve
3. **Load `quotes.csv`** — 6 swing products with bid/ask quotes
4. **Valuation loop** — for each product:
   - Flat valuation (`n_p = 0`) → baseline price per MWh
   - Intrinsic valuation → seasonal spread (profiled − flat)
   - Full valuation (`n_p = 30`) → extrinsic / optionality premium
5. **Scatter plot** — intrinsic vs. extrinsic: model vs. market bid/ask
6. **Line chart** — modelled swing premium vs. market bids/asks across all 6 products

---

## Portfolio Mark-to-Market Workflow (`portfolio.ipynb`)

Values the executed swing deals in `quotes_2.csv` as of a single valuation date and reports per-deal MtM and monthly forward exposures.

**Deal semantics** (forced full exercise — each deal must move `N_days × |daily volume|` MWh within its window; the optionality is *which* days):

| Product | Obligation | Payoff per MWh | Model mapping |
|---|---|---|---|
| call swing | buy at strike on the best (highest-price) days | `price − strike` | withdraw machinery: starts full, ends empty, strike → `w_cost` |
| put swing | sell at strike on the cheapest days | `strike − price` | inject machinery: starts empty, ends full, strike → `i_cost = −strike` |

**Conventions:** `direction = sign(daily volume)` — a negative volume is a sold/short deal whose value, MtM and exposures flip sign. `premium = executed price × N_days × |daily volume|`; `MtM = direction × (model value − premium)`. Per-deal `vol` / `MR` columns feed `sVol` / `sMR`.

**Cell flow:**

1. **Inputs** — `VAL_DATE`, file names, `n_p_full`, `clips_per_day` (1 clip = one day's volume)
2. **Daily curve** — nearest quote ≤ `VAL_DATE` with a full contract strip (rows before 2010-08-19 carry only 12 months) → stepped daily → DA-anchored → `smoothen_curve`. The xlsx parse is cached as parquet (rebuilt when the xlsx is newer)
3. **Portfolio load** — cleans the dirty CSV (padded headers, `" 2,400 "`, `" -   "` = 0), validates window length, curve coverage and put stock, plots the curve with deal windows shaded
4. **Valuation loop** — one `run_valuation` per deal on the shared daily curve (grid: `v_step = |daily volume|`, `n_states = N_days`)
5. **MtM table** — premium, direction-adjusted value and MtM per deal plus portfolio total
6. **Monthly exposures** — direction-adjusted daily deltas resampled to month sums (deals × months matrix + portfolio column), charted against the forward curve

### `quotes_2.csv`

| Column | Description |
|---|---|
| `Product` | `call swing` or `Put Swing` |
| `market` | Market (TTF) |
| `Start` / `End` | Exercise window (DD-MMM-YY) |
| `current stock, days` | Initial stock in days of daily volume (puts must hold ≥ `N_days`) |
| `daily volume, Mwh` | Volume moved per exercised day; **negative = sold/short deal** |
| `N_days` | Number of days that must be exercised within the window |
| `executed price, Eur/Mwh` | Premium paid per MWh of total obligation volume |
| `vol` / `MR` | Per-deal spot volatility and mean-reversion speed |
| `strike price, Eur/Mwh` | Strike at which gas is bought (call) / sold (put) |

---

## Data Files

### `curve.csv`

| Column | Description |
|---|---|
| `contractStart` | Monthly contract start date (DD-MMM-YY) |
| `contractEnd` | Monthly contract end date (DD-MMM-YY) |
| `value` | Forward price (€/MWh) |

### `quotes.csv`

| Column | Description |
|---|---|
| `Product` | Product identifier |
| `market` | Market (TTF) |
| `valDate` | Valuation date |
| `Start` | Contract start date |
| `End` | Contract end date |
| `N_days` | Duration (90, 120, or 180 days) |
| `bid` / `ask` | Market bid/ask prices (€/MWh) |
| `notes` | Optional notes |

#### Swing products

| Product | Start | End | N_days |
|---|---|---|---|
| 1 | Apr-26 | Jun-26 | 90 |
| 2 | Apr-26 | Sep-26 | 180 |
| 3 | Oct-26 | Dec-26 | 90 |
| 4 | Oct-26 | Mar-27 | 180 |
| 5 | Apr-27 | Jun-27 | 90 |
| 6 | Apr-27 | Sep-27 | 180 |

---

## Tests

[`tests/test_regression.py`](tests/test_regression.py) pins the independently
verified anchors (portfolio MtM = −56,901 EUR, the `n_p=0` DP equals greedy day
selection, intrinsic/extrinsic ≥ 0, the workbook bridge runs finite, the
curve-gap guard fires). Run with `pytest`, or as a plain script with no extra
dependency:

```bash
python -m pytest tests          # if pytest is installed
python tests/test_regression.py # plain-script fallback (prints PASS/FAIL)
```

---

## Dependencies

All dependencies are pinned in [`requirements.txt`](requirements.txt) — `numpy`, `numba`,
`scipy`, `pandas`, `matplotlib` (library + notebooks) plus `streamlit`, `openpyxl`,
`pyarrow` (the apps and the Excel/parquet data paths):

```bash
pip install -r requirements.txt
```

> **Note:** Numba JIT-compiles the core DP solver (`run_model`) and probability simulation (`probabilities`) with `nopython=True, parallel=True`. The first run triggers compilation (~20–40 s); the result is cached to disk, so subsequent runs start fast.

---

## Performance Notes

| Lever | Effect |
|---|---|
| `n_p = 0` | Fastest — single price path, intrinsic only |
| `n_p = 30` | Full optionality; tree has 61 price states |
| `v_step` | Larger step → fewer inventory states → faster |
| Numba parallel | Price loop parallelised across CPU cores automatically |
