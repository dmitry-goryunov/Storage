# Gas Storage / Swing Option Pricing Model

portfolio app https://storage-ksfhunyzfmkff3xptay3et.streamlit.app/

A quantitative library for valuing natural gas storage and swing contracts on the TTF market. Built around a trinomial price tree with Ornstein-Uhlenbeck mean reversion and a dynamic programming solver over a joint (time × price × volume) state space. The inner DP loop is JIT-compiled and parallelised with Numba for performance.

---

## Repository Structure

| File | Description |
|---|---|
| `storage_model.py` | Core library — curve utilities, `Storage` class, valuation wrappers, and metric computation |
| `storage_kernels.py` | Numba-compiled kernels (tree core, DP solver, probabilities). Kept separate so edits to `storage_model.py` do not invalidate the Numba disk cache (avoids 20-40s recompiles) |
| `streamlit_app.py` | Interactive Streamlit UI for running valuations |
| `portfolio_app.py` | Streamlit UI for the portfolio Mark-to-Market workflow (same logic as `portfolio.ipynb`: deal MtM table, monthly exposures, curve/window charts) |
| `Swing_new.ipynb` | Driver notebook — loads curve & quotes, runs intrinsic/extrinsic valuations for 6 products, plots results |
| `forward.ipynb` | Exploration notebook — forward curve work using `ttf q.xlsx` |
| `portfolio.ipynb` | Portfolio Mark-to-Market — values the swing trades in `quotes_2.csv` against one smoothed daily curve and reports per-deal MtM and monthly forward exposures |
| `pricing.ipynb` | Exploration notebook — pricing experiments |
| `finding.md` | Write-up of the put-swing delta investigation (January step-down, December amplification) |
| `curve.csv` | Monthly TTF forward curve (48 contracts) |
| `quotes.csv` | Market bid/ask quotes for 6 swing products |
| `quotes_2.csv` | Trade portfolio for `portfolio.ipynb` — 7 executed swing deals (product, window, daily volume, N_days, executed price, vol, MR, strike) |
| `products.xlsx`, `ratchets.xlsx` | Product parameter workbook and ratchet (rate-multiplier vs fullness) profiles used by `forward.ipynb` |
| `ttf q.xlsx` | Historical TTF quote data (used by `streamlit_app.py` and exploration notebooks) |
| `quotes.xlsx`, `curve_work.xlsx` | Reference data, not read by code |
| `requirements.txt` | Python dependencies |

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

- At each node, tests three actions — **inject (+1)**, **hold (0)**, **withdraw (−1)** — subject to:
  - Daily ratchet limits (`i_ratch`, `w_ratch`) by volume state
  - Exercise permission masks (`i_curve`, `w_curve`)
  - Per-unit transaction costs (`i_cost`, `w_cost`)
  - Inventory floor/ceiling tunnels (`mintunnel`, `max_tunnel`)
- Selects the action maximising expected discounted continuation value
- The volume dimension is vectorised; the price dimension is parallelised via `prange`
- Infeasible transitions are penalised with a large dummy value (`1e10`)

Returns `v` (optimal values) and `strat` (action indicator) over the full state grid.

### 4. Post-Processing

| Function | Purpose |
|---|---|
| `probabilities` | Forward simulation of joint (price, volume) state probabilities under the optimal strategy |
| `get_exercise` / `compute_all_metrics` | Expected daily exercise volume (MWh) and delta hedge ratios at each time step |
| `valuation` | Contract value at start, averaged over price states |

### 5. Value Decomposition

| Component | How computed |
|---|---|
| **Flat price** | `n_p = 0` (single price path) → zero-optionality baseline |
| **Intrinsic value** | Profiled price minus flat price; reflects seasonal spread from starting fully loaded |
| **Extrinsic value** | Full model (`n_p = 30`) minus intrinsic; reflects price uncertainty / optionality |

---

## API Reference

### `Storage` class

```python
Storage(
    valDate,        # str or date — contract valuation date
    storageStart,   # str or date — first day storage can operate
    storageEnd,     # str or date — last day storage can operate
    curve,          # pd.DataFrame with columns: contractStart, contractEnd, value
    n_p   = 0,      # int   — price tree half-width (0 = single path)
    v_step= 1000,   # float — MWh per inventory state
    sVol  = 0.6,    # float — daily spot volatility (60 %)
    sMR   = 1.0,    # float — mean-reversion speed (Ornstein-Uhlenbeck)
)
```

#### Methods

| Method | Returns | Description |
|---|---|---|
| `build()` | `None` | Runs full valuation pipeline; populates all result attributes |
| `flat()` | `float` | Flat price per MWh (zero-optionality, `n_p = 0`) |
| `profiled()` | `float` | Profiled price per MWh (starting fully loaded) |
| `set_volume_states(n_op_start)` | `None` | Override starting inventory state and rebuild arrays |

#### Result attributes (available after `build()`)

| Attribute | Shape | Description |
|---|---|---|
| `fwd` | `(n_t,)` | Daily forward prices |
| `x` | `(n_t, 2·n_p+1)` | Log-price deviations from forward |
| `v` | `(n_t, 2·n_p+1, n_op)` | Optimal contract value at each state |
| `strat` | `(n_t, 2·n_p+1, n_op)` | Optimal action: `+1` inject, `0` hold, `−1` withdraw |
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
import pandas as pd
from storage_model import Storage

# Load forward curve
curve = pd.read_csv("curve.csv")
curve['contractStart'] = pd.to_datetime(curve['contractStart'], format='mixed')
curve['contractEnd']   = pd.to_datetime(curve['contractEnd'],   format='mixed')

# --- Flat / intrinsic valuation (single price path) ---
s_flat = Storage(
    valDate      = "2026-01-01",
    storageStart = "2026-04-01",
    storageEnd   = "2026-09-30",
    curve        = curve,
    n_p          = 0,       # single path — no optionality
    v_step       = 1000,
    sVol         = 0.6,
    sMR          = 1.0,
)
s_flat.build()
print("Flat price   :", s_flat.flat(),     "€/MWh")
print("Profiled     :", s_flat.profiled(), "€/MWh")
print("Intrinsic    :", s_flat.profiled() - s_flat.flat(), "€/MWh")

# --- Full extrinsic valuation (trinomial tree) ---
s_full = Storage(
    valDate      = "2026-01-01",
    storageStart = "2026-04-01",
    storageEnd   = "2026-09-30",
    curve        = curve,
    n_p          = 30,      # 61-node price tree
    v_step       = 1000,
    sVol         = 0.6,
    sMR          = 1.0,
)
s_full.build()
print("Extrinsic    :", s_full.flat() - s_flat.flat(), "€/MWh")
```

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

## Dependencies

```
numpy
numba
scipy
pandas
matplotlib
```

```bash
pip install numpy numba scipy pandas matplotlib
```

> **Note:** Numba JIT-compiles the core DP solver (`run_model`) and probability simulation (`probabilities`) with `nopython=True, parallel=True`. First run will trigger compilation (~10–30 s); subsequent calls are fast.

---

## Performance Notes

| Lever | Effect |
|---|---|
| `n_p = 0` | Fastest — single price path, intrinsic only |
| `n_p = 30` | Full optionality; tree has 61 price states |
| `v_step` | Larger step → fewer inventory states → faster |
| Numba parallel | Price loop parallelised across CPU cores automatically |
