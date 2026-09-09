# Model conventions

What the model's inputs and outputs mean. Written because several of these were
discoverable only by reading the code, and two of them were being read the wrong way
round. Anything asserted here is checked in `tests/`; anything unknown is marked as such
rather than guessed.

Date: 2026-09-08.

---

## Prices and time

| | |
|---|---|
| Time step | `dt = 1/365.25` — daily, calendar, no business-day calendar anywhere |
| Grid | `n_t` steps from `valDate` to `backStop` = one month-end past `storageEnd` |
| Price process | Ornstein-Uhlenbeck in log price, mean-reverting to the forward curve |
| `sVol` | **Annualised** instantaneous log-volatility, as a fraction (0.6 = 60 %). The tree spacing is `dx = max(sVol) * sqrt(3*dt)` |
| `sMR` | Mean-reversion speed **per year**. In the kernel the drift is `sMR * dt * x`, and `dt` is in years |
| Stationary spread | `sVol / sqrt(2*sMR)` in log terms — 0.35 at `sVol=0.5, sMR=1.0`. Useful for checking `n_p` is wide enough: the tree half-width is `n_p * dx` |
| Forward fitting | At every step the tree is shifted so `E[S] = fwd[i]` exactly (to 1e-15). `exp(x[i,j])` is therefore an absolute price, not a ratio |

## Volumes

| | |
|---|---|
| `v_step` | The **clip**: MWh per inventory state |
| `clips_per_day` | Daily injection/withdrawal rate, in clips. Capacity per active day is `clips_per_day * v_step` |
| `n_op` | Number of inventory states = grid size + 1 |
| Working volume | `(n_op - 1) * v_step` |
| Rates are integers | A rate is a whole number of clips per day. Anything slower than one clip per day is **not expressible**; it needs a smaller `v_step`. See the roadmap |

**Two names carry two meanings — the most common source of error here.**

- ~~`n_op_start`~~ **fixed 2026-09-08.** It used to mean the grid size to
  `set_volume_states()` and the initial state to `build()`, so every caller set it twice.
  The two now have their own names: **`n_states`** (grid size, so `n_op = n_states + 1`)
  and **`initial_state`** (where inventory starts, in `0..n_states`), settable in one call
  as `set_volume_states(n_states, initial_state=...)`. `n_op_start` survives as a
  deprecated alias for `initial_state`. `build()` now rejects a start outside the grid,
  which nothing checked before.
- `inj_days` is the inventory **state count** in `resolve_grid` (legacy path), and
  **days-to-fill** in `params_for_run_valuation` (workbook path). `wdr_days` exists only in
  the second sense; passing it on the first path now raises rather than being ignored.

## Signs

Negative is buying, positive is selling, for both reported series:

| Product | `exp_ex` | `delta` |
|---|---|---|
| Put swing (obligation to buy) | negative | negative |
| Call swing (right to sell) | positive | positive |
| Storage | negative on injection days, positive on withdrawal days | same |

`exp_ex[i]` is expected MWh moved on day `i`; both series have length `n_t + 1`, with a
trailing `0.0` so they align with `date_span`.

## `delta` — an undiscounted hedge volume

`delta[i]` is **the number of MWh of forward for day `i` that you would trade to hedge**,
`E[S_i·Q_i] / F_i`. It is *not* a PV sensitivity, and it carries no discount factor.
(Decision D-O2, 2026-09-08.)

Why: hedging day `i` with `h` forwards gives `PV = h·DF_i·F_i·ε` against
`dV/dε = DF_i·E[S_i·Q_i]`, so the discount factor cancels and `h = E[S_i·Q_i]/F_i`. A
discounted delta would move when the yield curve moved and the gas did not, and would
under-hedge by `DF` — 4.6 % at 3 %, 7.7 % at 5 %, worse with tenor.

**`delta` ≠ expected volume.** `|delta| / |exp_ex| = E[S | exercise] / F`, so it exceeds
volume exactly when exercise is positively correlated with price. On the reference put
swing it runs from 0.45 in March (exercise chosen at cheap prices) to 1.71 in December
(exercise forced at expensive ones). Both numbers are correct; they answer different
questions. Use `exp_ex` for physical volume and `delta` for the hedge.

**It is a local derivative.** Validated against finite differences on the DP to ±0.4 % at a
±5 bp bump, every month. But it is strongly convex: at a 1 % bump, July is out by +8.6 %,
November by +43.9 %, while quota-forced December holds at −1.3 %. Re-hedge rather than
trusting a monthly bucket across a large move.

## Discounting and settlement

- `d_curve[i]` is a **discount factor to the valuation date** for a cash flow on day `i`.
  Set it from `discount_rate` — an annual, continuously compounded rate, so
  `d_curve[i] = exp(-r·i/365.25)` — as a `Storage` argument or a `run_valuation` param.
  It defaults to `0.0`, which gives all ones and changes nothing. Assign `d_curve`
  directly for a real, non-flat curve.
- **This is what makes the model prefer early withdrawal.** The DP multiplies every day's
  cash flow by `d_curve[i]`, so cash released sooner is worth more, and the optimiser
  reschedules accordingly. Before the rate existed, the timing of a withdrawal carried no
  value at all and the intrinsic leg could not see it.
- **The benchmark is PV'd too.** `daily_arithmetic_flat_metric` returns the mean of
  `DF·(F − K)` over the window, not the raw mean forward. Both legs of
  `intrinsic = profiled − flat` must be present-valued or the difference measures the
  discount factor rather than the day-selection spread — the same mistake as benchmarking
  a strike-net value against a raw average.
- **A hurdle rate and a funding rate answer different questions.** Discounting a hedgeable
  commodity cash flow at, say, a 10 % internal project rate is a capital-budgeting view —
  what the cash is worth to *this* business — not a mark-to-market. It will systematically
  favour early exercise relative to what a hedged book would do, and the resulting number
  is not comparable to a broker quote. Both are legitimate; choose deliberately and record
  which one a given valuation used.
- The DP discounts each day's cash flow once, by `d_curve[i]`; continuation values are not
  re-discounted.
- **Settlement is assumed on the exercise day.** There is no separate settlement-lag
  convention. A real gas contract paying month-end + N days would need `d_curve` built
  accordingly.

## The master invariant

    sum_i  d_curve[i] * delta[i] * fwd[i]  ==  v[0, n_p, n_op_start]

The value is linear in the price level, so the reported hedge must reprice the contract.
This ties the DP, the forward pass and the reported metrics together: if any of the three
disagrees about what the optimal strategy does, it breaks. Asserted in
`tests/test_reconciliation.py` to a relative 1e-9, with and without a live discount curve.
While `d_curve` is all ones it reduces to `sum(delta * fwd) == V0`.

## Per-MWh metrics

`price_per_mwh()` (aliased `profiled()`) is `v[0, n_p, n_op_start] / sum(exp_ex)` — value
divided by **MWh actually exercised**, not by `sum(delta)`. Dividing by delta was an 8.4 %
error that moved the wrong way as optionality was added. `flat()` is a different quantity:
the unweighted average forward over the exercise window, independent of `n_p` and of the
optimisation.

The reported `flat_metric` is that average **net of any strike**, because the value it
benchmarks is. Selling every day at the average forward earns `mean(F) - K` per MWh, not
`mean(F)`. Benchmarking a strike-net value against a raw average made `intrinsic` shift by
`K` — downwards for a call, upwards for a put — while the spread it is meant to measure
does not depend on the strike at all: a constant per-MWh amount cannot reorder the exercise
days. Fixed 2026-09-08; `flat()` on the `Storage` object is unchanged and still raw.

## Kernel constants

| Constant | Meaning |
|---|---|
| `-1e9` | A forbidden terminal inventory state in `t_p_curve` |
| `1e10` | An infeasible transition inside `run_model` (`bigdummy`) — a different quantity from the above |
| `1000 * v_step` | Penalty per clip outside a tunnel. Arbitrary; scale is on the roadmap |
| `1e-6` | An exercise whose gain over idling is smaller is snapped to idle. Absolute, not scale-aware; on the roadmap |

## Calibration — not established

This is the honest gap. The model is verified (it computes what it claims); it is not
validated (nothing here establishes the claims are right for a traded price).

- `sVol` defaults to 0.9 with **no recorded provenance** — no estimation window, method or
  as-of date.
- `sMR` defaults to 1.0, likewise.
- Forward curves come from `curve.csv` or the `ttf q.xlsx` quote matrix; the quote row is
  selected by date, but there is **no recorded as-of convention** tying a valuation to a
  quote date in the outputs.
- There is no parameter-sensitivity standard, so no statement of how much a value moves
  per unit of vol or mean reversion.

Treat every number this model produces as exploratory until these are recorded.
