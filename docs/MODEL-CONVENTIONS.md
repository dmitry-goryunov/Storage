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

**`delta` is not a procurement plan.** `|delta| / |exp_ex| = E[S | exercise] / F`, so it
exceeds volume exactly when exercise is positively correlated with price — and for a
mandatory swing that is systematically the back months, where paths still short of quota
must buy whatever the price. On the 10-day put swing over 2027, December expects 1,657 MWh
of gas but carries 2,481 MWh of delta, a ratio of 1.50 with `E[S|exercise]` at 39.86
against a 26.62 forward; January runs the other way at 0.46. Hedge December at its delta
and you buy 824 MWh you will not physically need; plan procurement off delta and the whole
book is 8 % short (−9,213 against a −10,000 quota).

Neither number is wrong. **`exp_ex` answers "how much gas", `delta` answers "how much price
risk"**, and the swing's volumetric uncertainty is not hedgeable with a forward at all —
that uncertainty *is* the optionality being valued. On the reference put
swing it runs from 0.45 in March (exercise chosen at cheap prices) to 1.71 in December
(exercise forced at expensive ones). Both numbers are correct; they answer different
questions. Use `exp_ex` for physical volume and `delta` for the hedge.

**Two hedge ratios, because there are two hedge instruments.** `delta` assumes an OTC
forward that settles with the deal: your exposure's PV sensitivity is `DF·E[S·Q]` and the
forward's is `DF·h·F`, so the discount factor cancels and you trade the physical volume.
Hedge instead with **exchange futures carrying daily variation margin** and it does not
cancel — margin cash moves today while the gas settles at delivery — so the hedge must be
*tailed* by the discount factor. `delta_pv = d_curve · delta` is that tailed series, and it
is also the right number for PV risk reporting. The two coincide when the rate is zero.

The difference is not small on a forward-dated book. The 10-day put swing over 2027 valued
2026-01-01 at 10 %:

| | delta | tailed | |
|---|---:|---:|---:|
| Jan-27 | −179 | −162 | −9.6 % |
| Jun-27 | −2,410 | −2,086 | −13.5 % |
| Dec-27 | −2,481 | −2,036 | −17.9 % |
| book | −9,213 | −7,819 | −15.1 % |

`delta_pv` also reprices the contract on its own — `sum(delta_pv · fwd) == V0` — which is
the master invariant with the weights already inside the series.

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
- **The two sides move in opposite directions.** A seller receives cash, so a positive rate
  pulls exercise *earlier*; a buyer pays cash, so it pushes exercise *later*. On a flat
  curve at 10 % both book about 1.0 EUR/MWh of pure timing gain (call 0.998, put 0.974).
  On a sloped curve the rate competes with the slope and each side flips once it wins: on a
  +18 %/yr curve the put buys on day 45 up to 20 % and on day 350 at 40 %, while the call
  does the reverse.
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

## Financing is a cash flow, and the curve can charge for it

**It is interest, not an artefact.** The deterministic schedule and the flat benchmark move
identical gas at identical prices, so their nominal totals match exactly and only the timing
differs. The gap is money the deal has not paid out; the PV of the interest it earns at
`discount_rate` *is* the financing number. On the 10-day put swing, flat 40, `K = 30` at
10 %: 100,000.00 EUR nominal on both legs, a balance peaking at 97,260.27 EUR on
2027-12-21, and 4,192.93 EUR of PV interest against 4,192.36 EUR reported — 0.01 %. You
receive it only if that balance genuinely earns the rate. It is a hurdle-rate gain on your
own cash, not something the gas market pays.

**A curve in contango at the discount rate charges the gas for it, with no new input.** The
optimiser never sees `F`; it sees `DF · F`. A flat forward curve beside a positive rate is
internally inconsistent — gas costs the same in December as in January while money costs
10 % — and the financing gain is the model reporting that inconsistency as free money. Put
the curve in contango at the same rate and `DF · F` is flat to 1e-14: `intrinsic` is exactly
zero, and its two halves come out large and equal-and-opposite, buying early being cheaper
on the curve by precisely what paying early costs in funding. Extrinsic is untouched (2.67
on the reference deal), because optionality comes from volatility, not slope.

**`borrow_rate` / `invest_rate` replace the single rate when the two directions differ.**
`discount_rate` assumes spare cash earns exactly what borrowed cash costs. Give the pair
instead and the rate follows the deal's own direction — a net payer funds at the borrow
rate, a net receiver places cash at the invest rate. Direction is taken from the sign of the
mean forward net of strike, not the product type, because a strike flips it: a put swing
struck above the curve receives rather than pays. Setting both forms raises, as does setting
one of the pair alone, as does a storage deal, which pays on injection and receives on
withdrawal and so has no single direction.

It is **one rate for the whole deal, not one per day.** A rate chosen from the sign of each
day's cash flow would make the value non-linear in the price level and break the master
invariant below.

## How the reported metrics compose

They nest. They are not terms to add side by side:

    flat                     the benchmark: F - K per MWh, with no choice of days
      -  intrinsic           what choosing days is worth on today's forward curve
           =  shape          ... from picking cheaper days
           +  financing      ... from picking later ones; zero without a discount rate
      -  extrinsic           what re-choosing as prices move adds
      =  price               what the deal actually pays

So `flat = price + intrinsic + extrinsic` for a buyer and `price = flat + intrinsic +
extrinsic` for a seller, with `shape` and `financing` the two halves of `intrinsic`
rather than two more terms beside it. `vs flat` in the notebook is measured off the
prices and `total` off the decomposition, so those agreeing is a cross-check; both are
asserted in `tests/` to 1e-9 on either side, struck and unstruck.

## `intrinsic` is not an option's intrinsic value

The names collide, and the two mean different things.

An option splits its value into **intrinsic** — the moneyness, `max(F - K, 0)` — plus time
value. A swing has many exercise days rather than one, so the split here is of a different
quantity: what you gain by **choosing days** rather than taking the window flat.

| | |
|---|---|
| `flat` | the moneyness, `F - K` per MWh. What the deal is worth with no choice of days at all |
| `intrinsic` | what the *deterministic* optimum adds over that by picking days on today's forward curve — plus financing, once a rate is on |
| `extrinsic` | what re-picking as prices move adds on top |

So with a flat curve at 40 and `K = 30`, `flat` is **10.000** — that is the 10 you would
look for — and `intrinsic` is **0**, correctly: no day beats any other, so choosing them is
worth nothing.

**A `put_swing` is also not a put.** It is the obligation to buy, so there is no
`max(·, 0)` anywhere. At a zero rate the value is exactly linear in the strike —
`dV/dK` is 10,000 MWh at every strike from 0 to 60 on the 10-day deal, the fixed volume,
with no kink at the money and a straight line to 1.6e-16 relative. A real 30-strike put
against a 40 forward is out of the money, worth zero, and would never be exercised; this
contract buys anyway.

Turn the rate on and a little genuine convexity in `K` does appear — `dV/dK` rises from
8,761 to 8,891 across the same range, 0.4 %. That is not option payoff convexity; it is the
schedule responding to the strike, which it can only do because `-K sum DF_i q_i` depends
on when you exercise. See the strike note above.

## Sense-checking on a flat curve

A curve with no shape has no day-selection value, so the model's outputs collapse to
answers you can state before running it. This is the configuration to check against when
a number looks wrong. Use `CURVE_SOURCE = "flat"` in `Products.ipynb`, which builds a
synthetic curve and touches no data file — **do not flatten a row of `ttf q.xlsx`**, which
silently changes every other valuation reading it and cannot be undone from the
spreadsheet.

At level 40, put swing over calendar 2027, valued 2026-03-06, `sMR = 1.0`:

| Setting | Must give | Measured |
|---|---|---|
| no vol, no rate | nothing to gain | I+E = 9e-06, pays 39.999991 |
| vol 50 %, no rate | **intrinsic exactly 0** — no shape, no deterministic edge | 0.00e+00 at every size |
| no vol, rate 10 % | **extrinsic exactly 0** — no optionality left, so all value is financing | −6e-15, intrinsic 1.677 |
| forced to take all 365 days | pays the curve exactly | 40.000000, I+E = 0 |
| deal size rising | price rises monotonically towards the level | 35.819 → 40.000, no reversal |

Two readings worth keeping:

- **On a flat curve, intrinsic *is* the discount rate.** It is 0 at 0 % and 1.677 at 10 %
  for the 10-day deal. None of that is day selection; there is none to be had. If
  intrinsic is not ~0 at 0 % on a flat curve, something other than the rate is moving.
- **Vol is what pulls exercise forward.** With no vol at 10 % the buyer defers to the very
  end — mean exercise day 359.5 of 365, because when you pay is the only thing left to
  optimise. Switch vol back on and it sits at 195: the financing pull towards the end
  against the option to wait for a dip.

**Put and call are mirror images, but not in euros.** At vol 50 % the put buys 4.181 below
the level and the call sells 4.395 above it. In log terms that is −0.1104 against +0.1043
— the call's *log* move is the smaller one, which is what a lognormal does: the fat right
tail turns a smaller log move into a larger euro move. The gap widens with vol (0.078 at
30 %, 0.215 at 50 %). A symmetric result in euros would be the bug.

## The master invariant

    sum_i  d_curve[i] * delta[i] * fwd[i]  ==  v[0, n_p, n_op_start]

The value is linear in the price level, so the reported hedge must reprice the contract.
This ties the DP, the forward pass and the reported metrics together: if any of the three
disagrees about what the optimal strategy does, it breaks. Asserted in
`tests/test_reconciliation.py` to a relative 1e-9, with and without a live discount curve.
While `d_curve` is all ones it reduces to `sum(delta * fwd) == V0`.

## Per-MWh metrics

`profiled()` is `v[0, n_p, initial_state] / sum(exp_ex)` — value
divided by **MWh actually exercised**, not by `sum(delta)`. Dividing by delta was an 8.4 %
error that moved the wrong way as optionality was added. `flat()` is a different quantity:
the unweighted average forward over the exercise window, independent of `n_p` and of the
optimisation.

The reported `flat_metric` is that average **net of any strike**, because the value it
benchmarks is. Selling every day at the average forward earns `mean(F) - K` per MWh, not
`mean(F)`. Benchmarking a strike-net value against a raw average made `intrinsic` shift by
`K` — downwards for a call, upwards for a put — while the spread it is meant to measure
barely moves. Fixed 2026-09-08; `flat()` on the `Storage` object is unchanged and still raw.

**A strike is neutral only at a zero rate.** Undiscounted, the cost is `sum (P_i - K) q_i`
and the `K` leg is a constant times a fixed volume, so it cannot reorder the exercise days —
schedule and split come out identical struck or not. Discounted it becomes
`sum DF_i (P_i - K) q_i`, and the `-K sum DF_i q_i` term rewards days with *large* discount
factors, pulling exercise earlier against the deferral the rate otherwise buys. Because the
financing gain scales with the net cash actually moving rather than the gross index, a deep
strike nearly removes it: on the seasonal test curve a 20.00 strike against a ~25 average
takes financing from 0.481 to 0.002 and returns the schedule to its undiscounted optimum.
Pass the run's strike to `intrinsic_components`, or the discount factors it divides out are
recovered from the wrong legs.

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
- Forward curves come from `curve.csv` or the `ttf q.xlsx` quote matrix. The quote row is
  selected by date — the last quote on or before the as-of date — and a curve built from a
  quote may **no longer start before that quote**: valuing 2026-01-01 off a 2026-03-06
  quote is look-ahead, and it used to back-fill the two intervening months with that
  quote's day-ahead price. `curve_df_for_storage` now raises. `curve.csv` carries no quote
  date at all, so nothing selects a row from it and an as-of date against it means nothing;
  `Products.ipynb` raises rather than ignoring one. The quote date is now carried into the
  notebook's curve label and results caption, but there is still **no such convention in
  the library's own outputs** — `run_valuation` returns no as-of stamp.
- There is no parameter-sensitivity standard, so no statement of how much a value moves
  per unit of vol or mean reversion.

Treat every number this model produces as exploratory until these are recorded.
