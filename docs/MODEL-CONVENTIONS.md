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

- **Everything money-denominated is a present value at `valDate`, with four labelled
  exceptions.** `d_curve[i] = exp(-r·i/365.25)` counts `i` from `valDate`, `d_curve[0]` is
  exactly 1, and the value is read at time index 0. Price the same window from two
  valuation dates and the deterministic value differs by exactly `exp(-r·ΔT)` — checked to
  3.6e-16 in `tests/`. So `flat_metric`, `profiled_metric`, `stochastic_metric`,
  `intrinsic`, `extrinsic`, `shape`, `financing`, every `value EUR` and every line of the
  P&L bridge are PVs to `valDate`.

  The exceptions, each labelled where it is reported:

  | | |
  |---|---|
  | The bridge's `Obligation at the forward (F - K)` | nominal on purpose — the next line is the effect of discounting it |
  | §2's window mean | printed "gross of strike and undiscounted", against the PV'd `flat` beside it |
  | §6b's nominal cash totals and peak balance | a cash balance *is* a balance on a date; the `PV of interest` line beside it is the PV |
  | `delta` | an undiscounted hedge volume by decision D-O2; `delta_pv` is the discounted twin |

  Volumes — `exp_ex`, MWh, mean exercise day — are not money and carry no discounting.

- `d_curve[i]` is a **discount factor to the valuation date** for a cash flow on day `i`.
  Set it from `discount_rate` — an annual, continuously compounded rate, so
  `d_curve[i] = exp(-r·i/365.25)` — as a `Storage` argument or a `run_valuation` param.
  It defaults to `0.0`, which gives all ones and changes nothing. Assign `d_curve`
  directly for a real, non-flat curve.
- **This is what makes the model prefer early settlement of receipts.** The DP multiplies
  every day's cash flow by `d_curve[i]`, so an earlier receipt has a larger PV and the
  optimiser reschedules accordingly. Payments move the other way. Before the rate existed,
  settlement timing carried no value in the optimiser.
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

## Discounting, settlement timing and funding are different

**The model's settlement-timing attribution can be represented as interest under the stated
scenario.** The deterministic schedule and the flat benchmark move
identical gas at identical prices, so their nominal totals match exactly and only the timing
differs. The gap is money the deal has not paid out; under this scenario, the PV of interest
at `discount_rate` matches the legacy timing attribution. On the 10-day put swing, flat 40, `K = 30` at
10 %: 100,000.00 EUR nominal on both legs, a balance peaking at 97,260.27 EUR on
2027-12-21, and 4,192.93 EUR of PV interest against 4,192.36 EUR reported — 0.01 %. You
receive it only if that balance genuinely earns or avoids the rate. This is not established
merely by discounting the commodity cash flow.

**A constructed curve growing at the discount rate makes `DF · F` flat.** The optimiser
never sees `F` in isolation; it sees `DF · F`. If the test curve is defined as
`F(t) = F(0)·exp(r·t)` and the same `r` is used for discounting, deterministic timing value
vanishes exactly. This is a useful invariant of the implementation. It is not a general
commodity-carry relationship, and it does not show that an observed flat gas forward curve
at positive rates is inconsistent. Storage costs, convenience yield, seasonality, transport
constraints and supply/demand can offset or dominate financial carry.

**`borrow_rate` / `invest_rate` are explicit treasury scenarios, not an FVA model.** Give
the pair together and set `funding_direction` to `borrow` or `invest`. The direction is not
inferred: optional exercise can select receipts even when the window-average `F-K` is a
payment, and stochastic prices can cross the strike. Setting both `discount_rate` and the
pair raises, as does setting one of the pair alone. Storage is refused because it pays on
injection and receives on withdrawal and has no single direction.

The same validation is used by `run_valuation`, optional columns in `products.xlsx`,
`Products.ipynb` and the single-deal Streamlit app. Presence is explicit:
`discount_rate=0.0` selects the market/valuation-rate mode and therefore still conflicts
with a borrow/invest pair. Blank fields are absent; zero and negative finite rates are not.

The helper still applies **one scenario rate for the whole deal**. A genuine asymmetric
funding valuation is nonlinear and requires the cash balance in the state, or an equivalent
nonlinear recursion. It should not be approximated by silently choosing a rate from the
mean forward or by applying different discount factors to individual expected cash flows.

For production use, record at least: rate purpose (market discount, treasury funding or
internal hurdle), curve currency and source, as-of date, day count, compounding,
interpolation, contractual settlement dates and payment lag. `run_valuation` currently
supports a scalar continuously compounded `discount_rate`; a production term-curve source
and settlement-date mapping remain roadmap work.

## How the reported metrics compose

They nest. They are not terms to add side by side:

    flat                     the benchmark: F - K per MWh, with no choice of days
      -  intrinsic           what choosing days is worth on today's forward curve
           =  shape          ... price selection at the benchmark discount factor
           +  timing         ... settlement timing at the benchmark price
           +  interaction    ... price-selection × timing cross-effect
      -  extrinsic           what re-choosing as prices move adds
      =  price               what the deal actually pays

So `flat = price + intrinsic + extrinsic` for a buyer and `price = flat + intrinsic +
extrinsic` for a seller. `vs flat` in the notebook is measured off the
prices and `total` off the decomposition, so those agreeing is a cross-check; both are
asserted in `tests/` to 1e-9 on either side, struck and unstruck.

### Intrinsic attribution is not unique

When both the forward curve and discount factors vary, price selection and settlement
timing interact. Writing `P` for the schedule's undiscounted average price net of strike,
`Fbar` for the window average and `DF_paid`/`DF_bench` for their effective discount factors:

    intrinsic = DF_bench·(P-Fbar)
              + Fbar·(DF_paid-DF_bench)
              + (P-Fbar)·(DF_paid-DF_bench)

The terms are day selection at the benchmark discount factor, settlement timing at the
benchmark price, and their interaction. `intrinsic_attribution()` reports all three.
Legacy `intrinsic_components()` preserves its two-number API by adding the interaction to
the second value. Calling that combined value “financing” is an attribution convention;
it is not proof that the balance can be funded or invested at the discount rate.

## `intrinsic` is not an option's intrinsic value

The names collide, and the two mean different things.

An option splits its value into **intrinsic** — the moneyness, `max(F - K, 0)` — plus time
value. A swing has many exercise days rather than one, so the split here is of a different
quantity: what you gain by **choosing days** rather than taking the window flat.

| | |
|---|---|
| `flat` | the moneyness, `F - K` per MWh. What the deal is worth with no choice of days at all |
| `intrinsic` | what the *deterministic* optimum adds over the flat schedule through price selection, settlement timing and their interaction |
| `extrinsic` | what re-picking as prices move adds on top |

So with a flat curve at 40 and `K = 30`, `flat` is **10.000** at a zero rate, and
`intrinsic` is **0**: no day beats any other and settlement timing has no value. At a
positive rate the same curve can have non-zero intrinsic solely because the exercise-day
settlement dates differ.

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
| no vol, rate 10 % | **extrinsic exactly 0** — no optionality left, so all intrinsic is settlement timing | −6e-15, intrinsic 1.677 |
| forced to take all 365 days | pays the curve exactly | 40.000000, I+E = 0 |
| deal size rising | price rises monotonically towards the level | 35.819 → 40.000, no reversal |

Two readings worth keeping:

- **On a flat curve, intrinsic is settlement-timing value under the selected rate.** It is
  0 at 0 % and 1.677 at 10 % for the 10-day deal. None of that is day selection. If
  intrinsic is not ~0 at 0 % on a flat curve, something other than the rate is moving.
- **Vol is what pulls exercise forward.** With no vol at 10 % the buyer defers to the very
  end — mean exercise day 359.5 of 365, because when you pay is the only thing left to
  optimise. Switch vol back on and it sits at 195: the settlement-timing pull towards the end
  against the option to wait for a dip.

**Put and call are mirror images, but not in euros.** At vol 50 % the put buys 4.181 below
the level and the call sells 4.395 above it. In log terms that is −0.1104 against +0.1043
— the call's *log* move is the smaller one, which is what a lognormal does: the fat right
tail turns a smaller log move into a larger euro move. The gap widens with vol (0.078 at
30 %, 0.215 at 50 %). A symmetric result in euros would be the bug.

## Fuel loss, and why it splits volume from hedge

`fuel_loss` is the fraction of injected gas retained for compression, so putting one clip
into inventory takes `1/(1 - fuel_loss)` clips out of the market. Real storage retains 1–2 %.
It defaults to `0.0`, which changes nothing.

**It scales the price, not the cost.** The fuel is taken in kind, so what it costs depends on
what gas is worth that day; `inj_cost` is a fixed EUR/MWh and cannot express that. The charge
therefore lands on the price leg in the kernel: the per-clip injection cash becomes
`-price * inj_fuel_mult - inj_cost`. On the reference store 1.5 % retention costs **8.9 %** of
value — 3,211,211 EUR falling to 2,925,871 — because it both taxes every cycle and removes
the marginal ones.

**Decision D-O3, 2026-09-10: `exp_ex` stays physical, `delta` becomes the traded volume.**
With a loss the two part company, and they answer their usual different questions:

| | |
|---|---|
| `exp_ex` | gas that moves in and out of the store. Still nets to zero over a cycle — a store gives back what it takes |
| `delta` | MWh of forward to trade, which on the way *in* is `1/(1 - fuel_loss)` times the inventory move, and on the way *out* is equal to it |

Measured on a deterministic tree, where `E[S | exercise] = F` makes the ratio exact: at 1.5 %
loss, `delta / physical` is 1.015228 on injection days and 1.000000 on withdrawal days.

The convention is not arbitrary — it is the one under which **the repricing identity still
closes**. `sum_i DF_i · delta_i · F_i == V0` holds at 1.8e-15 for every loss tested. Had
`delta` remained the inventory volume, the identity would have needed a separate fuel term
and the reported hedge would no longer have repriced the deal, which is the test that told
us which convention was right.

## Ratchets, and the grid they need

`ratchets` takes a table of fullness to rate multiplier, and the daily rate at inventory
level `l` becomes `rate * multiplier(l)`. The DP moves whole clips, so the kernel takes
`int(rate * multiplier)` — and **a positive multiplier that floors to zero silently freezes
the store**. Withdrawal at 1 clip/day with a 0.30 multiplier near empty gives 0.30 clips,
floors to 0, and the store can never take out its first clip: the deal prices at exactly
zero, with no error. On a 60-state grid an ordinary profile did precisely that, returning
0 EUR against 5,299,882 unratcheted.

`assert_ratchets_expressible` now refuses it, naming the offending fullness, the multiplier,
the resulting fractional clips and the grid that would fix it. **The smallest non-zero
multiplier `m` needs a base rate of at least `ceil(1/m)` clips per day** — divide `v_step` by
that factor and multiply `n_states` by it. The same profile on 240 states prices at
2,807,291 EUR against 5,299,882 without ratchets: a realistic ratchet costs **47 % of value**,
which is the number the silent version was hiding.

Two details worth knowing. A multiplier of **exactly zero** is left alone — that is the
legitimate way to say a rate is shut off at some fullness. And the table is **interpolated**
onto the states, so a profile ramping from 0 to 1 over several states puts intermediate
multipliers inside the truncation zone and is refused; put the knots on state boundaries if
you mean to shut off one state only.

None of the current notebooks set ratchets, so every storage valuation in them assumes rates
independent of fullness. That overstates flexibility, and on the reference deal by about
half.

## Dated inventory bounds

A storage contract that says "1 October inventory at least 70 %" is expressed as

    params["min_inventory"] = {"2027-10-01": 0.70}
    params["max_inventory"] = {"2027-04-01": 0.30}

date to **fraction of working volume** — what the contract says, and it survives a change of
clip size. Several dates may be given; each constrains that day only.

**The bound applies to the balance the day opens with**, before that day's injection or
withdrawal. That is where the model's penalty attaches, and it is what most contracts intend,
but it is not the only reading: the same schedule reported on the *closing* balance shows the
day's move already applied and can look a clip short. A 70 % floor on 1 April binds at
exactly 42.00 of 60 clips on the opening balance and 41 on the closing one. Roadmap P1.1
covers promoting this from documented to chosen.

**It is a penalty, not a hard constraint** — `1000 * v_step` per clip out of bounds. Large
against an ordinary deal, but a number rather than a guarantee, so a big enough contract can
pay it and breach the bound. `value_storage` therefore re-checks every bound on the built
policy and raises if one did not hold, rather than returning a valuation of a contract nobody
asked for. A floor that is physically unreachable — 100 % by 15 January on a 30-day fill from
empty — fails there.

Bounds are set after `set_volume_states`, which resets the tunnel arrays. A date outside the
model's grid raises rather than being ignored, as does a fraction outside [0, 1] and a floor
above its ceiling on the same day.

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
timing attribution scales with the net cash actually moving rather than the gross index, a deep
strike nearly removes it: on the seasonal test curve a 20.00 strike against a ~25 average
takes the legacy timing component from 0.481 to 0.002 and returns the schedule to its undiscounted optimum.
Pass the run's strike to `intrinsic_attribution` or `intrinsic_components`, or the discount factors they divide out are
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
