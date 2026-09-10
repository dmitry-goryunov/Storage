# Response to the independent review — 2026-09-10

[INDEPENDENT-REVIEW-2026-09-10.md](INDEPENDENT-REVIEW-2026-09-10.md) arrived after
[FINDINGS-2026-09-10.md](FINDINGS-2026-09-10.md) and
[DESIGN-P4.1-two-factor.md](DESIGN-P4.1-two-factor.md) were written. This is what happened
when its claims were checked, and what is outstanding as a result.

**Nothing here had been fixed when this was written.** The ranking at the end was a
proposal; it was then replied to in
[REVIEW-REPLY-AND-ACTION-PLAN-2026-09-10.md](REVIEW-REPLY-AND-ACTION-PLAN-2026-09-10.md),
which accepted the findings, raised three qualifications (accepted in place below) and set
the delivery order. **Since then: item 2, the inventory bounds, is repaired and the
correction notices and [`benchmarks.py`](../benchmarks.py) fixtures are in. Item 1, the
ratchet discretisation, is still open.**

---

## Provenance

The review ships an evidence archive with SHA-256 hashes of every file it inspected. All
**34 match this working copy byte for byte** — library, kernels, both test modules, all four
storage/swing notebooks, the workbook and every doc. It is a review of exactly this
snapshot, so nothing below can be explained away as drift. Its own suite run agrees with
ours: 101 passed.

Each claim was reproduced from the review's **prose**, not by running its scripts, so
agreement is independent evidence rather than a re-execution. Where a number below matches
the review to six decimals, two separate constructions arrived at it.

## Verdict

| § | Claim | Verdict |
|---|---|---|
| 2.1 | The bound check tests an expectation, not the state distribution | **Confirmed**, breach probabilities match exactly |
| 2.2 | Initial inventory is dropped from the cumulative series after day zero | **Confirmed**, false rejection reproduced |
| 2.3 | `round()` moves floors down and ceilings up | **Confirmed**, both directions |
| 3 | The 52.5 % cap and the infeasible 70 % floor are grid artefacts | **Confirmed**, all five rows match exactly |
| 4 | Fuel reproduces; the zero-net-hedge statement needs a fuel caveat | **Confirmed**, −1.522843 MWh/100 MWh |
| 5 | Correlations reproduce; the DA gap claim is wrong | **Confirmed**, six decimals; gap is 5 days |
| 6.3 | A common long factor adds nothing to a zero-fee store | **Confirmed** at machine precision |
| 6.4 | Log-ratio vol is not monotone in mean reversion | **Confirmed**, peak at κ = 1.385 |
| 6.5 | The ninefold shortfall is not established | **Confirmed**, it is ~3.1× |
| 6.6 | The 4.2 % extrinsic baseline is not reproducible | **Confirmed**, 19.62 % on the only recorded config |
| 7 | The repricing identity is not special to an exhaustive DP | **Confirmed** at 5.7e-14 on a deliberately bad policy |
| 8 | Memory budget and `int8` overflow | **Confirmed** by arithmetic |

Nothing in the review was found to be wrong. One qualification is noted at the end.

---

## What it overturns

### The ratchet result was discretisation, not physics

The same physical store — 600,000 MWh, 20,000 MWh/day in, 10,000 out, same ratchets, dates,
curve and 10 % funding — with only the clip size moving:

| inventory clips | clip, MWh | in c/d | out c/d | peak opening | value, EUR | 70 % floor |
|---:|---:|---:|---:|---:|---:|---|
| 240 | 2,500.00 | 8 | 4 | **52.50 %** | **1,279,390** | rejected |
| 480 | 1,250.00 | 16 | 8 | 72.08 % | 1,751,509 | accepted |
| 960 | 625.00 | 32 | 16 | 83.54 % | 2,019,763 | accepted |
| 1,920 | 312.50 | 64 | 32 | 88.54 % | 2,134,480 | accepted |
| 3,840 | 156.25 | 128 | 64 | **90.68 %** | **2,182,561** | accepted |

The 480-clip value is 36.90 % above the 240-clip value; the 3,840-clip value is 70.59 %
above it, and the last refinement still moves it 2.25 %. An independent continuous-volume
backward-reachability calculation — 92 withdrawal days, start-of-day ratchet convention —
permits **92.7565 %** opening inventory on 1 October, matching the review's figure to four
decimals.

The mechanism is `int(wdr_rate × multiplier)`. At 10 % full the contract allows
0.38 × 10,000 = **3,800 MWh/day**; the 240-clip grid delivers `int(4 × 0.38) = 1` clip =
**2,500 MWh/day**, a **34.2 %** shortfall. The store then refuses to fill above what that
crippled rate can drain before the window closes. `assert_ratchets_expressible` catches only
the truncate-to-*zero* case; 1.52 → 1 passes silently.

**So these are withdrawn:**

- FINDINGS "Ratchets cap a store through the exit… this store peaks at **52.5 %** full."
- FINDINGS "Ratchets and dated bounds can be **jointly infeasible** … so it was physical,
  not economic."
- FINDINGS "**Refining the clip is free and answer-neutral.**" — true unratcheted, and only
  ever checked there.
- The same three claims in the `Storage_30_60.ipynb` input cell, which names the truncation
  and then concludes physics from it.
- `test_ratchets_cap_the_store_through_the_exit_not_the_entry`, which asserts
  `0.4 < peak < 0.7` and that the 70 % floor raises. It passes by encoding the artefact.

The shipped notebook configuration is itself not converged:

| clips | total, EUR | intrinsic | extrinsic | extrinsic share | vs 240 |
|---:|---:|---:|---:|---:|---:|
| 240 | 2,941,740 | 2,364,556 | 577,184 | 19.62 % | — |
| 480 | 2,993,115 | 2,383,334 | 609,781 | 20.37 % | +1.75 % |
| 960 | 3,015,924 | 2,390,304 | 625,620 | 20.74 % | +2.52 % |

Without ratchets the same deal gives **3,210,646 EUR at 60, 240 and 480 clips — identical to
the euro**. That is the check that was run, and the one case where it could not fail.
Refinement costs 0.1–0.3 s here, so there is no performance argument for staying coarse.

### The two load-bearing arguments for a two-factor lattice do not hold

**The repricing identity is not a reason to prefer the lattice.** On 200,000 Monte Carlo
paths, with a deliberately silly, non-optimal, path-dependent policy — sell a unit whenever
today's price exceeds the running average, otherwise buy one:

```
  value of the silly policy   : 38.570578660141
  sum_i DF_i * delta_i * F_i  : 38.570578660141
  difference                  : 5.684e-14
```

With a EUR 2/MWh fee on each leg it closes once the cost leg is written down explicitly,
residual 2.1e-12. The identity is linearity of expectation applied to a definition: with
`delta_i := E[S_i Q_i]/F_i`, `sum_i DF_i delta_i F_i = E[sum_i DF_i S_i Q_i] = V_policy` for
any adapted policy. It cannot distinguish a good policy from a bad one, and LSMC satisfies
it too. [DESIGN-P4.1-two-factor.md](DESIGN-P4.1-two-factor.md) calls it "**Lost**" under
LSMC and rests the recommendation on that.

**A Schwartz–Smith long factor adds exactly nothing to a zero-fee store.** An independent
small DP — 24 decision dates, four inventory levels, OU short factor, independent
martingale long factor, hard terminal condition, zero fees:

| κ | one factor | two factors, σ_ξ = 0.8 | difference |
|---:|---:|---:|---:|
| 0.2 | 42.8418319181 | 42.8418319181 | 1.4e-14 |
| 1.0 | 43.5801756083 | 43.5801756083 | 7.1e-15 |
| 4.0 | 47.4214212411 | 47.4214212411 | 2.1e-14 |

*(A first attempt showed 1e-5. That was this test's own ξ lattice absorbing mass at its
boundary and breaking the martingale; widened past reach, it collapses to machine zero.)*

The proof is three lines. `L_t = exp(ξ_t − ½Var ξ_t)` is a martingale; zero-fee storage
cashflows are homogeneous of degree one in price; the admissible set does not depend on
price. So the optimal policy is scale-invariant, `V_t = L_t · W_t` where `W` is the
one-factor value, and `L_0 = 1`. The long factor integrates out. The review's continuous-time
version extends this to ρ ≠ 0 through a measure change that shifts χ's mean without touching
its OU coefficients — which forward-fitting absorbs.

**The boundary of this result matters as much as the result.** It requires zero fixed cash
fees, price-independent admissibility and a homogeneous terminal condition. A store with
non-zero `inj_cost`/`wdr_cost` breaks it, and so does a **struck** swing — `Products.ipynb`
carries a `STRIKE`. The reduction removes the *storage* case for a second factor. It does
not remove the swing case, which is untested.

---

## Corrections to what is written down

| where | says | measured |
|---|---|---|
| FINDINGS, DESIGN, STATUS | c6/c12 spread vol "**nine times** too little" | **about 3.1× at one point-maturity comparison** — σ·\|a₁−a₂\| = 0.11933 at σ 0.5, κ 1, against 0.373 pooled raw (3.13×) or 0.37776 excluding rolls (3.17×). The ninefold claim is unsupported as written; **3.1× is illustrative, not an established replacement** — delivery averaging, observation alignment, estimation window and parameter provenance are all still unspecified |
| FINDINGS, DESIGN | storage extrinsic is "**4.2 %** of value" | **19.62 %** on the shipped notebook, 23.77 % unratcheted. 4.2 % matches no recorded configuration |
| DESIGN | "1.1 % at sMR 0.2, 10.5 % at sMR 4.0" | **1.91 %** and **36.86 %** ratcheted; 2.64 % and 42.90 % unratcheted |
| DESIGN | the repricing invariant is "**Lost**" under LSMC | It is an identity for any adapted policy |
| DESIGN | a κ sign flip is "**the acceptance test for the whole item**" | Log-ratio vol is non-monotone in κ, peaking at κ = 1.385. A correct model can fail this gate |
| FINDINGS | DA prints have "**no gap over three days**" | Longest gap is **5 calendar days**, with **34** intervals over three. Three days is the *date index*, not the observations |
| FINDINGS | "Intrinsic delta nets to zero; the whole net hedge is extrinsic" | True at zero fuel. At 1.5 % the intrinsic delta nets to **−1.522843 MWh per 100 MWh cycled** — exactly the gas burnt |

The direction of the spread finding survives: one factor does understate the seasonal spread,
and extrinsic does rise with mean reversion (1.91 % at κ 0.2 to 36.86 % at κ 4.0, measured).
What fails is the magnitude, the arithmetic behind it, and the proposed acceptance gate.

The design's interim formula divides by `sqrt(2κ)`, which converts an OU factor's stationary
*level* standard deviation, not the annualised volatility of daily log-ratio *changes*. The
comparable quantity is `σ·|e^(−κτ₁) − e^(−κτ₂)|`. The 0.041 figure reconstructs from neither
formula at the reference deal's own σ = 0.5, κ = 1.

## What reproduced unchanged

- **Fuel.** 3,212,220.14 EUR at 0 %, 2,925,765.10 at 1.5 % — a **8.9177 %** reduction, and
  the multiplier 1/(1−0.015) = 1.0152284264. FINDINGS' 8.9 % stands.
- **Market statistics.** All six correlations to six decimals: c1/c3 0.917095, c1/c6
  0.801426, c6/c12 0.770636, c1/c12 0.742070, c12/c24 0.802850, c1/c24 0.632737. Workbook
  shape too: 4,171 rows, 4,117 numeric DA prints, 12 Mar 2010 to 6 Mar 2026.
- **Hedge instability.** Oct–Dec near-immovable, Apr–Aug swinging about half the book's
  largest position — consistent with what the stability table already reports.
- **The memory budget.** 580 × 41 × 31 × 61 = 44,967,980 nodes; at 240 clips 177,660,380
  (1.42 GB of float64 probability); at 390 clips 288,237,380 (2.31 GB). And `int8` for
  `strat` overflows: a refined 3,840-clip grid filled in 30 days needs **128 clips/day**
  against a 127 ceiling.

## Where to push back

One small thing, changing no conclusion.

- The review **understates its own ratchet case**. It cites 16.7 % rate loss, which is the
  shortfall at the bottom multiplier; at 10 % fullness on the same grid it is 34.2 %. The
  reply notes correctly that its figure was for a different inventory level and that it
  explicitly allowed larger errors elsewhere.

> **Withdrawn, later the same evening.** A second bullet here said the review "treats
> 'extrinsic rises with mean reversion' as an unsupported baseline". It does not: §6.4
> demonstrates that increase in its own two-factor example, and objects only to the
> unreproduced numerical baselines and to requiring a sign reversal as a gate. I had read
> §6.6's dispute of the endpoints as a dispute of the direction.

**Not checked here:** the exploratory covariance fit and PCA shares in §5, the claim that six
consecutive imposed floors reach only 59.2 %, and the review's process observations about
commit history and timings. The first would need a calibration harness that does not exist
yet.

---

## Outstanding work, ranked

Ranked by how wrong the answer is today, not by how interesting the work is.

### 1. Ratchet discretisation *(review §3; absorbs P1.4)* — **INSTRUMENTED**

The shipped notebook understates its own deal by ≥2.5 % and is still moving; the test suite
understates the harsher profile by 70 %; and three written claims are false. Not merely
"refine the grid" — it needs a stated discretisation policy: report effective MWh/day by
inventory level, and gate on value and feasibility convergence at fixed physical inputs.
Retain the zero-rate guard as an early check. Retract the notebook prose, the FINDINGS
bullets, and replace the test.

> **Done, later the same evening**, except the last decision. `describe_ratchet_rates()`
> reports contract against grid MWh/day per inventory level, and the loss turns out to be a
> **sawtooth** — exactly zero where `rate × multiplier` lands on an integer, **33 % just
> below one** — on the *mild* shipped profile, not the harsh one. A mild ratchet is not a
> safe ratchet. `benchmarks.py` carries the inventory ladder, a separate price-grid ladder
> and the convergence verdict (< 0.5 % on total *and* intrinsic over each of two successive
> doublings, with an absolute EUR floor). The shipped deal converges at **3,840 clips** at
> 3,032,172 EUR; the notebook now ships 960 behind a `REFINEMENT` knob and prints that it is
> **0.53 % low**. Still open: whether to pay ~8 s a valuation for the converged grid by
> default, fractional-volume actions if not, and `inj_days` meaning two things.

### 2. The three inventory-bound defects *(review §2; settles P1.1)* — **DONE**

Three bugs in about thirty lines of [storage_model.py](../storage_model.py). Two give a
wrong answer, one gives none:

- `round()` on both sides silently relaxes the contract — use `ceil` for floors, `floor` for
  ceilings, report the effective grid bound, and reject an empty admissible interval.
- The cumulative reconstruction drops `initial_state` after day zero. Read the balance from
  `prob[t, :, l]` instead of rebuilding it.
- The check compares an expectation. The exact inventory law is one line away; a contractual
  bound needs a probability tolerance, not an expected-inventory comparison.

The third was a decision, not a fix: hard admissibility in the DP versus the present
`1000·v_step` penalty. **Answered — hard.** The terminal condition was hardened with it: a
`-1e9` penalty is purchasable for the same reason, and the store was buying its way out of
it. Values on deals whose optimal policy already satisfied their bounds are unchanged to the
euro; only a breach that was profitable moves.

### 3. Correct the documents

The seven rows in the corrections table above, across FINDINGS, DESIGN, STATUS and the
notebook. Cheap, and it stops the next person building on numbers that do not reconstruct.

### 4. Rescope P4.1

The homogeneity reduction removes the storage rationale entirely. What survives:

- **Step 1, the Kalman calibration, becomes the whole of P4.1** — and absorbs **P3.4** (the
  unexplained `sVol = 0.9`) and **P4.3** (calibrate at the sensitivity that matters). Three
  roadmap items collapse into one.
- A second factor may still be needed for **struck swings**, where homogeneity fails. That
  is one experiment, and it should precede any lattice work.
- If a two-factor engine is built anyway, the review's §8 acceptance gates replace the
  design's — analytic moments, joint transition validity, hard feasibility, controlled
  one-factor and zero-volatility limits, convergence in both factor and inventory grids.

### 5. Calibration itself *(currently deferred)*

The spread understatement is real and σ and κ are unfit defaults, so this is the
highest-value new work in the repo and cheaper than the item it replaces. Worth reopening
ahead of the backtest, which stays deferred — a backtest answers "is it better", and nothing
is yet better.

> **Corrected, later the same evening.** This called it "a **one-factor** calibration
> problem", which is premature. The homogeneity reduction removes an extra *valuation state*
> for one particular factor — a common multiplicative one — under zero fees. It says nothing
> about a factor that changes *relative* prices, and a joint market calibration can move the
> short-factor estimate even where valuation later reduces to one state. **Market-model
> selection and valuation-state reduction are separate decisions.** The comparison should run
> a one-factor OU baseline, Schwartz–Smith, two differing mean-reversion speeds and a
> seasonal-loading candidate, without committing in advance to any of them or to Kalman.

One spread observation is one equation in two unknowns. At κ 0.2 the realised 0.373 needs
σ = 4.33; at κ 1.0, σ = 1.56; at κ 4.0, σ = 3.19. It pins a curve, not a point, so the
calibration needs the whole panel.

### 6. Unchanged and not pressing

| | |
|---|---|
| P1.2 | curve-shape acceptance criteria |
| P1.3 | discount-curve source, purpose and settlement timing — partly addressed |
| P2.1 | shortening the terminal backstop |
| P2.2 | the tie threshold — measured and declined 2026-09-10, leave it |
| — | the backtest — leave deferred |
| — | `ttf q.xlsx` row 4172, which needs a data source rather than a decision |

---

## The pattern worth keeping

Every claim the review overturned was **checked in the one configuration where it could not
fail**. Refinement was tested unratcheted, where the rates are exactly expressible. The
zero-net-hedge statement was measured at zero fuel, in the document that introduced fuel. The
bound checker was only ever run from an empty store. The tests passed throughout, and 101 of
them still pass over all three.

Passing tests are evidence about the tests.
