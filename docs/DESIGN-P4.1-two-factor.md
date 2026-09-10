# P4.1 — a second factor, so the seasonal spread can move

> ## Correction notice, 2026-09-10 (evening)
>
> **Do not implement this plan as written.** An
> [independent review](INDEPENDENT-REVIEW-2026-09-10.md) found three of its
> arguments wrong; all three were checked and the review is right. Measurements
> in [REVIEW-RESPONSE-2026-09-10.md](REVIEW-RESPONSE-2026-09-10.md).
>
> 1. **The long factor need not add any storage value.** With a common
>    multiplicative long factor, zero fixed fees, price-independent constraints
>    and a homogeneous terminal condition, storage cashflows are homogeneous of
>    degree one, the optimal policy is scale-invariant, and the factor integrates
>    out against its own martingale. An independent DP agrees to **1e-14** across
>    κ = 0.2, 1.0, 4.0. Fixed cash fees, a strike, or a factor that changes
>    *relative* prices break the reduction — so this narrows the case, it does
>    not close it. Market-model selection and valuation-state reduction are
>    separate decisions.
> 2. **The repricing invariant is not lost under LSMC.** With
>    `delta_i := E[S_i Q_i]/F_i`, `sum_i DF_i delta_i F_i = V_policy` is linearity
>    of expectation for *any* adapted policy — 5.7e-14 on a deliberately
>    non-optimal Monte Carlo policy. The fork table below is wrong on the one row
>    the recommendation rests on. Determinism and exhaustive policy enumeration
>    are still reasons to prefer a lattice; this is not.
> 3. **A mean-reversion sign flip is not a valid acceptance test.** The log-ratio
>    volatility `σ·|e^(−κτ₁) − e^(−κτ₂)|` is non-monotone in κ, peaking at
>    κ = 1.385. Step 5 would reject a correct model.
>
> **Numbers withdrawn:** the "nine times too little" comparison divided by
> `sqrt(2κ)`, which converts a stationary *level* standard deviation, not the
> annualised volatility of daily log-ratio *changes*. At σ 0.5, κ 1 the model
> gives **0.11933** against 0.373 realised — about **3.1×**, and that is one
> point-maturity comparison, not an established replacement. Delivery averaging,
> observation alignment, estimation window and parameter provenance all remain
> unspecified. The **4.2 %** extrinsic baseline and its 1.1 % / 10.5 % endpoints
> match no recorded configuration; the shipped notebook gives **19.62 %**, and
> 1.91 % / 36.86 % across the same κ range.
>
> **What survives.** The realised correlations (c1/c6 0.801, c6/c12 0.771,
> c1/c24 0.633) reproduce to six decimals, one factor does understate the
> seasonal spread, and extrinsic does rise with mean reversion — the review
> confirms that increase in its own example and objects only to using its
> reversal as a gate. **Step 1, the calibration, survives intact and is now the
> whole of the item.** Steps 2–6 need a fresh justification from struck swings or
> fee-bearing contracts, not from storage value.

Design note, 2026-09-10. Not yet implemented. Written because the item is large enough that
starting it without agreeing the numerical method would be a mistake, and because the case
for doing it at all is now measured rather than argued.

---

## The problem, in one line

The model has **one** state variable, so every forward is a deterministic function of it and
any two forwards correlate **exactly 1.000**. A storage contract is a bet on the *spread*
between two forwards. The model therefore cannot price the thing the asset is for.

## The evidence

Realised daily log returns from `ttf q.xlsx`, 2015 onward, roughly 2,800 observations:

| pair | realised correlation | spread vol (annualised) |
|---|---:|---:|
| c1 vs c3 | 0.917 | 0.282 |
| c1 vs c6 | 0.801 | 0.423 |
| **c6 vs c12** | **0.771** | **0.373** |
| c1 vs c12 | 0.742 | 0.475 |
| c12 vs c24 | 0.803 | 0.302 |
| c1 vs c24 | 0.633 | 0.560 |

The model says every one of these is 1.000. On the reference deal it gives the summer/winter
spread a log-vol of about **0.041** against a realised **0.373** — **nine times too little**.

The consequence is visible without any market data at all. Storage extrinsic is 4.2 % of
value and *rises* with mean reversion — 1.1 % at `sMR` 0.2, 10.5 % at `sMR` 4.0 — which is the
signature of short-term cycling value, not spread optionality. Faster reversion gives more
spot wiggle to trade against and does nothing whatever for the spread.

## The model

Schwartz–Smith. Log spot is the sum of a short-term deviation and a long-term equilibrium:

```
ln S_t = chi_t + xi_t

d chi = -kappa * chi dt + sigma_chi dW_chi        (mean reverting to zero)
d xi  =  mu_xi dt      + sigma_xi  dW_xi          (random walk)
corr(dW_chi, dW_xi) = rho
```

Forwards then carry both, with the short factor damped by maturity and the long factor not:

```
ln F(t,T) = exp(-kappa*(T-t)) * chi_t  +  xi_t  +  A(T-t)
```

Two forwards are no longer perfectly correlated, because they load differently on `chi` while
loading identically on `xi`. Correlation rises towards 1 for nearby maturities and falls for
distant ones — the shape the table above shows. The current model is the special case
`sigma_xi = 0`, which is the right regression test.

## The fork that has to be decided first

Everything downstream depends on this, so it is step 0, not step 3.

| | Two-factor lattice | Least-squares Monte Carlo |
|---|---|---|
| Grid | `v[time, chi, xi, inventory]` | paths × time, regress continuation |
| Cost | 580 × 41 × 31 × 61 ≈ **45 M nodes**, roughly 25× today | linear in paths; no dimensional blow-up |
| Exact policy | Yes — exhaustive over feasible moves, as now | No — regression approximates continuation |
| **Repricing invariant** | Should survive; needs re-derivation | **Lost.** `sum(DF·delta·F) == V0` is exact today because the DP is exhaustive |
| Extending to a third factor | Painful | Easy |
| Fits the existing code | Yes — same kernel shape, one more loop | No — a parallel implementation |

**The invariant is the reason to prefer the lattice.** It is this project's cornerstone: it
ties the DP, the forward pass and the reported metrics together, it is asserted at 1e-15
across every product, and it is what caught the `price_per_mwh` sign bug and settled decision
D-O3. Trading it for a numerical method that cannot support it would remove the main reason
to trust anything the model says. LSMC is the right answer for a five-factor book; it is the
wrong answer for a codebase whose main asset is that identity.

**Recommendation: the lattice**, with LSMC reconsidered only if the memory work below fails.

## Steps

### 0. Confirm the fork
Agree lattice over LSMC, or overturn it. One conversation; everything else assumes lattice.

### 1. Estimate the parameters before building anything
The Kalman filter on the forward panel is the standard route: the state `(chi, xi)` is hidden,
the observations are the 55 contract log-prices, and the measurement equation is the forward
formula above. `ttf q.xlsx` has 4,171 daily rows and 55 contracts — ample.

Do this **first**, because it is cheap, it is reusable regardless of the fork, and it answers
a question that decides the size of the prize: what are `sigma_chi`, `sigma_xi`, `kappa` and
`rho` actually worth on TTF? If `sigma_xi` comes back near zero the whole item collapses and
the current model is vindicated. The correlation table above says it will not, but measure.

Deliverable: a fitted parameter set with its estimation window and method recorded, and a
plot of model-implied against realised correlation by maturity pair.

### 2. Build the process, alone
`two_factor_tree` beside the current one, not replacing it.

- Rotate to independent coordinates so the lattice recombines. With `rho ≠ 0` the two
  Brownian motions are correlated; a Cholesky rotation gives independent axes and the tree
  recombines on each.
- Forward-fit as now: shift so `E[S_i] = fwd[i]` exactly at every step. This is what makes
  `exp(x)` an absolute price rather than a ratio, and it must survive.
- Probabilities in [0,1] with the same rejection of unstable configurations.

Tests before any DP work:
- terminal variance against the analytic Schwartz–Smith formula, across maturities;
- the correlation between two forwards, against the analytic value, across maturity pairs;
- **`sigma_xi = 0` reproduces the current one-factor tree to machine precision** — the
  regression test that makes the rest safe.

### 3. Extend the DP
- `v` and `strat` become 4-D: `[time, chi, xi, inventory]`.
- The Numba kernel gains one loop. The inner logic — the move search, the tie snap, the
  tunnel penalty, the ratchets — is unchanged.
- **Memory is the real constraint.** At 45 M nodes, `v` alone is 360 MB in float64. Two
  mitigations, both worth doing: keep only two time slices of `v` (backward induction needs
  `v[i+1]` only), and store `strat` as int8. `strat` must persist for the forward pass;
  `v` need not.
- Expect a rebuild of the Numba cache and a materially longer first compile.

### 4. Re-derive the metrics and the invariant
This is the step most likely to hold a surprise, so budget for it rather than assuming.

- `exp_ex` is unchanged — physical volume does not care how many factors there are.
- `delta` is `E[S_i Q_i] / F_i` as now, but the expectation is over a two-dimensional state.
  The formula should carry over; verify rather than assert.
- **Re-check `sum_i DF_i · delta_i · F_i == V0`.** The value is still linear in the forward
  *level*, which is what the identity rests on, so it ought to hold — but "ought" is not the
  standard used elsewhere in this repo. If it does not hold, stop and understand why before
  proceeding; that answer is more valuable than the feature.
- The hedge acquires a second dimension in principle — exposure to the short factor and to
  the long factor separately — which is a reporting question worth deciding once the numbers
  exist.

### 5. Measure what it bought
- Storage extrinsic under two factors against 4.2 % under one, on the same deal.
- Whether extrinsic still rises with mean reversion. It should not: under two factors the
  spread has its own volatility and mean reversion should reduce extrinsic, not raise it.
  **That sign flip is the acceptance test for the whole item.**
- The one-factor result should sit inside the two-factor result as `sigma_xi → 0`.

### 6. Decide what becomes of the one-factor path
Keep it for swings, where a single factor is defensible and the grid is cheap; or retire it.
Do not decide this before step 5.

## The cheap interim, if the above is too much

Keep one factor but calibrate it to the **spread's** volatility rather than the spot's.
Spread log-vol is `|exp(-kappa*tau1) - exp(-kappa*tau2)| * sigma / sqrt(2*kappa)`, so
`(sigma, kappa)` can be solved to match an observed spread vol at a chosen maturity pair.

The result is the wrong model with the right sensitivity for one product. It is defensible
**only if recorded in the output**, because the same parameters will then misprice anything
short-dated — a swing priced off a spread-calibrated vol would be badly wrong. If this route
is taken, the parameter set must be tagged with what it was fitted to, and the swing
notebooks must refuse it.

## Risks

- **The invariant may not survive.** Mitigation: step 4 is a gate, not a formality.
- **Memory.** Mitigation: two-slice `v`, int8 `strat`, and measure before optimising.
- **Calibration turns out to be the hard part**, not the lattice. Likely, and it is why step 1
  comes first.
- **Scope creep into the deferred backtest.** A two-factor model with fitted parameters will
  invite the question "is it better?", which only a backtest answers. Note it, resist it, or
  reopen the backtest deliberately.

## Effort

Rough, and honest that estimates on this kind of work are unreliable:

| step | |
|---|---|
| 1 — Kalman calibration | the largest single piece; a self-contained project |
| 2 — process and its tests | moderate, well-specified |
| 3 — DP extension | small in code, fiddly in memory |
| 4 — metrics and invariant | small if the identity holds, open-ended if it does not |
| 5–6 — measurement and decisions | small |

Step 1 is worth doing on its own merits even if steps 2 onward are never started: it converts
`sVol = 0.9` from an unexplained default into an estimate, which is P3.4 and P4.3 as well.
