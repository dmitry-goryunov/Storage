# Put Swing Delta Investigation

## Setup

- Put swing option: right/obligation to buy on exactly **30** cheapest days over 2027
- Parameters: valDate=2026-01-01, storageStart=2027-01-01, storageEnd=2027-12-31, days=30, vol=0.50, n_p=30, v_step=1000 MWh
- Hard terminal constraint: `t_p_curve[k≠30] = -1e9`, `t_p_curve[30] = 0`
- Flat benchmark price (flat_metric): ~25.07 EUR/MWh

## Observations

Two apparent anomalies in the delta profile:

1. **January**: sharp step-down at Jan 31
2. **December**: delta amplified relative to expected exercise (|delta|/|exp_ex| > 1)

## Finding 1 — January Step-Down at Jan 31

A "leading wave" of ~4.5% probability mass forms from the first day of the contract and exercises continuously from Jan 1, concentrated in extreme-low-price states (j=0..18, i.e. S/F < 1).

Day-by-day tracking:

| Date   | k  | prob    | n_buy_states |
|--------|----|---------|--------------|
| Jan 29 | 28 | 0.04609 | 19           |
| Jan 30 | 29 | 0.04483 | 19           |
| Jan 31 | 30 | 0.04371 | 0            |

On Jan 30: 97.5% of the k=29 wave exercises → flows to k=30 on Jan 31. At k=30 the quota is met, so 0 states exercise. This removes ~44 MWh/day from exp_ex, causing the step-down from ~−65 to ~−21 MWh/day.

**Conclusion**: correct model behavior. The leading wave is a direct consequence of the mandatory quota: paths that are in cheap-price states exercise every day from Jan 1, saturating the quota by Jan 30.

## Finding 2 — December Delta Amplification

By December, ~10% of paths are at k<30 and under mandatory quota pressure. December forward prices (~25.69 EUR) exceed the flat benchmark, so forced injections occur at S > F.

- Monthly December: delta = −3,710 MWh vs exp_ex = −2,169 MWh → ratio **1.71**
- Dec 31: |delta|/|exp_ex| = 1.71

Delta formula: `delta[i] = −Σ prob[i,j,k] × action[i,j,k] × exp(x[i,j]) / fwd[i]`

When exp(x) > fwd (i.e. S > F), each exercised unit contributes more than 1 to |delta|.

**Conclusion**: correct model behavior. The amplification reflects that quota-forced purchases happen at above-forward spot prices; those purchases are correctly hedged with more than 1:1 forward delta.

## Full-Year Sanity Checks

| Metric | Value |
|--------|-------|
| Total exp_ex | −30,000 MWh ✓ (mandatory quota met in all paths) |
| Total delta | −27,674 MWh (avg S/F at exercise ≈ 0.92) |
| Probability sum | 1.0 at all times ✓ |

## Soft Constraint Experiment (Conducted and Reverted)

Replaced the hard terminal penalty with a linear graduated penalty:

```python
t_p_soft[:days + 1] = -(days - np.arange(days + 1)) * (flat_metric * v_step)
```

Results at different penalty multipliers:

| Multiplier | Total ExpEx (MWh) | Jan31 prob_k30 | Dec Delta |
|------------|-------------------|----------------|-----------|
| 1.0×       | −20,495           | 0.029          | −1,210    |
| 2.0×       | −29,544           | 0.044          | −2,855    |
| 5.0×       | −30,000           | 0.044          | −3,710    |
| 10.0×      | −30,000           | 0.044          | −3,710    |

Threshold: multiplier ≥ **1.19×** (penalty must exceed max forward 29.72 EUR) to force full exercise. Below that threshold there is a volume shortfall. At or above threshold the behavior is identical to the hard constraint including the same January and December artifacts.

There is no "sweet spot" between smooth delta and full exercise — this is a fundamental property of the mandatory-quota formulation.

**Current state**: Hard constraint restored. Soft constraint was reverted.

---

# Peer Review of the Above — Independent Verification

*Reviewer: Claude (Opus 5), 2026-09-08. Everything below is reproducible with `python review_checks.py`; the section name for each block is given in brackets.*

## Reproduction [repro]

The base case reproduces exactly from a clean run (quote matrix, FDDate 2026-01-05, DA stub):

| Quantity | This run | finding.md |
|---|---|---|
| V₀ | −678,682.8 EUR | — |
| Total exp_ex | −30,000.0 MWh | −30,000 ✓ |
| Total delta | −27,674.4 MWh | −27,674 ✓ |
| flat_metric | 25.0724 EUR/MWh | ~25.07 ✓ |
| Jan 30 → Jan 31 exp_ex | −65.1 → −21.3 MWh/day | −65 → −21 ✓ |
| December delta / exp_ex | −3,709.8 / −2,169.0 = 1.71 | 1.71 ✓ |

One additional invariant worth keeping: **Σᵢ deltaᵢ · Fᵢ = V₀** (−678,682.8 vs −678,682.8). The value is linear in the price level, so the delta profile must reprice the contract exactly. It does — which means the metrics are internally consistent with the DP, and rules out any bug in `strat` → `prob` → `exp_ex/delta` for the base configuration. This is the single best regression test for the model and it currently exists nowhere in the repo.

## Finding 1 — confirmed, but the original test was confounded

The base case has **n_p = 30 and days = 30**. The tree's growing phase ends at step `i = n_p`, i.e. exactly at Dt+30 = **Jan 31** — the same date as quota saturation. From `_tree_core`, the dynamics genuinely change at that step: the full-width branch installs the truncated edge probabilities (`p_d[i,0] = 0`, `p_u[i,2n_p] = 0`). So two candidate mechanisms land on the same day, and the write-up picks one without excluding the other.

Separating them by varying `n_p` and `days` independently [np_invariance]:

| n_p | days | Step date | Day of window | exp_ex before → after |
|---|---|---|---|---|
| 20 | 30 | 2027-01-31 | 30 | −65.5 → −21.2 |
| 30 | 30 | 2027-01-31 | 30 | −65.1 → −21.3 |
| 45 | 30 | 2027-01-31 | 30 | −65.1 → −21.3 |
| 30 | **20** | **2027-01-21** | **20** | −78.5 → −20.7 |

The step tracks `days` and is invariant to `n_p` — the whole January profile is identical to within 0.4 MWh/day across n_p = 20/30/45. **Quota saturation confirmed; tree geometry excluded.**

Supporting tree diagnostics at these parameters [tree]: all transition probabilities lie in [0,1] (min p_m = 0.660), q sums to 1, forward fitting is exact to 1e-15, the terminal log-price std (0.351) matches the OU stationary value σ/√(2κ) = 0.354, and the mass parked on the two truncated boundary nodes is 1.5e-4 at the end of the grid. The boundary cannot produce a 44 MWh/day step.

**Verdict: Finding 1 stands as written.**

## Finding 2 — confirmed numerically, with a sharper statement available

The delta formula is a *multiplicative* (sticky-moneyness) bump: the tree is forward-fitted by shifting `x` additively in log space, so scaling F scales every node's spot. Under that bump `∂V/∂F · F = −E[S·Q]`, which is exactly what `compute_all_metrics` computes. The envelope theorem covers the quota constraint (the constraint does not depend on F), so no extra term is missing.

Verified by finite differences on the DP itself — ±5 bp multiplicative bump, one month at a time [delta_fd]:

| Month | Analytic Σδ·F (EUR) | FD @ 5 bp | Error | FD @ 1 % | Error |
|---|---|---|---|---|---|
| Jan-27 | −41,932 | −41,921 | −0.03 % | −41,785 | −0.35 % |
| Apr-27 | −68,308 | −68,093 | −0.31 % | −69,215 | +1.33 % |
| Jul-27 | −60,556 | −60,477 | −0.13 % | −65,746 | **+8.57 %** |
| Sep-27 | −43,611 | −43,620 | +0.02 % | −51,759 | **+18.68 %** |
| Nov-27 | −20,254 | −20,175 | −0.39 % | −29,135 | **+43.85 %** |
| Dec-27 | −95,403 | −95,424 | +0.02 % | −94,132 | −1.33 % |

Every bucket matches its finite difference to within ±0.4 % at a 5 bp bump. **The delta is a correct local derivative, month by month — Finding 2's conclusion is independently confirmed.**

Two refinements to the write-up:

1. **The ratio has a closed form.** |delta|/|exp_ex| = E[S · 1_exercise] / (F · P(exercise)) = **E[S | exercise] / F**. It is not really about "S > F on forced injections"; it is the conditional expected spot given exercise, so the ratio exceeds 1 exactly when exercise is positively correlated with price. The monthly profile makes the story readable [repro]:

   | | Jan | Mar | May | Jul | Sep | Oct | Dec |
   |---|---|---|---|---|---|---|---|
   | ratio | 0.53 | 0.45 | 0.78 | 0.96 | 1.07 | 1.24 | **1.71** |
   | E[S \| exercise] | 14.59 | 11.86 | 18.80 | 22.77 | 25.93 | 30.21 | **43.94** |

   The ratio rises monotonically from April onward and crosses 1.0 in August. Early exercise is *chosen* (cheap states, E[S|ex] = 12–19 vs a forward of 24–27); late exercise is *forced* (E[S|ex] = 44 against a December forward of 25.69). The 1.71 is the endpoint of a smooth trend, not a December anomaly — which is stronger evidence for "correct model behavior" than the single-month figure.

2. **The December delta is the most reliable bucket in the profile; the summer ones are not.** The convexity scan [bucket_convexity] shows December's bucketed delta is stable (−1.3 % at a 1 % bump, +8.7 % at 5 %) because quota-forced exercise is inelastic, while July's collapses (+8.6 % at 1 %, +32 % at 2 %, +119 % at 5 %) because those days sit on the exercise boundary and reallocate as soon as the month moves. For the monthly delta table in `streamlit_app.py` this is the practical caveat: the summer buckets are correct derivatives but decay within a single day's market move and need re-hedging; the winter buckets do not. Bumping each month by 1 % separately and summing gives −721,107 vs −678,683 for a parallel bump — a 6 % gap that is pure gamma.

**Verdict: Finding 2 stands; the ratio is better stated as E[S | exercise] / F, and the delta profile deserves a convexity caveat.**

## Soft-constraint experiment — reproduces, but the stated threshold is wrong

The experiment replicates exactly [soft_penalty]: 1.0× → −20,495 MWh, 2.0× → −29,544 MWh / Dec delta −2,855, 5.0× and 10.0× → −30,000 / −3,710. All four numbers match the table above.

The stated threshold does not. At the claimed **1.19×** the model buys only **−24,530 MWh** — 82 % of the quota, not 100 %:

| mult | 1.00 | 1.19 | 1.50 | 2.00 | 3.00 | **4.00** | 5.00 |
|---|---|---|---|---|---|---|---|
| total exp_ex | −20,495 | −24,530 | −27,747 | −29,544 | −29,983 | **−30,000** | −30,000 |

Full exercise first appears at **~4×**, and the table in the section above already shows this (2.0× is 456 MWh short). The reasoning behind 1.19× — "penalty must exceed max forward 29.72 EUR" — is the *deterministic* threshold. Under the tree the buyer can be forced to buy at spot, not at the forward, and the maximum spot reachable in the exercise window is ≈ max(F) · e^(n_p·dx) = 27.41 × 3.89 ≈ **107 EUR ≈ 4.26× flat_metric** — which is what the empirical 4× threshold reflects.

This makes the section's conclusion *stronger*, not weaker: the penalty needed to replicate the hard constraint is ~3.4× larger than thought, and — the more troubling part — it depends on `n_p` and `sVol`, i.e. on model discretisation parameters rather than on contract economics. A graduated penalty is therefore not merely a no-sweet-spot trade-off; it would make the answer a function of the tree width. Recommend recording that as the reason not to revisit it.

## Not covered by this investigation

The delta *profile* was validated; the **per-MWh metric built from it was not**, and that is where the actual bug is. `value_put_swing` reports `stochastic_metric = full_eur / np.sum(s.delta)` — value divided by the price-weighted delta (27,674 MWh) rather than by volume (30,000 MWh). At the base case this reports **24.52 EUR/MWh instead of 22.62**, an 8.4 % error, and it moves the wrong way as optionality is added (23.80 → 24.52 as n_p goes 0 → 30, while the true average purchase price falls 23.80 → 22.62). Finding 2's own headline — that delta ≠ volume by construction — is precisely why that denominator cannot be used. See `code_review.md` #1.
