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
