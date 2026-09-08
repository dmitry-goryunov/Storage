# Code Review — `storage_model.py`, `streamlit_app.py`

*Reviewer: Claude (Opus 5), 2026-09-08. Scope: the pricing library and the Streamlit front end, at the working-tree state of 2026-09-08 (uncommitted). Every claim below was executed, not inferred; reproduce with `python review_checks.py <section>`. The delta investigation itself is reviewed separately at the end of [finding.md](finding.md).*

## What is solid

Worth stating first, because it narrows where the problems can be:

- **The delta profile reprices the contract exactly.** Σᵢ deltaᵢ · Fᵢ = V₀ to 0.1 EUR on 678,683. Value → strategy → probabilities → metrics is internally consistent. [`repro`]
- **The tree is well built** at the parameters in use: transition probabilities in [0,1] (min p_m = 0.660), q sums to 1 to 1e-15, forward fitting exact to 1e-15, terminal log-std 0.351 against the OU stationary target σ/√(2κ) = 0.354, boundary mass 1.5e-4. [`tree`]
- **Degenerate cases return the right answer.** A quota equal to the window (days = 365 of 365) prices at exactly `flat_metric` = 25.0724 EUR/MWh. [`infeasible`]
- **The value converges in `n_p`** — 22.6215 at n_p = 20 vs 22.6228 at n_p = 60. [`convergence`]
- The 3-colour `prange` scheme in `probabilities` is race-free as claimed (j-values in a pass are 3 apart, writes touch rows j−1..j+1), and the backward `prange` over k writes row i while reading row i+1 only. Both are correct.

Findings are ordered by expected damage.

---

## 1. `stochastic_metric` divides value by delta instead of by volume — **8.4 % error, wrong sign of response**

`storage_model.py:312` (put) and `:368` (call)

```python
stochastic_metric = full_eur / np.sum(s.delta)
```

`sum(delta)` is the price-weighted forward-equivalent volume (27,674 MWh at the base case), not the contract volume (30,000 MWh). The idiom is inherited from the `n_p = 0` leg, where `delta ≡ exp_ex` because S = F at every node — at `:298`, inside the `n_p = 0` branch, the code does exactly that and is correct there. It stops being true the moment the tree has width.

Measured [`convergence`]:

| n_p | true EUR/MWh (V₀/volume) | as coded (V₀/Σδ) | extrinsic |
|---|---|---|---|
| 0 | 23.8017 | 23.8017 | 0.0000 |
| 10 | 22.7963 | 24.5236 | 1.0054 |
| 30 | **22.6228** | **24.5238** | 1.1790 |

Two separate defects. The level is wrong by **1.90 EUR/MWh (8.4 %)** — large next to the whole option premium of 2.45 EUR/MWh. And the *response* is inverted: adding optionality makes the reported price rise (23.80 → 24.52) while the true average purchase cost falls (23.80 → 22.62). For the call swing the bias runs the other way (selling concentrates in high-price states, so Σδ > volume and the price is understated).

Exposure: `result["total"]` whenever "Run intrinsic decomposition" is unchecked in the app — i.e. the headline number in the fast path. With the checkbox on, `total = intrinsic + extrinsic` is computed against `acq` from the `n_p = 0` run and is correct.

**Fix:** `denominator = -np.sum(s.exp_ex)` for the put (`+` for the call), or simply `params["days"] * v_step`. Guard against zero.

## 2. `Storage.flat()` and `Storage.profiled()` return 0 for any `n_p > 0`, and are the same function

`storage_model.py:150-157`

```python
def flat(self):      return self.v[0, 0, self.n_op_start] / np.sum(self.delta)
def profiled(self):  ACQ = np.sum(self.delta); return self.v[0, 0, self.n_op_start] / ACQ
```

Byte-identical bodies, and both index price state **0**. At time 0 only node `k = n_p` exists — `_tree_core` fills `x[i, j]` for `j ∈ [n_p−i, n_p+i]`, and `run_model`'s k-loop starts at `max(n_p−i, 0)`. Everything else in row 0 is untouched zeros. Verified: `v[0, 0, ·]` is exactly `0.000000` for every n_p ∈ {5,10,15,20,30,45,60} [`convergence`]. So both methods silently return 0.0 on a full tree.

The README's own usage example walks into this:

```python
s_full.build()                                   # n_p = 30
print("Extrinsic    :", s_full.flat() - s_flat.flat())   # 0 - flat  =>  minus the flat price
```

The notebooks avoid it only because they call these methods on `n_p = 0` models and distinguish "flat" from "profiled" by calling `set_volume_states()` in between (`Swing_new.ipynb` cell 4) — the method names carry no such meaning, which is exactly how `print(f"Profiled price = {s.flat():.2f}")` in that cell goes unnoticed.

**Fix:** index `self.v[0, self.n_p, self.n_op_start]`, delete one of the two methods, and make the volume-state meaning explicit (`price_per_mwh(start_state)`).

## 3. Fractional ratchets are interpreted three different ways

- `run_model:565-566` — `np.minimum(...).astype(np.int64)` → **truncates**
- `probabilities:654-655` — `int(round(min(...)))` → **rounds**
- `compute_all_metrics:701-702` — `np.minimum(...) * v_step` → **keeps the float**

`i_ratch`/`w_ratch` are documented user knobs (README: "Daily injection / withdrawal limits by volume state") and default to 1.0, where all three agree. Set anything else [`ratchets`]:

| i_ratch | V₀ | reported MWh | E[terminal clips] | Σδ·F − V₀ |
|---|---|---|---|---|
| 1.0 | −683,890 | 30,000 | 30.00 | 0.1 |
| **1.5** | **−683,890** (identical to 1.0 — DP truncated) | **22,500** | 30.00 | **171,504** |
| **0.5** | **−1e9** | 0 | 0.00 | 1e9 |
| 2.0 | −682,461 | 30,000 | 30.00 | −0.0 |

At 1.5 the DP prices a ratchet of 1, the forward pass moves 2 states per exercise, the reported volume assumes 1.5 — and the repricing identity breaks by 25 % of the contract value. At 0.5 the DP can never move, the quota is unreachable, and the sentinel is returned as a price.

**Fix:** pick one convention (integer clips, `int(np.floor(...))`), apply it in all three places, and validate at assignment — reject non-integers or document that ratchets are integer clip counts.

## 4. The `−1e9` infeasibility sentinel is returned as a price

`storage_model.py:117` sets the terminal penalty; nothing downstream checks whether the optimum actually reached the target state. When the quota cannot be met, V₀ *is* the sentinel [`infeasible`]:

| days | window | V₀ | reported EUR/MWh |
|---|---|---|---|
| 30 | 365 | −697,952 | 24.13 |
| 365 | 365 | −9,151,425 | 25.07 ✓ |
| 400 | 365 | **−1,000,000,000** | nan |
| 100 | 90 | **−1,000,000,000** | nan |

`days = 100` on a 90-day window is one keystroke away in the Streamlit sidebar, which allows `days` up to 3660 with no cross-check against `storageStart..storageEnd`. The app then divides by `np.sum(s.delta) = 0` and prints `-inf`/`nan` rather than an error.

**Fix:** after `build()`, assert that the terminal probability mass sits on the target state (`prob[n_t-1, :, term_inv].sum() ≈ 1`) or that `|V₀| < 1e8`, and raise a message naming the constraint. Add the sidebar cross-check `days ≤ window`.

## 5. The smoothed curve does not reprice the input monthly contracts

`smoothen_curve` (`:20-42`) resamples to monthly means, fits PCHIP through month midpoints, scales the slopes by `alpha`, and rebuilds as a Hermite spline. There is no correction step — despite the README stating "A one-shot additive correction is applied to each month so that smoothed daily averages exactly reproduce the original contract prices." `check_curve` at `:45` prints the discrepancy and is never called.

Measured over the 48-month curve [`curve_fit`]: max |error| **0.1023 EUR/MWh** (38 bp), mean 0.0218; `alpha = 1.0` is no better (0.1074). 2027 monthly errors run −0.102 to +0.076 EUR.

Small next to a 25 EUR curve, not small next to what the model is measuring: intrinsic value is 1.27 EUR/MWh and is *made of* the month-to-month shape that these errors distort. It also breaks hedge consistency — the monthly deltas the app reports are against contracts the model does not reprice.

**Fix:** after building the spline, add per-month `Δ_m = target_m − mean(smoothed_m)` to that month's days. One pass, exactly reproduces every input contract, keeps the daily shape. Then the README sentence becomes true.

## 6. `dx` is set from `vol_curve[0]` alone — a vol term structure silently destroys the tree

`build_tree:513`: `dx = vol_curve[0] * sqrt(3 * dt)`, while `_tree_core` uses `vol_arr[i]` per step. `Storage` builds `self.sVol` as a *per-step list*, which invites exactly this. With a 0.30 → 0.90 ramp [`tree`]:

```
negative-prob nodes = 28464   min p_m = -2.0063   terminal log-std = nan
```

No exception, no warning — the q-propagation overflows and the tree returns NaN. Nothing anywhere validates that p_u/p_m/p_d ∈ [0,1].

**Fix:** `dx = max(vol_curve) * sqrt(3 * dt)` (the standard Hull–White choice), and assert probability validity once after `_tree_core`. Cheap, and it converts a silent NaN into a clear error.

## 7. Streamlit's `n_p_full` default of 10 discards ~15 % of the extrinsic value

`streamlit_app.py:99` defaults `n_p_full = 10`. Against the converged value [`convergence`]:

| n_p | 5 | 10 | 15 | 20 | 30 | 60 |
|---|---|---|---|---|---|---|
| extrinsic (EUR/MWh) | 0.537 | **1.005** | 1.166 | 1.180 | 1.179 | 1.179 |
| shortfall | −54 % | **−15 %** | −1.3 % | — | — | — |

There is no performance reason: the full n_p = 60 build takes 0.22 s here (cost is linear in 2·n_p+1; a storage deal with ~200 volume states scales to a few seconds).

**Fix:** default 30, and note in the sidebar that the value should be checked for convergence.

## 8. `wdr_days` in the app is dead input

`streamlit_app.py:107` collects `wdr_days` and `:135` passes it; `value_storage` (`:386-400`) reads only `params["inj_days"]` — for `set_volume_states` *and* for `max_vol`. Withdrawal capacity is silently forced equal to injection capacity, and the per-MWh normalisation uses injection days. A user who sets asymmetric capacities gets a number that does not answer their question.

**Fix:** either use it (`n_op` from injection capacity, `w_ratch`/`max_vol` from withdrawal) or remove the input.

## 9. The delta ignores `d_curve`

`compute_all_metrics` (`:695-712`) never touches the discount curve, while `run_model:577` discounts every cash flow. With `d_curve` at a 3 % continuous rate [`discounting`]:

```
V0 = -652,146.1    sum(delta*F) = -684,200.5    gap = -32,054.5  (+4.92%)
```

The delta profile is undiscounted while the value is discounted — a 4.9 % overstatement of the hedge, and the repricing identity of §"What is solid" breaks. Invisible today because `d_curve` is `np.ones` everywhere, but it is a documented knob and the discrepancy is silent.

**Fix:** multiply the per-step delta and exp_ex weighting by `d_curve[i]` (delta), or document that `delta` is deliberately an undiscounted physical hedge volume — and then keep the identity test in discounted terms.

## 10. Bucketed deltas are strongly convex — the monthly table needs a caveat

The app's "Monthly Native Deltas" table is a correct local derivative (validated to ±0.4 % at 5 bp, see finding.md) but decays fast for the flexible months [`bucket_convexity`]:

| bump | Jul-27 error | Dec-27 error |
|---|---|---|
| 5 bp | −0.13 % | +0.02 % |
| 1 % | +8.57 % | −1.33 % |
| 2 % | +32.1 % | −0.85 % |
| 5 % | +119.2 % | +8.72 % |

Quota-forced winter months are inelastic and stable; summer months sit on the exercise boundary and reallocate as soon as the month moves. TTF monthlies move 1–3 % on an ordinary day, so a July hedge placed on this delta is materially wrong by the next close.

**Fix:** report a bucketed gamma (rerun at ±1 % — 24 extra builds, ~5 s) or at minimum caption the table as a local ratio requiring re-hedging.

---

## Minor

| # | Item | Location |
|---|---|---|
| 11 | Dead code: `get_exercise`, `get_delta` (superseded by `compute_all_metrics`), `valuation` (a one-element loop equal to `v[0, n_p, n_op_start]`), `check_curve` (never called) | `:623`, `:668`, `:682`, `:45` |
| 12 | `probabilities` takes `mintunnel`/`max_tunnel` and never reads them | `:634` |
| 13 | Tunnels can never bind as constructed: `mintunnel` is all zeros and `max_tunnel = n_op` while valid states are `0..n_op-1`. The penalty scale `1000.0 * v_step` is also arbitrary — 1e6 per unit, comparable to the whole contract value, so it under-penalises large deals and over-penalises small ones | `:117-119`, `:573` |
| 14 | The DP grid runs to `backStop` = one month past `storageEnd`, with all exercise masked off in the tail: 30 idle steps of 760 for a 1-year deal, but **29 of 119 (24 %) for a 3-month deal** | `:75-77` |
| 15 | A curve that does not extend to `backStop` fails inside SciPy with ``ValueError: `y` must contain only finite values`` — correct to fail, but the message does not say "your curve must cover one month past storageEnd", and `load_direct_curve` accepts uploads without a coverage check | `:82`, `streamlit_app.py:52` |
| 16 | `sVol` defaults to 0.9 in code, 0.6 in the README and 0.6/0.5 in the notebooks | `:67` |
| 17 | `strat` uses an absolute tolerance (`< 1e-6`) to call a state "no exercise" while `v` keeps the maximised value, so the two can disagree on a marginal state. The repricing identity holds to 0.1 EUR, so this is currently immaterial — worth making relative if `v_step` or price scale ever grows | `:614-616` |
| 18 | `round(..., 3)` on ~1e4-magnitude metrics, returned as Python lists that callers immediately re-wrap in `np.array` | `:707-710` |
| 19 | `Storage.n_op_start` is reassigned by callers *after* `set_volume_states()` set it, so the two meanings (state count vs. starting inventory) are entangled; any later `set_volume_states` call silently resets the start state | `:291-292`, `:117` |
| 20 | Five `debug_delta*.py` scripts duplicate the quote-matrix loading that now lives in `storage_model` (`monthly_curve_from_quote`, `curve_df_for_storage`) | repo root |

## Suggested regression tests

None of these exist today; all are one-liners against the base case and would have caught findings 1–4 and 6.

1. `abs(sum(delta_i * fwd_i) - v[0, n_p, 0]) < 1e-6 * abs(v0)` — the master consistency check.
2. `-sum(exp_ex) == days * v_step` for a mandatory quota.
3. `days == window` prices at exactly `flat_metric`.
4. `v[0, n_p, n_op_start] != 0` and `flat()/profiled()` non-zero for n_p > 0.
5. `extrinsic >= 0` and monotone non-decreasing in `n_p`; value converged between n_p = 30 and 45 to < 0.01 EUR/MWh.
6. `p_u, p_m, p_d ∈ [0,1]` and `q.sum(1) == 1` after `build_tree`, including a vol-term-structure case.
7. Smoothed monthly means reproduce input contracts to < 1e-9 (after fixing #5).
8. Infeasible quota raises rather than returning `-1e9`.
