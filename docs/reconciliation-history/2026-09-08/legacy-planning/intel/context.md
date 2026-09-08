# Context — synthesized intel

Running notes by topic, with source attribution. Nothing here is a requirement or a
constraint; it is background and evidence.

---

## Topic: What the project is
- source: README.md; docs/SPEC-remaining-work.md §Context

A quantitative library for valuing natural gas storage and swing contracts on the TTF
market. `storage_model.py` builds a trinomial price tree with Ornstein-Uhlenbeck mean
reversion and solves a dynamic program over a joint (time x price x volume) state space;
the inner DP loop is JIT-compiled and parallelised with Numba. `streamlit_app.py` is the
front end. `Swing_new.ipynb` is the driver notebook, running intrinsic/extrinsic
valuations for six TTF swing products against `curve.csv` (48 monthly contracts,
Jan 2026 - Dec 2029) and `quotes.csv` (bid/ask for six products, 90/120/180 days).

Pipeline: `map_curve_to_dates` -> `smoothen_curve` -> `build_tree` -> `run_model` (DP)
-> `probabilities` (forward pass) -> `compute_all_metrics` (exp_ex, delta).

## Topic: Review provenance
- source: code_review.md header; docs/SPEC-remaining-work.md §Context

A model review dated 2026-09-08 produced 10 findings plus 10 minor ones, executed rather
than inferred, reproducible with `python review_checks.py <section>`. Batches 0-2 fixed
findings 1, 2, 3, 4, 6, 15 and 18 and added `test_model.py` (13 tests; the suite went
3/13 -> 13/13). The SPEC covers what remains.

## Topic: Fixed findings (historical — not outstanding work)
- source: code_review.md §Status table, cross-checked against docs/SPEC-remaining-work.md

- **1 — `stochastic_metric` denominator.** Divided value by `sum(delta)` (price-weighted
  forward-equivalent volume, 27,674 MWh) instead of contract volume (30,000 MWh): level
  wrong by 1.90 EUR/MWh (8.4 %) against a 2.45 EUR/MWh option premium, and the response
  was inverted (reported price rose 23.80 -> 24.52 with optionality while true average
  purchase cost fell 23.80 -> 22.62). Fixed: now value / MWh exercised.
- **2 — `flat()` / `profiled()` returned 0 for any `n_p > 0`** and were byte-identical,
  both indexing price state 0, where only node `k = n_p` exists at time 0. Fixed: one
  `price_per_mwh()` with the correct price state and denominator. `flat` and `profiled`
  survive as aliases of `price_per_mwh` (storage_model.py:213-214), kept for the notebooks.
- **3 — Three ratchet conventions** (truncate in `run_model`, round in `probabilities`,
  float in `compute_all_metrics`). Fixed: one whole-clip convention; non-integer ratchets
  rejected.
- **4 — The `-1e9` sentinel was returned as a price** (e.g. `days = 100` on a 90-day
  window, one keystroke away in the sidebar, printed `-inf`/`nan`). Fixed: `build()`
  checks terminal feasibility and raises.
- **6 — `dx` from `vol_curve[0]`.** A 0.30 -> 0.90 vol ramp produced 28,464
  negative-probability nodes, `min p_m = -2.0063`, terminal log-std NaN, with no
  exception. Fixed: `dx` from `max(sVol)` plus a tree stability check.
- **15 — Curve coverage message** and `load_direct_curve` upload check. Fixed.
- **18 — `round(..., 3)` on ~1e4-magnitude metrics.** Removed; it was masking the
  repricing identity at 1e-7.

Effect on numbers: no valuation changed. `intrinsic`, `extrinsic`, `total` and `v0` are
identical before and after for all six quoted products, the finding.md put swing and the
storage case. Only `stochastic_metric` moved (+1.2 % to +5.5 % for the call swings,
-7.6 % for the put swing), and `flat()` / `profiled()` went from 0.0 to a real price.

## Topic: Put-swing delta investigation (evidence only)
- source: finding.md (original investigation + independent peer review)

Base case: put swing, right/obligation to buy on exactly 30 cheapest days over 2027;
valDate 2026-01-01, storageStart 2027-01-01, storageEnd 2027-12-31, days = 30, vol = 0.50,
n_p = 30, v_step = 1000 MWh; hard terminal constraint `t_p_curve[k != 30] = -1e9`;
flat_metric ~25.07 EUR/MWh. Reproduces exactly: V0 = -678,682.8 EUR, total exp_ex
-30,000.0 MWh, total delta -27,674.4 MWh.

- **Finding 1 (January step-down at Jan 31)** — a leading wave of ~4.5 % probability mass
  exercises continuously from Jan 1 in extreme-low-price states; on Jan 30, 97.5 % of the
  k = 29 wave exercises and flows to k = 30, where the quota is met and 0 states exercise
  (exp_ex steps -65 -> -21 MWh/day). Confirmed correct. The original test was confounded
  (n_p = 30 and days = 30 put quota saturation and the end of the tree's growing phase on
  the same date); varying them independently showed the step tracks `days` and is
  invariant to `n_p` (20/30/45 identical to within 0.4 MWh/day). Tree geometry excluded.
- **Finding 2 (December delta amplification, ratio 1.71)** — confirmed correct, and better
  stated as `|delta|/|exp_ex| = E[S | exercise] / F`. The ratio rises monotonically from
  April and crosses 1.0 in August (Jan 0.53, Mar 0.45, May 0.78, Jul 0.96, Sep 1.07,
  Oct 1.24, Dec 1.71), so 1.71 is the endpoint of a smooth trend, not a December anomaly.
  Early exercise is chosen (E[S|ex] 12-19 vs a forward of 24-27); late exercise is forced
  (E[S|ex] 43.94 against a December forward of 25.69). The delta is a multiplicative
  (sticky-moneyness) bump, `dV/dF * F = -E[S*Q]`, exactly what `compute_all_metrics`
  computes; the envelope theorem covers the quota constraint, so no term is missing.
  Validated by finite differences at +/-5 bp, month by month, to within +/-0.4 %.
- **Convexity caveat** — at a 1 % bump, Jul +8.57 %, Sep +18.68 %, Nov +43.85 % against
  Dec -1.33 %. Summer buckets sit on the exercise boundary and reallocate; winter buckets
  are quota-forced and inelastic. Feeds REQ-delta-convexity-caveat.
- **Not covered by the investigation** — the per-MWh metric built from the delta profile,
  which is where the actual bug was (review finding 1, now fixed).

## Topic: Soft-constraint experiment (conducted and reverted)
- source: finding.md, both the original section and its peer review

A linear graduated terminal penalty was tried and reverted. The peer review reproduced
all four reported multipliers exactly but corrected the stated threshold: full exercise
first appears at ~4x, not the claimed 1.19x (at 1.19x the model buys -24,530 MWh, 82 % of
quota). The 1.19x figure was the deterministic threshold (penalty must exceed the max
forward, 29.72 EUR); under the tree the buyer can be forced to buy at spot, and the
maximum reachable spot is ~max(F) * e^(n_p*dx) = 27.41 * 3.89 ~= 107 EUR ~= 4.26x
flat_metric. The correction strengthens the conclusion: the required penalty depends on
`n_p` and `sVol`, i.e. on discretisation parameters rather than contract economics, which
is the recommended reason not to revisit a graduated penalty. See decisions.md D-3.

## Topic: README documentation state
- source: README.md, with staleness recorded in its classification and in
  docs/SPEC-remaining-work.md §R5 / code_review.md

README.md is a DOC (precedence 3) and describes existing behaviour, not target behaviour.
Four claims are known stale and are the subject of REQ-dead-code-and-docs R5.3; two
further inaccuracies were found during synthesis and one suspected inaccuracy was checked
and found to be correct. Detail in INGEST-CONFLICTS.md INFO. Nothing in README.md may be
routed as a requirement.

## Topic: Deliberately excluded items
- source: code_review.md §Minor 17; docs/SPEC-remaining-work.md §R5 (which omits it)

Finding 17 — `strat` uses an absolute tolerance (`< 1e-6`) to call a state "no exercise"
while `v` keeps the maximised value, so the two can disagree on a marginal state. The
repricing identity holds to 0.1 EUR, so this is currently immaterial; worth making
relative if `v_step` or the price scale ever grows. Not in the SPEC; not a requirement.

## Topic: Repo inventory relevant to the remaining work
- source: repository state at 2026-09-08, cross-referenced against both SPECs

`storage_model.py`, `streamlit_app.py`, `test_model.py` (13 tests), `review_checks.py`,
`Swing_new.ipynb`, `curve.csv`, `quotes.csv`, and five repo-root `debug_delta*.py`
scripts (targets of R5.4). Dead-code targets confirmed still present: `check_curve`
(:45), `valuation` (:718), `get_exercise` (:766), `get_delta` (:780). Streamlit defaults
confirmed: `n_p_full` value 10 (:101), `wdr_days` collected (:107) and passed (:145).
