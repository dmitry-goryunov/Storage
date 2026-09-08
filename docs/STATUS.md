# Status — storage/swing pricing model

As of 2026-09-08, branch `review/model-fixes` (5 commits ahead of `main`, fast-forwardable).
Suite: `python test_model.py` → 13/13. Working tree clean.

Update this file as phases complete. The machine-readable state lives in
[.planning/STATE.md](../.planning/STATE.md) and [.planning/ROADMAP.md](../.planning/ROADMAP.md);
this is the human entry point.

---

## Where things stand

A model review found ten defects plus ten minor ones in `storage_model.py` and
`streamlit_app.py`. Seven are fixed and locked behind a regression suite; the rest are
scoped into a five-phase roadmap. **No valuation has changed yet** — the fixes so far
corrected how prices are *reported*, not what the model computes. Phase 2 is the only
remaining work that moves a valuation.

## What was done

### 1. Peer review of the put-delta investigation

Appended to [finding.md](../finding.md). Both original findings hold, with three
contributions:

- **The January test was confounded.** The base case ran `n_p = 30` and `days = 30`, and
  the tree's growing phase ends at step `n_p` — the same date as quota saturation, so two
  candidate causes landed on Jan 31. Varying them independently: the step stays at Jan 31
  for n_p = 20/30/45 and moves to Jan 21 when days = 20. It tracks the quota; tree
  geometry is ruled out.
- **The delta is correct, and now verified.** Every 2027 monthly bucket matches its
  finite-difference derivative to within ±0.4 % at a 5 bp bump. The ratio
  |delta|/|exp_ex| is exactly E[S | exercise] / F, which rises monotonically 0.45 (Mar)
  → 1.71 (Dec): early exercise is chosen at cheap prices, late exercise is forced at
  expensive ones. December is not an anomaly, it is the endpoint of a trend — and it is
  the *most* reliable bucket in the profile, because quota-forced exercise is inelastic.
- **The soft-penalty threshold was wrong.** Stated as 1.19× (max forward); actually ~4×.
  At 1.19× the model buys only 82 % of the quota — visible in the original table, where
  2.0× is still 456 MWh short. The binding quantity is the tail of the spot distribution
  (max F × e^(n_p·dx) ≈ 107 EUR ≈ 4.26× flat), not the maximum of the forward curve. This
  strengthens the original conclusion: the penalty needed to imitate a hard constraint
  depends on `n_p` and `sVol`, i.e. on discretisation rather than on contract economics.

### 2. Code review

[code_review.md](../code_review.md) — 10 findings plus 10 minor, severity-ranked, every
claim executed rather than inferred. Reproduce any number with
`python review_checks.py <section>`.

The two that mattered most:

- `stochastic_metric` divided value by `sum(delta)` — the price-weighted forward-equivalent
  volume — instead of by MWh exercised. 8.4 % error at the base case, and the response was
  inverted: adding optionality made the reported price *rise* (23.80 → 24.52) while the
  true average purchase price falls (23.80 → 22.62).
- `flat()` and `profiled()` were byte-identical and both read price state 0, which is
  untouched zeros for any `n_p > 0` — so they silently returned 0.0 on a full tree. The
  README's own usage example walks into it.

### 3. Fixes, behind a regression suite

[test_model.py](../test_model.py), 13 tests, built around the master invariant

    sum_i d_curve[i] * delta[i] * fwd[i] == v[0, n_p, n_op_start]

which ties the DP, the forward pass and the reported metrics together. Score went
**3/13 → 13/13**. Writing the tests first paid immediately: two apparent failures turned
out to be the model's own cosmetic `np.round(..., 3)`, which was masking the identity at
1e-7. Removed; it now holds at 2.4e-16.

Fixed: findings 1, 2 (reported metrics), 3, 4, 6, 15 (fail-loud guards) and 18.

**What moved, and what did not.** `flat_metric`, `profiled_metric`, `intrinsic`,
`extrinsic`, `total` and `v0` are identical before and after for all six products in
`quotes.csv`, the finding.md put swing and the storage case. Only `stochastic_metric`
moved — call swings +1.2 % to +5.5 %, put swing −7.6 % — and `flat()`/`profiled()` went
from 0.0 to a real price. Verified end-to-end in the running Streamlit app: the new guard
fires, a valid run completes in 0.4 s.

### 4. Commits

Every file including `storage_model.py` was untracked, so a naive commit would have shown
no fix diff at all. The pre-session state was reconstructed and verified by re-running the
suite against it (3/13 — the exact three that passed at the start), then the fixed files
were restored byte-for-byte (md5 match) and committed on top.

    f1c9a17  Bootstrap .planning/ by ingesting the review docs
    18d1fb0  Fix the per-MWh metric and price-state indexing; guard invalid configs   13/13
    0021a19  Add a regression suite for the pricing model                             3/13, red
    d999c1f  Add the model review: findings, peer review and reproduction script
    9492d90  Check in the working tree as the pre-review baseline

`git show 18d1fb0` is therefore a real review diff: 143 lines in `storage_model.py`, 16 in
the app.

### 5. Traceable planning

`.planning/` bootstrapped by ingesting four documents (2 SPEC, 2 DOC; no ADRs, so nothing
is locked). Conflict gate: 0 blockers, 6 warnings, 10 info — the report is kept at
[.planning/INGEST-CONFLICTS.md](../.planning/INGEST-CONFLICTS.md). 18 requirements, each
tracing to a numbered finding and a `file:line`.

Synthesis also checked the docs against the code and found README's `1e10` claim to be
**true** — `bigdummy` in `run_model` is the per-transition penalty, a different quantity
from the `-1e9` terminal sentinel — plus two further README inaccuracies.

## Decisions taken (2026-09-08, not ADR-locked)

| | Decision | Why |
|---|---|---|
| **O1** | Exact curve repricing is **default-on**, `exact=False` for comparison | A curve that does not reprice its own inputs is a defect, not a preference |
| **O2** | `delta` stays an **undiscounted physical hedge volume** | It is the number of forward MWh to trade: hedging with h forwards gives PV = h·DF·F·ε against dV/dε = DF·E[S·Q], so the DF cancels and h = E[S·Q]/F. I1 carries the discount weights instead |
| **R3** | **Remove** `wdr_days` from the app | It does nothing and implies a control the model lacks; capacity is set by `w_ratch` |
| **R4** | **Caption** the monthly delta table, no bucketed gamma | The convexity is measured and documented; a compute-bearing feature is not housekeeping |

## The plan

| Phase | What it does | Moves valuations? |
|---|---|---|
| **1 · Delta convention** | Document `delta` as a hedge volume, restate I1 with discount weights, test with a non-flat `d_curve` | No — provably a no-op while `d_curve` is ones, which is why it goes first |
| **2 · Exact curve repricing** | Solve the M×M knot system so the daily curve reprices every input contract to 1e-9; fix the partial-month knot bias | **Yes — the only phase that does.** Gated on a before/after table |
| **3 · Streamlit surface** | `n_p_full` 10 → 30, remove `wdr_days`, caption the delta table | Reported extrinsic moves ~15 % because a default changed, not the model |
| **4 · Dead code, tunnels, scale** | Delete `get_exercise`/`get_delta`/`valuation`/`check_curve`, fix tunnels and the `1000·v_step` penalty, disentangle `n_op_start` | Possibly — the penalty scale is a number the DP uses |
| **5 · Documentation** | README corrections, fold the five `debug_delta*.py` into `review_checks.py` | No. Last, because it depends on phases 2 and 4 |

Phase 1 was sequenced ahead of the curve work deliberately: with `d_curve` all ones,
restating I1 is provably a no-op, so the suite staying green proves the restatement alone.
Doing it after Phase 2 would change the master test in the same window as the only
valuation-changing fix, and a failure would not say which one broke it.

### Watch items

- **Phase 4 may not be housekeeping.** Replacing the arbitrary `1000.0 * v_step` tunnel
  penalty changes a number the DP uses; it may need Phase 2's before/after discipline.
- **Phase 3 changes a reported number without changing the model** — `n_p_full` 10 → 30
  shifts displayed extrinsic ~15 %. That is a converged default replacing an
  unconverged one, not a broken invariant.
- **Do not "fix" README's `1e10`.** It is correct; see above.

## Verify

```bash
python test_model.py                    # 13/13, ~10 s
python review_checks.py                 # reproduces every number in the review
python review_checks.py repro           # just the finding.md base case + the identity
streamlit run streamlit_app.py          # the app
```

## Where things live

| File | What it is |
|---|---|
| [code_review.md](../code_review.md) | The findings, with a status table |
| [finding.md](../finding.md) | Put-delta investigation + the peer review of it |
| [docs/SPEC-remaining-work.md](SPEC-remaining-work.md) | Invariants I1–I3, requirements R1–R6, decisions |
| [test_model.py](../test_model.py) | The regression suite |
| [review_checks.py](../review_checks.py) | Demonstration script — reproduces the review's numbers, not pass/fail |
| [.planning/ROADMAP.md](../.planning/ROADMAP.md) | The five phases |
| [.planning/REQUIREMENTS.md](../.planning/REQUIREMENTS.md) | 18 requirements + source traceability table |
| [.planning/intel/decisions.md](../.planning/intel/decisions.md) | Decisions, open and resolved |
