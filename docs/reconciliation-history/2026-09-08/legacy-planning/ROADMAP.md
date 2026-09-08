# Roadmap: Gas Storage & Swing Pricing Model

## Overview

The model has already been reviewed, tested and partly repaired: four commits on
`review/model-fixes` took `test_model.py` from 3/13 to 13/13 without moving a single
valuation. What remains is the tail of that review — six requirements from
`docs/SPEC-remaining-work.md`, exactly one of which changes what the model prices. The
roadmap sequences them so the master invariant is made final while nothing is moving
(Phase 1), the one valuation-changing fix then lands against a stable gate with a
published before/after (Phase 2), and the non-valuation-changing work follows in
descending order of user visibility: the app surface (Phase 3), the library's dead and
arbitrary parts (Phase 4), and finally the documentation, which can only tell the truth
once the code above it has settled (Phase 5).

## Completed History (pre-roadmap)

Not future work. Recorded so the roadmap starts from the real baseline.

Branch `review/model-fixes`, all four commits already on disk:

| Commit | What landed | Findings closed |
|--------|-------------|-----------------|
| `9492d90` | The working tree checked in as the pre-review baseline (previously untracked) | — |
| `d999c1f` | The model review: `code_review.md` (10 findings + 10 minor), `finding.md` peer review, `review_checks.py` | — |
| `0021a19` | Batch 0 — the regression suite `test_model.py`, 3/13 against the baseline | — |
| `18d1fb0` | Batch 1 (metric denominator + `price_per_mwh`) and Batch 2 (fail-loud guards) — suite green at 13/13 | 1, 2, 3, 4, 6, 15, 18 |

No valuation changed across Batches 0-2: `intrinsic`, `extrinsic`, `total` and `v0` are
identical before and after for all six quoted products, the `finding.md` put swing and the
storage case. Only `stochastic_metric` moved (+1.2 % to +5.5 % for the call swings, -7.6 %
for the put swing), and `flat()`/`profiled()` went from 0.0 to a real price.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Delta Convention and the Restated Invariant** - Fix what `delta` means and make I1 true under any discount curve, while no valuation is moving
- [ ] **Phase 2: Exact Monthly Curve Repricing** - Make the daily curve reprice the contracts it was built from; the only valuation-changing work
- [ ] **Phase 3: Streamlit Surface Truth** - Converged default, no dead input, and a delta table that admits its own convexity
- [ ] **Phase 4: Dead Code, Tunnels and Scale** - Remove what is never called and replace what is arbitrary
- [ ] **Phase 5: Documentation and Reproduction** - README describes the model that exists; one reproduction script, not six

## Phase Details

### Phase 1: Delta Convention and the Restated Invariant
**Goal**: The reported `delta` has a stated, defended meaning, and the suite's master test is written in a form that stays true under a real discount curve
**Depends on**: Nothing (first phase)
**Requirements**: DELTA-01, DELTA-02, DELTA-03
**Success Criteria** (what must be TRUE):
  1. A reader of `storage_model.py` can see, at the definition site, that `delta` is an undiscounted physical hedge volume — the forward MWh to trade — and why the discount factor cancels; no code path applies `d_curve` to `delta`.
  2. Invariant I1 reads `sum_i d_curve[i] * delta[i] * fwd[i] == v[0, n_p, n_op_start]` (relative error < 1e-9) in the SPEC and in the test that enforces it.
  3. `python test_model.py` passes with a new test that builds a contract at a 3 % continuous `d_curve` and asserts the restated identity; deleting the discount weights from that identity makes the test fail by ~4.9 %.
  4. Every existing valuation is unchanged — the restatement is provably a no-op while `d_curve` is all ones, which is every configuration in use today.
**Plans**: TBD

*Why first:* restating the master test is free of numerical consequence today, so this is the
only window in which it can be done without entangling it with a moving valuation. Phase 2 is
then gated by an invariant whose wording is already final.

### Phase 2: Exact Monthly Curve Repricing
**Goal**: The daily curve reprices every monthly contract it was built from, so intrinsic value and the monthly deltas refer to the market actually quoted
**Depends on**: Phase 1
**Requirements**: CURVE-01, CURVE-02, CURVE-03, CURVE-04, CURVE-05
**Success Criteria** (what must be TRUE):
  1. For the 48-month `curve.csv`, the smoothed curve's monthly means reproduce every input contract to < 1e-9, where today the worst miss is 0.1023 EUR/MWh against a 1.27 EUR/MWh intrinsic value — reproducible with `python review_checks.py curve_fit`.
  2. Exactness also holds when the first or last month is partial, because the solve uses the days actually present rather than a calendar midpoint.
  3. The curve stays smooth: no month-boundary step beyond the natural slope, so the DP's day-to-day exercise comparison is not corrupted by ~0.1 EUR jumps.
  4. `smoothen_curve(..., exact=False)` regenerates the old curve on demand, so any historical number can be reproduced; exact repricing is what runs by default.
  5. `code_review.md` carries a before/after valuation table for the six `quotes.csv` products, the `finding.md` put swing and the storage case, and `python test_model.py` is green.
**Plans**: TBD

*Risk note:* this is the only remaining item that changes valuations. Notebook and app results
shift on the next run; that shift is accepted (D-O1) and must be published, not absorbed.

### Phase 3: Streamlit Surface Truth
**Goal**: The app's defaults and labels tell the truth about what the model computes
**Depends on**: Phase 2
**Requirements**: APP-01, APP-02, APP-03
**Success Criteria** (what must be TRUE):
  1. A user opening the app gets `n_p_full = 30` and a sidebar note to check convergence, instead of a default that silently discards ~15 % of extrinsic value.
  2. The sidebar no longer offers `wdr_days`, nothing downstream reads it, and the storage inputs say plainly that capacity is symmetric and withdrawal capacity is expressed through `w_ratch`.
  3. The "Monthly Native Deltas" table is captioned as a local ratio requiring re-hedging, naming the measured convexity (July +8.6 %, September +18.7 %, November +43.9 % at a 1 % bump; quota-forced December -1.3 %) against TTF monthlies that move 1-3 % on an ordinary day.
  4. Re-running the app with the pre-change inputs reproduces the Phase 2 valuations exactly — only a default input value moved, not the model — and `python test_model.py` is green.
**Plans**: TBD
**UI hint**: yes

### Phase 4: Dead Code, Tunnels and Scale
**Goal**: The library exposes only code that is called, and no constant is arbitrary relative to deal size
**Depends on**: Phase 2
**Requirements**: CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04
**Success Criteria** (what must be TRUE):
  1. `get_exercise`, `get_delta`, `valuation` and `check_curve` no longer exist, and the suite, `review_checks.py`, the app and the notebooks all still run — proving nothing depended on them.
  2. `probabilities` and `run_model` no longer accept arguments they never read: the tunnels either bind, documented and tested, or `mintunnel`/`max_tunnel` are gone.
  3. The tunnel penalty scales with the deal rather than sitting at `1000.0 * v_step` (1e6 per unit, comparable to a whole contract), with the new scale justified in place and valuations unchanged — or accompanied by an I3 before/after table if any moves.
  4. `Storage` names the volume-state count and the starting inventory separately, callers set the start explicitly, and a test proves that a second `set_volume_states()` call no longer silently resets the start state.
**Plans**: TBD

### Phase 5: Documentation and Reproduction
**Goal**: The README describes the model that now exists, and there is one reproduction script instead of six ad-hoc ones
**Depends on**: Phase 4 (deleted API entries) and Phase 2 (the curve-correction claim)
**Requirements**: DOCS-01, DOCS-02, DOCS-03
**Success Criteria** (what must be TRUE):
  1. A reader following README finds no false claim: the curve correction is described as it now behaves, `sVol`'s default reads 0.9, `flat()`/`profiled()` are described as aliases (and the "Intrinsic = profiled - flat" decomposition is corrected), the usage example no longer labels minus the flat price as "Extrinsic", the smoother is named as PCHIP plus a cubic Hermite spline, and the deleted functions are gone from the API reference.
  2. README's `1e10` infeasibility statement survives untouched, with a note recording that it is the per-transition penalty and not the `-1e9` terminal sentinel, so a later reader does not "fix" a true statement.
  3. The repo root holds no `debug_delta*.py` scripts; every check they performed is reachable through `python review_checks.py <section>`, with no duplicated curve-loading helpers.
  4. The DP-grid truncation question is closed in writing — either the grid ends at the exercise window with an I3 before/after table, or the reason for keeping `backStop` is recorded alongside the measured cost (24 % idle steps on a three-month deal).
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Delta Convention and the Restated Invariant | 0/TBD | Not started | - |
| 2. Exact Monthly Curve Repricing | 0/TBD | Not started | - |
| 3. Streamlit Surface Truth | 0/TBD | Not started | - |
| 4. Dead Code, Tunnels and Scale | 0/TBD | Not started | - |
| 5. Documentation and Reproduction | 0/TBD | Not started | - |

## Gates That Apply to Every Phase

- **I1** — `sum_i d_curve[i] * delta[i] * fwd[i] == v[0, n_p, n_op_start]`, relative error < 1e-9 (wording final after Phase 1)
- **I2** — `python test_model.py` passes; 13/13 today, higher as phases add tests
- **I3** — no reported valuation changes without a documented before/after across the six `quotes.csv` products, the `finding.md` put swing and the storage case

---
*Roadmap created: 2026-09-08 from docs/SPEC-remaining-work.md (R1-R6) and code_review.md*
