# Constraints — synthesized intel

Sources are the two SPECs: docs/SPEC-remaining-work.md (precedence 0) and
code_review.md (precedence 1). Where they differ, precedence 0 wins and the
superseded value is recorded in decisions.md.

---

## C1 — Delta reprices the contract (invariant I1)
- source: docs/SPEC-remaining-work.md §Invariants (I1)
- type: nfr / master regression test
- constraint: `sum_i delta[i] * fwd[i] == v[0, n_p, n_op_start]`, relative error < 1e-9
- rationale: ties the DP, the forward pass and the reported metrics together; "it is the
  suite's master test"
- corroboration: finding.md peer review [repro] measured -678,682.8 vs -678,682.8;
  code_review.md §What is solid measured a 0.1 EUR gap on 678,683
- implemented at: test_model.py:62-77 (`repricing_gap`, asserted `< 1e-9` across `n_p`,
  plus a ratchet-of-two case)
- superseded looser bound: code_review.md suggested `< 1e-6 * abs(v0)` (see D-S2)
- **conditional on D-O2**: the SPEC states that with a discounting `d_curve` "I1 breaks by
  construction". Option (a) restates I1 in undiscounted terms; option (b) keeps it exact.
  The wording of this constraint is not final until D-O2 is decided.

## C2 — Regression suite must pass (invariant I2)
- source: docs/SPEC-remaining-work.md §Invariants (I2)
- type: nfr / gate
- constraint: `python test_model.py` passes 13/13
- verified: test_model.py currently defines 13 tests

## C3 — No unexplained valuation change (invariant I3)
- source: docs/SPEC-remaining-work.md §Invariants (I3)
- type: process gate
- constraint: no change to a reported valuation lands without a documented before/after
  across the six products in `quotes.csv`, the finding.md put swing, and the storage case
- precedent: code_review.md records exactly this table for the completed batches
  ("No valuation changed: `intrinsic`, `extrinsic`, `total` and `v0` are identical before
  and after for all 6 quoted products ... only `stochastic_metric` moved (+1.2 % to
  +5.5 % for the call swings, -7.6 % for the put swing)")
- interacts with R1, the only remaining item that changes valuations

---

## C4 — Curve accuracy target
- source: docs/SPEC-remaining-work.md §R1 Acceptance
- type: nfr
- constraint: smoothed monthly means reproduce input contracts to < 1e-9; no
  month-boundary discontinuity beyond the natural curve slope
- current measured state (code_review.md §5 [curve_fit], 48-month curve): max |error|
  0.1023 EUR/MWh (38 bp), mean 0.0218

## C5 — Convergence in `n_p`
- source: code_review.md §7 [convergence]; docs/SPEC-remaining-work.md §R2
- type: nfr
- constraint: reported extrinsic value must be checked for convergence in `n_p`;
  `n_p = 20` converges to 0.001, `n_p = 30` is the mandated Streamlit default, and a full
  `n_p = 60` build costs 0.22 s (cost is linear in `2*n_p+1`; a storage deal with ~200
  volume states scales to a few seconds)

## C6 — Tree validity (already enforced; must not regress)
- source: code_review.md §6 and §What is solid; finding.md peer review [tree]
- type: protocol / numerical
- constraints: `p_u, p_m, p_d` in [0,1] and `q.sum(1) == 1` after `build_tree`, including
  a vol-term-structure case; `dx = max(vol_curve) * sqrt(3*dt)` (Hull-White choice), not
  `vol_curve[0]`; forward fitting exact to 1e-15; terminal log-std 0.351 vs the OU
  stationary target sigma/sqrt(2*kappa) = 0.354; boundary mass 1.5e-4
- status: fixed in an earlier batch (finding 6); test_model.py:140-142 asserts the
  probability-sum and forward-fitting properties

## C7 — Ratchet convention (already enforced; must not regress)
- source: code_review.md §3
- type: api-contract
- constraint: one whole-clip convention applied consistently across `run_model`,
  `probabilities` and `compute_all_metrics`; non-integer ratchets rejected at assignment
- historical failure mode: at `i_ratch = 1.5` the DP priced a ratchet of 1, the forward
  pass moved 2 states per exercise, and the repricing identity broke by 171,504 (25 % of
  contract value); at 0.5 the sentinel was returned as a price

## C8 — Infeasibility must raise, never be priced (already enforced; must not regress)
- source: code_review.md §4
- type: api-contract
- constraint: after `build()`, assert the terminal probability mass sits on the target
  state (`prob[n_t-1, :, term_inv].sum() ~= 1`) or `|V0| < 1e8`, and raise naming the
  constraint; the Streamlit sidebar must cross-check `days <= window`
- sentinel: `t_p_curve` terminal penalty is `-1e9` (storage_model.py:123); a guard exists
  at storage_model.py:146-155

## C9 — Curve coverage (already enforced; must not regress)
- source: code_review.md §Minor 15 (marked fixed)
- type: api-contract
- constraint: the input curve must cover one month past `storageEnd` (to `backStop`);
  failure must say so explicitly, and `load_direct_curve` must apply a coverage check on
  upload rather than failing inside SciPy with "`y` must contain only finite values"

## C10 — Penalty scale must not be arbitrary
- source: docs/SPEC-remaining-work.md §R5; code_review.md §Minor 13
- type: nfr
- constraint: replace the `1000.0 * v_step` penalty scale (1e6 per unit, comparable to
  the whole contract value — under-penalises large deals, over-penalises small ones);
  tunnels must either bind or be removed (`mintunnel` is all zeros and
  `max_tunnel = n_op` while valid states are `0..n_op-1`)

---

## Verification checklist (code_review.md §Suggested regression tests)

Recorded as verification criteria, not as separate requirements. Items 1-6 and 8 are
already implemented in test_model.py (13 tests); item 7 lands with REQ-exact-curve-repricing.

1. `abs(sum(delta_i * fwd_i) - v[0, n_p, 0]) < 1e-6 * abs(v0)` — master consistency check
   (tightened to 1e-9 by C1)
2. `-sum(exp_ex) == days * v_step` for a mandatory quota
3. `days == window` prices at exactly `flat_metric`
4. `v[0, n_p, n_op_start] != 0` and `flat()` / `profiled()` non-zero for `n_p > 0`
5. `extrinsic >= 0` and monotone non-decreasing in `n_p`; value converged between
   `n_p = 30` and `45` to < 0.01 EUR/MWh
6. `p_u, p_m, p_d in [0,1]` and `q.sum(1) == 1` after `build_tree`, including a
   vol-term-structure case
7. Smoothed monthly means reproduce input contracts to < 1e-9 (after R1)
8. Infeasible quota raises rather than returning `-1e9`

---

## Out of scope (SPEC-mandated boundary)

- source: docs/SPEC-remaining-work.md §Out of scope
- Rewriting the DP or the tree — both verified correct
- The mandatory-quota artefacts in finding.md (the January step, the December delta
  amplification) — confirmed correct model behaviour and independently validated
