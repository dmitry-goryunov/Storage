# Decisions — synthesized intel

Ingest mode: new. Precedence applied: ADR > SPEC > PRD > DOC, with per-doc manifest
overrides (SPEC-remaining-work.md = 0, code_review.md = 1, finding.md = 2, README.md = 3).

**No ADRs were present in the ingest set. Zero LOCKED decisions exist.**
Nothing below may be treated as immutable by downstream consumers; every entry is
either OPEN (needs a user decision) or a non-locked position asserted by a SPEC.

---

## OPEN — decisions required before implementation

These are decision-shaped but explicitly **not decided**. They must not be extracted as
constraints or acceptance criteria until a user resolves them.

### D-O1 — Exact monthly curve repricing: default-on or behind a flag
- source: docs/SPEC-remaining-work.md (§R1, "Open question O1")
- status: OPEN — user decision required
- scope: `smoothen_curve` public behaviour; every downstream valuation
- option (a): default-on, with `exact=False` available for comparison
- option (b): opt-in behind a flag (default preserves today's behaviour)
- author's recommendation (a recommendation, NOT a decision): option (a), "on the grounds
  that a curve which does not reprice its own inputs is a defect rather than a preference"
- stated consequence of (a): "notebook results shift slightly on next run"
- coupling: R1 is "the only remaining item that changes valuations" (SPEC §R1, Risk)

### D-O2 — What is `delta`: physical hedge volume or value sensitivity
- source: docs/SPEC-remaining-work.md (§O2); evidence in code_review.md §9
- status: OPEN — user decision required
- scope: `compute_all_metrics`, reported `delta`, and the wording of invariant I1
- option (a): **physical hedge volume** — leave `delta` undiscounted, document it, and
  restate I1 in undiscounted terms
- option (b): **value sensitivity** — apply `d_curve` to `delta`, keeping I1 exact
- measured impact (code_review.md §9, `d_curve` at 3 % continuous):
  `V0 = -652,146.1`, `sum(delta*F) = -684,200.5`, gap `-32,054.5` (+4.92 %)
- currently invisible: `d_curve` is `np.ones` everywhere in use today
- **downstream coupling: this decision changes the definition of invariant I1** (see
  constraints.md C1). Resolve before I1 is restated or re-tested.

---

## RESOLVED — 2026-09-08, decided by the user at the ingest conflict gate

The four warnings raised in `.planning/INGEST-CONFLICTS.md` that required a user
decision were resolved before routing. These are now decided (though still not ADR-
locked; raise an ADR if any should become immutable).

### D-O1 RESOLVED — exact curve repricing is DEFAULT-ON
- decision: option (a) — exact monthly repricing on by default, with `exact=False`
  available to reproduce the old behaviour for comparison
- rationale: a curve that does not reprice its own input contracts is a defect, not a
  preference
- accepted consequence: notebook and app results shift slightly on the next run

### D-O2 RESOLVED — `delta` stays an UNDISCOUNTED PHYSICAL HEDGE VOLUME
- decision: option (a) — leave `delta` undiscounted; do NOT apply `d_curve` to it
- rationale: it is the number of forward MWh to trade. Hedging day i with h forwards
  gives PV = h*DF_i*F_i*eps against dV/deps = DF_i*E[S_i*Q_i], so DF cancels and
  h = E[S_i*Q_i]/F_i. A discounted delta would not be a tradeable quantity.
- **consequence for invariant I1**: restate as `sum_i DF_i * delta_i * fwd_i == V0`,
  which is exact both today (d_curve all ones, reducing to the current form) and under
  a real discount curve. constraints.md C1 must be updated to this wording.

### D-O3 RESOLVED — `wdr_days` is REMOVED from the Streamlit app (R3 variant b rejected)
- decision: remove the input rather than wire it through
- rationale: it does nothing today and implies a control the model does not have;
  withdrawal capacity is actually expressed through `w_ratch`
- note: asymmetric injection/withdrawal capacity remains a legitimate future feature,
  to be specified on its own rather than smuggled in through a dead input

### D-O4 RESOLVED — the monthly delta table gets a CAPTION, not a bucketed gamma
- decision: caption the table as a local ratio requiring re-hedging
- rationale: the convexity is documented and measured (July +8.6 % at a 1 % bump,
  November +43.9 %, quota-forced December -1.3 %); a compute-bearing gamma feature is
  not warranted as housekeeping
- note: bucketed gamma (~24 builds, ~5 s) remains available as a future feature

### Cycle downgrade ACCEPTED
- the `code_review.md -> finding.md -> code_review.md` cross-reference cycle was
  downgraded from BLOCKER to WARNING by the synthesizer and disclosed; accepted, because
  the edges are one-line mutual citations rather than transitive inclusions and the two
  documents agree

---

## Non-locked positions asserted by SPEC (precedence 0)

Recorded so downstream planning does not re-open them, but they are not locked and may
be revised by a user.

### D-1 — No rewrite of the DP solver or the trinomial tree
- source: docs/SPEC-remaining-work.md (§Out of scope)
- rationale (executed, not asserted): probabilities in [0,1], forward fitting exact to
  1e-15, value converged in `n_p`, race-free parallel passes
- corroborated by: code_review.md (§What is solid), finding.md (peer review, [tree])

### D-2 — The mandatory-quota artefacts are correct behaviour, not defects
- source: docs/SPEC-remaining-work.md (§Out of scope)
- covers: the January step-down at Jan 31, and the December delta amplification
- evidence: finding.md Findings 1 and 2, independently verified — the step tracks `days`
  and is invariant to `n_p` (20/30/45), and the bucketed delta matches finite differences
  to within +/-0.4 % at a 5 bp bump
- consequence: no requirement is derived from finding.md

### D-3 — The hard terminal constraint is retained; the graduated soft penalty is not revisited
- source: finding.md ("Current state: Hard constraint restored. Soft constraint was reverted.")
- reason recommended for the record by the peer review: the penalty multiplier needed to
  replicate the hard constraint is ~4x (not 1.19x), and it "depends on `n_p` and `sVol`,
  i.e. on model discretisation parameters rather than on contract economics"
- precedence note: finding.md is a DOC (precedence 2). SPEC §Out of scope places the
  quota artefacts out of scope but does not itself restate this rationale. Treat as
  supporting context, not as a constraint.

---

## Superseded by precedence (auto-resolved)

### D-S1 — Curve-repricing approach: knot solve supersedes additive per-month shift
- winner: docs/SPEC-remaining-work.md §R1 (precedence 0) — solve the M x M matrix for
  PCHIP knot values that reproduce every contract exactly, iterating solve -> recompute
  slopes -> re-solve (2-3 iterations)
- superseded: code_review.md §5 Fix (precedence 1) — "add per-month `delta_m = target_m -
  mean(smoothed_m)` to that month's days"
- rationale, stated by the winner: the flat additive shift "hits the constraint but
  leaves ~0.1 EUR steps at month boundaries, and the model compares adjacent days to
  decide exercise"
- see INGEST-CONFLICTS.md INFO

### D-S2 — Repricing tolerance: 1e-9 supersedes 1e-6
- winner: docs/SPEC-remaining-work.md §I1 (precedence 0) — relative error < 1e-9
- superseded: code_review.md §Suggested regression tests #1 (precedence 1) — `< 1e-6 * abs(v0)`
- note: the stricter bound is already implemented — test_model.py:72 asserts
  `repricing_gap(...) < 1e-9` across `n_p`. No action required.
