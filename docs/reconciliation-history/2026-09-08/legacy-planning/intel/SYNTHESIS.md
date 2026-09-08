# Synthesis — ingest of 2026-09-08

Entry point for downstream consumers (gsd-roadmapper). Mode: `new` — no pre-existing
`.planning/` context to merge against.

## Documents consumed (4)

- SPEC (2)
  - `docs/SPEC-remaining-work.md` — precedence 0 — the agreed remaining-work plan
    (invariants I1-I3, requirements R1-R5, open questions O1-O2)
  - `code_review.md` — precedence 1 — 10 findings + 10 minor, with a status table
- DOC (2)
  - `finding.md` — precedence 2 — put-delta investigation plus independent peer review
    (evidence, not requirements)
  - `README.md` — precedence 3 — model documentation, four claims known stale

All four classified `high` confidence with manifest-declared types. No UNKNOWN, no
low-confidence docs, no PRDs, no ADRs.

## Cross-reference graph

One cycle detected: `code_review.md -> finding.md -> code_review.md`. Inspected and found
to be mutual citation rather than transitive inclusion; both documents were synthesized
and the deviation from the default BLOCKER rule is disclosed as a WARNING in the conflicts
report. Traversal depth never approached the cap of 50. No other cycles;
`docs/SPEC-remaining-work.md` is referenced by nothing, `README.md` references only code
and data files.

## Decisions — 0 locked

- 2 OPEN decisions requiring user input: **D-O1** (exact curve repricing: default-on vs
  flag) and **D-O2** (`delta` as undiscounted physical hedge volume vs discounted value
  sensitivity). Neither was extracted as a constraint. D-O2 changes the wording of
  invariant I1.
- 3 non-locked positions asserted by the SPEC: no DP/tree rewrite (D-1); the
  mandatory-quota artefacts are correct behaviour (D-2); the hard terminal constraint is
  retained and the graduated soft penalty is not revisited (D-3).
- 2 precedence supersessions recorded: knot solve over additive shift (D-S1); 1e-9 over
  1e-6 repricing tolerance (D-S2).

## Requirements — 5 extracted

- `REQ-exact-curve-repricing` (SPEC R1) — open; blocked on D-O1; the only remaining item
  that changes valuations
- `REQ-converged-np-default` (SPEC R2) — open; default `n_p_full` 10 -> 30
- `REQ-wdr-days-input` (SPEC R3) — **variant-unresolved** (wire through vs remove)
- `REQ-delta-convexity-caveat` (SPEC R4) — **variant-unresolved** (bucketed gamma vs caption)
- `REQ-dead-code-and-docs` (SPEC R5) — open; five sub-items R5.1-R5.5; scope gap on
  review finding 19

Review findings 1, 2, 3, 4, 6, 15 and 18 are fixed per both SPECs and were routed to
context, not to requirements. Finding 17 is deliberately out of scope.

## Constraints — 10 plus a verification checklist

- 3 invariants (nfr / gate): C1 delta reprices the contract to < 1e-9 (master test,
  wording conditional on D-O2); C2 `python test_model.py` 13/13; C3 documented
  before/after on every valuation-affecting change
- 7 technical constraints: C4 curve accuracy, C5 convergence in `n_p`, C6 tree validity,
  C7 ratchet convention, C8 infeasibility must raise, C9 curve coverage, C10 penalty scale
- Type breakdown: 3 nfr/gate + 1 process gate + 3 api-contract + 3 numerical/nfr
- Plus the 8-item regression checklist from code_review.md, and the SPEC's out-of-scope
  boundary

## Context — 8 topics

Project overview; review provenance; fixed findings with measured before/after;
put-swing delta investigation and its peer-review refinements; the reverted
soft-constraint experiment and its corrected ~4x threshold; README documentation state;
deliberately excluded items; repo inventory (dead-code targets and Streamlit defaults
verified still present in the working tree).

## Conflicts

**0 blockers, 6 competing-variants/warnings, 10 auto-resolved/info.**

Warnings requiring user resolution before routing: O1, O2, the R3 variant, the R4
variant, the R5 finding-19 scope gap, and the cross-reference cycle downgrade.

Full detail: `.planning/INGEST-CONFLICTS.md`

## Files

- `.planning/intel/decisions.md`
- `.planning/intel/requirements.md`
- `.planning/intel/constraints.md`
- `.planning/intel/context.md`
- `.planning/INGEST-CONFLICTS.md`

Status: **AWAITING USER** — 6 warnings must be resolved before routing. No blockers.
