## Conflict Detection Report

Ingest mode: new. Docs: 4 (2 SPEC, 2 DOC). No ADRs, no PRDs, no locked decisions.
Precedence applied: manifest overrides — SPEC-remaining-work.md (0) > code_review.md (1)
> finding.md (2) > README.md (3).

### BLOCKERS (0)

None. No ADRs were ingested, so there are no LOCKED decisions and no LOCKED-vs-LOCKED
contradiction is possible. No document classified UNKNOWN or low-confidence. No existing
.planning/ context to contradict (mode: new).

### WARNINGS (6)

[WARNING] O1 undecided — exact curve repricing default-on or behind a flag
  Found: docs/SPEC-remaining-work.md §R1 "Open question O1" states two options and a
    recommendation (default-on with `exact=False` available for comparison), explicitly
    marked as an open question rather than a decision.
  Impact: R1 is "the only remaining item that changes valuations" (SPEC §R1, Risk).
    Choosing default-on shifts notebook results on the next run; choosing the flag leaves
    a curve that does not reprice its own inputs as the default path. Synthesis cannot
    pick without deciding a valuation-visible behaviour on the user's behalf.
  → Decide (a) default-on with `exact=False` escape hatch, or (b) opt-in flag, before
    REQ-exact-curve-repricing is routed.

[WARNING] O2 undecided — is `delta` a physical hedge volume or a value sensitivity
  Found: docs/SPEC-remaining-work.md §O2 offers two defensible conventions —
    (a) undiscounted physical hedge volume, documented, with I1 restated in undiscounted
    terms; (b) discounted value sensitivity, applying `d_curve` and keeping I1 exact.
    code_review.md §9 states the same fork and measures the gap: with `d_curve` at a 3 %
    continuous rate, V0 = -652,146.1 vs sum(delta*F) = -684,200.5, a 4.92 % overstatement.
  Impact: this decision changes the wording of invariant I1, the project's master
    regression test (constraints.md C1). Routing work against I1 before O2 is resolved
    risks encoding the wrong invariant. Invisible today only because `d_curve` is all
    ones everywhere in use.
  → Decide (a) or (b), then restate I1 accordingly. The SPEC's own guidance: "Decide
    before it becomes load-bearing."

[WARNING] Competing acceptance variants for REQ-wdr-days-input (SPEC R3)
  Found: docs/SPEC-remaining-work.md §R3 states "Either wire it through (`n_op` from
    injection capacity, `w_ratch`/`max_vol` from withdrawal) or remove the input."
    code_review.md §8 states the identical fork.
  Impact: the two variants produce different products — one adds asymmetric-capacity
    valuation, the other deletes a user-facing control. Merging them or silently picking
    one would lose intent and mis-size the work.
  → Choose "wire through" or "remove input" before routing. Both variants are preserved
    in intel/requirements.md.

[WARNING] Competing acceptance variants for REQ-delta-convexity-caveat (SPEC R4)
  Found: docs/SPEC-remaining-work.md §R4 states "Either report a bucketed gamma (+/-1 %
    rerun, ~24 builds, ~5 s) or caption the table as a local ratio requiring re-hedging."
    code_review.md §10 states the same fork with "or at minimum".
  Impact: variant (a) is a compute-bearing feature (~5 s per run); variant (b) is a
    display string. Estimation and sequencing differ by an order of magnitude.
  → Choose bucketed gamma or caption-only before routing.

[WARNING] R5 claims coverage of review finding 19 but supplies no acceptance bullet
  Found: docs/SPEC-remaining-work.md §R5 is titled "Dead code and documentation (findings
    11-14, 16, 19, 20)", but its five bullets cover findings 11, 12, 13, 14, 16 and 20
    only. Finding 19 — code_review.md §Minor 19, "`Storage.n_op_start` is reassigned by
    callers after `set_volume_states()` set it, so the two meanings (state count vs.
    starting inventory) are entangled; any later `set_volume_states` call silently resets
    the start state" — has no corresponding acceptance criterion.
  Impact: the highest-precedence document's header and body disagree. Synthesis would
    either drop a named finding or invent scope for it; both are wrong by default.
  → Confirm whether finding 19 is in R5 scope and state its acceptance, or strike 19 from
    the R5 heading.

[WARNING] Cross-reference cycle between code_review.md and finding.md — synthesized anyway
  Found: cycle detection on the `cross_refs` graph returns one cycle,
    code_review.md -> finding.md -> code_review.md. code_review.md's header points to
    finding.md ("The delta investigation itself is reviewed separately at the end of
    finding.md"); finding.md's closing line points back ("See `code_review.md` #1").
    Max traversal depth 50 was not approached; no other cycle exists. README.md
    references only code and data files; nothing references docs/SPEC-remaining-work.md.
  Impact: the default rule for this workflow is to treat any cross-ref cycle as a BLOCKER
    and to skip synthesis of the cyclic set. Both documents were inspected: the edges are
    citations, not transitive inclusions — each document is self-contained, was read once,
    and their content agrees rather than conflicts. Both were therefore synthesized, which
    is a deliberate downgrade from BLOCKER to WARNING and is disclosed here rather than
    resolved silently. If the downgrade is not accepted, all intel derived from
    code_review.md and finding.md must be re-reviewed.
  → Accept the downgrade, or re-run with one of the two cross-references removed to
    satisfy the rule literally.

### INFO (10)

[INFO] Auto-resolved: SPEC R1 knot solve supersedes code_review's additive per-month shift
  Note: code_review.md §5 (precedence 1) prescribes "add per-month `delta_m = target_m -
    mean(smoothed_m)` to that month's days". docs/SPEC-remaining-work.md §R1
    (precedence 0) explicitly rejects it — "Not the flat additive per-month shift: it hits
    the constraint but leaves ~0.1 EUR steps at month boundaries, and the model compares
    adjacent days to decide exercise" — and mandates the M x M knot solve instead. Higher
    precedence wins; requirements.md carries only the knot solve.

[INFO] Auto-resolved: repricing tolerance 1e-9 supersedes 1e-6
  Note: docs/SPEC-remaining-work.md §I1 requires relative error < 1e-9;
    code_review.md §Suggested regression tests #1 suggests `< 1e-6 * abs(v0)`. Precedence 0
    wins, and no work follows: test_model.py:72 already asserts `< 1e-9` across `n_p`.

[INFO] Auto-resolved: SPEC R5 supersedes README's stale claims
  Note: README.md is a DOC (precedence 3) describing existing behaviour. Its four known
    inaccuracies are content defects, not requirements; the requirement to fix them is
    docs/SPEC-remaining-work.md §R5 (precedence 0). Verified against the code during
    synthesis: `sVol` defaults to 0.9 (storage_model.py:67) against 0.6 documented; and
    `flat` / `profiled` are assignment aliases of one `price_per_mwh`
    (storage_model.py:213-214), so the documented "Intrinsic = profiled - flat"
    decomposition cannot hold. No requirement was extracted from README.md.

[INFO] Two further README inaccuracies detected during synthesis
  Note: beyond the four already recorded — (a) README §"Forward Curve" says the curve is
    "smoothed via a natural cubic spline fitted through monthly midpoints", while the code
    uses `PchipInterpolator` plus `CubicHermiteSpline` (storage_model.py:5, :36-37), which
    is also what R1 depends on; (b) README §"Post-Processing" documents `get_exercise` and
    `valuation` as part of the API, and R5.1 deletes both. Both fall inside R5.3's intent
    but are not named by it. Surfaced as INFO because R5.3 already owns README accuracy.

[INFO] README's `1e10` infeasibility claim was checked and is accurate — do not "fix" it
  Note: README §"Dynamic Programming Solver" states "Infeasible transitions are penalised
    with a large dummy value (`1e10`)". This looked stale against the `-1e9` sentinel cited
    in code_review.md §4 and finding.md, but they are two different quantities:
    `bigdummy = 1e10` inside `run_model` (storage_model.py:634) is the per-transition
    penalty, while `-1e9` (storage_model.py:123) is the terminal `t_p_curve` sentinel.
    Recorded so R5.3 does not "correct" a true statement.

[INFO] code_review findings 1, 2, 3, 4, 6, 15, 18 are fixed — historical, not work
  Note: code_review.md's status table and docs/SPEC-remaining-work.md §Context agree
    exactly on the fixed set. Where code_review.md still carries a "Fix:" paragraph for
    these findings, it is a record of what was done, not outstanding work. Detail moved to
    intel/context.md; none of it is in requirements.md.

[INFO] finding.md internal contradiction on the soft-penalty threshold — self-corrected
  Note: finding.md's original section states a threshold multiplier of 1.19x for full
    exercise; its own embedded peer review measures -24,530 MWh (82 % of quota) at 1.19x
    and puts the true threshold at ~4x, explaining 1.19x as the deterministic bound while
    the reachable spot maximum is ~max(F) * e^(n_p*dx) ~= 107 EUR ~= 4.26x flat_metric.
    The later peer review supersedes the earlier claim within the same document. No
    downstream requirement depends on either number — the soft-penalty route is reverted
    and the SPEC places the area out of scope — so the corrected value is recorded in
    context.md and decisions.md D-3 only.

[INFO] finding.md yields no requirements
  Note: finding.md is evidence (precedence 2). Both of its anomalies — the January
    step-down and the December delta amplification — are confirmed correct model behaviour
    and are explicitly placed out of scope by docs/SPEC-remaining-work.md. Its material
    contribution is validation data (finite-difference delta checks, the convexity scan
    feeding R4, and the sum(delta*F) = V0 invariant that became I1).

[INFO] code_review finding 17 is deliberately out of scope
  Note: `strat` uses an absolute tolerance (`< 1e-6`) to call a state "no exercise" while
    `v` keeps the maximised value. code_review.md §Minor 17 calls it "currently
    immaterial" since the repricing identity holds to 0.1 EUR;
    docs/SPEC-remaining-work.md §R5 omits it from every bullet and from its heading. Not
    routed as a requirement; recorded in context.md.

[INFO] No ADR layer in this ingest
  Note: the precedence chain ADR > SPEC > PRD > DOC resolved entirely within its SPEC and
    DOC tiers, using the per-doc manifest overrides. Because no decision is LOCKED, every
    entry in intel/decisions.md is revisable — including D-1, D-2 and D-3, which merely
    record positions asserted by the SPEC and by finding.md. If any of these should be
    binding, they need an ADR.
