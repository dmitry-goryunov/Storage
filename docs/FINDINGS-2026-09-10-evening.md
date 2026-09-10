# Findings — 2026-09-10, evening

An [independent review](INDEPENDENT-REVIEW-2026-09-10.md) of the day's storage work arrived,
was checked, was [replied to](REVIEW-REPLY-AND-ACTION-PLAN-2026-09-10.md) with an
implementation plan, and the first two items of that plan were built. Recorded separately
from [FINDINGS-2026-09-10.md](FINDINGS-2026-09-10.md) because the character is different:
that day widened what the model could be asked, this evening was finding out that three of
its answers were wrong.

`tests/` went **101 → 115** across five commits.

---

## What was done

**Verified the review rather than accepting it.** All 34 files it hashed match this working
copy byte for byte, so it reviews exactly this snapshot. Every claim was reproduced from its
**prose**, not by running its scripts — where a number matches to six decimals, two separate
constructions arrived at it. Nothing in it was found to be wrong. Measurements in
[REVIEW-RESPONSE-2026-09-10.md](REVIEW-RESPONSE-2026-09-10.md).

**P1.1 — dated inventory bounds became constraints.** A `1000·v_step` per-clip penalty is a
price, not a constraint: at a EUR 10,000 price level 19.52 % of paths opened a floored day
empty and the checker accepted every one. Bounds are now inadmissible states in the DP, with
infeasibility propagating back through anything that could reach them. Two further defects in
the same thirty lines: nearest rounding relaxed both sides (71 % → 70 %, 29 % → 30 %), and the
balance was rebuilt as `cumsum(net moves)` with the opening inventory added to day zero only,
so a full store that held everything was rejected against a 100 % floor.

**P1.4 — the discretisation loss became visible, then gated.** `describe_ratchet_rates()`
reports contract against grid MWh/day per inventory level; `value_storage` refuses a loss
over 10 %; `benchmarks.py` carries the inventory ladder, a separate price-grid ladder and a
convergence verdict. The notebook runs at 1,920 clips and prints its residual. `inj_days` no
longer means two things at once.

**Two executable fixtures**, because the numbers that could not be reproduced were the ones
that lived only in prose: [`benchmarks.py`](../benchmarks.py) for every contested
configuration with its source revision and package versions, and
[`two_factor_probe.py`](../two_factor_probe.py) for the P4.1 question.

**Correction notices** on FINDINGS, DESIGN, STATUS and the notebook, marked in place rather
than deleted.

## Discoveries

**The rate loss is a sawtooth, and it bites the *mild* profile.** The review found 16.7 % at
the bottom multiplier. The truth is worse and stranger: the loss is exactly zero wherever
`rate × multiplier` lands on an integer and worst just below one, so it swings between 0 %
and 33 % between adjacent inventory levels. The shipped notebook's softened ratchet — the one
chosen so the ratchets and the 70 % floor could coexist — loses **33.1 % of withdrawal rate
at 41 % full**. A mild ratchet is not a safe ratchet, and a spot check at one fullness proves
nothing about the next.

**The terminal condition was purchasable too.** Found only by fixing the bounds: with a hard
floor in place the store reached it and then paid the `-1e9` terminal penalty rather than
reporting that no schedule existed, surfacing as a misleading *"terminal inventory constraint
is infeasible"*. A fourth instance of the same defect class in one file. Hardened alongside.

**`ratchets.xlsx` is 101 rows of 1.0.** `forward.ipynb` has been running `use_ratchets = True`
against an inert table. Nothing was wrong with the valuations; nothing was ratcheted either.
The notebook now says so out loud.

**A calibrated second factor *costs* a store value — the opposite of what was first
reported.** This is the evening's real finding and it was mine to get wrong twice. Adding a
long factor on top of an unchanged `sigma_chi` gives the two-factor model strictly more total
volatility, and more volatility is worth more to any option; that measures a bigger number,
not a second factor. A jointly calibrated model fits the same observed volatility with both,
so `sigma_chi` comes down — and then a store **loses up to 8 %**, because a store monetises
short-horizon variance and the long factor is where that variance went.

So storage is *not* indifferent to the second factor after all. The exact homogeneity theorem
still holds — at a **fixed** short factor a homogeneous contract does not depend on the long
one, to ten decimals — but on its own it misleads. The channel is the short-factor estimate,
not the extra state. Which is precisely what the reply to the review said, and what I recorded
and then failed to apply.

**And for a struck swing even the sign depends on the calibration anchor**: +0.58 % holding
instantaneous spot variance fixed, −10.23 % holding terminal variance fixed, same contract and
the same `sigma_xi`. The two factors differ in their variance *term structure*, which is what a
real calibration fits and no scalar captures.

**The two accuracy gates disagree about a grid.** The 10 % rate gate clears 1,920 clips; the
0.5 % value-convergence gate wants 3,840. Both are defensible and they are not the same
question, so a number should say which one it was accepted under.

**The unstruck swing is the control that made the homogeneity result conclusive.** Same
contract, strike removed, effect vanishes exactly. Without it the result reads as "storage is
special"; with it, it reads as "homogeneity is what matters", which is the true statement and
the one that generalises.

## Claims corrected

Five from the review, all confirmed by measurement: the 52.5 % ratchet cap and the
"physically infeasible" 70 % floor are grid artefacts; "refining the clip is free" holds only
unratcheted, which is the one case that was checked; the day-ahead gap is five days, not
three; and "intrinsic delta nets to zero" needs a zero-fuel qualifier.

Three of my own, made while responding to it:

- **"About 1.3 % for a struck swing."** Measured at a fixed short factor. Withdrawn; see
  above.
- **"A variance-matched comparison does not isolate the factor."** I inferred this from the
  store moving under matching. Backwards — the store moving *is* the result.
- **"The review treats extrinsic-rises-with-mean-reversion as unsupported."** It does not;
  §6.4 demonstrates the increase in its own example. I read §6.6's dispute of the endpoints as
  a dispute of the direction.

## Process notes

- **The Bash heredoc ate a backslash escape for the fourth time.** `\n` inside notebook
  source became a real newline and broke an f-string, in the same session that documented the
  trap. Notebook and doc edits go through a file written with the editor; the heredoc is only
  safe for content with no escapes at all.
- **Every claim was reproduced from prose rather than by running the review's scripts.** That
  is slower and it is the reason the agreement means anything — and it is how the sawtooth
  turned up, which running their script would not have shown.
- **`benchmarks.py` reproduced the review's figures on different package versions** —
  NumPy 2.4.3 / pandas 2.3.3 / Numba 0.65.1 against its 2.5.3 / 3.0.5 / 0.67.0. A stronger
  reproduction than intended.
- **`origin` was still on the pre-rename URL**, a day after
  [yesterday's note](FINDINGS-2026-09-09.md) diagnosed exactly that. Consolidating the two
  remotes kept the wrong one, so every push since had been reaching
  `dmitry-goryunov/Storage` through a GitHub redirect and saying so in a line that reads
  like noise. Fixed. A redirect is a courtesy, not a guarantee.
- **A third Drive lock variant**: `AUTO_MERGE.lock` during a fast-forward merge, after two
  `packed-refs.lock` incidents yesterday. As before the operation succeeded and only the
  lock was left behind — which is the part worth knowing, because the error message says
  the opposite.

## What to do next

**Calibration is the only substantial item left**, and the probe sharpened what it has to
answer rather than merely confirming it was needed:

1. **What variance term structure does TTF actually have?** No scalar anchor decides what a
   second factor is worth — two defensible anchors gave a struck swing opposite signs. Only a
   panel fit resolves it, so the fit is load-bearing rather than preparatory.
2. **Does the current model overstate storage?** If a permanent component exists and
   `sVol = 0.9` was ever anchored to spot volatility, the one-factor model attributes all of
   that variance to the mean-reverting factor — the one storage monetises. The probe says a
   calibrated second factor takes up to 8 % off. That is a hypothesis about our own numbers.
3. **Delivery-aware returns.** `benchmarks.py` uses point maturities `tau = i/12`, flagged as
   an upper bound on the model side. Monthly delivery lowers it, so 3.1× is not yet
   like-for-like.

Two obstacles: **`ttf q.xlsx` ends 6 March 2026**, so it cannot support a current calibration
without newer data; and the review's §5 **exploratory covariance fit and PCA** are the one
part never checked, and belong inside this stage rather than being taken on trust.

Absorbs P3.4 and P4.3. The reply's §5 — whether an extra valuation dimension is justified —
folds in here too, because the probe's answer was that it cannot be decided ahead of the
calibration.

**Two small open decisions:** whether to pay ~8 s a valuation for the 3,840-clip grid that
clears the value gate, and whether `ratchets.xlsx` gets a real profile or a comment saying it
is deliberately inert.

**Unchanged:** P1.2 curve-shape criteria, P1.3 discount-curve source and settlement, P2.1
terminal backstop, P2.2 tie threshold (measured and declined), the backtest (deferred), and
`ttf q.xlsx` row 4172.

---

**Still verified, not validated.** Three answers were wrong this evening and are now right.
None of that establishes that the numbers are right for a traded price; that gate is still
calibration plus the backtest.

The pattern underneath all three is worth more than any of them. **Each wrong claim had been
checked in the one configuration where it could not fail** — refinement tested unratcheted,
where the rates are exactly expressible; the zero-net-hedge statement measured at zero fuel,
in the document that introduced fuel; the bound checker only ever run from an empty store; the
second factor compared without recalibrating the first. The tests passed throughout, and 101
of them still pass over all four.
