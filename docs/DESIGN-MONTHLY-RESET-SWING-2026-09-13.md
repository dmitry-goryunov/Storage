# Design proposal — a swing with a monthly-reset, market-indexed strike

13 September 2026. Proposal only: no code changes accompany this document. Written at the
user's request, choosing explicitly between two readings of "month-ahead strike, set one
month prior" — a pre-declared monthly schedule (cheap, already close to what the model
supports) and a market-indexed reset strike (expensive, genuinely new). This covers the
second.

## 1. What the contract is

For each delivery month `M` in the deal's operating window, the strike `K_M` applied to every
exercise decision during `M` is set from the realised **month-ahead index** observed during
the calendar month immediately before it, `M-1` — for example, the arithmetic average of the
daily month-ahead (front-month) quote across every trading day of `M-1`. `K_M` is unknown at
contract inception; it is fixed only once `M-1` has finished, and it is then a plain, known
number for the whole of `M`.

Three things must be pinned down as commercial terms before this is anything more than a
sketch, and none should be assumed the way this project has learned not to assume such things:

- **The index itself.** "Month-ahead" in this repository's own convention is the next
  calendar month's forward (`front_month_start` in `storage_model.py:364` — the same
  definition `quote_data.py`'s delivery panel uses). Whether the real contract means that,
  or day-ahead, or a specific published index, is a term-sheet fact, not a modelling choice.
- **The averaging window and day-count.** Every trading day of `M-1`, business days only,
  a specific settlement window inside `M-1` — these give different numbers. `quote_data.py`'s
  `build_returns` already had to make exactly this kind of decision explicit for realised
  returns; the reset index needs the same discipline.
- **What happens with no reset yet.** The first delivery month of the deal has no `M-1`
  inside the contract. Either the deal starts one month later than the nominal `storageStart`
  for indexing purposes, or an initial strike is supplied directly as a term-sheet input.

## 2. Why the current model cannot price this as written

`value_put_swing`/`value_call_swing` (`storage_model.py:1226`, `:1287`) take one scalar
`strike` and broadcast it uniformly: `s.i_cost[s.Dt:s._active] = -strike` for a put,
`s.w_cost[s.Dt:s._active] = strike` for a call. `i_cost`/`w_cost` are already **per-day
arrays** consumed by the kernel — a pre-declared, known-at-inception monthly schedule is a
half-day change to fill them non-uniformly, and needs no new state.

The reset structure is a different problem. `K_M` is a random variable at valuation time, and
its value depends on the *path* the price takes through `M-1`, not on where the lattice
happens to sit at any single date. The DP's state is `(time step i, price node j, inventory
state l)`. Nothing in that state remembers "the running average so far this window," so the
kernel has no way to know, on any given path, what `K_M` will turn out to be by the time it
matters.

## 3. The mechanism this design leans on, and the one thing that must be re-derived first

`storage_kernels.py::_tree_core` builds a trinomial lattice on a **raw**, mean-reverting
log-deviation coordinate (`mxi_dt = mr*dt*xi`, pulling every date's raw grid toward zero), then
adds a per-date, node-independent shift so that `E[exp(x[i,:])] = fwd[i]` exactly (the "Forward
distortion" step, `x[i, j_s:j_e] += log(fwd[i]/expected)`). The price at node `(i,j)` is
`exp(x[i,j])` after that shift; `q[i,:]` is the already-computed probability of being at each
node at date `i`.

That structure is exactly what a month-ahead reset needs: the forward price for a *future*
delivery month, as seen from a node sitting earlier on the lattice, is a projection of that
node's deviation from its own date's forward, decayed by the model's own mean reversion. This
project has solved a closely related problem before — `two_factor_probe.py`'s exact one-step
OU moments (S5, `docs/STATUS.md`'s S5 entry) — and found that the "obvious" continuous-time
projection formula, applied carelessly to a specific discretisation, silently mismatched the
lattice's own propagated variance by 8.95%. That was a *simpler*, non-forward-distorted
process than this one.

**Task 0, before any kernel code:** derive the exact conditional-forward-projection formula
for *this* lattice's specific discretisation — accounting for the per-date `expected[i]`
correction the forward-distortion step applies, not just the raw process's OU decay — and
verify it numerically against `Storage.x`/`Storage.build()`'s own `q`/`x` arrays at several
`sMR` values including zero and near-zero, the same protocol S5 used. Do not assume the
closed form from `two_factor_probe.py` transfers unchanged; that lattice has no forward
distortion to correct for. This is the single highest-risk step in the whole design, and it
is cheap to get wrong quietly.

## 4. Proposed state-space extension

Add a fourth DP dimension: a discretised **running average of the month-ahead index within
the current reset window**, `R`. Concretely:

1. **Scope of `R`.** It only needs to span one month's worth of index observations, not the
   whole deal's price range — averaging damps variance, so its grid can plausibly be narrower
   than the price tree's own `n_p` half-width. How much narrower is an empirical question
   (§6), not an assumption.
2. **Day-count is not a new state.** Days elapsed since the last reset boundary is a
   deterministic function of the time index `i` (the calendar is known in advance), so the
   running mean's update weight at each step is known without tracking a separate counter.
3. **Update rule.** At each step from node `(i,j)` to a successor node, compute that step's
   realised month-ahead quote via Task 0's projection formula, then
   `R_new = (R_old * days_so_far + quote_today) / (days_so_far + 1)`. `R_new` will generally
   fall between two of `R`'s discretised levels; interpolate the continuation value between
   them — the standard forward-shooting-grid technique for arithmetic-average (Asian-style)
   lattice pricing, not something specific to this repository.
4. **Reset boundary.** On the last day of `M-1`, `R` is finalised and becomes `K_M`: a known
   number from that point on. From the first day of `M` onward, the existing fixed-strike
   swing logic applies exactly as it does today, with `K_M` in place of the scalar `strike` —
   no new logic needed for the exercise decision itself, only for handing `R` off as `K_M` at
   the boundary.
5. **Repeating resets.** If every month resets the next month's strike (a chained, cliquet-style
   structure rather than a single reset), `R` re-initialises and starts accumulating again at
   the start of each `M-1` window, for every month of the deal. The DP carries this dimension
   throughout, not just once.

## 5. Sizing and cost

Total state space becomes `n_t × (2*n_p+1) × n_op × n_R` instead of `n_t × (2*n_p+1) × n_op` —
a multiplicative blow-up by whatever `n_R` (the number of average-price buckets) turns out to
need. Two things make this a real risk rather than a formality:

- **Storage's `n_op` is already large in practice** — the reference 30/60 store runs at 1,920
  clips after ratchet-driven refinement (`docs/STATUS.md`'s S2 entry). Multiplying that by a
  further `n_R` of, say, 30–50 could be prohibitive; a swing's own `n_op` (it tracks cumulative
  exercised volume against a quota, not physical inventory, but is still a real dimension) may
  be much smaller and this may be tractable there first.
- **`n_R` itself is unmeasured.** This design does not assume a number. Guide-issue S7's own
  standing instruction applies here without modification: *measure dimensions using the actual
  number of dates, factor nodes and inventory levels; do not assume a convenient size is
  adequate.* The right first step is a convergence study — price a small test case at `n_R` =
  10, 20, 40, 80 and see where the answer stops moving, exactly the discipline
  `benchmarks.convergence_verdict()` already enforces for the price and inventory grids.

## 6. Open decisions this design does not make unilaterally

- **The hedge convention.** Existing swing `delta` is a hedge against the *underlying* forward.
  Here, the strike is itself linked to a traded month-ahead instrument, so part of the
  position's risk is naturally hedgeable with that same instrument during the reset window.
  Whether `delta` should report against the underlying only, the index instrument only, or
  both separately is a genuine new convention decision — the same kind of decision D-O2/D-O3
  were for fuel loss and PV-tailing, and it deserves the same explicit, tested answer rather
  than an implicit default.
- **Behaviour with no `M-1` inside the deal** (§1, third bullet).
- **Interaction with ratchets and dated inventory bounds**, which already consume the
  inventory-state axis `l` — nothing here changes that axis, but the combination should be
  covered by a test, not assumed to compose safely by inspection.

## 7. Verification plan

Matching this repository's standing rule that every behavioural change starts with a failing
test, and that a green suite is not itself evidence for a case it never exercised:

1. **Degenerate limits with a hand-computable answer**, mirroring the "Independent follow-up
   hardening" pattern already in `docs/STATUS.md`: a one-day reset window (no averaging —
   reduces exactly to today's already-tested point-strike swing, priced identically) and
   zero volatility (the path is deterministic, so the realised average and hence `K_M` can be
   computed by hand and checked against the model's output exactly).
2. **An independent Monte Carlo cross-check** — simulate price paths under the same dynamics,
   compute each path's realised monthly average and resulting swing payoff directly, and
   compare the discounted mean to the lattice's reported value. This project already has a
   precedent for exactly this kind of independent, non-lattice oracle (the exhaustive
   four-day schedule enumeration, the Black-76 collapse case for a vanilla call).
3. **The repricing identity**, `sum(DF·delta·F) == V0`, re-derived for whatever hedge
   convention §6 settles on rather than assumed to still hold unchanged.
4. **A convergence gate on `n_R`** specifically (§5), reported and gated the same way
   `convergence_verdict()` already gates the price and inventory grids — this feature adds a
   third numerical dimension that needs its own recorded evidence, not a borrowed one.

## 8. Effort estimate

Comparable in scope to the two-factor design (`docs/DESIGN-P4.1-two-factor.md`) at the point
it was originally proposed: a genuine kernel-level extension (a new dimension in
`storage_kernels.py::run_model`), a new closed-form derivation with its own verification
protocol (§3's Task 0), a new hedge-convention decision, and a full independent-oracle test
suite before it can be trusted. Order of magnitude: comparable to that "self-contained
project," not a short add-on — a rough T-shirt size, not a day count I can responsibly commit
to before Task 0 and the `n_R` convergence study are actually done, since either could change
the shape of the rest.

## 9. Cheaper interim, offered for contrast

Approximate the index with a **single point observation** — the month-ahead quote on the last
day of `M-1` only, not the whole month's average — rather than a true running mean. This
removes the new state dimension entirely: at the reset date, the lattice already has the full
node distribution `q[i*,:]` and each node's price `exp(x[i*,j])`. Month `M`'s conditional
value, given each possible reset-date node `j`, is priced with the *existing* fixed-strike
swing code using that node's implied forward as `K_M(j)`, then weighted by `q[i*,j]` and
summed — no kernel change, only an orchestration layer calling the existing valuation
function once per reset-date node per month. This has a real, quantifiable bias against the
true averaged index (of the same character as the 0.11933-versus-0.12444 point-versus-delivery-
average gap `delivery_model.py` already measures for a different purpose), and that bias
should be measured before deciding it is acceptable — but it is implementable without
touching `storage_kernels.py` at all, and would give a working answer to price against while
the full running-average extension is built and verified.

## 10. Recommended next step

Do Task 0 (§3) first, on its own, before committing to the rest of this design: derive and
numerically verify the exact conditional-forward-projection formula for this lattice against
`Storage.build()`'s own arrays. It is the one piece every other section depends on, it is the
one place this project has already been burned by an analogous shortcut (S5), and it is
answerable in isolation without touching the kernel or committing to `n_R`'s eventual size.
