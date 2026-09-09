# Decision memo — what should `delta` mean?

Date: 2026-09-08
Status: **DECIDED 2026-09-08 — undiscounted.** The code now matches decision D-O2;
`delta` carries no discount factor and the invariant carries the weights instead. The
analysis below is kept as the reasoning behind that choice.
Reproduce: the table below comes from a mandatory put swing buying 30 days of 2027,
valued 2026-01-01 off `curve.csv`, `n_p = 20`, `sVol = 0.5`.

## The disagreement (resolved)

`compute_all_metrics` used to apply the discount factor to `delta`:

```python
discount = np.asarray(d_curve[:n_t], dtype=float)[:, None, None]
delta    = list(-(pa * discount * exp_x).sum(axis=(1, 2)) / fwd[:n_t]) + [0.0]
```

and `tests/test_reconciliation.py::test_delta_is_discounted_forward_sensitivity` locks that
in. The module docstring calls it a "discounted forward-price sensitivity (PV-equivalent
MWh)".

The decision recorded at the ingest conflict gate — `D-O2` in
`docs/reconciliation-history/2026-09-08/legacy-planning/intel/decisions.md`, and §R6 of the
archived SPEC — says the opposite: an **undiscounted physical hedge volume**, on the
grounds that the discount factor cancels in the hedge ratio.

Both are internally consistent. They are different quantities, and only one of the two
records can stand.

## What each is

**Discounted (as coded).** `∂V/∂F_i · F_i` — the PV sensitivity. The repricing identity is
`Σ delta_i · F_i = V₀` exactly.

**Undiscounted (as decided).** `E[S_i·Q_i] / F_i` — the MWh of forward you trade. Hedging
day *i* with *h* forwards gives `PV = h·DF_i·F_i·ε` against `dV/dε = DF_i·E[S_i·Q_i]`; the
discount factor cancels, so `h = E[S_i·Q_i]/F_i`. The identity becomes
`Σ DF_i · delta_i · F_i = V₀`, equally exact.

## What it costs, measured

| Discount rate | Reported (discounted) | Tradeable forward volume | Under-hedge |
|---|---:|---:|---:|
| 0 % | −27,740.4 MWh | −27,740.4 MWh | 0.00 % |
| 3 % | −26,457.0 MWh | −27,743.4 MWh | **−4.64 %** |
| 5 % | −25,614.9 MWh | −27,746.4 MWh | **−7.68 %** |

The repricing identity holds to 0.00 EUR in every row, so no consistency argument
separates them.

Two observations that do:

1. **The tradeable volume is rate-invariant** (−27,740 → −27,743 → −27,746 across 0–5 %),
   as a physical quantity must be. The reported number moves 7.7 % over the same range.
   A hedge ratio that changes when the discount curve changes, while nothing about the gas
   changes, is a hedge ratio the desk has to remember to correct.
2. **The error grows with tenor** — at 3 %, monthly buckets run −3.0 % in Jan-27 to −5.7 %
   in Dec-27. A hedger trading the reported MWh is short by 1,286 MWh on 30,000 MWh of
   physical volume, concentrated in the far months.

| Month (3 %) | Reported | Tradeable | Under-hedge |
|---|---:|---:|---:|
| Jan-27 | −739.9 | −763.1 | −3.0 % |
| Apr-27 | −3,605.1 | −3,747.7 | −3.8 % |
| Aug-27 | −1,947.0 | −2,044.2 | −4.8 % |
| Dec-27 | −4,962.8 | −5,262.9 | −5.7 % |

## Recommendation

Keep `delta` **undiscounted** and move the discount factor into the identity, per D-O2.
The number is labelled in MWh, plotted as MWh/day and summed into a monthly table a trader
reads as a hedge; it should be the quantity you can trade, and it should not move when the
yield curve does.

If the PV sensitivity is wanted as well, report it as a second, separately named series
rather than overloading `delta`.

## Subsequent implementation

PR #6 added `discount_rate` to `Storage` and `run_valuation`, so `d_curve` is no longer one
on every path. The original decision remains: `delta` is the undiscounted OTC-forward hedge
volume, while `delta_pv = d_curve · delta` is the PV sensitivity or futures-equivalent
series. Neither label removes the need to state the hedge instrument, margining and
settlement convention.

## If the decision is "keep discounted"

Then update, in one commit: `D-O2` in the archived decisions file (marked superseded, not
edited in place — it is history), §R6 of the archived SPEC, and item 2 of
`docs/reconciliation-history/2026-09-08/MIGRATION-LOG.md`. Add a note to the Streamlit
monthly-delta table that the figures are PV-equivalent MWh and must be divided by the
discount factor before trading.

## If the decision is "revert to undiscounted"

Then in `compute_all_metrics` drop the `discount` factor, restate the invariant as
`Σ DF_i · delta_i · fwd_i == V₀`, and rewrite
`test_delta_is_discounted_forward_sensitivity` to assert the discounted identity against an
undiscounted delta. The current test would otherwise fail, correctly.
