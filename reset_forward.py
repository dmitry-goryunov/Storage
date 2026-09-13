"""The conditional month-ahead quote a monthly-reset swing's strike is set from.

DESIGN-MONTHLY-RESET-SWING-2026-09-13.md sec.5 (Phase 1). Given the model's own
price lattice, computes -- exactly, by backward induction through the lattice's
own already-validated transition probabilities, not a separately-derived
closed-form projection formula -- the conditional expectation of a future date's
price given the node reached at an earlier date. That is what "the month-ahead
price observed on day t" means inside this model: not the unconditional forward
curve (which only describes day 0's view), and not the spot price at t (which
says nothing about a LATER delivery date).

The design doc's sec.5 explicitly prefers this over inserting a continuous-time
OU projection formula, precisely because deriving a closed form for this
lattice's own specific discretisation (raw mean-reverting coordinate plus a
per-date forward-distortion shift) is exactly the kind of thing S5 found silently
wrong by 8.95% on a DIFFERENT, simpler lattice. Reusing p_u/p_m/p_d directly
sidesteps that risk by construction: whatever the lattice actually implements is
what this projects forward, with no second formula to disagree with it.
"""
import numpy as np
import pandas as pd

import storage_model as sm


def build_lattice(val_date, storage_start, storage_end, vol, sMR, n_p,
                   daily_curve=None, curve=None, discount_rate=0.0):
    """The price lattice alone -- fwd, x, q, p_u, p_m, p_d -- with none of
    `Storage.build()`'s fixed-strike DP run alongside it. Reuses `Storage.__init__`
    for date-grid and curve-coverage handling (same errors on a gappy curve, same
    month-end backstop) rather than a second copy of that logic.
    """
    s = sm.Storage(val_date, storage_start, storage_end, curve=curve, n_p=n_p,
                   sVol=vol, sMR=sMR, daily_curve=daily_curve, discount_rate=discount_rate)
    fwd, x, q, p_u, p_m, p_d = sm.build_tree(s.price_curve, s.n_t, s.n_p, s.sVol, s.sMR)
    return dict(storage=s, fwd=fwd, x=x, q=q, p_u=p_u, p_m=p_m, p_d=p_d,
               date_span=s.date_span, d_curve=s.d_curve)


def _backward_conditional_expectation(p_u, p_m, p_d, terminal_i, terminal_values):
    """H(i,j) = E[terminal_values(j') | node j at date i], for every i <= terminal_i,
    by exact backward induction through the lattice's own one-step transitions --
    the same p_u[i,:-1]/p_d[i,1:] shift-and-add pattern `_tree_core` uses for its
    forward q propagation, run backward instead. No boundary special-casing is
    needed here beyond that shift: p_u/p_d are already zero in the direction a
    boundary node cannot move.
    """
    width = terminal_values.shape[0]
    H = np.zeros((terminal_i + 1, width))
    H[terminal_i, :] = terminal_values
    for i in range(terminal_i - 1, -1, -1):
        nxt = H[i + 1, :]
        cur = p_m[i, :] * nxt
        cur[:-1] += p_u[i, :-1] * nxt[1:]
        cur[1:] += p_d[i, 1:] * nxt[:-1]
        H[i, :] = cur
    return H


def project_month_end_quotes(lattice, month_end_dates):
    """For each date in `month_end_dates`, the conditional expectation of that
    date's own price -- exp(x[u,:]) -- projected back to every earlier date i,
    for every price node j.

    This is the "month-end point" convention: the index a reset uses is the
    model's conditional view of the price on the LAST day of the delivery month,
    not a delivery-period average. That mirrors the point-vs-delivery-averaged
    distinction `delivery_model.py`/`benchmarks.py` already document for a
    different purpose (the ~0.1193-vs-0.1244 point/average gap) -- a real,
    measurable simplification, not an invisible one, and the reason it is a
    documented prototype convention rather than an assumed one.

    Returns `{date_span_index_of(month_end): H}`, H of shape
    (that index + 1, 2*n_p+1) -- H[i,j] for i from 0 to the month-end index.
    """
    date_span = lattice["date_span"]
    x, p_u, p_m, p_d = lattice["x"], lattice["p_u"], lattice["p_m"], lattice["p_d"]
    result = {}
    for month_end in month_end_dates:
        u = date_span.get_loc(pd.Timestamp(month_end))
        result[u] = _backward_conditional_expectation(p_u, p_m, p_d, u, np.exp(x[u, :]))
    return result
