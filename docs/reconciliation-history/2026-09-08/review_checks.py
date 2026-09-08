"""
Reproduces every number quoted in the peer-review section of finding.md and in
code_review.md.

Run with:  python review_checks.py  [section ...]

Sections whose finding has been fixed print "rejected: ..." where the model now
refuses a configuration it used to mis-price. That output is the fix working,
not a failure -- the pass/fail regression suite is test_model.py.

Sections: repro np_invariance convergence delta_fd bucket_convexity soft_penalty
          curve_fit tree ratchets infeasible discounting
Default: all.
"""
import re
import sys
import time

import numpy as np
import pandas as pd

from storage_model import (Storage, build_tree, curve_df_for_storage,
                           daily_arithmetic_flat_metric, map_curve_to_dates,
                           quote_row_for_fd_date, smoothen_curve)

FDDATE       = pd.Timestamp("2026-01-05")
VALDATE      = pd.Timestamp("2026-01-01")
STORAGESTART = pd.Timestamp("2027-01-01")
STORAGEEND   = pd.Timestamp("2027-12-31")
DAYS, VOL, N_P, V_STEP = 30, 0.50, 30, 1000

_curve = None


def get_curve():
    """Curve used by finding.md: TTF quote matrix, FDDate 2026-01-05, DA stub."""
    global _curve
    if _curve is None:
        quotes = pd.read_excel("ttf q.xlsx")
        quotes = quotes.rename(columns={quotes.columns[0]: "quote_date"})
        quotes = quotes.dropna(subset=["quote_date"]).copy()
        quotes["quote_date"] = pd.to_datetime(quotes["quote_date"], format="mixed")
        quotes = quotes.sort_values("quote_date").reset_index(drop=True)
        cols = sorted([c for c in quotes.columns if re.fullmatch(r"TTFc\d+", str(c))],
                      key=lambda c: int(re.search(r"\d+", str(c)).group()))
        row = quote_row_for_fd_date(quotes, cols, FDDATE, exact=False)
        _curve = curve_df_for_storage(row, cols, curve_start=VALDATE, include_da=True)
    return _curve


def put_swing(n_p=N_P, days=DAYS, vol=VOL, v_step=V_STEP, storageEnd=STORAGEEND,
              ratch=None, eps=0.0, months=None, d_rate=0.0, t_p_curve=None, curve=None):
    """Same wiring as storage_model.value_put_swing (mandatory buy on `days` days)."""
    s = Storage(VALDATE, STORAGESTART, storageEnd,
                curve=get_curve() if curve is None else curve,
                n_p=0, v_step=v_step, sVol=vol)
    n = len(s.date_span)
    active = np.ones(n)
    active[s._active:] = 0.0
    active[:s.Dt] = 0.0
    s.i_curve = active.copy()
    s.w_curve = np.zeros(n)
    s.set_volume_states(days)
    s.n_op_start = 0
    if ratch is not None:
        s.i_ratch[:] = ratch
    s.t_p_curve = np.full(s.n_op + 2, -1e9) if t_p_curve is None else t_p_curve
    if t_p_curve is None:
        s.t_p_curve[days] = 0.0
    if d_rate:
        s.d_curve = np.exp(-d_rate * np.arange(s.n_t) / 365.25)
    if eps:                                   # multiplicative bump of the daily curve
        pc = s.price_curve.copy()
        if months is None:
            pc = pc * (1 + eps)
        else:
            m = pc.index.to_period("M").astype(str).isin(months)
            pc[m] = pc[m] * (1 + eps)
        s.price_curve = pc
    s.n_p = n_p
    s.build()
    return s


def series(s):
    idx = pd.DatetimeIndex(s.date_span[:s.n_t])
    return (pd.Series(np.array(s.exp_ex[:s.n_t]), index=idx),
            pd.Series(np.array(s.delta[:s.n_t]),  index=idx),
            pd.Series(s.fwd, index=idx))


# -- sections -----------------------------------------------------------------

def repro():
    print("\n=== 1. Reproduction of the finding.md base case ===")
    s = put_swing()
    ex, dl, fwd = series(s)
    print(f"  V0            = {s.v[0, N_P, 0]:,.1f} EUR")
    print(f"  total exp_ex  = {ex.sum():,.1f} MWh   (quota = {-DAYS * V_STEP:,} MWh)")
    print(f"  total delta   = {dl.sum():,.1f} MWh")
    print(f"  flat_metric   = {daily_arithmetic_flat_metric(s):.4f} EUR/MWh")
    print(f"  Jan 30 -> 31  = {ex['2027-01-30']:.1f} -> {ex['2027-01-31']:.1f} MWh/day")
    print(f"  IDENTITY  sum(delta_i * F_i) = {float((dl * fwd).sum()):,.1f} "
          f"vs V0 = {s.v[0, N_P, 0]:,.1f}   (must match)")
    m = pd.DataFrame({"exp_ex": ex.resample("MS").sum(), "delta": dl.resample("MS").sum(),
                      "fwd_avg": fwd.resample("MS").mean()}).loc["2027"]
    m["ratio"]   = (m.delta / m.exp_ex).round(3)
    m["E_S_ex"]  = (m.fwd_avg * m.delta / m.exp_ex).round(2)
    print(m.round(2).to_string())


def np_invariance():
    print("\n=== 2. Is the January step driven by the quota (days) or by the tree (n_p)? ===")
    rows = []
    for n_p, days in [(20, 30), (30, 30), (45, 30), (30, 20)]:
        ex, _, _ = series(put_swing(n_p=n_p, days=days))
        jan = ex.loc["2027-01-01":"2027-02-05"]
        step = jan.diff().abs().idxmax()
        rows.append(dict(n_p=n_p, days=days, step_date=step.date(),
                         day_of_window=(step - STORAGESTART).days,
                         before=round(ex[step - pd.Timedelta(days=1)], 1),
                         after=round(ex[step], 1)))
    print(pd.DataFrame(rows).to_string(index=False))
    print("  -> the step tracks days, not n_p.")


def convergence():
    print("\n=== 3. n_p convergence, and the per-MWh denominator that was fixed (finding 1) ===")
    s0 = put_swing(n_p=0)
    prof_eur, acq0 = s0.v[0, 0, 0], -np.sum(s0.delta)
    rows = []
    for n_p in [0, 5, 10, 15, 20, 30, 45, 60]:
        t = time.perf_counter()
        s = put_swing(n_p=n_p)
        el = time.perf_counter() - t
        v0, sd, se = s.v[0, n_p, 0], np.sum(s.delta), np.sum(s.exp_ex)
        rows.append(dict(n_p=n_p, v0=round(v0, 1),
                         EUR_per_MWh=round(v0 / se, 4),
                         old_v0_over_sum_delta=round(v0 / sd, 4),
                         extrinsic=round((v0 - prof_eur) / acq0, 4),
                         v_at_index_0=round(s.v[0, 0, 0], 6), secs=round(el, 2)))
    print(pd.DataFrame(rows).to_string(index=False))
    print("  v[0, 0, .] is exactly 0 for every n_p > 0 -- flat()/profiled() used to read it and")
    print("  return 0; price_per_mwh() now indexes n_p and divides by exercised volume.")
    print("  The old v0/sum(delta) metric rose with n_p (the contract looked worse with more")
    print("  optionality) while the true average purchase price falls: 23.80 -> 22.62.")


def delta_fd():
    print("\n=== 4. Finite-difference validation of the delta (5 bp bump) ===")
    s = put_swing()
    _, dl, fwd = series(s)
    per = dl.index.to_period("M").astype(str)
    rows = []
    for mth in sorted(set(per[(dl.index >= "2027-01-01") & (dl.index <= "2027-12-31")])):
        mask = per == mth
        analytic = float((dl[mask] * fwd[mask]).sum())

        def fd(e, mth=mth):
            return (put_swing(eps=+e, months=[mth]).v[0, N_P, 0]
                    - put_swing(eps=-e, months=[mth]).v[0, N_P, 0]) / (2 * e)

        f5, f1 = fd(0.0005), fd(0.01)
        rows.append(dict(month=mth, delta_MWh=round(float(dl[mask].sum()), 1),
                         analytic_EUR=round(analytic, 1),
                         fd_5bp=round(f5, 1), err_5bp_pct=round(100 * (f5 / analytic - 1), 2),
                         fd_1pct=round(f1, 1), err_1pct_pct=round(100 * (f1 / analytic - 1), 2)))
    print(pd.DataFrame(rows).to_string(index=False))
    print("  -> every bucket matches at 5 bp (delta is right); the 1 % column is convexity, not error.")


def bucket_convexity():
    print("\n=== 5. Convexity of the bucketed delta (eps scan) ===")
    s = put_swing()
    _, dl, fwd = series(s)
    per = dl.index.to_period("M").astype(str)
    for mth in ["2027-07", "2027-12"]:
        mask = per == mth
        analytic = float((dl[mask] * fwd[mask]).sum())
        out = []
        for e in [0.0005, 0.002, 0.01, 0.02, 0.05]:
            f = (put_swing(eps=+e, months=[mth]).v[0, N_P, 0]
                 - put_swing(eps=-e, months=[mth]).v[0, N_P, 0]) / (2 * e)
            out.append(f"eps={e:<7}{100 * (f / analytic - 1):+8.2f}%")
        print(f"  {mth} (analytic {analytic:,.0f} EUR): " + "  ".join(out))


def soft_penalty():
    print("\n=== 6. Soft terminal penalty: what multiple actually forces full exercise? ===")
    fm = daily_arithmetic_flat_metric(put_swing(n_p=0))
    rows = []
    for mult in [1.0, 1.19, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0]:
        tp = np.full(DAYS + 3, -1e9)
        tp[:DAYS + 1] = -(DAYS - np.arange(DAYS + 1)) * (fm * V_STEP) * mult
        ex, dl, _ = series(put_swing(t_p_curve=tp))
        rows.append(dict(mult=mult, penalty_per_clip=round(fm * mult, 2),
                         total_exp_ex=round(ex.sum(), 0),
                         dec_delta=round(dl.loc["2027-12"].sum(), 0)))
    print(pd.DataFrame(rows).to_string(index=False))
    s = put_swing(n_p=0)
    dx = VOL * np.sqrt(3 / 365.25)
    mx = s.price_curve.iloc[s.Dt:s._active].max()
    print(f"  max forward in window = {mx:.2f} ({mx / fm:.2f}x flat); "
          f"max spot reachable in the tree ~ {mx * np.exp(N_P * dx):.0f} "
          f"({mx * np.exp(N_P * dx) / fm:.2f}x flat)")
    print("  -> full exercise appears at ~4x, i.e. at the tail of the SPOT distribution,")
    print("     not at max(forward) = 1.19x.")


def curve_fit():
    print("\n=== 7. Does smoothen_curve reprice the input monthly contracts? (finding 5, OPEN - batch 3) ===")
    s = Storage(VALDATE, STORAGESTART, STORAGEEND, curve=get_curve(), n_p=0, sVol=VOL)
    stepped = map_curve_to_dates(s.date_span, get_curve())
    base = stepped.resample("ME").mean()
    for alpha, sm in [(1.2, s.price_curve), (1.0, smoothen_curve(stepped, alpha=1.0))]:
        err = sm.resample("ME").mean() - base
        print(f"  alpha={alpha}:  max |err| = {err.abs().max():.4f} EUR/MWh   "
              f"mean |err| = {err.abs().mean():.4f}   "
              f"max = {1e4 * (err / base).abs().max():.0f} bp")
    err = (s.price_curve.resample("ME").mean() - base).loc["2027"]
    print("  2027 monthly errors (EUR/MWh):", np.round(err.values, 3).tolist())


def tree():
    print("\n=== 8. Tree diagnostics: vol term structure now validated, not silently NaN (finding 6, fixed) ===")
    s = Storage(VALDATE, STORAGESTART, STORAGEEND, curve=get_curve(), n_p=0, sVol=VOL)
    for lbl, volc in [("flat 0.50", [VOL] * s.n_t),
                      ("0.70 -> 0.90 ramp", list(np.linspace(0.70, 0.90, s.n_t))),
                      ("0.30 -> 0.90 ramp", list(np.linspace(0.30, 0.90, s.n_t)))]:
        try:
            fwd, x, q, p_u, p_m, p_d = build_tree(s.price_curve, s.n_t, N_P, volc, [1.0] * s.n_t)
        except ValueError as exc:
            print(f"  {lbl:18s} rejected: {str(exc).split('.')[0]}.")
            print(f"  {'':18s}   (before the fix: 28464 negative-probability nodes and NaNs, "
                  f"returned silently)")
            continue
        neg = int(((p_u < -1e-12) | (p_m < -1e-12) | (p_d < -1e-12)).sum())
        var = np.array([np.dot(q[i], x[i] ** 2) - np.dot(q[i], x[i]) ** 2 for i in range(s.n_t)])
        fit = np.abs(np.array([np.dot(q[i], np.exp(x[i])) for i in range(N_P, s.n_t)])
                     / fwd[N_P:] - 1).max()
        print(f"  {lbl:18s} negative-prob nodes = {neg:6d}   min p_m = {p_m[N_P:].min():8.4f}   "
              f"terminal log-std = {np.sqrt(var[-1]):.4f} (OU target {volc[-1] / np.sqrt(2):.4f})   "
              f"fwd-fit err = {fit:.1e}")
    fwd, x, q, *_ = build_tree(s.price_curve, s.n_t, N_P, [VOL] * s.n_t, [1.0] * s.n_t)
    print(f"  edge mass q[0]+q[2n_p] at end of grid = {q[-1, 0] + q[-1, 2 * N_P]:.2e}  "
          f"(boundary truncation immaterial here)")


def ratchets():
    print("\n=== 9. Fractional ratchets: were read three ways, now rejected (finding 3, fixed) ===")
    rows = []
    for r in [1.0, 1.5, 0.5, 2.0]:
        try:
            s = put_swing(n_p=10, ratch=r)
        except ValueError as exc:
            rows.append(dict(i_ratch=r, V0="rejected", reported_MWh=str(exc)[:60] + "...",
                             E_terminal_clips="", identity_gap=""))
            continue
        ex, dl, fwd = series(s)
        v0 = s.v[0, 10, 0]
        term = s.prob[s.n_t - 1].sum(0)
        rows.append(dict(i_ratch=r, V0=round(v0, 1), reported_MWh=round(-ex.sum(), 0),
                         E_terminal_clips=round(float(np.dot(np.arange(s.n_op), term)), 2),
                         identity_gap=round(float((dl * fwd).sum()) - v0, 1)))
    print(pd.DataFrame(rows).to_string(index=False))
    print("  Was: run_model truncated, probabilities rounded, compute_all_metrics kept the float,")
    print("       so i_ratch=1.5 priced a ratchet of 1, moved 2 states and reported 1.5.")
    print("  Now: one whole-clip convention everywhere, non-integer ratchets rejected up front.")


def infeasible():
    print("\n=== 10. Infeasible quota: returned the -1e9 sentinel as a price, now raises (finding 4, fixed) ===")
    for days, end in [(30, STORAGEEND), (365, STORAGEEND), (400, STORAGEEND),
                      (100, pd.Timestamp("2027-03-31"))]:
        window = (pd.Timestamp(end) - STORAGESTART).days + 1
        try:
            s = put_swing(n_p=5, days=days, storageEnd=end)
        except ValueError as exc:
            print(f"  days={days:<4} window={window:<4} rejected: {str(exc).split('.')[0]}.")
            continue
        print(f"  days={days:<4} window={window:<4} V0={s.v[0, 5, 0]:>18,.1f}   "
              f"reported EUR/MWh = {s.price_per_mwh():,.2f}")
    print("  -> days=365 on a 365-day window prices at exactly the flat price;")
    print("     days > window used to return -1e9 as a price, now raises.")


def discounting():
    print("\n=== 11. delta ignores d_curve (finding 9, OPEN - needs a convention decision) ===")
    for r in [0.0, 0.03]:
        s = put_swing(n_p=10, d_rate=r)
        _, dl, fwd = series(s)
        v0 = s.v[0, 10, 0]
        rep = float((dl * fwd).sum())
        print(f"  d_curve rate={r:.0%}:  V0={v0:>14,.1f}   sum(delta*F)={rep:>14,.1f}   "
              f"gap={rep - v0:>12,.1f} ({100 * (rep / v0 - 1):+.2f}%)")


SECTIONS = dict(repro=repro, np_invariance=np_invariance, convergence=convergence,
                delta_fd=delta_fd, bucket_convexity=bucket_convexity,
                soft_penalty=soft_penalty, curve_fit=curve_fit, tree=tree,
                ratchets=ratchets, infeasible=infeasible, discounting=discounting)

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(SECTIONS)):
        SECTIONS[name]()
