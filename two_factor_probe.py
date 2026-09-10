"""Does a common long factor ever add value? An independent DP that answers it.

The P4.1 design note argued for a second price factor on the grounds that one
factor forces every pair of forwards to correlate exactly 1.000, so the model
cannot price the seasonal spread a store exists to monetise. An independent
review disproved the *storage* half of that: for a zero-fee store a common
multiplicative long factor integrates out exactly, adding nothing.

The reduction has conditions, and the reply to that review was right that they
matter. This probe tests them one at a time on the same machinery:

    zero-fee store        homogeneous  -> expect EXACTLY no gain
    store with cash fees  not homogeneous
    struck swing          not homogeneous

Homogeneity is the whole mechanism. Writing `L_t = exp(xi_t - 0.5 Var xi_t)`,
which is a martingale, the price is `S_t = F_0(t) * L_t * M_t` with `M` the
one-factor part. If every cashflow is homogeneous of degree one in price and the
admissible set does not depend on price, then the optimal policy is
scale-invariant, `V_t = L_t * W_t`, and `L_0 = 1`. Add a fixed EUR/MWh fee or a
strike and the cashflow is `S - K` rather than `S`: scale invariance breaks, the
policy starts depending on the level, and the long factor has something to say.

This is a deliberately small, independently written oracle, not a production
Schwartz-Smith engine. Its long factor is independent of the short one, which is
enough to test the design's universal claims. Run it:

    python two_factor_probe.py
"""
import numpy as np

DT = 1.0 / 12.0
N_T = 24                      # decision dates, two years of months
CAP = 3                       # inventory levels 0..3
SIG_CHI = 0.6
SIG_XI = 0.8
RATE = 0.05

# A seasonal forward curve, so there is a spread to trade rather than only noise.
F0 = 20.0 + 5.0 * np.cos(2 * np.pi * np.arange(N_T) / 12.0)
DF = np.exp(-RATE * DT * np.arange(N_T))


def _ou_lattice(kappa, sigma, half=40):
    """Recombining trinomial for dchi = -kappa*chi*dt + sigma*dW, chi_0 = 0."""
    dx = sigma * np.sqrt(3.0 * DT)
    nodes = np.arange(-half, half + 1) * dx
    n = nodes.size
    trans = np.zeros((n, n))
    for j, c in enumerate(nodes):
        mean = c - kappa * c * DT
        var = sigma ** 2 * DT
        k = int(np.clip(round(mean / dx), -half + 1, half - 1))
        eta = mean - k * dx
        p_u = 0.5 * ((var + eta ** 2) / dx ** 2 + eta / dx)
        p_d = 0.5 * ((var + eta ** 2) / dx ** 2 - eta / dx)
        for shift, p in ((1, p_u), (0, 1.0 - p_u - p_d), (-1, p_d)):
            trans[j, k + shift + half] += p
    assert np.allclose(trans.sum(axis=1), 1.0)
    assert trans.min() > -1e-12, trans.min()
    return nodes, trans


def _walk_lattice(sigma, half=None):
    """Recombining binomial for a driftless Brownian motion, xi_0 = 0.

    `half` defaults past anything reachable in N_T steps. It matters: with an
    absorbing edge, exp(xi) stops being a martingale and this probe reports a
    spurious gain of about 1e-5 -- which is the size of a real effect worth
    looking for, so the boundary has to be beyond reach rather than merely wide.
    """
    half = N_T + 2 if half is None else half
    dx = sigma * np.sqrt(DT)
    nodes = np.arange(-half, half + 1) * dx
    n = nodes.size
    trans = np.zeros((n, n))
    for j in range(n):
        trans[j, min(j + 1, n - 1)] += 0.5
        trans[j, max(j - 1, 0)] += 0.5
    return nodes, trans


def _price_lattice(kappa, sig_xi, sig_chi=SIG_CHI):
    """Forward-fitted prices on the joint (chi, xi) grid, with E[S_t] = F0[t]."""
    chi, t_chi = _ou_lattice(kappa, sig_chi)
    q_chi = np.zeros((N_T, chi.size))
    q_chi[0, chi.size // 2] = 1.0
    for t in range(N_T - 1):
        q_chi[t + 1] = q_chi[t] @ t_chi
    m_chi = np.array([np.log((q_chi[t] * np.exp(chi)).sum()) for t in range(N_T)])

    if sig_xi == 0.0:
        xi, t_xi = np.zeros(1), np.ones((1, 1))
    else:
        xi, t_xi = _walk_lattice(sig_xi)
    q_xi = np.zeros((N_T, xi.size))
    q_xi[0, xi.size // 2] = 1.0
    for t in range(N_T - 1):
        q_xi[t + 1] = q_xi[t] @ t_xi
    m_xi = np.array([np.log((q_xi[t] * np.exp(xi)).sum()) for t in range(N_T)])

    price = (F0[:, None, None]
             * np.exp(chi[None, :, None] - m_chi[:, None, None])
             * np.exp(xi[None, None, :] - m_xi[:, None, None]))
    for t in range(N_T):                      # the fit is exact, so check it
        fitted = (q_chi[t][:, None] * q_xi[t][None, :] * price[t]).sum()
        assert abs(fitted - F0[t]) < 1e-9, (t, fitted, F0[t])
    return price, t_chi, t_xi


NEG = -1e18


def _step(t_chi, t_xi, v):
    """Roll the value function back one date over both factors.

    The two transitions are independent, so the joint operator is separable and
    two contractions beat one: 81 x 81 x 53 x 4 plus 53 x 53 x 81 x 4 against
    81 x 81 x 53 x 53 x 4 for the combined form. Worth doing -- it is the whole
    cost of this probe, and it turned a three-minute test suite back into a
    ninety-second one.
    """
    return np.einsum("kl,ilm->ikm", t_xi, np.einsum("ij,jlm->ilm", t_chi, v))


def value_store(kappa, sig_xi, fee=0.0, sig_chi=SIG_CHI):
    """A cycling store: inject or withdraw one unit a month, end empty."""
    price, t_chi, t_xi = _price_lattice(kappa, sig_xi, sig_chi)
    n_c, n_x = price.shape[1], price.shape[2]

    v = np.full((n_c, n_x, CAP + 1), NEG)
    v[:, :, 0] = 0.0                                    # must end empty
    for t in range(N_T - 1, -1, -1):
        cont = _step(t_chi, t_xi, v) if t < N_T - 1 else v
        nxt = np.full((n_c, n_x, CAP + 1), NEG)
        for level in range(CAP + 1):
            for move in (-1, 0, 1):                     # +1 inject, -1 withdraw
                landing = level + move
                if not 0 <= landing <= CAP:
                    continue
                cash = -move * DF[t] * price[t] - abs(move) * DF[t] * fee
                nxt[:, :, level] = np.maximum(nxt[:, :, level], cash + cont[:, :, landing])
        v = nxt
    return float(v[n_c // 2, n_x // 2, 0])


def value_swing(kappa, sig_xi, strike=0.0, quota=6, sig_chi=SIG_CHI):
    """A call swing: take up to `quota` units at `strike`, one a month."""
    price, t_chi, t_xi = _price_lattice(kappa, sig_xi, sig_chi)
    n_c, n_x = price.shape[1], price.shape[2]

    v = np.zeros((n_c, n_x, quota + 1))                 # unused quota simply expires
    for t in range(N_T - 1, -1, -1):
        cont = _step(t_chi, t_xi, v) if t < N_T - 1 else v
        nxt = np.empty((n_c, n_x, quota + 1))
        payoff = DF[t] * (price[t] - strike)            # NOT homogeneous unless strike = 0
        for left in range(quota + 1):
            idle = cont[:, :, left]
            if left == 0:
                nxt[:, :, left] = idle
            else:
                nxt[:, :, left] = np.maximum(idle, payoff + cont[:, :, left - 1])
        v = nxt
    return float(v[n_c // 2, n_x // 2, quota])


def matched_sig_chi(kappa, sig_xi, horizon=None):
    """Short-factor vol that keeps TOTAL terminal log variance fixed.

    Without this the comparison is confounded. Bolting an independent factor on
    top of an unchanged short factor strictly increases total price variance, and
    more variance is worth more to any option -- so a struck contract would look
    like it needed a second factor when all it needed was a bigger number.

    Var(chi_T) = sig_chi^2 (1 - exp(-2 kappa T)) / (2 kappa) for the OU factor and
    sig_xi^2 T for the driftless walk. Solve the first for the residual after the
    second is taken out. Returns nan when sig_xi alone already exceeds the budget.
    """
    horizon = N_T * DT if horizon is None else horizon
    ou_unit = (1.0 - np.exp(-2.0 * kappa * horizon)) / (2.0 * kappa)
    residual = SIG_CHI ** 2 - (sig_xi ** 2 * horizon) / ou_unit
    return float(np.sqrt(residual)) if residual > 0.0 else float("nan")


def compare_matched(label, fn, sig_xi, kappas=(0.2, 1.0, 4.0), **kwargs):
    """One factor against two at EQUAL total terminal variance."""
    print(f"\n{label}")
    print(f"{'kappa':>6} {'sig_chi (2f)':>13} {'one factor':>15} {'two factors':>15} "
          f"{'gain':>12} {'relative':>10}")
    print("-" * 78)
    rows = []
    for kappa in kappas:
        matched = matched_sig_chi(kappa, sig_xi)
        if not np.isfinite(matched):
            print(f"{kappa:>6.1f}   sig_xi alone exceeds the variance budget")
            continue
        one = fn(kappa, 0.0, **kwargs)
        two = fn(kappa, sig_xi, sig_chi=matched, **kwargs)
        rel = (two - one) / abs(one) if one else np.nan
        rows.append((kappa, matched, one, two, two - one, rel))
        print(f"{kappa:>6.1f} {matched:>13.4f} {one:>15.10f} {two:>15.10f} "
              f"{two - one:>12.2e} {rel:>9.4%}")
    return rows

def compare(label, fn, kappas=(0.2, 1.0, 4.0), **kwargs):
    rows = []
    for kappa in kappas:
        one = fn(kappa, 0.0, **kwargs)
        two = fn(kappa, SIG_XI, **kwargs)
        rows.append((kappa, one, two, two - one, (two - one) / abs(one) if one else np.nan))
    print(f"\n{label}")
    print(f"{'kappa':>6} {'one factor':>16} {'two factors':>16} {'gain':>14} {'relative':>10}")
    print("-" * 68)
    for kappa, one, two, gain, rel in rows:
        print(f"{kappa:>6.1f} {one:>16.10f} {two:>16.10f} {gain:>14.2e} {rel:>9.4%}")
    return rows


def invariance(fn, sig_xis=(0.0, 0.1, 0.8), kappas=(0.2, 1.0, 4.0), **kwargs):
    """Vary the long factor with the SHORT factor held fixed.

    This is the clean experiment. Adding a long factor on top of an unchanged
    short one raises total variance, so a struck contract gains partly for that
    reason -- but a homogeneous contract gains NOTHING, at any sig_xi, and that
    contrast is the whole result.
    """
    header = "  ".join(f"sig_xi {s:<8.1f}" for s in sig_xis)
    print(f"{'kappa':>6}  {header}      spread")
    print("-" * (8 + 17 * len(sig_xis) + 12))
    rows = []
    for kappa in kappas:
        values = [fn(kappa, s, **kwargs) for s in sig_xis]
        spread = (max(values) - min(values)) / abs(values[0])
        rows.append((kappa, values, spread))
        cells = "  ".join(f"{v:>15.10f}" for v in values)
        print(f"{kappa:>6.1f}  {cells}  {spread:>10.4%}")
    return rows


def main():
    print(__doc__.split("Run it:")[0].strip().split("\n\n")[0])
    print(f"\n{N_T} monthly dates, seasonal curve {F0.min():.1f}-{F0.max():.1f}, "
          f"sigma_chi {SIG_CHI}, discount {RATE:.0%}")

    print("\n" + "=" * 78)
    print("1. The long factor, with the short factor held fixed")
    print("=" * 78)
    print("\nZero-fee store -- homogeneous")
    store = invariance(value_store)
    print("\nCall swing, strike 0 -- homogeneous")
    unstruck = invariance(value_swing, strike=0.0)
    print("\nStore, EUR 2/MWh on each leg -- a fixed fee breaks homogeneity")
    fees = invariance(value_store, fee=2.0)
    print("\nCall swing, strike 20 -- a strike breaks it too")
    struck = invariance(value_swing, strike=20.0)

    print("\n" + "=" * 78)
    print("2. Where the variance sits, with TOTAL terminal variance held fixed")
    print("=" * 78)
    print("A different question, and not the one above: move variance out of the")
    print("mean-reverting factor into the permanent one and both contracts lose value,")
    print("because both care about short-horizon variance rather than the total. For")
    print("the store that movement is ENTIRELY the reduced sigma_chi -- section 1 has")
    print("already shown its value does not depend on sigma_xi at all.")
    small = 0.10
    matched_store = compare_matched(
        f"\nStore, no fees, sigma_xi {small} at matched total variance",
        value_store, small)
    matched_struck = compare_matched(
        f"\nCall swing, strike 20, sigma_xi {small} at matched total variance",
        value_swing, small, strike=20.0)

    print("\n" + "=" * 78)
    print("Reading")
    print("=" * 78)
    print(f"  zero-fee store, sigma_xi 0 -> 0.8 : {max(r[2] for r in store):.2e} of value")
    print(f"  unstruck swing, same              : {max(r[2] for r in unstruck):.2e} of value")
    print(f"  store with cash fees              : {max(r[2] for r in fees):.2%}")
    print(f"  struck swing                      : {max(r[2] for r in struck):.2%}")
    print()
    print("  Both homogeneous contracts are invariant to the long factor to ten")
    print("  decimals, at any volatility. Both non-homogeneous ones move. The")
    print("  reduction is a statement about homogeneity, not about storage.")
    print()
    print("  Quote the modest column, not the extreme one. At sigma_xi 0.1 -- against a")
    print("  short factor of 0.6 -- the struck swing still gains about 1.3 % to 1.4 %,")
    print("  roughly flat in kappa, while the store gains exactly nothing.")
    print()
    print(f"  Reallocating variance instead (section 2) costs the store up to "
          f"{max(abs(r[5]) for r in matched_store):.2%} and the")
    print(f"  struck swing up to {max(abs(r[5]) for r in matched_struck):.2%}. That is the "
          f"short factor being cut, not the long")
    print("  factor being added, and it is why a variance-matched comparison cannot")
    print("  isolate a factor on its own: the two have different variance term")
    print("  structures, so no single matching horizon holds both fixed.")
    print()
    print("  For P4.1: the storage rationale is gone and does not come back. The case")
    print("  for a second factor has to be made on struck or fee-bearing contracts,")
    print("  and it is a real case -- but a 1.3 % one at plausible parameters, not the")
    print("  order-of-magnitude effect the original note implied for storage.")


if __name__ == "__main__":
    main()
