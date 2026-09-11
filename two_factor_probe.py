"""What is a second price factor actually worth? An independent DP that asks it.

The P4.1 design note argued for a second factor because one factor forces every
pair of forwards to correlate exactly 1.000, so the model cannot price the
seasonal spread a store exists to monetise. An independent review disproved the
*storage* half of that: for a zero-fee store a common multiplicative long factor
integrates out exactly.

That exact result stands (section 1). But it is stated at a FIXED short factor,
and on its own it answers the wrong question. Bolting an independent factor on
top of an unchanged `sigma_chi` strictly increases total volatility, and more
volatility is worth more to any option -- so a struck contract looks like it
needs a second factor when all it was given was a bigger number. **A jointly
calibrated two-factor model fits the same observed volatility with both factors,
so `sigma_chi` comes down.** Section 2 does that instead.

"Calibrated" here is narrower than it sounds: mean reversion is FIXED (not
jointly estimated) and the two factors are held INDEPENDENT, then sigma_chi is
solved algebraically to hold one scalar variance anchor at its one-factor
value. That is a variance-ALLOCATION scenario, not a fit to any market
observation, and the direction it moves sigma_chi is conditional on those two
choices -- introduce a negative short/long correlation and the algebra can
require sigma_chi to go UP instead of down for the same target variance (see
docs/PROJECT-REVIEW-2026-09-10-evening.md finding 3 for a worked
counterexample). Treat every percentage below as "what this scalar scenario
gives", not as "what an estimated two-factor model would price a store at."

It changes the answer, including its sign, and it changes the answer for STORAGE
too -- which section 1 alone would tell you is unaffected. The reply to the
review put it exactly right: a joint calibration can move the short-factor
estimate even where valuation later reduces to one state, so market-model
selection and valuation-state reduction are separate decisions.

    contract               homogeneous   value depends only on chi
    zero-fee store         yes           yes -- exactly
    unstruck swing         yes           yes -- exactly
    store with cash fees   no            no
    struck swing           no            no

This is a deliberately small, independently written oracle, not a production
Schwartz-Smith engine. Its long factor is independent of the short one, which is
enough for the questions above. Run it:

    python two_factor_probe.py

2026-09-11: the short factor's own lattice (`_ou_lattice`) was rebuilt on the
OU process's EXACT one-step moments in place of an Euler discretisation, and
`matched_sig_chi`'s "terminal" anchor was corrected to the actual last decision
date rather than one step past it -- IMPLEMENTATION-GUIDE-2026-09-11.md §8.
Together these had produced an 8.95 % gap between what the anchor matched and
what the lattice actually propagated; the gap is now zero to machine precision.
The qualitative story below is unchanged -- a calibrated second factor still
costs a store value, and the struck swing's answer still flips sign between
anchors -- only the specific percentages moved, generally smaller than before.
"""
import numpy as np

DT = 1.0 / 12.0
N_T = 24                      # decision dates, two years of months
CAP = 3                       # inventory levels 0..3
SIG_CHI = 0.6                 # the one-factor model's short-factor volatility
RATE = 0.05

# The ACTUAL decision dates: 0, DT, ..., (N_T-1)*DT. Until 2026-09-11 the
# "terminal" anchor below defaulted to N_T*DT = 2.0y -- one step PAST the last
# decision at (N_T-1)*DT = 23/12 = 1.9167y -- and matched a variance nothing in
# this probe is ever actually valued at. IMPLEMENTATION-GUIDE-2026-09-11.md
# §8.2. Everything that indexes a decision date already used range(N_T)
# correctly (F0, DF); only the anchor's own horizon was wrong.
DECISION_TIMES = np.arange(N_T) * DT

# A seasonal forward curve, so there is a spread to trade rather than only noise.
F0 = 20.0 + 5.0 * np.cos(2 * np.pi * np.arange(N_T) / 12.0)
DF = np.exp(-RATE * DECISION_TIMES)


# -- Lattices -----------------------------------------------------------------

def _ou_lattice(kappa, sigma, half=40):
    """Recombining trinomial for dchi = -kappa*chi*dt + sigma*dW, chi_0 = 0.

    Built from the EXACT one-step conditional moments over an interval `DT`,
    not an Euler discretisation:

        a    = exp(-kappa*DT)                                  -- mean(t+DT) = a * chi(t)
        var  = sigma^2 * (1 - exp(-2*kappa*DT)) / (2*kappa)     [kappa > 0]
             = sigma^2 * DT                                     [kappa -> 0, the limit]

    Until 2026-09-11 this used Euler's `(1 - kappa*DT)*c` and `sigma^2*DT`
    instead -- first-order approximations of the two exact expressions above,
    good only as `kappa*DT -> 0`. At kappa=4, DT=1/12 (kappa*DT = 1/3, not
    small), Euler's mean-reversion factor is 0.6667 against the exact 0.7165:
    a 7 % per-step error that compounds over every one of the N_T-1 steps,
    not just the last -- which is why matching only the FINAL variance to a
    closed-form target (the probe's original "immediate correction" option)
    would still have left every earlier transition biased. Built once,
    correctly, instead. `1 - exp(-2*kappa*DT)` is computed as `-expm1(...)`
    to avoid cancellation at small `kappa*DT`.

    **`dx` is derived from this same exact variance, not from `sigma^2*DT`.**
    Mean reversion makes the exact one-step variance SMALLER than the Euler
    proxy (`(1-exp(-2*kappa*DT))/(2*kappa) < DT` for any kappa > 0), so a `dx`
    sized off `sigma*sqrt(3*DT)` becomes too wide for the trinomial's own
    stability condition (`dx <= sqrt(3*var)`) once kappa is large enough --
    confirmed by construction: at kappa=4 it produced probabilities as
    negative as -0.0033, not a rounding-noise sliver. This is not a boundary
    or a half-width problem, so widening `half` would not have fixed it; `dx`
    itself has to track the variance it is meant to discretise.
    IMPLEMENTATION-GUIDE-2026-09-11.md §8.3.
    """
    a = np.exp(-kappa * DT)
    var = (sigma ** 2 * (-np.expm1(-2.0 * kappa * DT)) / (2.0 * kappa)
          if kappa > 0.0 else sigma ** 2 * DT)
    dx = np.sqrt(3.0 * var)
    nodes = np.arange(-half, half + 1) * dx
    n = nodes.size

    trans = np.zeros((n, n))
    for j, c in enumerate(nodes):
        mean = a * c
        raw_k = round(mean / dx)
        k = int(np.clip(raw_k, -half + 1, half - 1))
        if k != raw_k:
            # CLAMPED: the unrestricted target lies at or past the edge (only
            # possible for a `j` already at or adjacent to the boundary, which
            # is unreachable by forward propagation for a `half` this
            # generous -- confirmed separately, not assumed). The usual
            # eta-based split assumes |eta| is a small fraction of dx; here
            # clamping can make it a FULL dx or more (seen concretely at
            # kappa=0, j=0: eta = -dx exactly), which drives one of p_u/p_d/p_m
            # negative. A deterministic jump to the clamped node sums to
            # exactly 1 by construction, so it needs no renormalisation and
            # cannot be negative -- and it changes nothing anyone ever
            # computes, because this row is never visited.
            trans[j, k + half] = 1.0
            continue
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
    spurious gain of about 1e-5 -- the size of a real effect worth looking for,
    so the boundary has to be beyond reach rather than merely wide.
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

    The transitions are independent, so the joint operator is separable and two
    contractions beat one: 81 x 81 x 53 x 4 plus 53 x 53 x 81 x 4 against
    81 x 81 x 53 x 53 x 4 combined. It is the whole cost of this probe.
    """
    return np.einsum("kl,ilm->ikm", t_xi, np.einsum("ij,jlm->ilm", t_chi, v))


# -- Contracts ----------------------------------------------------------------

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
            nxt[:, :, left] = (idle if left == 0
                               else np.maximum(idle, payoff + cont[:, :, left - 1]))
        v = nxt
    return float(v[n_c // 2, n_x // 2, quota])


CONTRACTS = (
    ("zero-fee store", value_store, {}, True),
    ("unstruck swing", value_swing, dict(strike=0.0), True),
    ("store, EUR 2/MWh legs", value_store, dict(fee=2.0), False),
    ("call swing, strike 20", value_swing, dict(strike=20.0), False),
)


# -- Calibration anchors ------------------------------------------------------

def matched_sig_chi(sig_xi, anchor="spot", kappa=None, horizon=None):
    """The short-factor volatility a two-factor FIT would use, given the anchor.

    A second factor is not free volatility. Which observable the two models are
    made to agree on is a CHOICE, and the choice changes the answer -- including
    its sign, which is why more than one is reported.

    * ``"none"``     -- leave `sigma_chi` alone. Adds variance as well as a
      factor, so it answers a question nobody asked. Kept because it is what the
      first version of this probe did, and the mistake is worth naming.
    * ``"spot"``     -- hold the instantaneous variance of `d log S` fixed:
      `sigma_chi^2 + sigma_xi^2` for independent factors. Horizon-free and
      kappa-free, and what fitting both factors to one observed spot volatility
      would give.
    * ``"terminal"`` -- hold `Var(log S_T)` fixed at `horizon`, which defaults
      to `DECISION_TIMES[-1]` -- the ACTUAL last decision date, `(N_T-1)*DT`,
      not `N_T*DT` (one step past it; the pre-2026-09-11 default, since
      corrected). Depends on both kappa and the horizon, because the OU
      factor's variance saturates at `sigma^2 / (2 kappa)` while the walk's
      grows linearly.

    Returns nan when the long factor alone already exceeds the budget.
    """
    if anchor == "none":
        return SIG_CHI
    if anchor == "spot":
        residual = SIG_CHI ** 2 - sig_xi ** 2
    elif anchor == "terminal":
        horizon = DECISION_TIMES[-1] if horizon is None else horizon
        ou_unit = (1.0 - np.exp(-2.0 * kappa * horizon)) / (2.0 * kappa)
        residual = SIG_CHI ** 2 - (sig_xi ** 2 * horizon) / ou_unit
    else:
        raise ValueError(f"unknown anchor {anchor!r}")
    return float(np.sqrt(residual)) if residual > 0.0 else float("nan")


# -- Reports ------------------------------------------------------------------

def invariance(fn, sig_xis=(0.0, 0.1, 0.8), kappas=(0.2, 1.0, 4.0), **kwargs):
    """Vary the long factor with the short factor held fixed (anchor "none")."""
    rows = []
    for kappa in kappas:
        values = [fn(kappa, s, **kwargs) for s in sig_xis]
        rows.append((kappa, values, (max(values) - min(values)) / abs(values[0])))
    return rows


def anchored(fn, anchor, sig_xis=(0.0, 0.1, 0.3, 0.5), kappas=(0.2, 1.0, 4.0), **kwargs):
    """Vary the long factor with `sigma_chi` cut to hold the anchor fixed."""
    rows = []
    for kappa in kappas:
        values, chis = [], []
        for sig_xi in sig_xis:
            chi = matched_sig_chi(sig_xi, anchor, kappa=kappa)
            chis.append(chi)
            values.append(float("nan") if not np.isfinite(chi)
                          else fn(kappa, sig_xi, sig_chi=chi, **kwargs))
        rows.append((kappa, chis, values, (values[-1] - values[0]) / abs(values[0])))
    return rows


def _print_anchored(title, anchor, sig_xis=(0.0, 0.1, 0.3, 0.5)):
    print(f"\n{title}")
    header = " ".join(f"{'xi ' + format(s, '.2f'):>14}" for s in sig_xis)
    print(f"{'contract':>23} {'kappa':>6} {header} {'0 -> ' + format(sig_xis[-1], '.1f'):>11}")
    print("-" * (32 + 15 * len(sig_xis) + 10))
    out = {}
    for label, fn, kwargs, _ in CONTRACTS:
        rows = anchored(fn, anchor, sig_xis=sig_xis, **kwargs)
        out[label] = rows
        for index, (kappa, _chis, values, rel) in enumerate(rows):
            name = label if index == 0 else ""
            cells = " ".join(f"{v:>14.8f}" for v in values)
            print(f"{name:>23} {kappa:>6.1f} {cells} {rel:>10.2%}")
    return out


def main():
    print(__doc__.split("Run it:")[0].strip().split("\n\n")[0])
    print(f"\n{N_T} monthly dates, seasonal curve {F0.min():.1f}-{F0.max():.1f}, "
          f"one-factor sigma_chi {SIG_CHI}, discount {RATE:.0%}")

    print("\n" + "=" * 92)
    print("1. The exact result: at a FIXED short factor, a homogeneous contract does")
    print("   not depend on the long factor at all")
    print("=" * 92)
    print(f"{'contract':>23} {'homogeneous':>12}  {'spread over sigma_xi 0 .. 0.8':>31}")
    print("-" * 70)
    for label, fn, kwargs, homogeneous in CONTRACTS:
        rows = invariance(fn, **kwargs)
        print(f"{label:>23} {str(homogeneous):>12}  {max(r[2] for r in rows):>30.2e}")
    print("\n  Zero to machine precision for the homogeneous pair, at ANY long-factor")
    print("  volatility. That is a theorem rather than a measurement: with L a")
    print("  martingale and cashflows homogeneous of degree one, V_t = L_t * W_t and")
    print("  L_0 = 1.")
    print("\n  It is stated at a fixed sigma_chi, and on its own it misleads.")

    print("\n" + "=" * 92)
    print("2. The fair comparison: a variance-allocation scenario that takes volatility")
    print("   out of the short factor rather than adding it on top -- at FIXED kappa and")
    print("   INDEPENDENT factors, not a fit to any market observation. Correlation could")
    print("   move sigma_chi the other way; see PROJECT-REVIEW-2026-09-10-evening.md #3")
    print("=" * 92)
    _print_anchored(
        f"Anchor: instantaneous spot variance, sigma_chi^2 + sigma_xi^2 held at "
        f"{SIG_CHI ** 2:.2f}", "spot")
    # The terminal anchor runs out of budget far sooner: at kappa 4 the OU
    # factor's variance has already saturated, so the walk eats it quickly.
    _print_anchored(
        f"Anchor: Var(log S_T) held fixed at T = {DECISION_TIMES[-1]:.4f}y "
        f"(the LAST DECISION DATE -- a tighter budget, see sigma_chi below)",
        "terminal", sig_xis=(0.0, 0.05, 0.10, 0.14))

    print("\n" + "=" * 92)
    print("Reading")
    print("=" * 92)
    # Same contract, same long-factor volatility, two defensible anchors.
    common = (0.0, 0.10)
    print(f"  At sigma_xi {common[1]:.2f} and kappa 4 -- one long factor, two anchors:")
    print(f"  {'':<23}{'spot anchor':>14}{'terminal anchor':>18}")
    for label, fn, kwargs, _ in CONTRACTS:
        by_spot = anchored(fn, "spot", sig_xis=common, kappas=(4.0,), **kwargs)[0][3]
        by_term = anchored(fn, "terminal", sig_xis=common, kappas=(4.0,), **kwargs)[0][3]
        print(f"  {label:<23}{by_spot:>+13.2%}{by_term:>+17.2%}")
    print()
    print("  A calibrated second factor COSTS a store value, under BOTH anchors and at")
    print("  every kappa above 0.2: -4.97 % at sigma_xi 0.5 on the spot anchor, -5.55 % at")
    print("  sigma_xi 0.14 on the terminal one (kappa 4 throughout). A store monetises")
    print("  short-horizon variance and the long factor is where that variance went.")
    print("  Section 1 on its own would tell you storage is unaffected. It is not -- the")
    print("  channel is the short-factor estimate, which is exactly why market-model")
    print("  selection and valuation-state reduction are separate decisions.")
    print()
    print("  The effect grows with kappa and vanishes as kappa -> 0, which is the sense")
    print("  of it: at slow mean reversion the two factors are nearly the same process,")
    print("  so it hardly matters which one holds the variance. At fast mean reversion")
    print("  they are very different, and moving variance into the permanent factor")
    print("  destroys the cycling the store lives on.")
    print()
    print("  The struck swing's answer DEPENDS ON THE ANCHOR, and so does its SIGN:")
    print("  +0.81 % against -7.76 % at one and the same sigma_xi (0.10, kappa 4). Holding")
    print("  spot variance fixed it gains, because the walk's variance keeps accumulating")
    print("  where the OU factor's saturates, so a longer-dated option sees more terminal")
    print("  variance. Holding terminal variance fixed removes precisely that, and it loses.")
    print()
    print("  So this probe cannot say what a second factor is worth, and nor can any")
    print("  single-number anchor: the two factors differ in their variance TERM")
    print("  STRUCTURE, which is the thing a real calibration fits and no scalar")
    print("  captures. What it does establish is that the question is a CALIBRATION")
    print("  question rather than a valuation-architecture one -- the order the reply")
    print("  to the review set, and the reason to fit the forward panel first.")


if __name__ == "__main__":
    main()
