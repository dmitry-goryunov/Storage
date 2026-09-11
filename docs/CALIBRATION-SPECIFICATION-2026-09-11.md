# Current TTF calibration specification

Date: 11 September 2026

Scope: S6 data and observation layer, followed by the first reproducible S7 model comparison

Status: implemented and run against the repository workbook

This specification replaces the active implementation steps in
[`DESIGN-P4.1-two-factor.md`](DESIGN-P4.1-two-factor.md). That earlier design remains dated
history. Its claim that a second factor was already required did not survive independent
review. The current result is narrower: the available historical panel does not pass the
predeclared gates for adding a second factor, and it does not identify valuation-measure
dynamics.

## 1. Decision reached

Keep the one-factor valuation architecture for the present project scope. Do not replace the
illustrative valuation parameters with the historical estimates yet.

The primary comparison finds only a 0.00222 improvement in held-out negative log likelihood
per scalar observation from the independent two-factor model, below the declared 0.01 gate,
while its training BIC is 6.02 worse. The correlated model gains just 0.000084 per held-out
observation, has BIC 6.27 worse, and drives correlation to the imposed -0.95 boundary. Its
transformed parameters are almost perfectly dependent in the local-curvature diagnostic.

A pre-crisis sensitivity fit reaches the same model-selection decision for a more serious
reason: both two-factor candidates improve training BIC substantially and then forecast
2020-2021 worse than the one-factor model. The one-factor volatility also moves from 0.3793
in the pre-crisis fit to 0.7396 in the primary 2015-2022 fit. This is evidence of regime
instability, not a stable Q calibration.

The result therefore closes the architecture question provisionally and leaves valuation
calibration open. A richer factor structure may still be needed after the observation-noise
and regime specification is improved, or for a struck or fee-bearing payoff whose reduction
to one valuation state fails.

## 2. Reproduction command and artefacts

Run from the repository root in the pinned environment:

```bash
python calibration.py calibration-config.json \
  --output calibration-results-2026-09-11.json
```

The run is deterministic. It uses no random method and therefore records no decorative
random seed.

| File | Purpose |
|---|---|
| `calibration-config.json` | Frozen source, windows, ranks, candidates, bounds-related gates and P/Q restriction |
| `calibration-data-manifest-2026-09-11.json` | Portable source identity, units, dates, cleaning and filter record |
| `calibration-results-2026-09-11.json` | Parameters, every optimiser start, objectives, holdout scores, covariance diagnostics and decision |
| `quote_data.py` | Content-addressed loading, delivery panel, matched returns and manifest construction |
| `delivery_model.py` | Point and monthly-delivery observation functions and nonlinear Jacobian |
| `calibration.py` | Candidate likelihoods, fitting, diagnostics, gates and report generation |

## 3. S6 data construction, step by step

1. Read the exact `ttf q.xlsx` bytes and calculate their SHA-256 before parsing. The same
   bytes are parsed and used as the cache identity. A damaged derived parquet cache is
   discarded and rebuilt from those bytes.
2. Apply the shared cleaning rule. The first column becomes `quote_date`; unparseable dates
   are removed; contract cells are coerced to numeric; rejected values become missing rather
   than silently changing the row set.
3. Map each continuous column to a fixed delivery identity. On quote date `t`, `TTFc1` maps
   to the next calendar month and `TTFcN` advances `N-1` further months. The manifest marks
   this as a project convention because provider documentation was not supplied.
4. Represent delivery as a half-open period. February is `[1 February, 1 March)`, so every
   delivery day enters an integral or daily sum exactly once.
5. Retain missing and non-positive quotations in the long panel with a validity flag. They
   are data-quality observations, not rows to erase before interval construction.
6. Group by fixed `contract_identifier`, order by quote date and calculate a log return only
   for adjacent panel rows where both observations are valid. This preserves a
   `TTFc2 -> TTFc1` month-end roll for the same delivery while preventing a missing quote
   from being bridged.
7. Record `return_start`, `return_end`, elapsed calendar days and year fraction. Friday to
   Monday is three calendar days, not a relabelled one-day observation.
8. Freeze the manifest before fitting. The source has SHA-256
   `b9403ff09f0424aa477ff6351073febed688bfbb611eee2433213c741f7e9729`, 4,171 quote rows
   from 12 March 2010 through 6 March 2026, 55 continuous monthly columns, 6,071 missing
   observations and no non-positive observations. The original retrieval date is absent and
   is recorded as unknown rather than inferred.

The full workbook produces 229,405 delivery-panel rows across 247 delivery months and
223,023 matched returns. Of those, 178,471 span one calendar day and 44,552 span three days.

## 4. Delivery observation

For a point-delivery forward, the short-factor loading is

```text
a(t,T) = exp(-kappa * (T-t)).
```

For a flat monthly forward delivered over `[A,B)`, the implemented loading is

```text
a_bar(t;A,B) =
    [exp(-kappa*(A-t)) - exp(-kappa*(B-t))] / [kappa*(B-A)].
```

The `kappa = 0` limit is exactly one. A discrete delivery function separately implements
price and volume weighting. The nonlinear monthly observation is a weighted average of
factor forwards; its analytic Jacobian is checked against central finite differences.

The synthetic acceptance case includes a month-end roll, a weekend, a missing quote and two
delivery identities. It also reproduces the independent-review counterexample: at
`sigma = 0.5`, `kappa = 1`, delivery averaging gives 0.12443854 for the declared spread,
against 0.11932561 for the old month-end point comparison. Averaging is not assumed to move
volatility in one universal direction.

## 5. S7 measures and estimands

The comparison estimates physical-measure covariance parameters from historical log-forward
returns:

```text
P one factor:
    d chi = -kappa * chi dt + sigma_chi dW_chi

P two factor:
    d chi = -kappa * chi dt + sigma_chi dW_chi
    d xi  =                         sigma_xi dW_xi
    corr(dW_chi, dW_xi) = rho
```

The fit does not estimate a market price of risk, Q drift or a P-to-Q mean-reversion map.
The configuration consequently says that there is no valuation-measure restriction. This is
why the result does not feed its P estimates into `sVol` or `sMR` and does not report a new
storage value.

The observed vector at each return interval uses ranks 1, 6, 12 and 24 at the interval end.
The return remains keyed to fixed delivery identity, so selecting a rank does not recreate
the old continuous-column roll error.

For delivery loadings `a`, elapsed time `dt`, quote-level measurement-noise standard
deviation `tau` and identity matrix `I`, the conditional covariances are:

```text
one factor:
    C = dt * sigma_chi^2 * a a' + 2*tau^2*I

two factor:
    C = dt * [sigma_chi^2*a a' + sigma_xi^2*1 1'
              + rho*sigma_chi*sigma_xi*(a 1' + 1 a')]
        + 2*tau^2*I.
```

Factor variance scales with actual calendar time. Measurement noise is separate. The
`2*tau^2` return term assumes independent quote errors at consecutive observations. This is
a composite return likelihood, not a Kalman likelihood, and its serial-noise restriction is
one of the next issues to test.

## 6. Frozen estimation sequence

1. Primary training is 1 January 2015 through 31 December 2022. The untouched holdout is
   1 January 2023 through 6 March 2026.
2. The predeclared sensitivity trains through 31 December 2019 and holds out 2020-2021.
3. Every snapshot must contain all four declared ranks. The primary sample has 2,087
   training snapshots and 830 holdout snapshots, or 8,348 and 3,320 scalar observations.
4. Fit one-factor, independent two-factor and correlated two-factor candidates against the
   same snapshots and delivery rule.
5. Optimise log coordinates for positive parameters and an inverse-hyperbolic-tangent
   coordinate for correlation. Physical bounds are `kappa` 0.01-10 per year,
   factor volatility 0.01-5 for `sigma_chi`, 0.001-5 for `sigma_xi`, correlation
   -0.95 to 0.95, and quote noise 0.00001-0.20 in log-price units.
6. Run four dispersed deterministic starting points for every candidate. Save every start,
   optimiser status, iteration count, objective and final parameters.
7. Require at least 75% of starts to finish successfully within 0.0001 NLL per scalar
   observation of the best result. Record transformed-parameter boundary hits.
8. Calculate a central finite-difference Hessian of NLL per observation. Require positive
   curvature and condition number at most 100 million for the richer-model identification
   gate. Retain the transformed-parameter correlation matrix so confounding remains visible.
9. Require both training BIC improvement of at least 10 and holdout NLL improvement of at
   least 0.01 per scalar observation before selecting a richer model. These are project
   decision gates, not universal statistical constants.
10. Compare observed and average predicted covariance and correlation, every delivery-pair
    spread RMS, and lag-one correlation of marginally standardised returns.

## 7. Results

### Primary 2015-2022 training, 2023-2026 holdout

| Candidate | kappa | sigma_chi | sigma_xi | rho | tau | Training BIC | Holdout NLL/obs | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| One factor | 0.4674 | 0.7396 | - | - | 0.01483 | -36,017.69 | -2.37243 | Baseline; all four starts agree |
| Two factor, independent | 0.5057 | 0.7323 | 0.1388 | 0 | 0.01467 | -36,011.67 | -2.37465 | Holdout gain 0.00222; fails fit gate |
| Two factor, correlated | 0.2768 | 1.0816 | 0.3801 | -0.95 | 0.01470 | -36,011.43 | -2.37251 | Correlation at boundary; fails fit and identification gates |

The one-factor local Hessian is positive definite with condition number 13.6. The independent
two-factor condition number is 1,001.3. The correlated fit's is 23,165.5, and its transformed
factor parameters have absolute local correlations above 0.94. The numerical optimiser is
stable across starts; the richer parameter interpretation is not.

The holdout c1-c6 observed RMS log-return spread is 0.02525. The one-factor model predicts
0.03065; the independent two-factor model predicts 0.03048. At the long end the mismatch is
larger: observed c12-c24 is 0.01335 against 0.03151 for the baseline. The sizeable fitted
quote noise helps the marginal likelihood but injects independent spread variation across
all ranks. This diagnostic argues for a proper state-space treatment of measurement noise
before interpreting factor count economically.

### Pre-crisis training, 2020-2021 holdout

| Candidate | kappa | sigma_chi | sigma_xi | rho | Training BIC gain vs one factor | Holdout NLL gain/obs |
|---|---:|---:|---:|---:|---:|---:|
| One factor | 0.5445 | 0.3793 | - | - | baseline | baseline |
| Two factor, independent | 5.3889 | 0.4614 | 0.2265 | 0 | +312.38 | -0.51723 |
| Two factor, correlated | 7.1558 | 0.4965 | 0.2130 | 0.5826 | +501.77 | -0.49384 |

The richer candidates fit the quiet training regime and fail badly out of sample. The
one-factor model itself predicts a c1-c6 RMS spread of 0.01881 against 0.05019 observed in
2020-2021, while standardised c12 returns retain lag-one correlation of 0.228. This is not a
cleanly white, stable residual process.

## 8. What is complete and what remains

S6 is complete for the repository dataset: source identity, delivery identity, validity,
return intervals and monthly observation functions are explicit and tested. The provider's
own continuous-rank definition and the original retrieval date remain unavailable, so the
project convention must not be presented as independently sourced market metadata.

The first S7 model comparison is complete and reproducible. It supports retaining one factor
for the current architecture because neither richer candidate passes the predeclared combined
fit and identification gates. It does not validate the fitted dynamics for valuation. The
window sensitivity, residual structure, measurement-noise allocation and missing P-to-Q link
make a new storage-price claim unjustified.

The next course of action is:

1. obtain provider documentation and an updated, timestamped market extract;
2. replace independent return noise with a level-state observation equation, using the S6
   nonlinear delivery function and its Jacobian in an extended Kalman likelihood;
3. add an explicit regime or time-varying-volatility candidate because the pre-crisis model
   fails the crisis holdout, then repeat the same frozen-window gates;
4. estimate or declare the P-to-Q restriction from option or forward-risk-premium evidence;
5. only after that, propagate parameter and model uncertainty through unchanged storage,
   fee-bearing storage, unstruck swing and struck swing fixtures;
6. build a second valuation state only if the accepted market model and target payoff cannot
   be reduced and the value difference is material after numerical refinement.

Until those steps pass, the old 8% second-factor haircut remains withdrawn and `sVol=0.9`
remains an illustrative input rather than a calibrated parameter.
