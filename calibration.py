"""Reproducible S7 covariance-model comparison for the TTF delivery panel.

This module estimates physical-measure (P) covariance parameters from aligned
log-forward returns.  It deliberately does not infer a risk premium or silently
label the estimates risk-neutral (Q) parameters.  The monthly delivery loading
is the S6 flat-forward average over ``[delivery_start, delivery_end)``.

The likelihood is a deliberately small first comparison rather than a hidden
research platform.  Conditional on each observed interval ``dt`` it uses

    one factor: C = dt * sigma_chi^2 * a a' + 2*tau^2*I

    two factors: C = dt * (sigma_chi^2*a a' + sigma_xi^2*11'
                           + rho*sigma_chi*sigma_xi*(a1' + 1a'))
                     + 2*tau^2*I

where ``a`` is the delivery-averaged short loading and ``tau`` is quote-level
measurement noise.  The factor covariance scales with actual calendar time;
measurement noise is separate and therefore does not.  Treating return noise as
``2*tau^2`` assumes consecutive quote errors are independent.  The report names
that restriction so a later state-space model can replace it without changing
the data contract or pretending the likelihood was more exact than it is.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import quote_data as qd


LOG_TWO_PI = math.log(2.0 * math.pi)
YEAR_DAYS = 365.25

CANDIDATES = {
    "one_factor": ("kappa", "sigma_chi", "observation_sigma"),
    "two_factor_independent": (
        "kappa", "sigma_chi", "sigma_xi", "observation_sigma"),
    "two_factor_correlated": (
        "kappa", "sigma_chi", "sigma_xi", "rho", "observation_sigma"),
}

PHYSICAL_BOUNDS = {
    "kappa": (0.01, 10.0),
    "sigma_chi": (0.01, 5.0),
    "sigma_xi": (0.001, 5.0),
    "rho": (-0.95, 0.95),
    "observation_sigma": (1e-5, 0.20),
}


@dataclass(frozen=True)
class ReturnSnapshot:
    """One aligned cross-section of delivery-contract returns."""

    return_start: pd.Timestamp
    return_end: pd.Timestamp
    dt: float
    values: np.ndarray
    delivery_start_offsets: np.ndarray
    delivery_end_offsets: np.ndarray
    source_columns: tuple[str, ...]


def _rank(column):
    text = str(column)
    if not text.startswith("TTFc") or not text[4:].isdigit():
        raise ValueError(f"invalid continuous-rank column {column!r}")
    return int(text[4:])


def select_calibration_returns(returns, ranks):
    """Select declared end-of-interval ranks without losing roll identity.

    The return itself is already matched by delivery contract.  Selection uses
    the rank at ``return_end`` so a month-end ``TTFc2 -> TTFc1`` observation is
    retained as the new front rank while its fixed delivery identity remains the
    key used to calculate the return.
    """
    required = {"return_start", "return_end", "contract_identifier", "log_return",
                "year_fraction", "end_source_column"}
    missing = required - set(returns.columns)
    if missing:
        raise ValueError(f"returns are missing required column(s): {sorted(missing)}")
    ranks = tuple(int(r) for r in ranks)
    if not ranks or any(r <= 0 for r in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("ranks must be distinct positive integers")
    chosen = {f"TTFc{r}" for r in ranks}
    selected = returns.loc[returns["end_source_column"].isin(chosen)].copy()
    selected["return_start"] = pd.to_datetime(selected["return_start"])
    selected["return_end"] = pd.to_datetime(selected["return_end"])
    return selected.sort_values(["return_end", "end_source_column"]).reset_index(drop=True)


def make_snapshots(returns, start_date, end_date, ranks, min_contracts=2):
    """Build aligned return vectors and exact delivery-period offsets."""
    selected = select_calibration_returns(returns, ranks)
    start_date, end_date = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if end_date < start_date:
        raise ValueError("end_date must not precede start_date")
    selected = selected.loc[
        selected["return_end"].between(start_date, end_date, inclusive="both")]
    snapshots = []
    for (return_start, return_end), group in selected.groupby(
            ["return_start", "return_end"], sort=True):
        group = group.sort_values("end_source_column", key=lambda s: s.map(_rank))
        if len(group) < int(min_contracts):
            continue
        if group["end_source_column"].duplicated().any():
            raise ValueError(f"duplicate selected rank on return interval ending {return_end:%Y-%m-%d}")
        dt = float(group["year_fraction"].iloc[0])
        if not np.isfinite(dt) or dt <= 0.0 or not np.allclose(group["year_fraction"], dt):
            raise ValueError("an aligned return vector has inconsistent elapsed time")
        periods = pd.PeriodIndex(group["contract_identifier"], freq="M")
        delivery_start = periods.to_timestamp(how="start")
        delivery_end = (periods + 1).to_timestamp(how="start")
        start_offsets = np.asarray((delivery_start - return_start).days, dtype=float) / YEAR_DAYS
        end_offsets = np.asarray((delivery_end - return_start).days, dtype=float) / YEAR_DAYS
        values = group["log_return"].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(start_offsets).all()
                or (start_offsets < 0.0).any() or (end_offsets <= start_offsets).any()):
            raise ValueError("invalid value or delivery interval in calibration returns")
        snapshots.append(ReturnSnapshot(
            pd.Timestamp(return_start), pd.Timestamp(return_end), dt, values,
            start_offsets, end_offsets, tuple(group["end_source_column"])))
    if not snapshots:
        raise ValueError("the declared window contains no aligned return snapshots")
    return snapshots


def _delivery_loadings(kappa, snapshot):
    if kappa == 0.0:
        return np.ones(len(snapshot.values))
    width = snapshot.delivery_end_offsets - snapshot.delivery_start_offsets
    return (np.exp(-kappa * snapshot.delivery_start_offsets)
            * (-np.expm1(-kappa * width)) / (kappa * width))


def covariance_matrix(model, parameters, snapshot):
    """Conditional covariance for one aligned return snapshot."""
    if model not in CANDIDATES:
        raise ValueError(f"unknown candidate {model!r}")
    a = _delivery_loadings(parameters["kappa"], snapshot)
    sigma_chi = parameters["sigma_chi"]
    process = sigma_chi**2 * np.outer(a, a)
    if model != "one_factor":
        sigma_xi = parameters["sigma_xi"]
        ones = np.ones(len(a))
        process += sigma_xi**2 * np.outer(ones, ones)
        rho = 0.0 if model == "two_factor_independent" else parameters["rho"]
        process += rho * sigma_chi * sigma_xi * (
            np.outer(a, ones) + np.outer(ones, a))
    observation = 2.0 * parameters["observation_sigma"]**2
    return snapshot.dt * process + observation * np.eye(len(a))


def _encode(model, parameters):
    values = []
    for name in CANDIDATES[model]:
        value = float(parameters[name])
        values.append(np.arctanh(value) if name == "rho" else np.log(value))
    return np.asarray(values)


def _decode(model, transformed):
    return {
        name: float(np.tanh(value) if name == "rho" else np.exp(value))
        for name, value in zip(CANDIDATES[model], transformed)
    }


def _transformed_bounds(model):
    bounds = []
    for name in CANDIDATES[model]:
        low, high = PHYSICAL_BOUNDS[name]
        bounds.append((np.arctanh(low), np.arctanh(high)) if name == "rho"
                      else (np.log(low), np.log(high)))
    return bounds


def negative_log_likelihood(transformed, model, snapshots):
    parameters = _decode(model, transformed)
    widths = {len(snapshot.values) for snapshot in snapshots}
    if len(widths) == 1:
        # The configured comparison requires all four declared ranks, so the
        # whole likelihood can normally be evaluated as one stack of small
        # matrices.  Keeping the general loop below preserves the public
        # function for diagnostic samples with varying cross-section sizes.
        values = np.stack([snapshot.values for snapshot in snapshots])
        starts = np.stack([snapshot.delivery_start_offsets for snapshot in snapshots])
        ends = np.stack([snapshot.delivery_end_offsets for snapshot in snapshots])
        dt = np.asarray([snapshot.dt for snapshot in snapshots])[:, None, None]
        kappa = parameters["kappa"]
        if kappa == 0.0:
            loadings = np.ones_like(starts)
        else:
            width = ends - starts
            loadings = (np.exp(-kappa*starts) * (-np.expm1(-kappa*width))
                        / (kappa*width))
        process = parameters["sigma_chi"]**2 * (
            loadings[:, :, None] * loadings[:, None, :])
        if model != "one_factor":
            sigma_xi = parameters["sigma_xi"]
            process += sigma_xi**2
            rho = 0.0 if model == "two_factor_independent" else parameters["rho"]
            process += rho * parameters["sigma_chi"] * sigma_xi * (
                loadings[:, :, None] + loadings[:, None, :])
        observation = 2.0 * parameters["observation_sigma"]**2
        covariance = dt*process + observation*np.eye(values.shape[1])[None, :, :]
        try:
            chol = np.linalg.cholesky(covariance)
            solved = np.linalg.solve(chol, values[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            return 1e100
        total = 0.5 * (
            values.size*LOG_TWO_PI
            + 2.0*np.log(np.diagonal(chol, axis1=1, axis2=2)).sum()
            + np.square(solved).sum())
        return float(total) if np.isfinite(total) else 1e100

    total = 0.0
    for snapshot in snapshots:
        covariance = covariance_matrix(model, parameters, snapshot)
        try:
            chol = np.linalg.cholesky(covariance)
            solved = np.linalg.solve(chol, snapshot.values)
        except np.linalg.LinAlgError:
            return 1e100
        total += 0.5 * (
            len(snapshot.values) * LOG_TWO_PI
            + 2.0 * np.log(np.diag(chol)).sum()
            + solved @ solved)
    return float(total) if np.isfinite(total) else 1e100


def default_starts(model):
    """Dispersed deterministic starting points in physical coordinates."""
    common = [
        dict(kappa=0.15, sigma_chi=0.35, observation_sigma=0.003),
        dict(kappa=0.75, sigma_chi=0.70, observation_sigma=0.010),
        dict(kappa=2.50, sigma_chi=1.20, observation_sigma=0.025),
        dict(kappa=6.00, sigma_chi=2.00, observation_sigma=0.050),
    ]
    if model == "one_factor":
        return common
    starts = []
    rhos = [0.0, 0.0, -0.50, 0.50]
    for base, sigma_xi, rho in zip(common, [0.15, 0.40, 0.80, 1.50], rhos):
        row = dict(base, sigma_xi=sigma_xi)
        if model == "two_factor_correlated":
            row["rho"] = rho
        starts.append(row)
    return starts


def _finite_hessian(function, optimum, step=2e-3):
    """Central finite-difference Hessian of a per-observation objective."""
    x = np.asarray(optimum, dtype=float)
    n = len(x)
    hessian = np.empty((n, n), dtype=float)
    f0 = function(x)
    for i in range(n):
        ei = np.zeros(n); ei[i] = step
        hessian[i, i] = (function(x + ei) - 2.0*f0 + function(x - ei)) / step**2
        for j in range(i):
            ej = np.zeros(n); ej[j] = step
            value = (function(x+ei+ej) - function(x+ei-ej)
                     - function(x-ei+ej) + function(x-ei-ej)) / (4.0*step**2)
            hessian[i, j] = hessian[j, i] = value
    return hessian


def _identification_diagnostics(model, optimum, snapshots):
    scalar_count = sum(len(s.values) for s in snapshots)
    objective = lambda x: negative_log_likelihood(x, model, snapshots) / scalar_count
    hessian = _finite_hessian(objective, optimum)
    eigenvalues = np.linalg.eigvalsh(hessian)
    positive = bool(np.all(eigenvalues > 0.0))
    condition = float(eigenvalues.max() / eigenvalues.min()) if positive else None
    correlations = None
    if positive:
        covariance = np.linalg.inv(hessian)
        scale = np.sqrt(np.diag(covariance))
        correlations = covariance / np.outer(scale, scale)
    return {
        "hessian_eigenvalues_per_observation": eigenvalues.tolist(),
        "hessian_positive_definite": positive,
        "hessian_condition_number": condition,
        "transformed_parameter_correlation": None if correlations is None else correlations.tolist(),
    }


def score(model, parameters, snapshots):
    nll = negative_log_likelihood(_encode(model, parameters), model, snapshots)
    scalar_count = sum(len(s.values) for s in snapshots)
    sizes = {len(snapshot.values) for snapshot in snapshots}
    diagnostics = None
    if len(sizes) == 1:
        values = np.stack([snapshot.values for snapshot in snapshots])
        predicted = np.stack([
            covariance_matrix(model, parameters, snapshot) for snapshot in snapshots])
        observed_covariance = np.einsum("ti,tj->ij", values, values) / len(values)
        predicted_covariance = predicted.mean(axis=0)

        def correlation(covariance):
            scale = np.sqrt(np.diag(covariance))
            return covariance / np.outer(scale, scale)

        source_columns = snapshots[0].source_columns
        if not all(snapshot.source_columns == source_columns for snapshot in snapshots):
            raise ValueError("diagnostic snapshots do not share one declared rank order")
        standardised = values / np.sqrt(np.diagonal(predicted, axis1=1, axis2=2))
        lag_one = {}
        for i, column in enumerate(source_columns):
            x, y = standardised[:-1, i], standardised[1:, i]
            lag_one[column] = None if len(x) < 2 else float(np.corrcoef(x, y)[0, 1])
        pair_spreads = {}
        for i in range(len(source_columns)):
            for j in range(i+1, len(source_columns)):
                key = f"{source_columns[i]}-{source_columns[j]}"
                observed = np.mean(np.square(values[:, i]-values[:, j]))
                expected = np.mean(
                    predicted[:, i, i] + predicted[:, j, j] - 2.0*predicted[:, i, j])
                pair_spreads[key] = {
                    "observed_rms_log_return": float(np.sqrt(observed)),
                    "predicted_rms_log_return": float(np.sqrt(expected)),
                }
        diagnostics = {
            "rank_order": list(source_columns),
            "observed_return_covariance": observed_covariance.tolist(),
            "mean_predicted_return_covariance": predicted_covariance.tolist(),
            "covariance_rmse": float(np.sqrt(np.mean(
                np.square(observed_covariance-predicted_covariance)))),
            "observed_return_correlation": correlation(observed_covariance).tolist(),
            "mean_predicted_return_correlation": correlation(predicted_covariance).tolist(),
            "correlation_rmse": float(np.sqrt(np.mean(np.square(
                correlation(observed_covariance)-correlation(predicted_covariance))))),
            "standardised_return_lag_one_correlation": lag_one,
            "delivery_pair_spread_rms": pair_spreads,
        }
    return {
        "negative_log_likelihood": nll,
        "mean_negative_log_likelihood_per_observation": nll / scalar_count,
        "snapshot_count": len(snapshots),
        "scalar_observation_count": scalar_count,
        "diagnostics": diagnostics,
    }


def fit_candidate(model, snapshots, *, maxiter=500):
    """Fit one candidate from multiple dispersed deterministic starting points."""
    if model not in CANDIDATES:
        raise ValueError(f"unknown candidate {model!r}")
    bounds = _transformed_bounds(model)
    attempts = []
    best = None
    for start in default_starts(model):
        result = minimize(
            negative_log_likelihood, _encode(model, start), args=(model, snapshots),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-11, "gtol": 1e-7},
        )
        attempt = {
            "start": start,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "negative_log_likelihood": float(result.fun),
            "parameters": _decode(model, result.x),
        }
        attempts.append(attempt)
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError(f"all optimisation attempts failed for {model}")

    parameters = _decode(model, best.x)
    training = score(model, parameters, snapshots)
    parameter_count = len(CANDIDATES[model])
    scalar_count = training["scalar_observation_count"]
    boundary_hits = []
    for name, value, (low, high) in zip(CANDIDATES[model], best.x, bounds):
        if min(abs(value-low), abs(high-value)) <= 1e-3 * max(1.0, high-low):
            boundary_hits.append(name)
    close = [
        a for a in attempts
        if a["success"] and
        (a["negative_log_likelihood"] - best.fun) / scalar_count <= 1e-4
    ]
    return {
        "model": model,
        "parameters_P": parameters,
        "parameter_units": {
            "kappa": "1/year", "sigma_chi": "log-price/sqrt(year)",
            "sigma_xi": "log-price/sqrt(year)", "rho": "dimensionless",
            "observation_sigma": "log-price per quote",
        },
        "transformed_parameter_order": list(CANDIDATES[model]),
        "transformed_optimum": best.x.tolist(),
        "success": bool(best.success),
        "status": int(best.status),
        "message": str(best.message),
        "iterations": int(best.nit),
        "boundary_hits": boundary_hits,
        "training": {
            **training,
            "aic": 2*parameter_count + 2*training["negative_log_likelihood"],
            "bic": parameter_count*math.log(scalar_count)
                   + 2*training["negative_log_likelihood"],
        },
        "start_stability": {
            "attempt_count": len(attempts),
            "converged_to_best_count": len(close),
            "fraction": len(close) / len(attempts),
            "criterion": "successful and within 1e-4 NLL per scalar observation of best",
        },
        "attempts": attempts,
        "identification": _identification_diagnostics(model, best.x, snapshots),
    }


def compare_models(returns, config):
    """Fit declared candidates, score untouched holdout data and decide."""
    ranks = config["ranks"]
    train = make_snapshots(
        returns, config["training_start"], config["training_end"], ranks,
        config.get("minimum_contracts_per_snapshot", 2))
    holdout = make_snapshots(
        returns, config["holdout_start"], config["holdout_end"], ranks,
        config.get("minimum_contracts_per_snapshot", 2))
    candidates = config.get("candidates", list(CANDIDATES))
    unknown = set(candidates) - set(CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    fits = {}
    for model in candidates:
        fitted = fit_candidate(model, train, maxiter=config.get("max_iterations", 500))
        fitted["holdout"] = score(model, fitted["parameters_P"], holdout)
        fits[model] = fitted

    thresholds = config["decision_thresholds"]
    baseline = fits["one_factor"]
    eligible, weak = [], []
    for model, fitted in fits.items():
        if model == "one_factor":
            continue
        heldout_gain = (
            baseline["holdout"]["mean_negative_log_likelihood_per_observation"]
            - fitted["holdout"]["mean_negative_log_likelihood_per_observation"])
        bic_gain = baseline["training"]["bic"] - fitted["training"]["bic"]
        identification = fitted["identification"]
        identified = (
            identification["hessian_positive_definite"]
            and identification["hessian_condition_number"]
                <= thresholds["maximum_hessian_condition_number"]
            and not fitted["boundary_hits"]
            and fitted["start_stability"]["fraction"]
                >= thresholds["minimum_start_stability_fraction"])
        fitted["comparison_to_one_factor"] = {
            "holdout_nll_gain_per_observation": heldout_gain,
            "training_bic_gain": bic_gain,
            "passes_fit_improvement": (
                heldout_gain >= thresholds["minimum_holdout_nll_gain_per_observation"]
                and bic_gain >= thresholds["minimum_training_bic_gain"]),
            "passes_identification": identified,
        }
        if fitted["comparison_to_one_factor"]["passes_fit_improvement"]:
            (eligible if identified else weak).append(model)

    if eligible:
        selected = min(eligible, key=lambda name: fits[name]["holdout"][
            "mean_negative_log_likelihood_per_observation"])
        conclusion = "richer model selected by the predeclared fit and identification gates"
    elif weak:
        selected = None
        conclusion = (
            "inconclusive: a richer model improves fit but fails the predeclared "
            "identification or optimisation-stability gates")
    else:
        selected = "one_factor"
        conclusion = "retain the one-factor baseline for the stated scope"
    return {
        "sample": {
            "training_window": [config["training_start"], config["training_end"]],
            "holdout_window": [config["holdout_start"], config["holdout_end"]],
            "ranks": list(ranks),
            "training_snapshot_count": len(train),
            "holdout_snapshot_count": len(holdout),
            "training_scalar_observation_count": sum(len(s.values) for s in train),
            "holdout_scalar_observation_count": sum(len(s.values) for s in holdout),
        },
        "observation_specification": {
            "returns": "log returns matched by fixed delivery identity",
            "delivery_loading": "flat-forward calendar-day average over [delivery_start, delivery_end)",
            "time_scale": "actual elapsed calendar days / 365.25",
            "mean": "zero conditional mean over daily/weekend return intervals",
            "measurement_noise": "independent quote-level Gaussian error; return variance 2*tau^2",
        },
        "measure_specification": {
            "P": "kappa, factor volatilities and correlation estimated from historical returns",
            "Q": config["valuation_measure_restriction"],
        },
        "decision_thresholds": thresholds,
        "fits": fits,
        "decision": {"selected_model": selected, "conclusion": conclusion},
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def run_configuration(config_path):
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    root = os.path.dirname(os.path.abspath(config_path))
    source = config["source_path"]
    source = source if os.path.isabs(source) else os.path.join(root, source)
    quotes, provenance = qd.load_quote_matrix(
        source, cache_dir=os.path.join(root, config.get("cache_directory", ".cache")))
    panel = qd.build_delivery_panel(quotes, provenance["source_sha256"])
    returns = qd.build_returns(panel)
    manifest = qd.build_data_manifest(
        quotes, provenance, retrieval_date=config.get("retrieval_date"),
        as_of_date=config.get("as_of_date"), units=config.get("units", "EUR/MWh"))
    # Keep checked-in output portable; the fingerprint, byte count and relative
    # locator establish identity without embedding a machine-specific path.
    manifest["source_path"] = config["source_path"]
    primary = compare_models(returns, config)
    sensitivity = {}
    for window in config.get("sensitivity_windows", []):
        alternative = dict(config)
        alternative.update(window)
        alternative.pop("name", None)
        alternative.pop("sensitivity_windows", None)
        sensitivity[window["name"]] = compare_models(returns, alternative)

    decisions = [primary["decision"], *[value["decision"] for value in sensitivity.values()]]
    selections = [decision["selected_model"] for decision in decisions]
    if len(set(selections)) == 1:
        across_windows = f"all declared windows select {selections[0]}"
    else:
        across_windows = "model selection is not stable across the declared windows"
    result = {
        "configuration": config,
        "data_manifest": manifest,
        "comparison": primary,
        "sensitivity_comparisons": sensitivity,
        "architecture_assessment": {
            "window_selections": selections,
            "conclusion": across_windows,
            "valuation_status": (
                "no repricing: the historical P comparison does not identify Q dynamics "
                "or a market price of risk"),
        },
    }
    return _json_safe(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="JSON calibration configuration")
    parser.add_argument("--output", help="write JSON report here; stdout if omitted")
    args = parser.parse_args(argv)
    result = run_configuration(args.config)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
