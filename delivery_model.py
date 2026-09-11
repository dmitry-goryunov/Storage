"""Delivery-period observation functions for the S6 calibration layer."""
import math

import numpy as np
import pandas as pd

YEAR_DAYS = 365.25


def _time_offsets(t, dates):
    values = np.asarray(dates)
    if np.issubdtype(values.dtype, np.datetime64) or isinstance(t, (pd.Timestamp, str)):
        origin = pd.Timestamp(t)
        parsed = pd.to_datetime(values)
        return np.asarray((parsed - origin).days, dtype=float) / YEAR_DAYS
    return np.asarray(values, dtype=float) - float(t)


def _validate_kappa(kappa):
    kappa = float(kappa)
    if not math.isfinite(kappa) or kappa < 0.0:
        raise ValueError("kappa must be finite and non-negative")
    return kappa


def flat_forward_delivery_loading(kappa, t, delivery_start, delivery_end):
    """Average exp(-kappa*(u-t)) over a uniformly weighted delivery period."""
    kappa = _validate_kappa(kappa)
    offsets = _time_offsets(t, [delivery_start, delivery_end])
    start, end = map(float, offsets)
    if not np.isfinite(offsets).all() or end <= start:
        raise ValueError("delivery_end must be later than delivery_start")
    if start < -1e-12:
        raise ValueError("delivery starts before the observation time")
    if kappa == 0.0:
        return 1.0
    width = end - start
    return float(np.exp(-kappa * start) * (-np.expm1(-kappa * width)) / (kappa * width))


def delivery_averaged_loading(kappa, t, dates, forward_prices, weights=None):
    """Price-weighted short-factor loading for discrete delivery points."""
    kappa = _validate_kappa(kappa)
    offsets = _time_offsets(t, dates)
    prices = np.asarray(forward_prices, dtype=float)
    if offsets.ndim != 1 or prices.ndim != 1 or len(offsets) != len(prices) or len(prices) == 0:
        raise ValueError("dates and forward_prices must be equally sized non-empty vectors")
    weights = np.ones(len(prices), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != prices.shape:
        raise ValueError("weights must match forward_prices")
    if (not np.isfinite(offsets).all() or not np.isfinite(prices).all()
            or not np.isfinite(weights).all() or (offsets < -1e-12).any()
            or (prices <= 0.0).any() or (weights < 0.0).any()):
        raise ValueError("dates, prices and weights must define positive finite delivery weights")
    economic_weights = weights * prices
    denominator = economic_weights.sum()
    if denominator <= 0.0:
        raise ValueError("delivery weights sum to zero")
    loadings = np.exp(-kappa * offsets)
    return float(economic_weights @ loadings / denominator)


def log_delivery_forward_and_jacobian(chi, xi, loadings, base_forwards, weights=None):
    """Log price and derivatives for a weighted average of factor forwards.

    Each delivery point is `base_forward * exp(loading*chi + xi)`. The result
    supplies the nonlinear observation and analytic Jacobian needed by an
    extended Kalman implementation in S7.
    """
    loadings = np.asarray(loadings, dtype=float)
    base = np.asarray(base_forwards, dtype=float)
    if loadings.ndim != 1 or base.shape != loadings.shape or len(base) == 0:
        raise ValueError("loadings and base_forwards must be equally sized non-empty vectors")
    weights = np.ones(len(base), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != base.shape or (weights < 0.0).any() or (base <= 0.0).any():
        raise ValueError("weights must be non-negative and base forwards positive")
    terms = weights * base * np.exp(loadings * float(chi) + float(xi))
    total_weight = weights.sum()
    total = terms.sum()
    if total_weight <= 0.0 or not np.isfinite(total) or total <= 0.0:
        raise ValueError("delivery forward is not finite and positive")
    price = total / total_weight
    d_chi = float(terms @ loadings / total)
    return float(np.log(price)), np.array([d_chi, 1.0])
