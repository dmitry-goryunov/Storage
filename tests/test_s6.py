"""Independent acceptance checks for the S6 data and observation layer."""
import hashlib
import math

import numpy as np
import pandas as pd
import pytest

import delivery_model as dm
import quote_data as qd


SOURCE_HASH = hashlib.sha256(b"synthetic S6 panel").hexdigest()


def test_delivery_panel_follows_the_delivery_month_across_a_rank_roll():
    quotes = pd.DataFrame({
        "quote_date": pd.to_datetime(["2026-01-30", "2026-02-02"]),
        "TTFc1": [90.0, 101.0],
        "TTFc2": [100.0, 110.0],
    })
    panel = qd.build_delivery_panel(quotes, SOURCE_HASH)
    march = panel[panel["contract_identifier"] == "2026-03"]
    assert march["source_column"].tolist() == ["TTFc2", "TTFc1"]
    assert march["price_eur_mwh"].tolist() == [100.0, 101.0]
    assert march["delivery_start"].unique().tolist() == [pd.Timestamp("2026-03-01")]
    assert march["delivery_end"].unique().tolist() == [pd.Timestamp("2026-04-01")]
    returns = qd.build_returns(panel)
    march_return = returns[returns["contract_identifier"] == "2026-03"].iloc[0]
    assert march_return["elapsed_calendar_days"] == 3
    assert march_return["start_source_column"] == "TTFc2"
    assert march_return["end_source_column"] == "TTFc1"
    assert march_return["log_return"] == pytest.approx(math.log(1.01))


def test_a_missing_observation_breaks_the_return_chain():
    quotes = pd.DataFrame({
        "quote_date": pd.to_datetime(["2026-01-28", "2026-01-29", "2026-01-30"]),
        "TTFc1": [10.0, np.nan, 12.0],
    })
    panel = qd.build_delivery_panel(quotes, SOURCE_HASH)
    assert panel["validity_flag"].tolist() == ["valid", "missing", "valid"]
    assert qd.build_returns(panel).empty


def test_non_positive_quotes_are_retained_and_flagged():
    quotes = pd.DataFrame({
        "quote_date": pd.to_datetime(["2026-01-28", "2026-01-29", "2026-01-30"]),
        "TTFc1": [10.0, 0.0, -1.0],
    })
    panel = qd.build_delivery_panel(quotes, SOURCE_HASH)
    assert panel["validity_flag"].tolist() == ["valid", "non_positive", "non_positive"]
    assert qd.build_returns(panel).empty


def test_duplicate_delivery_observations_are_refused_before_fitting():
    quotes = pd.DataFrame({
        "quote_date": pd.to_datetime(["2026-01-30", "2026-01-30"]),
        "TTFc1": [10.0, 11.0],
    })
    panel = qd.build_delivery_panel(quotes, SOURCE_HASH)
    with pytest.raises(ValueError, match="duplicate observation"):
        qd.build_returns(panel)


def test_manifest_records_source_range_and_data_quality_counts():
    quotes = pd.DataFrame({
        "quote_date": pd.to_datetime(["2026-01-30", "2026-01-30", "2026-02-02"]),
        "TTFc1": [10.0, np.nan, 0.0],
        "TTFc2": [20.0, 21.0, 22.0],
    })
    provenance = dict(source_path="synthetic.xlsx", source_sha256=SOURCE_HASH,
                      byte_count=123, cleaning_version=1, cache="rebuilt")
    manifest = qd.build_data_manifest(quotes, provenance)
    assert manifest["quote_date_min"] == "2026-01-30T00:00:00"
    assert manifest["quote_date_max"] == "2026-02-02T00:00:00"
    assert manifest["duplicate_quote_dates"] == 1
    assert manifest["missing_observations"] == 1
    assert manifest["non_positive_observations"] == 1
    assert manifest["contract_columns"] == ["TTFc1", "TTFc2"]
    assert manifest["units"] == "EUR/MWh"
    assert manifest["as_of_date"] == "2026-02-02T00:00:00"
    assert manifest["retrieval_date"] is None
    assert manifest["delivery_interval"] == "delivery_start inclusive, delivery_end exclusive"


def test_empty_returns_keep_the_declared_schema():
    quotes = pd.DataFrame({
        "quote_date": pd.to_datetime(["2026-01-28"]),
        "TTFc1": [10.0],
    })
    returns = qd.build_returns(qd.build_delivery_panel(quotes, SOURCE_HASH))
    assert returns.empty
    assert "log_return" in returns.columns


def test_flat_forward_delivery_loading_reproduces_the_review_example():
    kappa, sigma, width = 1.0, 0.5, 1.0 / 12.0
    first = dm.flat_forward_delivery_loading(kappa, 0.0, 0.5-width, 0.5)
    second = dm.flat_forward_delivery_loading(kappa, 0.0, 1.0-width, 1.0)
    delivery_spread_vol = sigma * (first-second)
    point_spread_vol = sigma * (math.exp(-0.5)-math.exp(-1.0))
    assert point_spread_vol == pytest.approx(0.11932560927059555)
    assert delivery_spread_vol == pytest.approx(0.12443854388643175)
    assert delivery_spread_vol > point_spread_vol


def test_delivery_loading_has_the_zero_kappa_and_point_limits():
    assert dm.flat_forward_delivery_loading(0.0, 0.0, 0.5, 1.0) == 1.0
    point = math.exp(-1.2 * 0.75)
    averaged = dm.flat_forward_delivery_loading(1.2, 0.0, 0.75, 0.75000001)
    assert averaged == pytest.approx(point, rel=1e-8)


def test_discrete_delivery_loading_is_price_and_volume_weighted():
    got = dm.delivery_averaged_loading(1.0, 0.0, [0.5, 1.0], [20.0, 40.0], [2.0, 1.0])
    expected = (40.0*math.exp(-0.5)+40.0*math.exp(-1.0))/80.0
    assert got == pytest.approx(expected)


def test_delivery_observation_jacobian_matches_finite_differences():
    chi, xi = 0.2, -0.1
    loadings = np.array([0.8, 0.4])
    forwards = np.array([20.0, 30.0])
    weights = np.array([1.0, 2.0])
    value, jac = dm.log_delivery_forward_and_jacobian(
        chi, xi, loadings, forwards, weights)
    eps = 1e-6
    up_chi = dm.log_delivery_forward_and_jacobian(
        chi+eps, xi, loadings, forwards, weights)[0]
    down_chi = dm.log_delivery_forward_and_jacobian(
        chi-eps, xi, loadings, forwards, weights)[0]
    up_xi = dm.log_delivery_forward_and_jacobian(
        chi, xi+eps, loadings, forwards, weights)[0]
    down_xi = dm.log_delivery_forward_and_jacobian(
        chi, xi-eps, loadings, forwards, weights)[0]
    assert np.isfinite(value)
    assert jac[0] == pytest.approx((up_chi-down_chi)/(2*eps), rel=1e-8)
    assert jac[1] == pytest.approx((up_xi-down_xi)/(2*eps), rel=1e-8)
