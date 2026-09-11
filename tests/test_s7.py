"""Independent checks for the executable S7 model comparison."""
import math

import numpy as np
import pandas as pd
import pytest

import calibration as cal


def _return_rows():
    return pd.DataFrame({
        "contract_identifier": ["2026-03", "2026-08", "2027-02", "2028-02"],
        "return_start": pd.to_datetime(["2026-01-30"] * 4),
        "return_end": pd.to_datetime(["2026-02-02"] * 4),
        "year_fraction": [3/365.25] * 4,
        "log_return": [0.01, 0.02, -0.01, 0.005],
        "end_source_column": ["TTFc1", "TTFc6", "TTFc12", "TTFc24"],
    })


def test_snapshot_selection_preserves_actual_time_and_delivery_periods():
    snapshots = cal.make_snapshots(
        _return_rows(), "2026-02-01", "2026-02-03", [1, 6, 12, 24], 4)
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.dt == pytest.approx(3/365.25)
    assert snapshot.source_columns == ("TTFc1", "TTFc6", "TTFc12", "TTFc24")
    assert snapshot.delivery_start_offsets[0] == pytest.approx(30/365.25)
    assert snapshot.delivery_end_offsets[0] == pytest.approx(61/365.25)


def test_two_factor_covariance_matches_the_declared_algebra():
    snapshot = cal.make_snapshots(
        _return_rows(), "2026-02-01", "2026-02-03", [1, 6, 12, 24], 4)[0]
    parameters = dict(kappa=1.0, sigma_chi=0.6, sigma_xi=0.3,
                      rho=-0.4, observation_sigma=0.01)
    got = cal.covariance_matrix("two_factor_correlated", parameters, snapshot)
    a = cal._delivery_loadings(1.0, snapshot)
    expected_process = (
        0.6**2*np.outer(a, a) + 0.3**2*np.ones((4, 4))
        - 0.4*0.6*0.3*(np.outer(a, np.ones(4)) + np.outer(np.ones(4), a)))
    expected = snapshot.dt*expected_process + 2*0.01**2*np.eye(4)
    assert np.allclose(got, expected)
    assert np.linalg.eigvalsh(got).min() > 0.0


def test_actual_elapsed_time_scales_process_but_not_measurement_noise():
    one_day = cal.make_snapshots(
        _return_rows().assign(
            return_end=pd.Timestamp("2026-01-31"), year_fraction=1/365.25),
        "2026-01-31", "2026-01-31", [1, 6, 12, 24], 4)[0]
    three_day = cal.make_snapshots(
        _return_rows(), "2026-02-01", "2026-02-03", [1, 6, 12, 24], 4)[0]
    parameters = dict(kappa=1.0, sigma_chi=0.6, observation_sigma=0.01)
    c1 = cal.covariance_matrix("one_factor", parameters, one_day)
    c3 = cal.covariance_matrix("one_factor", parameters, three_day)
    noise = 2*0.01**2*np.eye(4)
    # Loadings differ slightly because the observation dates differ; recompute
    # the process matrices and test their own declared dt scaling directly.
    a1 = cal._delivery_loadings(1.0, one_day)
    a3 = cal._delivery_loadings(1.0, three_day)
    assert np.allclose(c1-noise, one_day.dt*0.6**2*np.outer(a1, a1))
    assert np.allclose(c3-noise, three_day.dt*0.6**2*np.outer(a3, a3))


def test_likelihood_prefers_the_generating_one_factor_covariance():
    template = cal.make_snapshots(
        _return_rows(), "2026-02-01", "2026-02-03", [1, 6, 12, 24], 4)[0]
    truth = dict(kappa=0.8, sigma_chi=0.7, observation_sigma=0.006)
    covariance = cal.covariance_matrix("one_factor", truth, template)
    rng = np.random.default_rng(20260911)
    snapshots = [cal.ReturnSnapshot(
        template.return_start, template.return_end, template.dt,
        rng.multivariate_normal(np.zeros(4), covariance),
        template.delivery_start_offsets, template.delivery_end_offsets,
        template.source_columns) for _ in range(500)]
    true_nll = cal.negative_log_likelihood(cal._encode("one_factor", truth),
                                           "one_factor", snapshots)
    wrong = dict(kappa=4.0, sigma_chi=0.15, observation_sigma=0.04)
    wrong_nll = cal.negative_log_likelihood(cal._encode("one_factor", wrong),
                                            "one_factor", snapshots)
    assert math.isfinite(true_nll)
    assert true_nll < wrong_nll


@pytest.mark.parametrize("ranks", [[], [0, 1], [1, 1]])
def test_rank_selection_rejects_an_invalid_fit_definition(ranks):
    with pytest.raises(ValueError, match="distinct positive"):
        cal.select_calibration_returns(_return_rows(), ranks)
