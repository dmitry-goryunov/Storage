"""Independent checks for the unified pricing application's I/O boundary."""

import numpy as np
import pandas as pd
import pytest

from pricing_app_core import (
    build_output_tables,
    build_pricing_params,
    parse_inventory_bounds,
    parse_ratchet_table,
    prepare_direct_curve,
)


def base_values(product_type="storage"):
    return {
        "product_type": product_type,
        "FDDate": "2026-01-05",
        "valDate": "2026-01-05",
        "storageStart": "2026-04-01",
        "storageEnd": "2027-03-30",
        "uses_quote_matrix": False,
        "vol": 0.60,
        "sMR": 1.0,
        "n_p_full": 30,
        "run_intrinsic": True,
        "discount_rate": 0.04,
        "capacity_mwh": 600_000.0,
        "n_states": 60,
        "inj_days": 30.0,
        "wdr_days": 60.0,
        "initial_storage_mwh": 100_000.0,
        "terminal_storage_mwh": 20_000.0,
        "inj_cost": 0.55,
        "wdr_cost": 0.45,
        "fuel_loss": 0.012,
        "max_ratchet_rate_loss": 0.08,
    }


def test_storage_term_sheet_maps_to_exact_asymmetric_grid():
    ratchets = (
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    )
    bounds = ({"2026-10-01": 0.70}, {"2027-03-01": 0.30})

    params, effective = build_pricing_params(
        base_values(), ratchets=ratchets, inventory_bounds=bounds
    )

    assert params["v_step"] == 10_000.0
    assert params["inj_rate"] == 2
    assert params["wdr_rate"] == 1
    assert params["initial_inv_clips"] == 10
    assert params["terminal_inv_clips"] == 2
    assert params["fuel_loss"] == 0.012
    assert params["ratchets"] is ratchets
    assert params["min_inventory"] == bounds[0]
    assert params["max_inventory"] == bounds[1]
    assert effective["injection_mwh_per_day"] == 20_000.0
    assert effective["withdrawal_mwh_per_day"] == 10_000.0


def test_storage_refuses_grid_that_changes_the_requested_rate():
    values = base_values()
    values.update({"n_states": 30, "wdr_days": 90.0})

    with pytest.raises(ValueError, match="smallest grid.*n_states=90"):
        build_pricing_params(values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("initial_storage_mwh", 5_000.0, "initial_storage_mwh"),
        ("terminal_storage_mwh", 605_000.0, "terminal_storage_mwh"),
    ],
)
def test_storage_refuses_unrepresentable_or_out_of_range_inventory(
    field, value, message
):
    values = base_values()
    values[field] = value

    with pytest.raises(ValueError, match=message):
        build_pricing_params(values)


def test_call_swing_preserves_strike_treasury_scenario_and_optionality():
    values = base_values("call_swing")
    for key in (
        "discount_rate",
        "n_states",
        "inj_days",
        "wdr_days",
        "initial_storage_mwh",
        "terminal_storage_mwh",
    ):
        values.pop(key)
    values.update(
        {
            "daily_max": 20_000.0,
            "clips_per_day": 2,
            "strike": 31.25,
            "zero_penalty": True,
            "borrow_rate": 0.07,
            "invest_rate": 0.03,
            "funding_direction": "invest",
        }
    )

    params, effective = build_pricing_params(values)

    assert params["strike"] == 31.25
    assert params["zero_penalty"] is True
    assert params["invest_rate"] == 0.03
    assert params["funding_direction"] == "invest"
    assert params["v_step"] == 10_000.0
    assert effective["n_states"] == 60
    assert effective["volume_obligation"] == "optional up to maximum"


def test_storage_rejects_one_direction_treasury_scenario():
    values = base_values()
    values.pop("discount_rate")
    values.update(
        {
            "borrow_rate": 0.07,
            "invest_rate": 0.03,
            "funding_direction": "borrow",
        }
    )
    with pytest.raises(ValueError, match="not valid for storage"):
        build_pricing_params(values)


def test_quote_date_cannot_be_after_valuation_date():
    values = base_values()
    values.update(
        {
            "uses_quote_matrix": True,
            "FDDate": "2026-01-06",
            "valDate": "2026-01-05",
        }
    )
    with pytest.raises(ValueError, match="cannot be after"):
        build_pricing_params(values)


def test_ratchets_accept_percent_fullness_sort_and_validate():
    frame = pd.DataFrame(
        {
            "Fullness": [100, 0, 50],
            "Injection": [0, 1, 0.8],
            "Withdrawal": [1, 0, 0.7],
        }
    )
    fullness, injection, withdrawal = parse_ratchet_table(frame)
    np.testing.assert_allclose(fullness, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(injection, [1.0, 0.8, 0.0])
    np.testing.assert_allclose(withdrawal, [0.0, 0.7, 1.0])

    frame.loc[2, "Fullness"] = 0
    with pytest.raises(ValueError, match="unique"):
        parse_ratchet_table(frame)


def test_inventory_bounds_allow_one_sided_rows_and_refuse_crossed_bounds():
    frame = pd.DataFrame(
        {
            "date": ["2026-10-01", "2027-03-01"],
            "minimum": [0.70, np.nan],
            "maximum": [np.nan, 0.30],
        }
    )
    minimum, maximum = parse_inventory_bounds(frame)
    assert minimum == {"2026-10-01": 0.70}
    assert maximum == {"2027-03-01": 0.30}

    crossed = pd.DataFrame(
        {"date": ["2026-10-01"], "minimum": [0.8], "maximum": [0.7]}
    )
    with pytest.raises(ValueError, match="exceeds"):
        parse_inventory_bounds(crossed)


def test_direct_curve_normalisation_and_validation():
    frame = pd.DataFrame(
        {
            " contractEnd ": ["2026-02-28", "2026-01-31"],
            "contractStart": ["2026-02-01", "2026-01-01"],
            "value": ["31.5", "30.0"],
        }
    )
    curve = prepare_direct_curve(frame)
    assert curve["contractStart"].tolist() == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-02-01"),
    ]
    assert curve["value"].tolist() == [30.0, 31.5]

    bad = frame.copy()
    bad.loc[0, "value"] = "not-a-price"
    with pytest.raises(ValueError, match="unparseable"):
        prepare_direct_curve(bad)


def test_output_tables_keep_physical_and_pv_tailed_delta_separate():
    dates = pd.date_range("2026-01-01", "2026-01-31")
    payload = {
        "date_span": dates,
        "Dt": 4,
        "active": 20,
        "n_t": 30,
        "price_curve": np.arange(len(dates), dtype=float) + 20,
        "physical_delta": np.ones(30),
        "pv_tailed_delta": np.full(30, 0.9),
    }
    result = {
        "intrinsic_profile_raw": np.full(len(dates), 2.0),
        "extrinsic_profile_raw": np.full(len(dates), 3.0),
    }

    daily, monthly = build_output_tables(
        payload, result, "storage", "2026-01-05", "2026-01-20"
    )

    assert list(daily.columns)[-3:-1] == [
        "physical_forward_delta_mwh",
        "pv_tailed_forward_delta_mwh",
    ]
    assert daily["physical_forward_delta_mwh"].sum() == 16.0
    assert daily["pv_tailed_forward_delta_mwh"].sum() == pytest.approx(14.4)
    assert monthly.loc[0, "physical_forward_delta_mwh"] == 16.0
    assert monthly.loc[0, "pv_tailed_forward_delta_mwh"] == pytest.approx(14.4)
