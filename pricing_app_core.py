"""Pure input and output helpers for the unified pricing application.

The Streamlit page collects physical term-sheet quantities.  This module is
the tested boundary that converts them into the discrete parameters used by
``storage_model.run_valuation``.  Keeping it free of Streamlit imports makes
the conversion rules independently testable.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd

from quote_data import parse_quote_bytes, source_fingerprint
from storage_model import (
    normalise_rate_parameters,
    normalise_storage_contract,
    resolve_grid,
)


PRODUCT_TYPES = ("put_swing", "call_swing", "storage")


def _finite_number(value, name, *, minimum=None, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number, got {value!r}.") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}, got {number:g}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum:g}, got {number:g}.")
    return number


def _positive_integer(value, name, *, allow_zero=False):
    number = _finite_number(value, name, minimum=0 if allow_zero else 1)
    if isinstance(value, (bool, np.bool_)) or not number.is_integer():
        word = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {word} integer, got {value!r}.")
    return int(number)


def contract_columns(quotes):
    """Return the TTFc<N> columns in numeric rank order."""
    import re

    columns = [c for c in quotes.columns if re.fullmatch(r"TTFc\d+", str(c))]
    if not columns:
        raise ValueError("No TTFc<N> forward-contract columns were found.")
    return sorted(columns, key=lambda c: int(re.search(r"\d+", str(c)).group()))


def read_uploaded_table(raw_bytes, file_name):
    """Read an uploaded CSV or Excel table without applying product semantics."""
    if not raw_bytes:
        raise ValueError(f"{file_name or 'Uploaded file'} is empty.")
    buffer = io.BytesIO(raw_bytes)
    if str(file_name).lower().endswith(".csv"):
        return pd.read_csv(buffer)
    return pd.read_excel(buffer)


def prepare_quote_matrix(raw_bytes, file_name):
    """Parse a quote matrix using the shared S6 cleaning rule."""
    quotes, stats = parse_quote_bytes(raw_bytes, file_name)
    return quotes, contract_columns(quotes), {
        "source_name": str(file_name),
        "source_sha256": source_fingerprint(raw_bytes),
        **stats,
    }


def prepare_local_quote_matrix(path):
    """Read a bundled quote matrix through the same byte-based path as uploads."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find local {str(path)!r}.")
    return prepare_quote_matrix(path.read_bytes(), path.name)


def prepare_direct_curve(frame):
    """Validate and normalise a direct contract curve."""
    frame = frame.rename(columns={c: str(c).strip() for c in frame.columns})
    required = {"contractStart", "contractEnd", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Direct curve file is missing columns: " + ", ".join(sorted(missing))
        )
    curve = frame[["contractStart", "contractEnd", "value"]].copy()
    curve["contractStart"] = pd.to_datetime(curve["contractStart"], format="mixed")
    curve["contractEnd"] = pd.to_datetime(curve["contractEnd"], format="mixed")
    curve["value"] = pd.to_numeric(curve["value"], errors="coerce")
    if curve.empty:
        raise ValueError("Direct curve contains no rows.")
    if curve.isna().any().any():
        raise ValueError("Direct curve contains a missing or unparseable date/price.")
    if (curve["contractEnd"] < curve["contractStart"]).any():
        raise ValueError("Direct curve contains a contractEnd before contractStart.")
    if not np.isfinite(curve["value"].to_numpy(dtype=float)).all():
        raise ValueError("Direct curve prices must be finite numbers.")
    return curve.sort_values(["contractStart", "contractEnd"]).reset_index(drop=True)


def parse_ratchet_table(frame):
    """Convert an editable ratchet table to the model's tuple representation."""
    frame = frame.rename(columns={c: str(c).strip().lower() for c in frame.columns})
    required = ["fullness", "injection", "withdrawal"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError("Ratchet table is missing columns: " + ", ".join(sorted(missing)))
    table = frame[required].dropna(how="all").copy()
    if table.empty:
        raise ValueError("Ratchets are enabled but the ratchet table is empty.")
    if table.isna().any().any():
        raise ValueError("Every ratchet row needs fullness, injection and withdrawal values.")
    for column in required:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table.isna().any().any() or not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError("Ratchet values must be finite numbers.")

    fullness = table["fullness"].to_numpy(dtype=float)
    if fullness.max() > 1.5:
        fullness = fullness / 100.0
    if (fullness < 0.0).any() or (fullness > 1.0).any():
        raise ValueError("Ratchet fullness must be between 0 and 1, or between 0 and 100 percent.")
    if pd.Series(fullness).duplicated().any():
        raise ValueError("Ratchet fullness levels must be unique.")
    injection = table["injection"].to_numpy(dtype=float)
    withdrawal = table["withdrawal"].to_numpy(dtype=float)
    if (injection < 0.0).any() or (withdrawal < 0.0).any():
        raise ValueError("Ratchet multipliers cannot be negative.")
    order = np.argsort(fullness)
    return fullness[order], injection[order], withdrawal[order]


def parse_inventory_bounds(frame):
    """Return dated minimum/maximum inventory mappings from an editable table."""
    frame = frame.rename(columns={c: str(c).strip().lower() for c in frame.columns})
    required = {"date", "minimum", "maximum"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Inventory-bound table is missing columns: " + ", ".join(sorted(missing))
        )
    table = frame[["date", "minimum", "maximum"]].dropna(how="all").copy()
    if table.empty:
        return {}, {}
    if table["date"].isna().any():
        raise ValueError("Every inventory-bound row needs a date.")
    table["date"] = pd.to_datetime(table["date"], format="mixed")

    minimum, maximum = {}, {}
    for row in table.itertuples(index=False):
        stamp = pd.Timestamp(row.date)
        key = stamp.strftime("%Y-%m-%d")
        if pd.isna(row.minimum) and pd.isna(row.maximum):
            raise ValueError(f"Inventory-bound row {key} has neither a minimum nor a maximum.")
        if not pd.isna(row.minimum):
            if key in minimum:
                raise ValueError(f"Duplicate minimum inventory bound for {key}.")
            minimum[key] = _finite_number(row.minimum, f"minimum[{key}]", minimum=0, maximum=1)
        if not pd.isna(row.maximum):
            if key in maximum:
                raise ValueError(f"Duplicate maximum inventory bound for {key}.")
            maximum[key] = _finite_number(row.maximum, f"maximum[{key}]", minimum=0, maximum=1)
        if key in minimum and key in maximum and minimum[key] > maximum[key]:
            raise ValueError(f"Minimum inventory exceeds maximum inventory on {key}.")
    return minimum, maximum


def build_pricing_params(values, *, ratchets=None, inventory_bounds=None):
    """Validate physical UI values and build one canonical valuation dictionary.

    Returns ``(params, effective)``.  ``effective`` contains the grid quantities
    worth showing to the user before a valuation is relied on.
    """
    product_type = str(values.get("product_type", "")).strip()
    if product_type not in PRODUCT_TYPES:
        raise ValueError(f"product_type must be one of {PRODUCT_TYPES}, got {product_type!r}.")

    val_date = pd.Timestamp(values["valDate"])
    start = pd.Timestamp(values["storageStart"])
    end = pd.Timestamp(values["storageEnd"])
    if end < start:
        raise ValueError("Contract end must be on or after contract start.")
    if start < val_date:
        raise ValueError("Contract start must be on or after the valuation date.")
    fd_date = pd.Timestamp(values.get("FDDate", val_date))
    if values.get("uses_quote_matrix") and fd_date > val_date:
        raise ValueError("Forward-curve date cannot be after the valuation date.")

    params = {
        "product_type": product_type,
        "FDDate": fd_date,
        "valDate": val_date,
        "storageStart": start,
        "storageEnd": end,
        "vol": _finite_number(values["vol"], "vol", minimum=0),
        "sMR": _finite_number(values["sMR"], "sMR", minimum=0),
        "n_p_full": _positive_integer(values["n_p_full"], "n_p_full", allow_zero=True),
        "run_intrinsic": bool(values.get("run_intrinsic", True)),
    }

    rate_inputs = {
        key: values[key]
        for key in ("discount_rate", "borrow_rate", "invest_rate", "funding_direction")
        if key in values and values[key] is not None
    }
    params.update(normalise_rate_parameters(rate_inputs))

    capacity = _finite_number(values["capacity_mwh"], "capacity_mwh", minimum=1e-12)
    if product_type == "storage":
        if "borrow_rate" in params:
            raise ValueError(
                "Storage has both payment and receipt cash flows. Use one market/valuation "
                "discount rate; the one-direction treasury scenario is not valid for storage."
            )
        n_states = _positive_integer(values["n_states"], "n_states")
        physical = {
            "capacity_mwh": capacity,
            "n_states": n_states,
            "inj_days": _finite_number(values["inj_days"], "inj_days", minimum=1e-12),
            "wdr_days": _finite_number(values["wdr_days"], "wdr_days", minimum=1e-12),
            "initial_storage_mwh": _finite_number(
                values.get("initial_storage_mwh", 0), "initial_storage_mwh", minimum=0
            ),
            "terminal_storage_mwh": _finite_number(
                values.get("terminal_storage_mwh", 0), "terminal_storage_mwh", minimum=0
            ),
        }
        derived = normalise_storage_contract(physical)
        params.update(physical)
        params.update(derived)
        params.update({
            "daily_max": None,
            "clips_per_day": max(derived["inj_rate"], derived["wdr_rate"]),
            "inj_cost": _finite_number(values.get("inj_cost", 0), "inj_cost"),
            "wdr_cost": _finite_number(values.get("wdr_cost", 0), "wdr_cost"),
            "fuel_loss": _finite_number(values.get("fuel_loss", 0), "fuel_loss", minimum=0, maximum=0.999999),
            "ratchets": ratchets,
            "max_ratchet_rate_loss": _finite_number(
                values.get("max_ratchet_rate_loss", 0.10),
                "max_ratchet_rate_loss",
                minimum=0,
                maximum=1,
            ),
        })
        if inventory_bounds:
            minimum, maximum = inventory_bounds
            if minimum:
                params["min_inventory"] = dict(minimum)
            if maximum:
                params["max_inventory"] = dict(maximum)
        effective = {
            "capacity_mwh": capacity,
            "n_states": n_states,
            "v_step_mwh": derived["v_step"],
            "injection_clips_per_day": derived["inj_rate"],
            "withdrawal_clips_per_day": derived["wdr_rate"],
            "injection_mwh_per_day": derived["inj_rate"] * derived["v_step"],
            "withdrawal_mwh_per_day": derived["wdr_rate"] * derived["v_step"],
            "initial_inventory_clips": derived["initial_inv_clips"],
            "terminal_inventory_clips": derived["terminal_inv_clips"],
        }
        return params, effective

    daily_max = _finite_number(values["daily_max"], "daily_max", minimum=1e-12)
    clips_per_day = _positive_integer(values.get("clips_per_day", 1), "clips_per_day")
    params.update({
        "capacity_mwh": capacity,
        "daily_max": daily_max,
        "clips_per_day": clips_per_day,
        "strike": _finite_number(values.get("strike", 0), "strike"),
        "zero_penalty": bool(values.get("zero_penalty", False)) if product_type == "call_swing" else False,
    })
    v_step, n_states, _ = resolve_grid(params, "days")
    params["v_step"] = v_step
    effective = {
        "maximum_volume_mwh": capacity,
        "daily_max_mwh": daily_max,
        "n_states": n_states,
        "v_step_mwh": v_step,
        "clips_per_day": clips_per_day,
        "volume_obligation": (
            "optional up to maximum" if params["zero_penalty"] else "mandatory full quantity"
        ),
    }
    return params, effective


def build_output_tables(payload, result, product_type, start, end):
    """Create the daily schedule and monthly hedge tables used on screen/export."""
    date_span = pd.DatetimeIndex(payload["date_span"])
    exercise_dates = date_span[payload["Dt"]:payload["active"]]
    delta_dates = date_span[:payload["n_t"]]
    prices = pd.Series(np.asarray(payload["price_curve"], dtype=float), index=date_span)
    intrinsic = pd.Series(
        np.asarray(result["intrinsic_profile_raw"], dtype=float)[:len(date_span)],
        index=date_span,
    )
    stochastic = pd.Series(
        np.asarray(result["extrinsic_profile_raw"], dtype=float)[:len(date_span)],
        index=date_span,
    )
    physical_delta = pd.Series(
        np.asarray(payload["physical_delta"], dtype=float)[:len(delta_dates)],
        index=delta_dates,
    )
    pv_tailed_delta = pd.Series(
        np.asarray(payload["pv_tailed_delta"], dtype=float)[:len(delta_dates)],
        index=delta_dates,
    )

    daily = pd.DataFrame(index=date_span)
    daily.index.name = "date"
    daily["forward_eur_mwh"] = prices
    daily["intrinsic_expected_volume_mwh"] = intrinsic
    daily["stochastic_expected_volume_mwh"] = stochastic
    daily["physical_forward_delta_mwh"] = physical_delta.reindex(date_span)
    daily["pv_tailed_forward_delta_mwh"] = pv_tailed_delta.reindex(date_span)
    daily["active"] = daily.index.isin(exercise_dates)
    daily = daily.loc[pd.Timestamp(start):pd.Timestamp(end)].reset_index()

    monthly = pd.concat(
        {
            "physical_forward_delta_mwh": physical_delta.loc[
                pd.Timestamp(start):pd.Timestamp(end)
            ].resample("MS").sum(),
            "pv_tailed_forward_delta_mwh": pv_tailed_delta.loc[
                pd.Timestamp(start):pd.Timestamp(end)
            ].resample("MS").sum(),
        },
        axis=1,
    ).reset_index(names="month")
    monthly["product_type"] = product_type
    return daily, monthly
