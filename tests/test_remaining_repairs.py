"""Focused checks for independently reproduced gaps beyond the original pack."""
import math

import numpy as np
import pandas as pd
import pytest

import benchmarks as b
import quote_data as qd
import storage_model as sm
import two_factor_probe as probe


def _stable_table(grids, refused=None):
    table = pd.DataFrame({"n_states": grids, "total_eur": [1e6]*len(grids),
                          "intrinsic_eur": [1e6]*len(grids)})
    if refused is not None:
        table["refused"] = pd.Series(refused, dtype=object)
    return table


def test_convergence_requires_two_connected_successive_doublings():
    table = _stable_table([240,480,960,1920,3840],
                          [None,None,"refused",None,None])
    status, _, message = b.convergence_verdict(table)
    assert status == "insufficient"
    assert "connected" in message


@pytest.mark.parametrize("bad", [[240.9,480.9,960.9], [0,1,2], [240,np.inf,960]])
def test_convergence_requires_finite_positive_integer_grid_counts(bad):
    status, _, _ = b.convergence_verdict(_stable_table(bad))
    assert status == "invalid"


def test_a_corrupt_derived_cache_is_rebuilt_from_the_source(tmp_path):
    source = tmp_path/"source.xlsx"
    pd.DataFrame({"quote_date":[pd.Timestamp("2026-01-05")],
                  "TTFc1":[999.0]}).to_excel(source,index=False)
    first, _ = qd.load_quote_matrix(source, cache_dir=tmp_path)
    cache = next(tmp_path.glob("quote_matrix_*.parquet"))
    cache.write_bytes(b"corrupt cache fixture")
    rebuilt, provenance = qd.load_quote_matrix(source, cache_dir=tmp_path)
    assert rebuilt.equals(first)
    assert provenance["cache"] == "rebuilt after invalid cache"


def test_zero_volatility_ou_is_a_deterministic_single_state():
    nodes, transition = probe._ou_lattice(1.0, 0.0)
    assert np.array_equal(nodes, [0.0])
    assert np.array_equal(transition, [[1.0]])


def test_zero_mean_reversion_terminal_anchor_uses_the_brownian_limit():
    got = probe.matched_sig_chi(0.1, "terminal", kappa=0.0)
    assert got == pytest.approx(math.sqrt(0.6**2-0.1**2))


@pytest.mark.parametrize("bad", [60.9, True, np.inf, "sixty"])
def test_storage_grid_count_is_a_positive_integer(bad):
    with pytest.raises(ValueError, match="positive integer"):
        sm.normalise_storage_contract(dict(
            product_type="storage", n_states=bad, capacity_mwh=600000.0,
            inj_days=30, wdr_days=60, initial_storage_mwh=0.0,
            terminal_storage_mwh=0.0))


def test_conflicting_explicit_clip_rate_is_refused():
    with pytest.raises(ValueError, match="conflicts"):
        sm.normalise_storage_contract(dict(
            product_type="storage", n_states=60, capacity_mwh=600000.0,
            inj_days=30, wdr_days=60, inj_rate=99,
            initial_storage_mwh=0.0, terminal_storage_mwh=0.0))
