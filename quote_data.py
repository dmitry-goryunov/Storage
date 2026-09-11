"""Shared TTF quote-matrix loading and cleaning.

One cleaner, one cache-identity policy, used by `benchmarks.py` and
`portfolio_app.py` -- IMPLEMENTATION-GUIDE-2026-09-11.md S3 (checklist items
7-9). Deliberately does NOT import `portfolio_app` or `streamlit`: that module
configures a Streamlit page and has import-time side effects, and this one is
imported by a plain script (`benchmarks.py`) and by `pytest` as well.

Until 2026-09-11 the cache was validated by MODIFICATION TIME, against a
SINGLE fixed path (`benchmarks._PARQUET_CACHE`) regardless of what workbook
was actually requested. Reproduced independently: a synthetic one-row
workbook with `TTFc1=999`, given an older modification time than the
repository's own cache, returned the repository's 4,171 rows with `TTFc1=10`
-- a request for a different dataset silently analysed the original one. A
same-path edit with its modification time restored (routine after some
editor saves, and after certain version-control checkouts) stayed silently
stale for the same reason.

Both are the same defect class `storage_model.normalise_storage_contract`
fixed for grid sizing: a lookup that trusts something OTHER than the content
it is supposed to represent -- a path string or a timestamp, not the bytes.
The fix here is the same shape: identify the source by a SHA-256 of its
actual bytes and address the cache BY that fingerprint, so two different
files can never collide on one cache entry and a byte-identical re-read of
the same content always finds its own cache regardless of path or mtime.
"""
import hashlib
import io
import math
import os
import re

import pandas as pd

from storage_model import front_month_start, month_end

#: Bump when `clean_quote_matrix`'s logic changes. Baked into the cache
#: filename, so a cleaning change invalidates every existing cache entry by
#: construction -- there is no separate "is this cache still valid" check to
#: get wrong, because a stale-rule cache simply is not the file being looked
#: for any more.
CLEANING_VERSION = 1


def source_fingerprint(data):
    """SHA-256 hex digest of file bytes -- `data` is bytes-like or a path."""
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    else:
        with open(data, "rb") as handle:
            raw = handle.read()
    return hashlib.sha256(raw).hexdigest()


def clean_quote_matrix(raw):
    """The one cleaning rule, shared so the cache and a fresh parse can never
    disagree about what "clean" means: first column -> `quote_date`, drop rows
    with no date, coerce every other column to numeric (junk strings such as
    "Retrieving..." become NaN), sort by date.

    Returns `(cleaned, stats)`. `stats` records what changed rather than
    silently inventing or discarding observations: `rows_raw`, `rows_dated`
    (after dropping unparseable dates), `duplicate_dates` (quote_date values
    seen more than once, a data-quality fact for a calibration to decide about,
    not something this function resolves), and `rejected_cells` (numeric
    coercion failures across all non-date columns).
    """
    quotes = raw.rename(columns={raw.columns[0]: "quote_date"})
    rows_raw = len(quotes)
    quotes = quotes.dropna(subset=["quote_date"]).copy()
    quotes["quote_date"] = pd.to_datetime(quotes["quote_date"], format="mixed")
    rows_dated = len(quotes)
    duplicate_dates = int(quotes["quote_date"].duplicated().sum())

    rejected_cells = 0
    for column in quotes.columns[1:]:
        original = quotes[column]
        coerced = pd.to_numeric(original, errors="coerce")
        was_text = original.notna() & coerced.isna()
        rejected_cells += int(was_text.sum())
        quotes[column] = coerced

    quotes = quotes.sort_values("quote_date").reset_index(drop=True)
    stats = dict(rows_raw=rows_raw, rows_dated=rows_dated,
                duplicate_dates=duplicate_dates, rejected_cells=rejected_cells)
    return quotes, stats


def parse_quote_bytes(raw_bytes, file_name):
    """Parse+clean raw file bytes (xlsx or csv) by extension. Returns the
    same `(cleaned, stats)` shape as `clean_quote_matrix`."""
    buf = io.BytesIO(raw_bytes)
    raw = pd.read_csv(buf) if str(file_name).lower().endswith(".csv") else pd.read_excel(buf)
    return clean_quote_matrix(raw)


def _cache_path(cache_dir, fingerprint):
    return os.path.join(cache_dir, f"quote_matrix_v{CLEANING_VERSION}_{fingerprint}.parquet")


def load_quote_matrix(source_path, cache_dir=None, use_cache=True):
    """Load and clean a TTF-style quote matrix, with a content-addressed cache.

    Returns `(quotes, provenance)`. `provenance` carries `source_path`
    (resolved), `source_sha256`, `format`, `byte_count`, `cleaning_version`,
    `cache` ("hit" / "rebuilt" / "rebuilt (not persisted)" / "n/a (explicit
    parquet source)"), and the cleaning `stats` above when a workbook was
    actually parsed. Make it part of any research output or valuation
    manifest that depends on this data -- an `n_states` that reprices a
    contract needs the same discipline as a dataset that reprices a
    correlation.

    The cache lives at `<cache_dir>/quote_matrix_v<version>_<sha256>.parquet`:
    content-addressed, so two different sources can never collide on one
    cache entry, and there is no timestamp or metadata sidecar to fall out of
    sync with what it is meant to validate -- the filename IS the validation.
    `cache_dir` defaults to the source file's own directory. Writing is best
    effort (temp file + atomic replace): a read-only checkout still returns
    valid data, with that recorded in `provenance["cache"]` rather than
    silently swallowed.

    An explicit `.parquet` `source_path` is a source in its own right -- a
    requested derived dataset, not a cache of something else -- and is read
    directly, fingerprinted on its own bytes.

    `use_cache=False` skips the cache entirely, both read and write, for a
    reproducible comparison: the same source and cleaner must give equivalent
    data whether or not caching is enabled.
    """
    source_path = os.fspath(source_path)
    resolved = os.path.abspath(source_path)
    with open(source_path, "rb") as handle:
        raw_bytes = handle.read()
    fingerprint = source_fingerprint(raw_bytes)
    byte_count = len(raw_bytes)

    if source_path.lower().endswith(".parquet"):
        quotes = pd.read_parquet(io.BytesIO(raw_bytes))
        return quotes, dict(source_path=resolved, source_sha256=fingerprint,
                            format="parquet", byte_count=byte_count,
                            cleaning_version=None, cache="n/a (explicit parquet source)")

    cache_dir = os.path.dirname(resolved) if cache_dir is None else cache_dir
    cache_file = _cache_path(cache_dir, fingerprint)

    if use_cache and os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file), dict(
                source_path=resolved, source_sha256=fingerprint, format="xlsx",
                byte_count=byte_count, cleaning_version=CLEANING_VERSION, cache="hit")
        except Exception:
            # A cache is disposable derived data. Rebuild it from the already
            # fingerprinted source bytes rather than turning cache damage into
            # a source-data failure.
            pass

    quotes, stats = parse_quote_bytes(raw_bytes, source_path)
    cache_status = "rebuilt after invalid cache" if use_cache and os.path.exists(cache_file) else "rebuilt"
    if use_cache:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = cache_file + f".tmp{os.getpid()}"
            quotes.to_parquet(tmp)
            os.replace(tmp, cache_file)
        except Exception:
            cache_status = "rebuilt (not persisted)"    # read-only checkout, still valid data

    provenance = dict(source_path=resolved, source_sha256=fingerprint, format="xlsx",
                      byte_count=byte_count, cleaning_version=CLEANING_VERSION,
                      cache=cache_status, **stats)
    return quotes, provenance


def _contract_columns(quotes, columns=None):
    """Return validated TTF continuous-rank columns in rank order."""
    selected = ([c for c in quotes.columns if re.fullmatch(r"TTFc\d+", str(c))]
                if columns is None else list(columns))
    invalid = [c for c in selected if c not in quotes or not re.fullmatch(r"TTFc\d+", str(c))]
    if invalid:
        raise ValueError(f"invalid or missing TTF contract column(s): {invalid}")
    if not selected:
        raise ValueError("no TTFc<N> contract columns were found")
    return sorted(selected, key=lambda c: int(re.search(r"\d+", str(c)).group()))


def build_delivery_panel(quotes, source_hash, columns=None):
    """Map continuous-rank quotes to their actual delivery-month identities.

    Rows with missing or non-positive prices remain in the panel and carry a
    validity flag. Their presence breaks a return interval, preventing a later
    valid quote from being silently differenced across the missing observation.
    """
    if "quote_date" not in quotes:
        raise ValueError("quotes must contain quote_date")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_hash):
        raise ValueError("source_hash must be a SHA-256 hexadecimal digest")
    contract_columns = _contract_columns(quotes, columns)
    wide = quotes[["quote_date", *contract_columns]].copy()
    wide["quote_date"] = pd.to_datetime(wide["quote_date"], errors="coerce")
    if wide["quote_date"].isna().any():
        raise ValueError("quote_date contains an unparseable value")
    panel = wide.melt(id_vars="quote_date", value_vars=contract_columns,
                      var_name="source_column", value_name="price_eur_mwh")
    panel["price_eur_mwh"] = pd.to_numeric(panel["price_eur_mwh"], errors="coerce")
    rank = panel["source_column"].str.extract(r"(\d+)", expand=False).astype(int)
    panel["delivery_start"] = [
        front_month_start(q) + pd.DateOffset(months=int(r) - 1)
        for q, r in zip(panel["quote_date"], rank)
    ]
    # `delivery_end` is exclusive: a February monthly contract is represented
    # as [1 February, 1 March).  That convention makes the integration width
    # include every delivery day and avoids losing the final calendar day when
    # the observation model evaluates the monthly average.
    panel["delivery_end"] = panel["delivery_start"].map(month_end) + pd.DateOffset(days=1)
    panel["contract_identifier"] = panel["delivery_start"].dt.strftime("%Y-%m")
    panel["source_hash"] = source_hash.lower()
    panel["validity_flag"] = "valid"
    panel.loc[panel["price_eur_mwh"].isna(), "validity_flag"] = "missing"
    panel.loc[panel["price_eur_mwh"].notna() & (panel["price_eur_mwh"] <= 0.0),
              "validity_flag"] = "non_positive"
    return panel[["quote_date", "contract_identifier", "delivery_start", "delivery_end",
                  "price_eur_mwh", "source_column", "source_hash", "validity_flag"]].sort_values(
        ["contract_identifier", "quote_date", "source_column"]).reset_index(drop=True)


def build_returns(panel):
    """Create log returns between adjacent valid observations of one delivery.

    Missing and non-positive panel rows break the chain. Weekend and holiday
    gaps are retained with their actual elapsed calendar days.
    """
    required = {"quote_date", "contract_identifier", "price_eur_mwh",
                "source_column", "validity_flag"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"delivery panel is missing required column(s): {sorted(missing)}")
    work = panel.copy()
    work["quote_date"] = pd.to_datetime(work["quote_date"], errors="coerce")
    if work["quote_date"].isna().any():
        raise ValueError("delivery panel contains an unparseable quote_date")
    duplicates = work.duplicated(["contract_identifier", "quote_date"], keep=False)
    if duplicates.any():
        sample = work.loc[duplicates, ["contract_identifier", "quote_date"]].iloc[0]
        raise ValueError(
            f"duplicate observation for contract {sample['contract_identifier']} on "
            f"{sample['quote_date']:%Y-%m-%d}; resolve duplicate source rows before fitting")
    rows = []
    for contract, group in work.sort_values("quote_date").groupby("contract_identifier", sort=True):
        records = list(group.to_dict("records"))
        for left, right in zip(records[:-1], records[1:]):
            if left["validity_flag"] != "valid" or right["validity_flag"] != "valid":
                continue
            elapsed = (right["quote_date"] - left["quote_date"]).days
            if elapsed <= 0:
                raise ValueError(f"non-positive return interval for contract {contract}")
            rows.append({
                "contract_identifier": contract,
                "return_start": left["quote_date"], "return_end": right["quote_date"],
                "elapsed_calendar_days": elapsed, "year_fraction": elapsed / 365.25,
                "start_price_eur_mwh": left["price_eur_mwh"],
                "end_price_eur_mwh": right["price_eur_mwh"],
                "log_return": math.log(right["price_eur_mwh"] / left["price_eur_mwh"]),
                "start_source_column": left["source_column"],
                "end_source_column": right["source_column"],
            })
    columns = [
        "contract_identifier", "return_start", "return_end",
        "elapsed_calendar_days", "year_fraction", "start_price_eur_mwh",
        "end_price_eur_mwh", "log_return", "start_source_column",
        "end_source_column",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_data_manifest(quotes, provenance, columns=None, *, retrieval_date=None,
                        as_of_date=None, units="EUR/MWh"):
    """Return a JSON-serialisable audit record for a calibration dataset."""
    contract_columns = _contract_columns(quotes, columns)
    dates = pd.to_datetime(quotes["quote_date"], errors="coerce")
    numeric = quotes[contract_columns].apply(pd.to_numeric, errors="coerce")
    quote_date_max = None if dates.dropna().empty else dates.max().isoformat()
    retrieval = provenance.get("retrieval_date") if retrieval_date is None else retrieval_date
    as_of = quote_date_max if as_of_date is None else pd.Timestamp(as_of_date).isoformat()
    return {
        "source_path": provenance.get("source_path"),
        "source_sha256": provenance.get("source_sha256"),
        "byte_count": provenance.get("byte_count"),
        "retrieval_date": None if retrieval is None else pd.Timestamp(retrieval).isoformat(),
        "retrieval_date_status": "not recorded by source" if retrieval is None else "recorded",
        "as_of_date": as_of,
        "cleaning_version": provenance.get("cleaning_version"),
        "cache": provenance.get("cache"),
        "quote_date_min": None if dates.dropna().empty else dates.min().isoformat(),
        "quote_date_max": quote_date_max,
        "quote_rows": int(len(quotes)),
        "units": units,
        "contract_columns": [str(c) for c in contract_columns],
        "column_definitions": {
            "quote_date": "date on which the forward quotation was observed",
            "TTFc<N>": "continuous rank N monthly TTF forward; TTFc1 maps to the next calendar month",
            "price": f"monthly forward quotation in {units}",
        },
        "duplicate_quote_dates": int(dates.duplicated().sum()),
        "unparseable_quote_dates": int(dates.isna().sum()),
        "rejected_cells_during_cleaning": provenance.get("rejected_cells"),
        "missing_observations": int(numeric.isna().sum().sum()),
        "non_positive_observations": int((numeric <= 0.0).sum().sum()),
        "delivery_convention": "TTFc1 is the calendar month after quote_date; ranks advance monthly",
        "delivery_interval": "delivery_start inclusive, delivery_end exclusive",
        "delivery_convention_status": "project convention; provider definition not independently supplied",
        "return_filter": "same delivery identity; adjacent panel rows; both prices positive and present",
        "filters": [
            "unparseable quote dates removed by cleaning",
            "non-numeric price cells coerced to missing and retained in the panel",
            "missing and non-positive prices retained and flagged in the panel",
            "returns require adjacent panel rows for the same delivery identity",
            "missing or non-positive observations break rather than bridge a return chain",
        ],
    }
