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
import os

import pandas as pd

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
    fingerprint = source_fingerprint(source_path)
    byte_count = os.path.getsize(source_path)

    if source_path.lower().endswith(".parquet"):
        quotes = pd.read_parquet(source_path)
        return quotes, dict(source_path=resolved, source_sha256=fingerprint,
                            format="parquet", byte_count=byte_count,
                            cleaning_version=None, cache="n/a (explicit parquet source)")

    cache_dir = os.path.dirname(resolved) if cache_dir is None else cache_dir
    cache_file = _cache_path(cache_dir, fingerprint)

    if use_cache and os.path.exists(cache_file):
        return pd.read_parquet(cache_file), dict(
            source_path=resolved, source_sha256=fingerprint, format="xlsx",
            byte_count=byte_count, cleaning_version=CLEANING_VERSION, cache="hit")

    quotes, stats = clean_quote_matrix(pd.read_excel(source_path))
    cache_status = "rebuilt"
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
