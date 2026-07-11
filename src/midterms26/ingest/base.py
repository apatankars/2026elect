"""Ingest framework: raw cache, provenance, gated fetch, warehouse upsert.

Design:
  * Downloads are *immutable* and cached under ``data/raw/<source>/``. A JSONL
    provenance manifest records url, sha256, byte size, and fetch time so a
    backtest can prove exactly which bytes produced a feature.
  * Live network fetches are **gated**: ``RunContext.allow_fetch`` must be True
    (Phase 1 otherwise runs entirely against cached raw files). This keeps the
    default pipeline offline and credential-free.
  * ``upsert_dataframe`` writes a Polars frame into a warehouse table with
    ``INSERT OR REPLACE`` on the table's primary key — so re-running backfill is
    idempotent.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl

from midterms26.context import RunContext
from midterms26.logging import get_logger

log = get_logger("ingest")

MANIFEST_NAME = "_manifest.jsonl"


class FetchNotAllowedError(RuntimeError):
    """Raised when a live download is attempted without ``allow_fetch``."""


def source_dir(ctx: RunContext, source: str) -> Path:
    """Return (and create) the raw cache directory for ``source``."""
    d = ctx.raw_dir / source
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_provenance(
    ctx: RunContext,
    source: str,
    *,
    filename: str,
    url: str,
    data: bytes,
    as_of: str | None = None,
) -> None:
    """Append a provenance row for a cached raw file."""
    manifest = source_dir(ctx, source) / MANIFEST_NAME
    row = {
        "filename": filename,
        "url": url,
        "sha256": _sha256(data),
        "bytes": len(data),
        "fetched_at": datetime.now(UTC).isoformat(),
        "as_of": as_of,
    }
    with manifest.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def fetch(
    ctx: RunContext, source: str, *, url: str, filename: str, as_of: str | None = None
) -> Path:
    """Download ``url`` into the raw cache, gated on ``ctx.allow_fetch``.

    Returns the cached path. If the file already exists it is reused (immutable
    cache). Requires the ``ingest`` extra (httpx) only when actually fetching.
    """
    dest = source_dir(ctx, source) / filename
    if dest.exists():
        log.info("fetch.cached", source=source, filename=filename)
        return dest
    if not ctx.allow_fetch:
        raise FetchNotAllowedError(
            f"{source}:{filename} not cached and allow_fetch=False; "
            f"place the file at {dest} or run with allow_fetch=True (needs network/keys)."
        )
    import httpx  # local import: only needed on a real fetch

    log.info("fetch.download", source=source, url=url)
    resp = httpx.get(url, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    record_provenance(ctx, source, filename=filename, url=url, data=resp.content, as_of=as_of)
    return dest


def cached_files(ctx: RunContext, source: str, *, pattern: str = "*") -> list[Path]:
    """Return cached raw files for ``source`` (excludes the manifest)."""
    d = ctx.raw_dir / source
    if not d.exists():
        return []
    return sorted(p for p in d.glob(pattern) if p.name != MANIFEST_NAME and p.is_file())


def upsert_dataframe(conn: duckdb.DuckDBPyConnection, table: str, df: pl.DataFrame) -> int:
    """Idempotently upsert ``df`` into ``table`` on its primary key.

    Only the columns present in ``df`` are written (table defaults fill the rest).
    Returns the number of rows upserted.
    """
    if df.height == 0:
        return 0
    cols = df.columns
    collist = ", ".join(f'"{c}"' for c in cols)
    conn.register("_ingest_src", df)
    try:
        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({collist}) SELECT {collist} FROM _ingest_src"
        )
    finally:
        conn.unregister("_ingest_src")
    return df.height
