"""Shared normalization: canonical ids, two-party margin, office/state coding.

Canonical race id: ``{cycle}-{office}-{state}-{district}``
  * office   ∈ {HOUSE, SENATE, GOV}
  * state    USPS 2-letter, uppercased
  * district HOUSE -> 2-digit zero-padded ('00' = at-large); SENATE -> 'SEN';
             GOV -> 'GOV'
"""

from __future__ import annotations

import polars as pl

OFFICES = frozenset({"HOUSE", "SENATE", "GOV"})


def date_col(name: str) -> pl.Expr:
    """Parse ``name`` to Date, accepting ISO strings or existing dates.

    Avoids the deprecated implicit String->Date cast by routing through
    ``str.to_date`` (non-strict, so unparseable values become null).
    """
    return pl.col(name).cast(pl.Utf8).str.to_date(strict=False)


# USPS codes incl. DC and territories that appear in House/Gov returns.
_USPS = frozenset(
    [
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
        "DC",
        "PR",
        "GU",
        "VI",
        "AS",
        "MP",
    ]
)

_OFFICE_ALIASES = {
    "HOUSE": "HOUSE",
    "US HOUSE": "HOUSE",
    "REPRESENTATIVE": "HOUSE",
    "H": "HOUSE",
    "SENATE": "SENATE",
    "US SENATE": "SENATE",
    "SEN": "SENATE",
    "S": "SENATE",
    "GOVERNOR": "GOV",
    "GOV": "GOV",
    "G": "GOV",
}


class NormalizationError(ValueError):
    """Raised when a value cannot be coerced to canonical form."""


def normalize_office(raw: str) -> str:
    """Map a source office label to one of :data:`OFFICES`."""
    key = raw.strip().upper()
    if key in OFFICES:
        return key
    if key in _OFFICE_ALIASES:
        return _OFFICE_ALIASES[key]
    raise NormalizationError(f"unrecognized office {raw!r}")


def normalize_state(raw: str) -> str:
    """Validate and uppercase a USPS state code."""
    code = raw.strip().upper()
    if code not in _USPS:
        raise NormalizationError(f"unrecognized USPS state {raw!r}")
    return code


def normalize_district(office: str, raw: object) -> str:
    """Return the canonical district token for ``office``."""
    if office == "SENATE":
        return "SEN"
    if office == "GOV":
        return "GOV"
    # HOUSE
    if raw is None:
        raise NormalizationError("house race requires a district number")
    s = str(raw).strip().upper()
    if s in {"AL", "AT-LARGE", "ATLARGE", "0", "00"}:
        return "00"
    if not s.isdigit():
        raise NormalizationError(f"invalid house district {raw!r}")
    return f"{int(s):02d}"


def race_id(cycle: int, office: str, state: str, district: str | int | None) -> str:
    """Build the canonical race id from raw-ish parts (each is normalized)."""
    off = normalize_office(office)
    st = normalize_state(state)
    dist = normalize_district(off, district)
    return f"{cycle}-{off}-{st}-{dist}"


def two_party_margin(dem_votes: float, rep_votes: float) -> float | None:
    """Two-party margin D% − R% in percentage points, or ``None`` if no two-party vote."""
    total = dem_votes + rep_votes
    if total <= 0:
        return None
    return (dem_votes - rep_votes) / total * 100.0
