"""Canonical id / margin / office-district-state normalization."""

from __future__ import annotations

import pytest

from midterms26.ingest.normalize import (
    NormalizationError,
    normalize_district,
    normalize_office,
    normalize_state,
    race_id,
    two_party_margin,
)


@pytest.mark.parametrize(
    "raw,expected",
    [("US HOUSE", "HOUSE"), ("Senate", "SENATE"), ("Governor", "GOV"), ("gov", "GOV")],
)
def test_normalize_office(raw: str, expected: str) -> None:
    assert normalize_office(raw) == expected


def test_normalize_office_rejects_president() -> None:
    with pytest.raises(NormalizationError):
        normalize_office("US PRESIDENT")


def test_normalize_state_validates() -> None:
    assert normalize_state("ca") == "CA"
    with pytest.raises(NormalizationError):
        normalize_state("ZZ")


@pytest.mark.parametrize(
    "office,raw,expected",
    [
        ("HOUSE", 1, "01"),
        ("HOUSE", "12", "12"),
        ("HOUSE", "AL", "00"),
        ("HOUSE", "at-large", "00"),
        ("SENATE", None, "SEN"),
        ("GOV", 7, "GOV"),
    ],
)
def test_normalize_district(office: str, raw: object, expected: str) -> None:
    assert normalize_district(office, raw) == expected


def test_house_requires_district() -> None:
    with pytest.raises(NormalizationError):
        normalize_district("HOUSE", None)


def test_race_id_roundtrip() -> None:
    assert race_id(2018, "US HOUSE", "ca", 12) == "2018-HOUSE-CA-12"
    assert race_id(2018, "Senate", "TX", None) == "2018-SENATE-TX-SEN"


def test_two_party_margin() -> None:
    assert two_party_margin(200, 120) == pytest.approx(25.0)
    assert two_party_margin(0, 0) is None
    assert two_party_margin(300, 0) == pytest.approx(100.0)
