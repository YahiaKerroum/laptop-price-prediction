"""CPU name normalisation and benchmark lookup.

Listings write CPU names however the seller felt like it. The tests that matter
here are the real-world formats: "11TH GEN INTEL CORE I5 1135G7" is how a large
share of Ouedkniss listings describe a CPU, and it has to resolve.
"""

from __future__ import annotations

import pytest

from laptop_price.cleaning.cpu import (
    extract_family,
    extract_manufacturer,
    load_corrections,
    load_cpu_reference,
    match_cpu,
    normalize,
)


class TestNormalize:
    def test_clock_suffix_is_dropped(self):
        """PassMark names carry '@ 2.40GHz'; listings never do."""
        assert normalize("Intel Core i5-1135G7 @ 2.40GHz") == "i5 1135g7"

    def test_vendor_noise_is_dropped(self):
        assert normalize("Intel Core i7-11800H") == normalize("i7 11800h")

    def test_generation_prefix_is_kept_as_tokens(self):
        assert normalize("11TH GEN INTEL CORE I5 1135G7") == "11th gen i5 1135g7"

    @pytest.mark.parametrize("value", [None, "", float("nan")])
    def test_blank_normalises_to_empty(self, value):
        assert normalize(value) == ""


class TestMatching:
    @pytest.mark.parametrize(
        ("query", "expected_prefix"),
        [
            ("INTEL CORE I7 11800H", "Intel Core i7-11800H"),
            ("AMD RYZEN 7 5800HS", "AMD Ryzen 7 5800HS"),
            ("Apple M3 8 Core", "Apple M3 8 Core"),
            ("INTEL CORE I5 8250U", "Intel Core i5-8250U"),
        ],
    )
    def test_direct_names_resolve(self, query, expected_prefix):
        match = match_cpu(query)
        assert match is not None
        assert str(match["name"]).startswith(expected_prefix)

    @pytest.mark.parametrize(
        ("query", "expected_prefix"),
        [
            ("11TH GEN INTEL CORE I5 1135G7", "Intel Core i5-1135G7"),
            ("10TH GEN INTEL CORE I5 10210U", "Intel Core i5-10210U"),
        ],
    )
    def test_generation_prefixed_listings_resolve(self, query, expected_prefix):
        """These scored 69.2 - just under the threshold - before the clock-suffix fix."""
        match = match_cpu(query)
        assert match is not None, f"{query} should resolve"
        assert str(match["name"]).startswith(expected_prefix)

    def test_the_typo_table_is_consulted(self):
        """'i5 1135u' is a known typo for the i5-1135G7."""
        match = match_cpu("i5 1135u")
        assert match is not None
        assert match["match_type"] == "correction"
        assert str(match["name"]).startswith("Intel Core i5-1135G7")

    def test_gibberish_returns_nothing(self):
        """Better no benchmark than one belonging to a different chip."""
        assert match_cpu("total gibberish zzz") is None

    def test_blank_returns_nothing(self):
        assert match_cpu("") is None
        assert match_cpu(None) is None

    def test_corrections_table_loads(self):
        assert len(load_corrections()) > 100

    def test_reference_table_has_a_normalised_key(self):
        reference = load_cpu_reference()
        assert "normalized" in reference.columns
        assert reference["normalized"].str.contains("@").sum() == 0


class TestAttributes:
    @pytest.mark.parametrize(
        ("name", "manufacturer"),
        [
            ("Intel Core i7-11800H", "INTEL"),
            ("AMD Ryzen 7 5800HS", "AMD"),
            ("Apple M3 8 Core", "APPLE"),
            ("Intel Celeron N4500", "INTEL"),
        ],
    )
    def test_manufacturer(self, name, manufacturer):
        assert extract_manufacturer(name) == manufacturer

    @pytest.mark.parametrize(
        ("name", "family"),
        [
            ("Intel Core i7-11800H", "i7"),
            ("AMD Ryzen 7 5800HS", "Ryzen 7"),
            ("Apple M3 8 Core", "Apple M"),
            ("Intel Celeron N4500", "Celeron"),
            ("Something Unrecognised", "UNKNOWN"),
        ],
    )
    def test_family(self, name, family):
        assert extract_family(name) == family
