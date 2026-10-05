"""Tests for price parsing, troll detection and unit-scale resolution.

These functions used to be notebook cells whose only verification was a `print`
statement. The unit-correction rule in particular rewrites the *target variable*,
so it is the single most consequential piece of logic in the project.
"""

from __future__ import annotations

import math

import pytest

from laptop_price.cleaning.price import (
    brand_multiplier,
    estimate_component_cost,
    is_troll_price,
    parse_ram_size,
    parse_storage_size,
    ram_price,
    resolve_price_scale,
    storage_price,
)


class TestTrollPrices:
    @pytest.mark.parametrize("value", [111, 999, 4444, 111111, 999999, 22222])
    def test_repeated_digits_are_trolls(self, value):
        assert is_troll_price(value)

    @pytest.mark.parametrize("value", [123, 1234, 12345, 123456, 12345678])
    def test_sequential_digits_are_trolls(self, value):
        assert is_troll_price(value)

    @pytest.mark.parametrize("value", [95_000, 120_000, 45_500, 1_250_000])
    def test_real_prices_are_not_trolls(self, value):
        assert not is_troll_price(value)

    def test_two_repeated_digits_is_a_real_price(self):
        # 11 is short; the rule needs at least MIN_REPEAT_LENGTH digits.
        assert not is_troll_price(11)

    @pytest.mark.parametrize("value", [None, float("nan"), "", "abc"])
    def test_missing_and_junk_are_not_trolls(self, value):
        assert not is_troll_price(value)

    def test_9999999_is_caught(self):
        """The audit found this survived into the final dataset as 99999.99."""
        assert is_troll_price(9_999_999)


class TestCapacityParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("16GB", 16.0), ("8 GB", 8.0), ("512MB", 0.5), ("32", 32.0)],
    )
    def test_parse_ram_size(self, text, expected):
        assert parse_ram_size(text) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("512GB", 512.0), ("1TB", 1024.0), ("2TB", 2048.0), ("256GB+1TB", 1280.0)],
    )
    def test_parse_storage_size(self, text, expected):
        assert parse_storage_size(text) == pytest.approx(expected)

    @pytest.mark.parametrize("value", [None, "", float("nan")])
    def test_blank_capacity_is_zero(self, value):
        assert parse_ram_size(value) == 0.0
        assert parse_storage_size(value) == 0.0


class TestComponentPricing:
    def test_premium_brands_cost_more(self):
        assert brand_multiplier("MACBOOK PRO") > brand_multiplier("VIVOBOOK")
        assert brand_multiplier("ASPIRE") < 1.0

    def test_unknown_brand_is_neutral(self):
        assert brand_multiplier("SOME UNKNOWN MODEL") == 1.0

    def test_more_ram_costs_more(self):
        assert ram_price(16, "DDR4") > ram_price(8, "DDR4")

    def test_newer_memory_costs_more(self):
        assert ram_price(16, "DDR5") > ram_price(16, "DDR3")

    def test_ram_is_capped(self):
        """A scraped 512GB "RAM" value must not dominate the component estimate."""
        assert ram_price(512, "DDR4") == ram_price(128, "DDR4")

    def test_zero_ram_assumes_a_mainstream_configuration(self):
        assert ram_price(0, "DDR4") == ram_price(8, "DDR4")

    def test_storage_price_adds_both_drives(self):
        assert storage_price(256, 1000) == pytest.approx(
            storage_price(256, 0) + storage_price(0, 1000)
        )

    def test_estimate_uses_all_components(self):
        row = {
            "mapped_cpu_name": "Intel Core i7-11800H",
            "gpu_name": "NVIDIA GeForce RTX 3060",
            "RAM_SIZE": "16GB",
            "RAM_TYPE": "DDR4",
            "SSD_SIZE": "512GB",
            "HDD_SIZE": None,
            "model_name": "ROG",
        }
        cost = estimate_component_cost(
            row,
            {"Intel Core i7-11800H": 40_000},
            {"NVIDIA GeForce RTX 3060": 60_000},
        )
        # CPU 40k + GPU 60k + RAM 24k + SSD 16k = 140k, times the 1.25 ROG premium.
        assert cost == pytest.approx(175_000, rel=0.02)


class TestPriceScaleResolution:
    """The rescoped unit correction (roadmap A1)."""

    def test_dinars_are_left_alone(self):
        result = resolve_price_scale(95_000)
        assert result.price == 95_000
        assert result.method == "unchanged"
        assert not result.ambiguous

    def test_spoken_shorthand_is_scaled_up(self):
        # "75" on a listing means 75,000 DZD.
        result = resolve_price_scale(75)
        assert result.price == 75_000
        assert result.method == "price_only"
        assert not result.ambiguous

    def test_out_of_band_prices_are_scaled_down(self):
        # 9,500,000 is above the plausible band; one decade down lands inside it.
        result = resolve_price_scale(9_500_000)
        assert result.price == 950_000
        assert result.method == "price_only"

    def test_a_consistent_estimate_cannot_move_a_price_only_answer(self):
        """Rows the price axis settles must not be touched by the features.

        That is the whole point of the rescoping: the component estimate only
        gets a vote when it *strongly* contradicts the price-only reading.
        """
        without = resolve_price_scale(9_500_000)
        with_estimate = resolve_price_scale(9_500_000, 900_000)
        assert with_estimate.price == without.price == 950_000
        assert with_estimate.method == "price_only"
        assert not with_estimate.ambiguous

    def test_a_contradicting_estimate_overrules_and_flags(self):
        """A build cost 47x below the price-only reading does get to intervene."""
        result = resolve_price_scale(9_500_000, 20_000)
        assert result.price == 95_000
        assert result.method == "component"
        assert result.ambiguous

    def test_ambiguous_row_falls_back_to_components_and_is_flagged(self):
        # 6,000,000 -> both 600,000 and 60,000 are plausible laptop prices.
        result = resolve_price_scale(6_000_000, 55_000)
        assert result.price == 60_000
        assert result.method == "component"
        assert result.ambiguous

    def test_ambiguous_row_without_an_estimate_stays_on_the_price_rule(self):
        result = resolve_price_scale(6_000_000)
        assert result.price == 600_000
        assert not result.ambiguous

    def test_rescaling_is_bounded(self):
        """The original loop was unbounded; this one must terminate and give up."""
        result = resolve_price_scale(1e-30)
        assert result.method == "unresolved"
        assert math.isfinite(result.price)

    @pytest.mark.parametrize("value", [0, -5, None, "abc", float("nan")])
    def test_junk_prices_are_not_invented(self, value):
        result = resolve_price_scale(value)
        assert result.method == "unresolved"
