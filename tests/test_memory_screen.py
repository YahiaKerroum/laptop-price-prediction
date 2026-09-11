"""Tests for RAM/storage and screen normalisation.

The swap detector and the dual-drive parser are the two places where a silent
mistake changes a laptop's specification rather than merely losing a row, so they
carry the most tests here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from laptop_price.cleaning.memory import (
    encode_ram_type,
    is_ram_value,
    is_storage_value,
    needs_swap,
    normalize_storage_to_gb,
    parse_dual_storage,
    to_gb,
)
from laptop_price.cleaning.screen import (
    RESOLUTION_ORDINAL,
    encode_resolution,
    fill_by_model_mode,
    normalize_resolution,
    parse_screen_size,
    resolution_tier,
    snap_screen_size,
)


class TestValueClassification:
    @pytest.mark.parametrize("value", ["4GB", "8GB", "16 GB", "32gb", "128GB"])
    def test_realistic_ram_sizes(self, value):
        assert is_ram_value(value)

    @pytest.mark.parametrize("value", ["512GB", "1TB", "256GB", ""])
    def test_storage_is_not_ram(self, value):
        assert not is_ram_value(value)

    @pytest.mark.parametrize("value", ["256GB", "512GB", "1TB", "256GB+1TB"])
    def test_storage_values(self, value):
        assert is_storage_value(value)

    def test_128gb_is_ambiguous_and_treated_as_ram(self):
        # 128GB is real RAM on workstations, so the storage threshold sits at 256.
        assert is_ram_value("128GB")
        assert not is_storage_value("128GB")


class TestSwapDetection:
    def test_obvious_swap(self):
        assert needs_swap("512GB", "16GB")

    def test_correct_assignment_is_left_alone(self):
        assert not needs_swap("16GB", "512GB")

    def test_ram_empty_and_storage_holds_a_ram_value(self):
        assert needs_swap("", "16GB")

    def test_ram_present_storage_empty_is_fine(self):
        assert not needs_swap("16GB", "")

    def test_two_storage_values_are_not_a_swap(self):
        assert not needs_swap("256GB", "512GB")


class TestStorageParsing:
    def test_dual_storage_splits(self):
        assert parse_dual_storage("1TB+240GB") == ("1TB", "240GB")

    def test_single_drive_has_no_secondary(self):
        assert parse_dual_storage("512GB") == ("512GB", None)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("1TB", "1000GB"), ("2TB", "2000GB"), ("512GB", "512GB"), ("256", "256GB")],
    )
    def test_normalize_to_gb_uses_the_marketing_terabyte(self, text, expected):
        assert normalize_storage_to_gb(text) == expected

    def test_space_separated_dual_drive_keeps_the_first(self):
        assert normalize_storage_to_gb("1TB 512GB") == "1000GB"

    @pytest.mark.parametrize(
        ("text", "expected"), [("1TB", 1000.0), ("512GB", 512.0), ("256GB+1TB", 1256.0)]
    )
    def test_to_gb(self, text, expected):
        assert to_gb(text) == pytest.approx(expected)


class TestRamTypeEncoding:
    def test_generations_are_ordered(self):
        assert (
            encode_ram_type("DDR3")
            < encode_ram_type("DDR4")
            < encode_ram_type("DDR5")
            < encode_ram_type("LPDDR5X")
        )

    def test_unknown_is_nan_not_zero(self):
        """NaN lets a tree learn its own split; 0 would rank it below SDRAM."""
        assert np.isnan(encode_ram_type(None))
        assert np.isnan(encode_ram_type("SOMETHING ELSE"))

    def test_longest_match_wins(self):
        assert encode_ram_type("LPDDR5X") != encode_ram_type("DDR5")


class TestScreen:
    @pytest.mark.parametrize(("text", "expected"), [("15.6", 15.6), ("15,6", 15.6), ('14"', 14.0)])
    def test_parse_screen_size_handles_european_decimals(self, text, expected):
        assert parse_screen_size(text) == pytest.approx(expected)

    @pytest.mark.parametrize("value", ["3.5", "27", "0", "abc"])
    def test_implausible_sizes_become_nan(self, value):
        assert np.isnan(parse_screen_size(value))

    def test_snapping_to_canonical_sizes(self):
        assert snap_screen_size(15.5) == 15.6
        assert snap_screen_size(13.4) == 13.3

    def test_rare_but_valid_sizes_are_kept(self):
        # 18.4" is genuine, just uncommon; forcing it to 17.3 would be a lie.
        assert snap_screen_size(18.4) == 18.4

    def test_resolution_normalisation(self):
        assert normalize_resolution("1920 x 1080") == "1920x1080"
        assert normalize_resolution("FullHD") == "fullhd"

    @pytest.mark.parametrize(
        ("text", "tier"),
        [
            ("1920x1080", "FHD"),
            ("1366x768", "HD"),
            ("3840x2160", "4K"),
            ("2560x1440", "QHD"),
            ("1920 x 1200", "WUXGA"),
        ],
    )
    def test_resolution_tiers(self, text, tier):
        assert resolution_tier(text) == tier

    def test_unknown_resolution_is_nan(self):
        assert pd.isna(resolution_tier("991x123"))

    def test_encoding_is_monotonic_in_quality(self):
        assert (
            encode_resolution("HD")
            < encode_resolution("FHD")
            < encode_resolution("QHD")
            < encode_resolution("4K")
        )
        assert len(RESOLUTION_ORDINAL) == 9

    def test_fill_by_model_mode(self):
        frame = pd.DataFrame(
            {
                "model_name": ["THINKPAD", "THINKPAD", "THINKPAD", "MACBOOK", "MACBOOK"],
                "tier": ["FHD", "FHD", None, "QHD+", None],
            }
        )
        filled = fill_by_model_mode(frame, "tier")
        assert filled.tolist() == ["FHD", "FHD", "FHD", "QHD+", "QHD+"]

    def test_fill_falls_back_to_the_global_mode(self):
        frame = pd.DataFrame({"model_name": ["A", "A", "B"], "tier": ["FHD", "FHD", None]})
        assert fill_by_model_mode(frame, "tier").tolist() == ["FHD", "FHD", "FHD"]
