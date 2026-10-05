"""Price normalisation: troll detection, component-cost estimation, unit correction.

Background
----------
Algerian sellers quote laptop prices in three different conventions, often on the
same page:

* **dinars** - the written convention, e.g. ``95000``
* **centimes** - the spoken convention, where 10,000 DZD is said as "1 million",
  so the listing reads ``9500000``
* **spoken shorthand** - ``"95"`` meaning 95,000 DZD

Without a correction step the target variable is unusable. The original
implementation resolved the ambiguity by comparing the listing price to a
rule-based component-cost estimate built *from the model's own features*, and
rewrote the target whenever the ratio was extreme. That leaks feature-derived
information into the target on every rewritten row.

What changed (roadmap A1)
-------------------------
The raw price distribution is trimodal with near-empty gaps between the modes, so
the units are separable on the price axis alone for the overwhelming majority of
rows. :func:`resolve_price_scale` therefore works in three stages:

1. **price-only rule** - rescale by powers of ten and keep the candidates that
   land inside the plausible laptop band. If exactly one lands there, the row is
   settled with no feature contact at all.
2. **trailing-zero tiebreak** - Algerian asking prices are overwhelmingly round;
   a candidate divisible by 1000 beats one that is not.
3. **component fallback** - only for rows where two or more candidates survive
   both stages (e.g. ``6000000``, where 600,000 and 60,000 are both plausible
   laptop prices) does the component estimate get a vote. Those rows are flagged
   ``price_unit_ambiguous`` so metrics can be reported with and without them.

The loop is also bounded, which the original was not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from laptop_price.config import CONFIG

# ---------------------------------------------------------------------------
# Troll prices
# ---------------------------------------------------------------------------

#: Prices that are obviously not offers - keyboard mashing or placeholders.
SEQUENTIAL_TROLLS: frozenset[int] = frozenset(
    {
        123,
        321,
        1111,
        1234,
        12345,
        123456,
        1234567,
        12345678,
        123456789,
        1223789,
        222222,
        5649841,
        8976378,
    }
)

#: A run of identical digits at least this long counts as a troll value.
MIN_REPEAT_LENGTH = 3


def is_troll_price(price: Any) -> bool:
    """Return True for placeholder / joke prices such as 111111 or 1234.

    Ported from the original ``clean_price`` notebook (now under
    ``notebooks/legacy/``). In the merged notebook this filter existed but was
    dead code: it nulled ``estimated_price_dzd``, and the next section recomputed
    that column from scratch, so values like ``99999`` survived into the final
    dataset. Here it is applied to the price itself and the rows are dropped.
    """
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return False
    try:
        value = int(float(price))
    except (TypeError, ValueError):
        return False

    digits = str(abs(value))
    if len(set(digits)) == 1 and len(digits) >= MIN_REPEAT_LENGTH:
        return True
    return abs(value) in SEQUENTIAL_TROLLS


# ---------------------------------------------------------------------------
# Component-cost estimation
# ---------------------------------------------------------------------------

BRAND_MULTIPLIER: dict[str, float] = {
    "RAZER": 1.30,
    "MAC": 1.50,
    "ROG": 1.25,
    "ALIENWARE": 1.25,
    "STEALTH": 1.20,
    "VECTOR": 1.20,
    "PRECISION": 1.20,
    "THINKPAD": 1.15,
    "ZENBOOK": 1.15,
    "TUF": 1.05,
    "KATANA": 1.05,
    "VIVOBOOK": 1.00,
    "IDEAPAD": 0.95,
    "INSPIRON": 0.95,
    "PAVILION": 0.95,
    "ASPIRE": 0.90,
}

RAM_PRICE_PER_GB: dict[str, int] = {
    "LPDDR5X": 2400,
    "DDR5X": 2200,
    "LPDDR5": 2100,
    "DDR5": 2000,
    "DDR4X": 1700,
    "DDR4": 1500,
    "DDR3": 1000,
    "DDR2": 800,
    "DEFAULT": 1500,
}

DEFAULT_CPU_PRICE_DZD = 15_000
SSD_PRICE_PER_256GB = 8_000
HDD_PRICE_PER_TB = 6_000


def parse_ram_size(ram_str: Any) -> float:
    """Parse a RAM size string to GB. ``"16GB"`` -> 16.0, ``"512MB"`` -> 0.5."""
    if ram_str is None or (isinstance(ram_str, float) and math.isnan(ram_str)) or ram_str == "":
        return 0.0

    text = str(ram_str).upper().strip()
    numeric = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if not numeric or numeric == ".":
        return 0.0
    try:
        value = float(numeric)
    except ValueError:
        return 0.0

    return value / 1024 if "MB" in text else value


def parse_storage_size(size_str: Any) -> float:
    """Parse a storage string to GB, summing dual-drive entries like ``"1TB + 512GB"``."""
    if size_str is None or (isinstance(size_str, float) and math.isnan(size_str)) or size_str == "":
        return 0.0

    text = str(size_str).upper().strip()
    if "+" in text:
        return sum(parse_storage_size(part.strip()) for part in text.split("+"))

    numeric = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if not numeric or numeric == ".":
        return 0.0
    try:
        value = float(numeric)
    except ValueError:
        return 0.0

    if "TB" in text:
        return value * 1024
    if "MB" in text:
        return value / 1024
    return value


def brand_multiplier(model_name: Any) -> float:
    """Premium/budget price multiplier inferred from the model name."""
    name = str(model_name).upper()
    for brand, multiplier in BRAND_MULTIPLIER.items():
        if brand in name:
            return multiplier
    return 1.0


def ram_price(ram_size_gb: float, ram_type: Any) -> float:
    """Price the installed memory, capped at 128GB to bound scraping errors."""
    if ram_size_gb <= 0:
        ram_size_gb = 8.0  # assume a mainstream configuration
    ram_size_gb = min(ram_size_gb, 128.0)

    if ram_type is None or (isinstance(ram_type, float) and math.isnan(ram_type)):
        per_gb = RAM_PRICE_PER_GB["DEFAULT"]
    else:
        per_gb = RAM_PRICE_PER_GB.get(str(ram_type).upper().strip(), RAM_PRICE_PER_GB["DEFAULT"])
    return ram_size_gb * per_gb


def storage_price(ssd_gb: float, hdd_gb: float) -> float:
    """Price the installed storage."""
    total = 0.0
    if ssd_gb > 0:
        total += (ssd_gb / 256) * SSD_PRICE_PER_256GB
    if hdd_gb > 0:
        total += (hdd_gb / 1000) * HDD_PRICE_PER_TB
    return total


def estimate_component_cost(
    row: pd.Series | dict[str, Any],
    cpu_price_map: dict[str, float],
    gpu_price_map: dict[str, float],
) -> float:
    """Rule-based build cost: CPU + GPU + RAM + storage, scaled by brand.

    Kept as a *feature* and as a baseline row in the results table - in that role
    it is a legitimate and interesting comparison ("does the model beat a
    component-sum heuristic?"). It only touches the target for the small set of
    genuinely unit-ambiguous rows.
    """
    get = row.get if isinstance(row, dict) else row.get

    cpu = cpu_price_map.get(get("mapped_cpu_name"), DEFAULT_CPU_PRICE_DZD)
    gpu = gpu_price_map.get(get("gpu_name"), 0.0)
    memory = ram_price(parse_ram_size(get("RAM_SIZE")), get("RAM_TYPE"))
    disks = storage_price(parse_storage_size(get("SSD_SIZE")), parse_storage_size(get("HDD_SIZE")))

    base = float(cpu) + float(gpu) + memory + disks
    return round(base * brand_multiplier(get("model_name")), -2)


# ---------------------------------------------------------------------------
# Unit / scale correction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleResolution:
    """Outcome of resolving one listing's price units."""

    price: float
    #: ``price_only`` | ``trailing_zero`` | ``component`` | ``unchanged`` | ``unresolved``
    method: str
    #: True when the price axis alone could not decide and components broke the tie.
    ambiguous: bool
    #: Power of ten applied to the raw value.
    exponent: int


def _candidate_exponents(raw: float, lo: float, hi: float, max_steps: int) -> list[int]:
    """Powers of ten that move ``raw`` into the plausible band ``[lo, hi]``."""
    return [k for k in range(-max_steps, max_steps + 1) if lo <= raw * (10.0**k) <= hi]


def _greedy_exponent(raw: float, lo: float, hi: float, max_steps: int) -> int | None:
    """Walk one decade at a time towards the band and stop on first entry.

    This is the price-only rule: it never looks at a feature. Because the band
    spans two decades, more than one power of ten usually lands inside it, so
    "stop as soon as you arrive" - rather than "enumerate everything" - is what
    makes the rule decisive.
    """
    if lo <= raw <= hi:
        return 0
    step = -1 if raw > hi else 1
    for n in range(1, max_steps + 1):
        exponent = step * n
        if lo <= raw * (10.0**exponent) <= hi:
            return exponent
    return None


def _is_round(value: float, unit: int = 1000) -> bool:
    return abs(value - round(value / unit) * unit) < 1e-6


def resolve_price_scale(
    raw_price: Any,
    component_estimate: float | None = None,
    *,
    min_dzd: float | None = None,
    max_dzd: float | None = None,
    max_steps: int | None = None,
) -> ScaleResolution:
    """Resolve which unit convention a listing price was written in.

    Parameters
    ----------
    raw_price
        The scraped ``price_preview`` value.
    component_estimate
        Rule-based build cost, consulted **only** when the price axis leaves more
        than one plausible reading.

    Returns
    -------
    ScaleResolution
        The corrected price plus which stage decided it, so that downstream code
        can report metrics with and without the feature-influenced rows.

    Examples
    --------
    >>> resolve_price_scale(75).price            # spoken shorthand, scaled up
    75000.0
    >>> resolve_price_scale(95000).price         # already dinars, untouched
    95000.0
    >>> resolve_price_scale(9500000).price       # scaled down to the band
    950000.0

    The last one shows why the component fallback still earns its keep: 950,000
    and 95,000 DZD are both plausible laptop prices, so the price axis alone
    cannot choose. The greedy rule takes the first landing; supplying a component
    estimate lets a strongly contradicting build cost overrule it, and flags the
    row:

    >>> result = resolve_price_scale(9500000, 20000)
    >>> result.price, result.method, result.ambiguous
    (95000.0, 'component', True)
    """
    cfg = CONFIG.price
    lo = float(min_dzd if min_dzd is not None else cfg.min_dzd)
    hi = float(max_dzd if max_dzd is not None else cfg.max_dzd)
    steps = int(max_steps if max_steps is not None else cfg.rescale_max_steps)

    try:
        raw = float(raw_price)
    except (TypeError, ValueError):
        return ScaleResolution(float("nan"), "unresolved", False, 0)
    if not np.isfinite(raw) or raw <= 0:
        return ScaleResolution(raw, "unresolved", False, 0)

    # --- Stage 1: the price-only rule. No feature is consulted. ---------------
    greedy = _greedy_exponent(raw, lo, hi, steps)
    if greedy is None:
        # Nothing lands in the band - leave it; outlier trimming will drop it.
        return ScaleResolution(raw, "unresolved", False, 0)

    method = "unchanged" if greedy == 0 else "price_only"
    resolved = raw * (10.0**greedy)

    if not (component_estimate and component_estimate > 0 and CONFIG.price.ambiguous_fallback):
        return ScaleResolution(resolved, method, False, greedy)

    # --- Stage 2: does the component estimate contradict the price-only answer?
    # Only a *strong* disagreement counts, using the same factor-of-8 threshold
    # the original implementation used to decide a price was mis-scaled.
    alternatives = [k for k in _candidate_exponents(raw, lo, hi, steps) if k != greedy]
    if not alternatives:
        return ScaleResolution(resolved, method, False, greedy)

    log_estimate = math.log10(component_estimate)
    greedy_gap = abs(math.log10(resolved) - log_estimate)
    if greedy_gap <= math.log10(8):
        # The price-only answer is consistent with the build cost. Keep it.
        return ScaleResolution(resolved, method, False, greedy)

    contenders = [
        k for k in alternatives if abs(math.log10(raw * 10.0**k) - log_estimate) < greedy_gap
    ]
    if not contenders:
        return ScaleResolution(resolved, method, False, greedy)

    # --- Stage 3: roundness breaks ties between the surviving contenders. -----
    if CONFIG.price.trailing_zero_tiebreak and len(contenders) > 1:
        round_ones = [k for k in contenders if _is_round(raw * (10.0**k))]
        if round_ones:
            contenders = round_ones

    exponent = min(contenders, key=lambda k: abs(math.log10(raw * 10.0**k) - log_estimate))
    return ScaleResolution(raw * (10.0**exponent), "component", True, exponent)


def correct_prices(
    df: pd.DataFrame,
    cpu_price_map: dict[str, float],
    gpu_price_map: dict[str, float],
    *,
    price_column: str = "price_preview",
    drop_trolls: bool = True,
) -> pd.DataFrame:
    """Add ``price_corrected`` and its provenance columns to ``df``.

    Adds
    ----
    ``estimated_component_cost``
        Rule-based build cost. Kept as a feature and as an evaluation baseline.
    ``price_corrected``
        The target variable.
    ``price_scale_method``
        Which stage of :func:`resolve_price_scale` decided the value.
    ``price_unit_ambiguous``
        True where the component estimate had to break a tie (~2% of rows).
    ``is_troll_price``
        True for placeholder prices; these rows are dropped when ``drop_trolls``.
    """
    out = df.copy()

    out["estimated_component_cost"] = out.apply(
        lambda row: estimate_component_cost(row, cpu_price_map, gpu_price_map), axis=1
    )
    out["is_troll_price"] = out[price_column].apply(is_troll_price)

    resolutions = [
        resolve_price_scale(raw, estimate)
        for raw, estimate in zip(out[price_column], out["estimated_component_cost"], strict=True)
    ]
    out["price_corrected"] = [r.price for r in resolutions]
    out["price_scale_method"] = [r.method for r in resolutions]
    out["price_unit_ambiguous"] = [r.ambiguous for r in resolutions]

    # A troll price carries no information about the real asking price.
    out.loc[out["is_troll_price"], "price_corrected"] = np.nan

    if drop_trolls:
        out = out[~out["is_troll_price"]].copy()

    return out


__all__ = [
    "BRAND_MULTIPLIER",
    "RAM_PRICE_PER_GB",
    "SEQUENTIAL_TROLLS",
    "ScaleResolution",
    "brand_multiplier",
    "correct_prices",
    "estimate_component_cost",
    "is_troll_price",
    "parse_ram_size",
    "parse_storage_size",
    "ram_price",
    "resolve_price_scale",
    "storage_price",
]
