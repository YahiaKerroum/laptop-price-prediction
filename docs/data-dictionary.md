# Data dictionary

Per-column source, unit, transformations and known issues.

The mechanical half of this — dtype, null rate, cardinality, range — is **generated from the
data** by `make pipeline` into [`data-card.md`](data-card.md) and `data-card.csv`. That is
deliberate: the original reports drifted away from the data they described (citing 15,517
rows where the split ran on 15,418, and crediting feature selection with dropping a column
that was never in the feature list). A generated table cannot drift.

This page carries what a generator cannot know: provenance and caveats.

---

## Provenance

| Source | What it is | Trust |
|---|---|---|
| `data/raw/data.csv` | Scraped Ouedkniss listings, 16,394 rows, 2018–2025 | Free text, seller-entered, unvalidated |
| `data/raw/cpus.csv`, `gpus.csv` | PassMark benchmark reference | Reliable; joined by fuzzy name match |
| `data/raw/cpu_prices.csv`, `gpu_prices.csv` | Component prices in DZD | **Derived, not observed** — see below |
| `data/mappings/*` | CPU → DDR type, CPU → storage, name corrections | Hand-curated |

### The component prices are estimates, not observations

`cpu_prices.csv` and `gpu_prices.csv` were produced by a rule in notebook 01: estimate a USD
price from `cpumark`, cores and TDP; convert at the parallel ("square") DZD rate; apply a
market-competitiveness adjustment. No Algerian component price was ever observed.

They are fine as a *feature* and as an evaluation baseline. They are not a ground truth.

---

## The target

### `price_corrected`

The asking price in DZD after two corrections.

**This is an asking price, not a sale price.** Nothing in this dataset records what a laptop
actually sold for. The difference is real, seller-specific and entirely unmeasured. Every
model, metric and claim in this project inherits that caveat.

| Step | Effect |
|---|---|
| Troll removal | 62 rows dropped (`111111`, `99999`, `1234`, …) |
| Unit correction | 97.6% resolved on the price axis alone; 247 rows (1.6%) used the component estimate and are flagged `price_unit_ambiguous` |

Rows outside 10,000–1,000,000 DZD are dropped. Median: 95,000 DZD.

---

## Columns with caveats

### `spec_Etat` — condition

`MOYEN=1`, `BON ÉTAT=2`, `JAMAIS UTILISÉ=3`. **41.4% missing, encoded NaN.**

Missingness is informative: unstated-condition listings have a mean price *between* buckets 2
and 3. The original encoded missing as `0`, placing the largest group at the bottom of the
scale. Paired with `etat_is_missing`.

The scraper stripped accents, so the raw values are `BON TAT` and `JAMAIS UTILIS`. Both
spellings are accepted.

### `city` → `city_grouped`

470 raw levels collapsed to 59 (minimum 30 listings; the rest become `OTHER`).

**Heavily skewed.** A single district (Bab Ezzouar) accounts for over a third of all
listings. Estimates for southern wilayas rest on very little data, and the fairness question
— does the model systematically underprice southern listings? — has not been investigated.

### `created_at` → `listing_year`, `listing_month`, …

Emitted as `"2021 10 01T18:01:44.000Z"` — space-separated, not ISO-hyphenated. A default
parser silently yields `NaT` on some pandas versions; `parse_created_at` normalises the
separators first.

Coverage is very uneven: 10,816 of 16,255 listings are from 2025, and just 1 from 2018. The
time-based split therefore trains on ~5,400 rows and tests on ~10,800 — unusual, and worth
knowing when reading that metric.

### `model_name` → `brand`

46 raw levels collapsed to 39. It is a **model family** (THINKPAD, LATITUDE, MACBOOK), not a
manufacturer — there is no Lenovo/Dell/HP column. 528 rows have no model name.

### `cpu_mark`, `gpu_g3d_mark`, `gpu_g2d_mark`

PassMark scores, attached by fuzzy name match. Scraped **with thousands separators**
(`"19,108"`), so a bare `to_numeric` nulls 98% of the column. Stripped before coercion.

~4.8% missing where the CPU string could not be matched above the confidence threshold. Left
missing rather than guessed.

`gpu_g2d_mark` and `gpu_g3d_mark` are strongly collinear, as are `cpu_mark`/`tdp`/`cores`.
Both are retained — trees tolerate collinearity. The original report's claim that
"multicollinearity was addressed" was not accurate; `gpu_g2d_mark` was selected, not
eliminated.

### `RAM_SIZE`, `SSD_SIZE`, `HDD_SIZE`

Parsed to float GB, using the marketing terabyte (1 TB = 1000 GB).

Sellers write multiple capacities three ways, handled differently:

| Form | Meaning | Handling |
|---|---|---|
| `"256GB+1TB"` | dual drive | summed |
| `"256GB/512GB"` | a choice | first taken |
| `"1TB 128GB"` | a choice | first taken |

RAM and storage are **routinely swapped** between columns; `needs_swap` detects and repairs
it, but only when the swap improves the row — exchanging two storage values achieves nothing.

Plausibility clamps null what survives: RAM outside 1–128 GB, storage above 16,384 GB. These
caught 23 RAM values (0.125 GB megabyte misparses, 512 GB storage leakage) and one
`"250320500GB"` — three drive options run together with no separator.

A missing disk is `0` (no drive); an *implausible* one is `NaN` (unknown).

### `SCREEN_SIZE_SNAPPED`, `SCREEN_RESOLUTION_ENC`

Sizes snap to the nearest of eight canonical panels when within 0.3"; rare but genuine sizes
(18.4") are left alone. Resolutions map to a 9-tier ordinal scale, HD(1) → 5K(9).

Both are imputed from the **per-model mode** where missing — manufacturers ship one panel
across a product line. `screen_size_imputed` and `screen_resolution_imputed` record which
rows this touched. `SCREEN_FREQUENCY` is dropped: missing for almost every listing and not
recoverable.

### `spec_signature`

Bookkeeping, not a feature. Concatenates RAM, SSD, HDD, `cpu_mark`, `gpu_g3d_mark`, screen
size and resolution to identify a configuration.

**16,255 rows carry only 6,944 unique signatures.** Effective sample size is closer to 6,944
than to the row count, and identical signatures sell up to 9× apart — the noise ceiling.

### `price_unit_ambiguous`

True for the 247 rows where the component estimate broke a unit tie. Lets metrics be reported
with and without the rows whose target was feature-influenced.

---

## Known dataset-level issues

| Issue | Status |
|---|---|
| `original_data.csv` was byte-identical to `data_cleaned.csv` — two names, one file | Duplicate dropped |
| 57% of rows are duplicate configurations | Measured; grouped split reports the effect |
| Algiers-heavy geography | Documented, not corrected |
| 2025-heavy time distribution | Documented; affects the time-split read |
| 875 rows dropped for unmappable CPU names | Accepted; an LLM parser is a roadmap item |
| No listing text, photos, seller identity, or view counts | The single biggest limitation — see [`roadmap.md`](roadmap.md) §2 |

---

## Generated tables

- [`data-card.md`](data-card.md) — every column with dtype, null rate, cardinality, range
- `data-card.csv` — the same, machine-readable

Both are regenerated by `make pipeline`.
