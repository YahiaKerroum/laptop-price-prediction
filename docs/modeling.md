# Modelling

## Target

`price_corrected` — the listing's asking price in DZD after troll removal and unit
correction. Trained on `log1p(price)` and inverted inside the artifact, so callers always
work in dinars.

Predictions are clamped to the trained band [10,000 – 1,000,000 DZD]. Outside it the model
has no evidence, and a pricing service quoting 40 million dinars for a laptop is worse than
one quoting the ceiling. Before this guard a Ridge baseline emitted a single extrapolated
prediction large enough to overflow `expm1` to infinity, which poisoned every aggregate
metric computed from it.

## Features

36 columns: 32 numeric and 4 categorical.

### Restored from deletion

The original pipeline dropped `created_at`, `city` and `model_name`, and replaced brand with
`model_family` — a four-level tier computed from `cpu_mark` and `gpu_g3d_mark`, i.e. a lossy
re-encoding of two features the model already had, with 76% of rows in one bucket.

| Column | Why it matters |
|---|---|
| `city_grouped` | Median price spans 2.35× between Algiers districts; 470 levels collapsed to 59 (min 30 listings) |
| `brand` | ThinkPad vs IdeaPad at identical specs is a large used-market difference; 46 levels → 39 |
| `listing_year`, `listing_month`, `listing_month_index`, `month_sin/cos` | Listings span 2018–2025 — seven years of dinar inflation, tech depreciation and the 2021–22 GPU spike |

`model_family` is retained, but only so the results table can show what real brand
information bought over the proxy.

### Condition

41.4% of `spec_Etat` is missing. The original encoded `MOYEN=1, BON ÉTAT=2, JAMAIS UTILISÉ=3,
missing=0` — placing the largest group at the bottom of the scale when its mean price sits
*between* buckets 2 and 3.

Here, missing is `NaN` and `etat_is_missing` is an explicit flag. `HistGradientBoosting`
learns its own split for missing values, so the information is used rather than fabricated.

The original also contained a 60-line `infer_laptop_state()` that was **never called**. It
read `price_preview` to infer condition, which would have leaked the target directly. It is
deleted, not wired in.

### Engineered

| Feature | Definition |
|---|---|
| `gpu_to_cpu_ratio` | `gpu_g3d_mark / cpu_mark` — gaming vs productivity build |
| `storage_per_ram` | `SSD_SIZE / RAM_SIZE` — balanced vs lopsided |
| `total_storage`, `has_hdd`, `is_dual_drive` | Storage shape |
| `pixels`, `ppi` | Actual pixel count and density, instead of only an ordinal tier |
| `total_tdp` | `cpu_tdp + gpu_tdp` — ultrabook through desktop-replacement |
| `estimated_component_cost` | Rule-based build cost; also an evaluation baseline |
| `perf_per_expected_dinar` | `cpu_mark / estimated_component_cost` |

Keeping `estimated_component_cost` as a *feature* is legitimate and interesting — it answers
"does the model beat a component-sum heuristic?" The problem was only ever letting it edit
the target.

### Encoding

Numeric columns pass through untouched: tree ensembles handle raw magnitudes and NaN
natively, and not imputing is the point of the condition fix. Categoricals are one-hot
encoded with `handle_unknown="ignore"` and `min_frequency=20`, so an unseen city at inference
time cannot raise.

Monotonic constraints are **enforced** for `RAM_SIZE`, `SSD_SIZE`, `cpu_mark` and
`gpu_g3d_mark` — more must never mean cheaper. Beyond correctness this keeps the
"what would raise the value" panel from producing an embarrassing recommendation.

They are passed to the estimator as a dict keyed by feature name, which requires the
preprocessor to emit a DataFrame (`set_output(transform="pandas")`) — the width of the
transformed matrix depends on how many one-hot levels survive `min_frequency` and is not
known until fit time, so a positional array could not be built in advance. The quantile
models carry the same constraints: an upper bound that falls when RAM rises is indefensible.

A parametrised test sweeps each constrained feature across its range and asserts the
prediction never decreases.

## Splits

Three, all reported, so the headline cannot be quietly chosen from whichever flatters most.

| Split | What it answers |
|---|---|
| Stratified random 60/20/20 | Comparable to the original report |
| Grouped on spec signature | Are duplicate configurations inflating the score? |
| **Time-based: train ≤ 2024, test 2025** | **Can we price a laptop listed tomorrow?** |

The dataset has 16,255 rows but only **6,944 unique spec signatures** — 57% are duplicates in
feature space, so a random split can put the same configuration on both sides. The grouped
split gives R² 0.810 against 0.848 random. The duplicates were *not* inflating the score
much, which is worth being able to say having actually checked.

## Results

| Model | Split | R² | MAE (DZD) | RMSE | MAPE | MedAPE | Within 20% |
|---|---|---|---|---|---|---|---|
| Global median | time | −0.107 | 63,033 | 103,493 | 59.4% | 42.1% | 24.0% |
| Median of identical spec | time | 0.074 | 51,194 | 94,672 | 47.0% | 27.8% | 41.0% |
| Component-cost sum | time | 0.095 | 49,617 | 93,604 | 37.5% | 24.2% | 43.5% |
| Ridge | time | 0.735 | 25,057 | 50,650 | 20.5% | 14.0% | 65.9% |
| RandomForest | time | 0.812 | 21,267 | 42,660 | 19.5% | 12.0% | 70.0% |
| **HistGradientBoosting** | **time** | **0.810** | **21,319** | 42,911 | 19.5% | **12.2%** | 70.5% |
| HistGradientBoosting | grouped | 0.810 | 18,021 | 39,520 | 18.2% | 10.5% | 76.6% |
| HistGradientBoosting | random | 0.848 | 18,028 | 36,937 | 18.4% | 10.6% | 75.4% |

Against the original (R² 0.827, MAE 21,070, random split), **MAE fell 14%** on the
comparable split. That came from restoring three deleted columns, not from a better model —
RandomForest and HistGradientBoosting are within noise of each other here.

### The unit-ambiguity check

| Test set | R² | MAE |
|---|---|---|
| All rows | 0.8279 | 18,998 |
| Excluding `price_unit_ambiguous` | 0.8278 | 18,910 |

247 rows (1.6%) had their target influenced by a feature-derived estimate. Dropping them
changes R² by 0.0001. The sharpest methodological criticism of the original is now a
demonstrated non-issue rather than an open question.

## Prediction intervals

Three `HistGradientBoosting` quantile models at q = 0.1 / 0.5 / 0.9.

| Metric | Value |
|---|---|
| Nominal coverage | 80% |
| **Observed coverage** | **72.6%** |
| Median width | 34,648 DZD (36% of price) |

The interval under-covers by about 7 points. Reported as measured rather than tuned to look
right; conformal prediction (MAPIE) would give a coverage guarantee and is the obvious next
step.

A range is the correct output shape regardless. Identical configurations sell 2–9× apart, so
a point estimate claims precision the data does not contain.

## The ceiling

| Quantity | Value |
|---|---|
| Rows | 16,255 |
| Unique spec signatures | 6,944 |
| Median max/min price ratio for repeated configurations | ~1.5× |
| Worst observed | ~9× |

Predicting the perfect per-configuration mean would cap out near R² 0.95 in log space. At
0.81 on the time split there is real headroom, but **most of it is not in the spec columns**.
It lives in listing text, seller reputation, photo count, negotiability and urgency — none of
which were scraped. Squeezing another 0.01 out of a better gradient booster is a rounding
error next to getting listing text into the model.

## Anomaly detection

| Component | Method |
|---|---|
| Scam / bait | IsolationForest, LocalOutlierFactor, ECOD, COPOD on the joint (specs, log-price) distribution, majority vote |
| Deal finder | `(predicted − actual) / predicted`, with joint outliers filtered out |
| Spec consistency | High-confidence association rules; a listing violating `{RTX 4060} → {16GB}` is probably mistyped |

Evaluating unsupervised detection without labels is mostly impossible, so two honest things
are reported: pairwise Jaccard agreement between detectors, and precision@k once ~200
listings are hand-labelled. `notebooks/09` writes a stratified labelling template to
`data/labels/` so the effort is spent on rows that matter.

### What the detectors actually agree on

At 2% contamination each detector flags 326 listings. They do **not** flag the same ones:

| | IsolationForest | LOF | ECOD | COPOD |
|---|---|---|---|---|
| **IsolationForest** | 1.00 | 0.00 | 0.45 | 0.71 |
| **LOF** | 0.00 | 1.00 | 0.00 | 0.00 |
| **ECOD** | 0.45 | 0.00 | 1.00 | 0.49 |
| **COPOD** | 0.71 | 0.00 | 0.49 | 1.00 |

The three global methods overlap substantially. `LocalOutlierFactor` overlaps with none of
them — Jaccard 0.00 across the board. That is not a bug: LOF scores *local* density, so it
finds listings that are odd relative to their immediate neighbourhood rather than odd in the
distribution as a whole. Majority vote yields 288 listings (1.8%); the union yields 815
(5.0%).

The deal filter uses the **union**, not the majority. For a feed that tells people where to
spend money, a false "not a bargain" costs far less than a false "bargain".

**This is why four detectors were benchmarked rather than one.** A single detector would have
looked perfectly reasonable and quietly covered a quarter of the anomaly space.

Section 6c is the piece that makes the project one argument rather than three assignments:
the association rules mined for their own deliverable become an input to fraud detection.

## Reproducing

```bash
make build && make pipeline && make train
```

`models/<version>/metadata.json` records the git SHA, library versions and full metric panel
for every run.
