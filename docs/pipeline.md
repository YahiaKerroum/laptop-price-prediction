# Pipeline

Each stage, what it consumes, what it produces, and how to re-run it.

---

## Stage 0 — Raw inputs

Read-only. Nothing in the project writes to `data/raw/`.

| File | Rows | What it is |
|---|---|---|
| `data/raw/data.csv` | 16,394 | The original scrape: price, date, city, condition, model, CPU/GPU text, RAM, storage, screen |
| `data/raw/cpus.csv` | 1,530 | PassMark CPU reference: name, cpumark, tdp, cores, integrated GPU, estimated price |
| `data/raw/gpus.csv` | 629 | PassMark GPU reference: G3D, G2D, TDP, estimated price |
| `data/raw/cpu_prices.csv` | 1,530 | CPU → estimated component price, DZD |
| `data/raw/gpu_prices.csv` | 628 | GPU → estimated component price, DZD |
| `data/mappings/cpu_ddr_map.csv` | 823 | CPU → supported DDR generation |
| `data/mappings/cpu_storage_map.csv` | 1,430 | CPU → typical storage configuration |
| `data/mappings/cpu_corrections.csv` | 126 | Normalised typo → canonical CPU name |

`cpu_corrections.csv` was lifted out of a notebook cell where it was 126 dict literals. It
is data, so it lives in `data/`, where it can be inspected and extended without editing code.

---

## Stage 1 — Preprocessing

**`notebooks/01_preprocessing.ipynb`** → `data/processed/pre_processed_data.csv` (16,392 rows)

The largest stage, and the only one still notebook-owned, because it is mostly hand-curated
knowledge rather than reusable rules.

| Section | What it does |
|---|---|
| 2 | CPU name normalisation → fuzzy match against PassMark → `cpu_mark`, `cores`, `tdp` |
| 3 | GPU matching, including inferring integrated graphics from the CPU |
| 3b | CPU generation extraction and normalisation |
| 4 | CPU price estimation: USD → DZD at the parallel rate, then a market adjustment |
| 5 | RAM and storage: un-swap reversed columns, split dual drives, normalise to GB |
| 6 | Screen: snap sizes to canonical panels, map resolutions to a 9-tier scale |
| 7 | Price: component-cost estimate and unit-convention correction |

Re-run it with `make notebooks`, or open it in `make lab`.

**Changed from the original:** the CPU price chain read and rewrote `data/raw/cpus.csv` five
times in sequence. It now writes to `data/interim/cpus_priced.csv`, keeping raw data
immutable. The logic is unchanged.

---

## Stage 2a — Original encoding *(kept for comparability)*

**`notebooks/03_encoding_feature_engineering.ipynb`** → `data/processed/model_ready_data.csv`

Reproduces the original course artifact exactly: 15 numeric columns, `spec_Etat` ordinally
encoded with missing→0, `city`/`created_at`/`model_name` dropped. Consumed by notebooks 04
and 05 so the original results stay checkable.

**Verified reproducible.** Re-running this notebook from `pre_processed_data.csv` regenerates
`model_ready_data.csv` **byte-for-byte identically** to the file shipped in the original
project folder — `make verify` reports it as identical rather than changed. That is a
stronger guarantee than the original had of itself: before the path and nbformat repairs,
the notebook could not be executed top-to-bottom at all.

---

## Stage 2b — Repaired feature build *(what the service uses)*

**`laptop_price.data.build_model_ready()`** → `data/processed/features.csv` (16,255 rows)

```bash
make pipeline
```

Three things happen:

### Troll-price removal

`is_troll_price` flags placeholder values — repeated digits (`111111`, `99999`) and
sequential runs (`1234`, `12345678`). 62 such rows are dropped.

This filter existed in the original but was dead code: it nulled `estimated_price_dzd`, and
the next section recomputed that column from scratch, wiping the effect. The report claimed
the values had been removed; they were still in the final dataset.

### Price-unit correction, rescoped

The original resolved all three price conventions by comparing each listing to a rule-based
component-cost estimate built *from the model's own features*, rewriting the target on 826
rows. That lets feature-derived information into the target.

The replacement runs in stages:

1. **Price-only rule.** Walk one decade at a time toward the plausible band
   (10,000–1,000,000 DZD) and stop on first entry. No feature is consulted. This reproduces
   the original's output on **97.62%** of rows.
2. **Component check.** Only where the build cost *strongly* contradicts the price-only
   reading — by more than the factor of 8 the original itself used — does the estimate get a
   vote.
3. **Roundness tiebreak.** Algerian asking prices are overwhelmingly round; a candidate
   divisible by 1000 beats one that is not.

Result: **247 rows (1.6%)** are feature-influenced instead of 826, and each is flagged
`price_unit_ambiguous` so metrics can be reported with and without them. Excluding them from
the test set moves R² by 0.0001.

The loop is also bounded. The original's `while` had no iteration cap.

### Feature construction

Restores `city`, `created_at` and `model_name`; encodes missing `spec_Etat` as NaN with an
`etat_is_missing` flag; adds ratio, temporal and geographic features. See
[`modeling.md`](modeling.md).

Output is validated against the pandera contract in `laptop_price.features.schema`. When
first enabled it immediately caught 23 rows with RAM of 0.125 GB or 512 GB — megabyte values
misparsed as gigabytes, and storage capacities that leaked through the swap detector.

---

## Stage 3 — Training

**`laptop_price.models.train`** → `models/<version>/`

```bash
make train
```

Fits `HistGradientBoostingRegressor` on a log target inside one `sklearn.Pipeline`, plus
three quantile models for the interval. Evaluates on all three splits against all three
baselines, then refits on train+validation for the shipped artifact.

`HistGradientBoosting` is the default specifically because it handles NaN natively — which
is what the `spec_Etat` fix depends on. Imputing a value there would undo it.

---

## Stage 4 — Analysis notebooks

| Notebook | Input | Produces |
|---|---|---|
| `02_eda.ipynb` | `pre_processed_data.csv` | Distributions, correlations |
| `04_regression.ipynb` | `model_ready_data.csv` | The original model comparison |
| `05_clustering.ipynb` | `model_ready_data.csv` | K-Means / Agglomerative / DBSCAN |
| `06_clustering_encoding.ipynb` | `pre_processed_data.csv` | `clustering_optimized_data.csv` |
| `07_clustering_optimized.ipynb` | `clustering_optimized_data.csv` | The improved segmentation |
| `08_association_rules.ipynb` | `pre_processed_data.csv` | `reports/rules/*.csv` |
| `09_anomaly_detection.ipynb` | `features.csv` + `models/latest` | Scam flags, deal ranking, consistency checks |
| `10_market_segmentation.ipynb` | `features.csv` | Consensus segments, named and price-profiled |

`clustering_optimized_data.csv` was an input to notebook 07 that had never been committed,
so that notebook could not run. It is now generated by notebook 06.

---

## Running everything

```bash
make build
make pipeline     # features.csv + docs/data-card.csv
make train        # models/<version>/
make notebooks    # all 9, top-to-bottom, via papermill
make test         # 211 tests
make verify       # repo vs the original project folder
```

`make notebooks` writes executed copies and per-notebook logs to `reports/executed/`.
A failure there is a reproducibility regression, not a warning.

### Runtime

The full sweep runs in **about 31 minutes**:

| Notebook | Time |
|---|---|
| 01 preprocessing | 27s |
| 02 eda | 6s |
| 03 encoding | 10s |
| 04 regression | 42s |
| 05 clustering | 222s |
| 06 clustering encoding | 11s |
| 07 clustering optimized | 314s |
| 08 association rules | 334s |
| 09 anomaly detection | 13s |
| 10 market segmentation | 862s |

Notebook 10 is the longest because consensus clustering fits the embedding eight
times over — that repetition is the point, not overhead. See
[`modeling.md`](modeling.md#the-base-pipeline-is-not-reproducible-and-that-is-the-main-finding).

That 47 seconds for notebook 04 is **FAST mode**. Its original hyperparameter search is a
576-candidate LightGBM grid at 5 folds — 2,880 fits — plus a 250-fit random search for
Random Forest and two XGBoost grids, all on ~9,300 rows. Left alone it runs for hours; a
2.5-hour attempt during this consolidation was still only 60% through the notebook.

So `04_regression.ipynb` takes a papermill parameter:

```bash
make notebooks                        # FAST: single-point searches, for verification
./scripts/run_notebooks.sh --full     # the original searches, hours
```

`FAST` changes nothing but the size of those searches — the same cells run, the same
models are fitted, the same artifacts are written. The full grids remain the default when
the notebook is opened by hand, because notebooks 01–08 exist to reproduce the original work
rather than improve on it. Replacing `GridSearchCV` with Optuna is
[`roadmap.md`](roadmap.md) §4a.

### Notebook bugs fixed to make this pass

| Notebook | Bug |
|---|---|
| 04 | `best_grid` referenced but never defined; a duplicate grid search silently rebound `xgb_grid` to a different search space; the scaler was written before its directory existed |
| 05 | A block copied from the regression notebook sorted an always-empty DataFrame by a column it never had; the scaler was refitted over the full dataset, discarding the train-only fit |
| 07 | `n_init=1` during the k search but `n_init=20` for the final fit — k was chosen under noisier conditions than the model fitted |
| 08, 09 | `mlxtend` 0.23 requires `num_itemsets` |
| all | Markdown cells carried `outputs: null`, which is invalid nbformat |
