# Changelog

All notable changes to this project.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] — 2026-09-11

Consolidation of the scattered course project into one operational, deployable repository,
plus the accuracy repairs and the missing deliverable identified in the audit.

### Repository

- **Consolidated** `docs_for_claude/DataminingProject (1)/` — which was gitignored, so none
  of the latest work was under version control — into the organized tree. All 27 source
  files accounted for; `scripts/verify_sync.py` proves it and runs in CI.
- **Renumbered** the eight notebooks into execution order (`01_preprocessing` …
  `08_association_rules`).
- **Archived** the five superseded split notebooks under `notebooks/legacy/`. One
  (`clean_price.ipynb`) had a JSON-corrupted first cell; another read a file that never
  existed.
- **Normalised** all CSVs to LF and dropped `original_data.csv`, which was byte-identical to
  `data_cleaned.csv`.
- **Made `data/raw/` immutable.** Notebook 01's CPU price chain read and rewrote
  `data/raw/cpus.csv` five times in sequence; it now writes to `data/interim/cpus_priced.csv`.

### Added

- `src/laptop_price/` — the pipeline as an importable, tested package: `cleaning`,
  `features`, `models`, `evaluation`, `anomaly`, `serving`, `api`, `app`.
- **Anomaly detection** (`anomaly/`, `notebooks/09`) — the deliverable the brief required and
  the original never built. Scam detection (IsolationForest, LOF, ECOD, COPOD), an
  underpriced-deal finder, and spec-consistency checks driven by the association rules.
- **FastAPI service** — `/predict` (with optional SHAP contributions), `/deals`, `/schema`,
  `/health`, `/reload`.
- **Streamlit app** — spec form → price range with explanation, deal feed, market charts.
- **Docker** — multi-stage image (`dev`/`lab`/`api`/`app`) and a compose stack with profiles.
- **Makefile** — `build`, `pipeline`, `train`, `test`, `verify`, `notebooks`, `serve`, `docs`.
- **135 tests** over the parsers, splits, metrics, and the artifact round-trip.
- **pandera contracts** asserted at stage boundaries.
- **Config file** (`config.yaml`) replacing magic numbers scattered across notebook cells.
- **Documentation**: architecture, pipeline, data dictionary, modelling, model card, API,
  deployment, audit, roadmap. The data card and model card are *generated* from the data and
  the artifact so they cannot drift.

### Fixed — accuracy

- **Restored `created_at`, `city` and `model_name`**, deleted by the original with no stated
  reason. City median price spans 2.35×; listings span seven years of dinar inflation.
  **MAE fell from 21,070 to 18,028 DZD (−14%)** on the comparable split.
- **`spec_Etat` missing is now NaN, not 0.** 41% of rows were being placed at the bottom of a
  1–2–3 ordinal scale when their true price level sits between buckets 2 and 3.
- **Rescoped the price-unit correction.** A price-only rule settles 97.62% of rows with zero
  feature contact; the component estimate now intervenes on **247 rows (1.6%)** instead of
  826, and each is flagged `price_unit_ambiguous`. Excluding them moves R² by 0.0001.
- **Applied the troll-price filter.** It existed but was dead code — it nulled a column the
  next section recomputed from scratch. 62 placeholder prices removed.
- **Added MAPE and median APE**, plus three baselines (global median, median of identical
  spec, component-cost sum). Median APE is 12.2%.
- **Added group-aware and time-based splits.** The time-based figure (R² 0.810) is now the
  headline; it is lower than the random-split number and is the honest one.
- **Removed** the unused polynomial expansion and the never-called `infer_laptop_state()`,
  which read `price_preview` to infer condition and would have leaked the target.

### Fixed — artifacts

- **One `sklearn.Pipeline`** replaces three mismatched pickles: `scaler.pkl` expected 14
  features, `best_model.pkl` expected 10, and the model had been trained *unscaled*. Loading
  them and following the obvious path produced silent garbage.
- **Versioned bundles** under `models/<version>/` with `metadata.json` publishing the feature
  contract, served by `GET /schema`. The old pickles are kept under `models/legacy_v0/` for
  provenance; nothing loads them.
- **Predictions clamped** to the trained price band. A Ridge extrapolation previously
  overflowed `expm1` to infinity and poisoned every aggregate metric.

### Fixed — reproducibility

Before this release no notebook could run top-to-bottom.

- `04_regression`: `best_grid` was referenced but never defined; a duplicate grid search
  silently rebound `xgb_grid` to a different search space; the scaler was written before its
  directory existed.
- `05_clustering`: a block copied from the regression notebook sorted an always-empty
  DataFrame by a column it never had; the scaler was refitted over the full dataset,
  discarding the train-only fit.
- `07_clustering_optimized`: `n_init=1` during the k search but `n_init=20` for the final fit
  — k was chosen under noisier conditions than the model that was fitted.
- `08`, `09`: `mlxtend` 0.23 requires `num_itemsets`.
- All notebooks: markdown cells carried `outputs: null`, which is invalid nbformat and made
  papermill fail while *reporting* errors, masking the real ones.
- `clustering_optimized_data.csv`, an input to notebook 07, had never been committed. It is
  now generated by notebook 06.

### Fixed — data quality

Each of these was surfaced by the new pandera contracts or the generated data card.

- 23 rows with RAM of 0.125 GB (megabyte values misparsed) or 512 GB (storage leaking through
  the swap detector) — nulled at source.
- One `HDD_SIZE` of `"250320500GB"` — three drive options run together with no separator.
  Storage now clamped to 16,384 GB.
- `"256GB/512GB"` and `"1TB 128GB"` — a *choice* of configurations, not a dual drive. Now
  distinguished from `"256GB+1TB"`, which is summed.
- `cpu_mark` arrives as `"19,108"`; a bare `to_numeric` nulled 98% of the column.
- `needs_swap("256GB", "512GB")` returned `True` while its own docstring said `False`.
  Swapping two storage values achieves nothing; the docstring's behaviour is now implemented.
- `created_at` is `"2021 10 01T…"`, space-separated rather than ISO-hyphenated, which some
  pandas versions silently parse to `NaT`.

### Known limitations

- Prediction intervals under-cover: **72.6%** observed against a nominal 80%. Reported as
  measured rather than tuned; conformal prediction is the next step.
- The clustering rebuild (Gower distance, UMAP→HDBSCAN, silhouette in the clustering space,
  bootstrap stability) is **not** done. Only the scaler double-fit and the `n_init`
  inconsistency were corrected. See [`roadmap.md`](docs/roadmap.md) §5.
- RAM imputation still hardcodes 16 CPU→RAM pairs for 21 rows.
- `GET /deals` refits the anomaly detectors on every request.
- No authentication or rate limiting on the API; CORS is open.
- The three committed PDF reports are **unchanged** and quote the original numbers. They are
  kept as the historical deliverable; current numbers are in `docs/modeling.md` and the
  generated model card.
