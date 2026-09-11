# Audit — what was wrong, and what was done about it

The original project was a competent course deliverable with a missing deliverable, a
degenerate clustering result reported as a success, a target-correction step that let
feature-derived estimates edit the target, and three strong predictive columns deleted for no
stated reason.

This document records each finding, the evidence, and the resolution. The forward-looking
half of the original review is in [`roadmap.md`](roadmap.md).

---

## Findings and resolutions

### A1 · The price-unit correction let features touch the target — 🟢 fixed

Algerian sellers use three price conventions: dinars, centimes (×100), and spoken shorthand
(`"75"` = 75,000 DZD). A correction step is genuinely necessary; without it the target is
unusable. **That call was correct.**

The problem was *how*. `fix_price_scale()` compared each price to a rule-based component-cost
estimate built from the model's own features and rewrote the target on **826 rows**. For
those rows, part of what the model then learned was the hand-written price table.

**Measured:** a price-only rule — walk one decade at a time toward the plausible band
(10,000–1,000,000 DZD), stop on first entry, consult no feature — reproduces
`fix_price_scale` on **15,928 of 16,317 rows (97.62%)**. The raw log₁₀ distribution is
trimodal with near-empty gaps, so the units are separable on the price axis alone.

**Resolution.** `resolve_price_scale` runs in stages: price-only rule first; a roundness
tiebreak (90.6% of corrected prices end in `000`); and the component estimate only where it
*strongly* contradicts the price-only reading, using the same factor-of-8 threshold the
original used. The loop is bounded — the original's `while` had no iteration cap.

| | Original | Now |
|---|---|---|
| Rows where features touched the target | 826 | **247 (1.6%)** |
| Those rows flagged | no | **yes** (`price_unit_ambiguous`) |
| Agreement with the original target | — | 99.21% |

Excluding the flagged rows from the test set moves R² from 0.8279 to 0.8278. The criticism
is now a demonstrated non-issue rather than an open question.

`estimated_component_cost` is retained as a *feature* and as an evaluation baseline, where it
is entirely legitimate.

---

### A2 · `spec_Etat` encoded in the worst possible way — 🟢 fixed

41.4% of the condition column is missing. The original encoded it ordinally as
`MOYEN=1, BON ÉTAT=2, JAMAIS UTILISÉ=3, missing=0`.

The notebook's own output shows why that is wrong:

| Code | Meaning | Mean price |
|---|---|---|
| 1 | MOYEN | 51,412 |
| 2 | BON ÉTAT | 88,700 |
| 3 | JAMAIS UTILISÉ | 190,083 |
| **0** | **missing (41%)** | **114,649** |

The 1→2→3 ordering is cleanly monotonic, and then the largest group is placed at the bottom
of the scale when its true price level sits *between* buckets 2 and 3.

**Resolution.** Missing is `NaN`, with an explicit `etat_is_missing` flag.
`HistGradientBoostingRegressor` handles NaN natively and learns its own split — which is why
it is the default estimator, and why the numeric columns are not imputed.

**Also:** the 60-line `infer_laptop_state()` written to fill those values was **never
called**. That was lucky — it read `price_preview` to infer condition, which would have been
direct target leakage. It is deleted, not wired in.

---

### A3 · Three of the strongest signals were thrown away — 🟢 fixed

`created_at`, `city` and `model_name` were dropped before modelling. Brand was replaced by
`model_family`, a four-level tier computed from `cpu_mark` and `gpu_g3d_mark` — a lossy
re-encoding of two features the model already had, with 76% of rows in one bucket.

| Column | Signal |
|---|---|
| `city` | Median price 161,950 DZD (Hydra) to 69,000 (Dar El Beïda) among cities with 100+ listings — a **2.35× spread** |
| `created_at` | Listings span 2018–2025: seven years of dinar inflation, tech depreciation, the 2021–22 GPU spike |
| `model_name` | ThinkPad vs IdeaPad at identical specs is a large used-market difference |

**Resolution.** All three restored, plus derived temporal features. `model_family` retained
only as a comparison column.

**Measured effect:** MAE fell from 21,070 to 18,028 on the comparable split — **−14%**,
better than the −11% the original review predicted.

---

### A4 · 43% duplicate configurations — 🟢 checked, not a problem

16,255 rows carry only **6,944 unique spec signatures**.

**Resolution.** A `grouped_split` on spec signature is now reported alongside the random
split. R² 0.810 grouped vs 0.848 random — the duplicates were *not* inflating the score much.
Worth being able to say having actually checked.

Two things do follow: effective sample size is closer to 6,944 than 16,255, and there is a
hard noise ceiling. Identical configurations sell up to **9× apart**; predicting the perfect
per-configuration mean would cap out near R² 0.95 in log space.

---

### A5 · The clustering is degenerate — 🟡 partially addressed

`clustering.ipynb` reported a silhouette of **0.9796** at k=4 and called it a success. Cluster
sizes were 14,161 / 498 / 550 — that is not a segmented market, it is a few outliers peeled
off one blob.

**Done:** the scaler double-fit is fixed (it was `fit_transform`-ed a second time over the
full dataset, silently discarding the train-fitted scaler), and `n_init` is now consistent
between the k search and the final fit in the optimised notebook — previously k was chosen
under noisier conditions than the model that was fitted.

**Not done:** the deeper rebuild — Gower distance, UMAP→HDBSCAN, reporting silhouette in the
space actually clustered rather than in PCA space, and bootstrap stability — is
[`roadmap.md`](roadmap.md) §5. `clustering_optimized.ipynb` still measures silhouette after
PCA reduction, which flatters the score.

---

### A6 · The saved artifacts did not fit together — 🟢 fixed

```
saved_models/scaler.pkl      → expects 14 features
saved_models/best_model.pkl  → expects 10 features
selected_features.json       → lists 10
```

Verified by loading them. Worse, the model was trained on `X_train_selected` — **unscaled**.
The shipped scaler was both the wrong shape and something the model had never seen. Anyone
following the obvious path got silent garbage.

**Resolution.** One `sklearn.Pipeline` carrying its own preprocessing, serialised whole, with
`metadata.json` publishing the feature contract and `GET /schema` serving it. A round-trip
test (fit → save → load → predict) is part of the suite. The old pickles are kept under
`models/legacy_v0/` for provenance; nothing loads them.

---

### A7 · Smaller things — 🟢 mostly fixed

| Finding | Status |
|---|---|
| Troll-price filter was dead code — it nulled a column the next section recomputed, so `11111`, `99999`, `111111` survived | **Fixed.** Applied to the price itself; 62 rows dropped |
| Polynomial features computed (10→55) and never used | **Removed** |
| `cell 40` referenced undefined `best_grid` — the notebook was not reproducible top-to-bottom | **Fixed** → `xgb_grid` |
| Duplicate XGBoost grid search silently rebound `xgb_grid` to a different space | **Disabled**, with a note |
| No MAPE anywhere | **Fixed.** MAPE and median APE are first-class; median APE is 12.2% |
| `n_init=1` in the k search vs `n_init=20` for the final fit | **Fixed** |
| `data_cleaned.csv` and `original_data.csv` byte-identical | **Fixed.** Duplicate dropped, recorded in the data dictionary |
| `cpu_generation_normalized` dropped for being 3.9% NaN | **Restored** — the model handles NaN |
| RAM imputation hardcoded 16 CPU→RAM pairs for 21 rows | **Still hardcoded.** Roadmap item |
| Report/notebook drift (15,517 vs 15,418 rows; `match_score` credited as dropped when it was never in the feature list) | **Addressed** by generating the model card from the artifact metadata |
| `gpu_g2d_mark`/`gpu_g3d_mark` collinear; "multicollinearity addressed" was wrong | **Acknowledged**, both retained — trees tolerate it. Not re-litigated |

---

### A8 · Anomaly detection was never built — 🟢 built

The brief required it. There was no `IsolationForest`, no `LocalOutlierFactor`, no one-class
model anywhere in the repo; the closest thing was DBSCAN labelling 130 points as noise.

**Resolution.** `src/laptop_price/anomaly/` and `notebooks/09_anomaly_detection.ipynb`:

- **Scam detection** — IsolationForest, LOF, ECOD and COPOD on the joint (specs, log-price)
  distribution, with a majority vote and a pairwise-agreement matrix.
- **Deal finder** — residual-ranked bargains with joint outliers filtered out, exposed as
  `GET /deals`.
- **Spec consistency** — high-confidence association rules used as a fraud signal. This is
  the piece that makes regression, association rules and anomaly detection one argument
  rather than three separate assignments.
- **Evaluation** — detector agreement now, precision@k once ~200 listings are hand-labelled;
  the notebook writes a stratified labelling template.

---

## Found during the consolidation

Not in the original review; surfaced by making the project run.

| Finding | Resolution |
|---|---|
| `notebooks/cleaning/clean_price.ipynb` cell 0 was **JSON-corrupted** — notebook keys leaked into the cell source | Archived under `notebooks/legacy/`; superseded by the merged notebook |
| The repo's `association_rules.ipynb` read `data_final_cleaned.csv`, **a file that never existed** | Replaced by the current version |
| `clustering_optimized_data.csv` was an input to notebook 07 that was **never committed** | Now generated by notebook 06 |
| Markdown cells carried `outputs: null` — **invalid nbformat**, which made papermill fail while *reporting* errors, masking the real ones | Repaired across all notebooks |
| `needs_swap("256GB", "512GB")` returned `True`; the function's own docstring said `False`. Swapping two storage values achieves nothing | Implemented the docstring's behaviour |
| The CPU price chain read and rewrote `data/raw/cpus.csv` five times in sequence | Redirected to `data/interim/cpus_priced.csv`; raw data is now immutable |
| Notebook 03 and the package both wrote `model_ready_data.csv`; re-running the notebook silently clobbered the package's matrix | Separated into `model_ready_data.csv` and `features.csv` |
| 23 rows had RAM of 0.125 GB or 512 GB — megabyte values misparsed as GB, and storage leaking through the swap detector | Caught by the new pandera contract on its first run; nulled at source |
| An unbounded component-cost baseline drove R² to **−48,502** via one extreme row | Clipped to the plausible band |
| A Ridge extrapolation overflowed `expm1` to infinity, poisoning every aggregate metric | Predictions clamped to the trained price band |
| `cpu_mark` arrives as `"19,108"` — a bare `to_numeric` nulls 98% of it | Thousands separators stripped before coercion |

---

## Where this leaves the numbers

| Metric | Original | Now (comparable split) | Now (honest headline) |
|---|---|---|---|
| Split | random | random | **time-based** |
| R² | 0.827 | 0.848 | 0.810 |
| MAE | 21,070 | **18,028** | 21,319 |
| Median APE | not reported | 10.6% | **12.2%** |

The headline is the time-based figure and it is *lower* than the original's. That is not a
regression — it is the first number in this project that answers "can we price a laptop
listed tomorrow", which is the only question a deployed service is ever asked.
