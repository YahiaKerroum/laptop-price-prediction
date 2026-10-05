# Roadmap

> **Note.** This document was written as a single audit-plus-roadmap review of the original
> project. **Part A** (the audit) has been acted on — see [`audit.md`](audit.md) for what was
> fixed and what the numbers look like now. **Part B** is the forward roadmap and is
> unchanged. Items marked as done in `audit.md` are no longer open.

---

> From a course deliverable to a real product.
>
> This document has two halves. **Part A** is an honest audit of what we built — the bugs,
> the shortcuts, and the methodological mistakes. Every claim in it is backed by a number
> re-measured from the actual CSVs and notebooks in this folder, not by vibes.
> **Part B** is the roadmap: what to build next, ordered so the cheap high-impact things
> come first.
>
> Read Part A first. Half of the "accuracy improvements" in Part B are just *undoing*
> mistakes from Part A, and those are free.

---

## 0. Where we actually stand

| Thing | Current state |
|---|---|
| Raw listings scraped | 16,392 |
| Rows surviving to the model | 15,517 → 15,418 after outlier trim |
| Features fed to the final model | 10 (all numeric) |
| Best model | XGBoost, log-target |
| Reported test R² | 0.827 |
| Reported test MAE | 21,070 DZD |
| Median listing price | 95,500 DZD → **MAE is ~22% of a typical laptop** |
| Deliverables from the brief | Regression ✅ · Clustering ⚠️ · **Anomaly detection ❌ (never built)** |

That last row matters. The brief asked for anomaly detection and there is no
`IsolationForest`, no `LocalOutlierFactor`, no one-class model anywhere in the repo. The
closest thing is DBSCAN labelling 130 points as noise inside `clustering.ipynb`.
**A whole deliverable is missing** — and it also happens to be the most commercially
useful model we could ship (see §6, scam / underpriced-deal detection).

---

# PART A — What went wrong (the honest audit)

## A1. 🟠 The price-unit fix works — but it lets the features touch the target, and it only needs to for 2.4% of rows

**First, credit where it's due: the problem is real and the fix was necessary.** Algerian
prices are quoted in dinars in writing but in *centimes* in speech (10,000 DZD = "1 million"),
and Ouedkniss sellers use both conventions — plus a third, the spoken shorthand where "75"
means 75,000 DZD. All three are in the data. Without a correction step this dataset is
unusable. That call was correct.

The issue is *how* the ambiguity gets resolved. In `PreProcessing.ipynb`, `fix_price_scale()`:

```python
ratio = real_price / estimated_price      # estimated_price = rule-based CPU+GPU+RAM+SSD+brand sum
while ratio >= 8:   real_price /= 10
while ratio <= 1/8: real_price *= 10
```

826 rows had their price rewritten this way, and `price_corrected` became the training
target. So a hand-crafted pricing model built *from the features* was used to reshape the
target that a second model then predicts *from those same features*. For rewritten rows,
part of what XGBoost learns is our own price table.

**But I measured how much of the correction actually needs the features, and the answer is:
almost none of it.** A rule that uses only the price — repeatedly ×10 or ÷10 until it lands
in the plausible laptop band of 10,000–1,000,000 DZD, no features at all — **reproduces
`fix_price_scale` on 15,928 of 16,317 rows (97.62%)**.

The raw log₁₀ distribution shows why it's that easy:

```
1e1.0   219 ###     <- spoken shorthand ("75" = 75,000)
1e1.5   236 ###
1e2.0    37
1e2.5     9         <- near-empty gap
1e3.0    26
1e3.5    32
1e4.0   694 ###########
1e4.5  7570 #############################################################  <- dinars
1e5.0  6707 #############################################
1e5.5   630 ##########
1e6.0    27         <- near-empty gap
1e6.5    33         <- centimes
1e7.0     3
1e7.5     2
```

Three modes with near-empty gaps between them. The units are separable on the price axis
alone. (Also worth noting the split of what the correction actually did: **679 rows were
scaled *up*** — the "75" shorthand — and only **72 were scaled down** from centimes. The
shorthand convention was the bigger problem, and it isn't mentioned in the report.)

**The 389 rows (2.4%) where the two approaches disagree are the genuinely hard cases, and
there your approach is the better one:**

| raw | `fix_price_scale` | price-only rule |
|---|---|---|
| 6,000,000 | 60,000 | 600,000 |
| 9,900,000 | 99,000 | 990,000 |
| 12,000,000 | 1,200,000 | 120,000 |

`6000000` is ambiguous by construction — 600,000 DZD and 60,000 DZD are *both* plausible
laptop prices, so the price axis alone genuinely cannot decide. If the components are worth
~55k, then 60,000 is obviously right and the component estimate earned its keep. **My
original "never let a feature-derived estimate touch the target" was too strong; for these
rows there is no purely price-based answer.**

**So the fix isn't to abandon the approach — it's to scope it:**

1. **Run the price-only rule first.** It settles 97.6% of cases with zero feature contact,
   so 97.6% of the target is provably uncontaminated.
2. **Fall back to the component estimate only for the ~389 genuinely ambiguous rows** —
   i.e. only when *two or more* candidate rescalings both land in the plausible band.
3. **Flag those rows** with `price_unit_ambiguous = True`. Then report test metrics
   **with and without them**. At 2.4% of the data, dropping them from the *test set* costs
   nothing and removes every doubt about the headline number. That single extra column
   converts a criticism into a demonstrated non-issue.
4. Use trailing-zero structure as a tiebreaker before falling back to features — 90.6% of
   corrected prices end in `000`, and `6000000` having 6 trailing zeros is itself evidence.
5. Two small bugs in the current loop regardless: it's unbounded (add an iteration cap), and
   it produced `9999999 → 99999.99`, a non-round price that is also one of the surviving
   troll values from §A7.

Keeping `estimated_component_cost` as a **feature** and as a **baseline row in the results
table** is also worth doing — it's a genuinely interesting comparison ("does XGBoost beat a
component-sum heuristic?") and in that role it's completely legitimate.

## A2. 🔴 41% of `spec_Etat` is missing, and we encoded it in the worst possible way

`spec_Etat` (condition) is ordinally encoded as `MOYEN=1 · BON ÉTAT=2 · JAMAIS UTILISÉ=3 · missing=0`.

But our own notebook prints the mean price per bucket:

| encoded | meaning | mean price |
|---|---|---|
| 1 | MOYEN | 51,412 |
| 2 | BON ÉTAT | 88,700 |
| 3 | JAMAIS UTILISÉ | 190,083 |
| **0** | **missing (6,437 rows, 41%)** | **114,649** |

The 1→2→3 ordering is beautifully monotonic. Then we put the largest group of all at
**0**, at the bottom of the scale, when its true price level sits *between* buckets 2 and 3.
Trees can partially route around this; for linear models it is actively poisonous — and
it's part of why Ridge/Lasso plateaued at R² ≈ 0.48.

**Worse:** we wrote a 60-line `infer_laptop_state()` to fill those 6,437 values… and
**never called it**. Cell 36 only applies `encode_etat_ordinal`. It's dead code. (Which
is lucky — that function reads `price_preview` to infer condition, which would have been
direct target leakage. We dodged that by accident.)

**Fixes, increasing in sophistication:**
1. Encode missing as `NaN` and use a model that handles it natively (LightGBM, XGBoost,
   HistGradientBoosting, CatBoost all do). Missingness gets its own learned split.
2. Add an explicit `etat_is_missing` boolean. Missingness is informative here — listings
   without a stated condition have a 9% lower median price (90,000 vs 99,000 DZD).
3. Impute properly: train a small classifier on the 9,470 labelled rows using specs +
   brand + city + listing age, **never price**. Report its accuracy and keep the
   predicted-vs-observed flag as a feature.

## A3. 🔴 We threw away three of the strongest signals in the dataset

`created_at`, `city`, and `model_name` were all dropped before modelling. Brand wasn't
exactly dropped — it was crushed into a 4-level `model_family` tier that is itself computed
from mean `cpu_mark` and `gpu_g3d_mark` per model. So "brand" was replaced by a lossy
re-encoding of two features the model already has. 47 brands → 4 buckets, with 11,800 of
15,517 rows (76%) in one bucket. That column is nearly constant.

Meanwhile the raw columns carry real signal:

- **City** — median price ranges from 161,950 DZD (Hydra) to 69,000 DZD (Dar El Beïda)
  among cities with 100+ listings. **A 2.35× spread.** Algiers-centre neighbourhoods
  command a large premium. Deleted.
- **`created_at`** — listings span 2018→2025, 10,900 of 16,392 in 2025 alone. Seven years
  of DZD inflation, tech depreciation and the 2021-22 GPU spike sit in that column.
  Deleted. A 2019 listing and a 2025 listing for the same laptop are *not* the same
  observation.
- **`model_name`** — ThinkPad vs IdeaPad at identical specs is a large price difference in
  the used market. Deleted.

**The cost of this is measurable.** Same gradient-boosting model, same target, same split —
the only change is adding these columns back as-is, with zero clever engineering:

| Feature set | R² | MAE (DZD) | Median APE |
|---|---|---|---|
| Our current numeric features | 0.7930 | 22,497 | 12.4% |
| + brand (`model_name`, 47 levels) | 0.8013 | 21,655 | 12.3% |
| + city (top-200 + OTHER) | 0.8033 | 21,489 | 12.1% |
| **+ listing month index** | **0.8142** | **19,977** | **11.0%** |

**MAE drops 11% for free.** No new models, no tuning, no scraping — just not deleting
columns we already have. Cheapest win available; it should be step 1.

## A4. 🟠 43% of rows are duplicate configurations — and identical laptops sell 2–9× apart

Of 15,517 rows there are only **8,810 unique feature combinations**. 6,707 rows (43.2%)
are exact duplicates in feature space; 1,439 (9.3%) are exact duplicates including price.

I checked whether this inflated our score through train/test leakage. Honest answer:
**it didn't, much.** Grouping identical configs so they can't straddle the split gives
R² = 0.8224 vs 0.8016 for the random split — essentially unchanged. Our reported number
isn't fake. But two real things follow:

1. **Our effective sample size is ~8,810, not 15,517.** Model capacity and CV design
   should be calibrated to that.
2. **There is a hard noise ceiling.** Identical spec rows sell at wildly different prices:

   | RAM | SSD | CPU mark | listings | min | median | max | spread |
   |---|---|---|---|---|---|---|---|
   | 8GB | 256GB | 5,800 | 67 | 10,000 | 52,000 | 88,000 | **8.8×** |
   | 8GB | 256GB | 14,133 | 72 | 40,000 | 118,000 | 230,000 | **5.8×** |
   | 8GB | 256GB | 5,800 | 76 | 39,000 | 52,950 | 119,000 | **3.1×** |

   Median max/min ratio across all repeated configs: **1.5×**.

   Predicting the perfect group mean for every config would give R² ≈ 0.954 in log space —
   and that's an optimistic upper bound. **We are at 0.82 against a ceiling near 0.95, and
   most of the remaining gap isn't in the spec columns at all.** It's in seller reputation,
   negotiability, listing photos, title text, urgency, whether the charger is included, and
   whether the price is a real ask or an anchor.

   **This reframes the accuracy conversation.** Squeezing 0.83 out of a better XGBoost is a
   rounding error. Getting listing text and seller context into the model is where the
   remaining 0.13 lives. See §3.

## A5. 🟠 The clustering is degenerate

`clustering.ipynb` reports a silhouette of **0.9796** at k=4 and calls it a success. A
silhouette that high on real tabular data is a red flag, not a trophy — and the cluster
sizes give it away: **14,161 / 498 / 550**. We didn't segment the market, we peeled a
handful of extreme outliers off one blob.

Causes:
- `RobustScaler` on features whose IQR is zero (`HDD_SIZE` is 0 at both the 25th and 75th
  percentile — 92% of listings have no HDD). Scaling is a no-op there and raw magnitudes
  dominate.
- Raw `price_preview` (1,800 → 3,550,000) fed in alongside `SSD_SIZE` (up to 12,800).
  Euclidean distance then lives in one or two dimensions.
- The scaler fitted on the train split gets `fit_transform` called on it *again* over the
  full dataset in the clustering cell, silently discarding the earlier fit.

`clustering_optimized.ipynb` is better (silhouette 0.67, sizes 5,226/2,979/2,944/4,057) but
measures silhouette **in PCA space after reducing** — which flatters the score, because PCA
discards exactly the directions that make clusters look messy. Report silhouette in the
space you clustered in, and report Davies-Bouldin and Calinski-Harabasz alongside it.

## A6. 🟡 The saved model artifacts don't fit together

```
saved_models/scaler.pkl      → expects 14 features
saved_models/best_model.pkl  → expects 10 features
selected_features.json       → lists 10
```

Also, the XGBoost model was trained on `X_train_selected` — **unscaled**. So the shipped
scaler is both the wrong shape *and* something the model never saw. Anyone loading these
three files and following the obvious path (scale → predict) gets silent garbage. This is
the exact bug that turns a working notebook into a broken deployment, and it's why §9
proposes one `Pipeline` object instead of three loose pickles.

## A7. 🟡 Smaller things, worth a cleanup pass

- **The troll-price filter is dead code.** It sets `estimated_price_dzd = NaN` for junk
  prices, but the next section *recomputes that column from scratch*, wiping the effect.
  Prices of `11111`, `99999`, `111111`, `999999` are still in the final dataset. 10 rows —
  not fatal, but the report claims they were removed.
- **Polynomial interaction features** are computed (10 → 55) and never used by any model.
- **`gpu_g2d_mark`/`gpu_g3d_mark`** are heavily collinear, as are `cpu_mark`/`tdp`/`cores`.
  The report's "multicollinearity addressed" claim is wrong — `gpu_g2d_mark` was
  *selected*, not eliminated.
- **Report/notebook drift.** The regression report says feature selection "correctly
  dropped `match_score` (noise)" — `match_score` isn't in the feature list at all. It also
  cites 15,517 rows and 60/20/20, but the split ran on 15,418 rows after outlier trimming.
  Small, but it costs marks and credibility.
- **Cell 40 references `best_grid`**, never defined in the notebook, and reports
  R² = 0.8253 where cell 29 says 0.8271. The notebook isn't reproducible top-to-bottom.
- **RAM imputation hardcodes 16 CPU→RAM pairs by hand** for 21 rows. Fine for a course,
  won't survive new data.
- **`cpu_generation_normalized` was dropped** because 607 rows (3.9%) were NaN. We deleted
  a whole feature rather than impute or let the model handle NaN.
- **No MAPE anywhere.** For a price product, "typically within X%" is the metric users
  understand. Our median APE is ~12%; that's the number for the slide.
- **`n_init=1` in the optimized clustering k-search** but `n_init=20` for the final fit —
  the k we selected was chosen under noisier conditions than the model we fitted.
- **`data_cleaned.csv` and `original_data.csv` are byte-identical.** Two names, one file.

---

# PART B — Where we take it next

## 1. 🎯 Fix the free stuff first (a weekend)

Before any new model, library, or web app:

1. Put `created_at`, `city`, `model_name` back. (§A3 — MAE −11%, measured.)
2. Encode missing `spec_Etat` as `NaN`, add `etat_is_missing`. (§A2)
3. Reorder `fix_price_scale`: price-only rule first (settles 97.6%), component estimate only
   as the tiebreaker for the ~389 ambiguous rows, and flag those with
   `price_unit_ambiguous`. (§A1)
4. Ship one `sklearn.Pipeline` instead of three mismatched pickles. (§A6)
5. Report MAPE / median APE alongside R². (§A7)
6. Add **`GroupKFold` on spec-signature** so we can honestly say duplicates aren't
   inflating anything — and prove it, since we now know they don't.
7. Delete the dead code (troll filter, polynomial features, unused inference function) or
   wire it in.

That's the whole difference between "a course project" and "a project that survives a code
review."

## 2. 📡 Rebuild the data layer

The dataset is the ceiling. Everything else is negotiating with it.

**Re-scrape with much more per listing.** The current 15 columns are a fraction of what a
Ouedkniss ad actually contains:

- **Listing title and full description text** — the biggest missed feature by far. Sellers
  write "clavier rétroéclairé", "sous garantie 2 ans", "batterie à changer", "écran
  tactile", "prix négociable", "échange possible", "facture disponible". Every one of those
  moves the price and none are in our dataset.
- **Photo count and the photos themselves** — listings with 8 photos convert differently
  from listings with 1. A CLIP embedding of the hero image encodes condition and whether
  it's a stock photo (likely a shop, likely new) or a desk photo (likely private sale).
- **Seller identity, shop-vs-private flag, listing count, member-since, response rate.**
  A registered shop in Bab Ezzouar and a student selling their old laptop price the same
  machine very differently.
- **`is_negotiable` / "prix fixe"**, **delivery offered**, **exchange accepted**.
- **View count, favourite count, days-online, bump count** — demand signal. A laptop
  relisted four times at a falling price tells you the first price was wrong.
- **Listing lifecycle** — the big one. Snapshot the same ad IDs weekly; when an ad
  disappears it probably sold. That converts our *asking-price* dataset into a
  **transaction-price** dataset, which is a categorically more valuable thing.

**Honesty note:** every model in this repo predicts *asking price*, not *sale price*. The
reports should say so. It's the difference between "what will this sell for" and "what will
someone put on the internet hoping to get."

**Enrich from outside:**
- Full PassMark / Geekbench / Cinebench / 3DMark tables (we have partial CPU/GPU maps).
- Laptop release year per model → **true machine age**, distinct from listing age.
  `age = listing_year − release_year` is probably worth more than either alone.
- Original MSRP per model → **`price_retention_ratio = current_price / MSRP`**, which
  normalises across tiers and is what a used-market pricing model actually wants.
- **DZD/EUR/USD parallel-market exchange rate by month.** In Algeria the "square" rate
  moves imported-electronics prices directly. A monthly FX series joined on `created_at`
  would likely explain much of the temporal drift we're blind to.
- Algerian CPI / electronics inflation index, to deflate prices to constant DZD before
  modelling, then re-inflate at prediction time.

**Data quality infrastructure:**
- A **Great Expectations** or `pandera` schema so bad data fails loudly instead of silently
  producing a 43%-duplicate dataset nobody noticed.
- A **data card** per column: source, unit, null rate, transformations, known issues. We'd
  have caught the scaler mismatch in a week instead of never.
- **DVC** to version the CSVs, so we stop having byte-identical files under two names.

## 3. 🧬 Feature engineering, properly

### 3a. Text features (highest expected value)

The listing title/description is unmined. Three tiers of ambition:

| Tier | Method | Effort |
|---|---|---|
| Cheap | TF-IDF char+word n-grams on title → SVD to 50 dims → concat to tabular | hours |
| Good | Keyword flags: `garantie`, `négociable`, `échange`, `facture`, `tactile`, `rétroéclairé`, `batterie`, `fissure`, `original`, `import` | hours |
| Best | Multilingual sentence embeddings — the text is French + Arabic + Darja + English, often in one sentence. `paraphrase-multilingual-MiniLM-L12-v2` or `multilingual-e5-base` | days |

The Algerian-dialect mix is a real research angle — most laptop-price papers use clean
English e-commerce data. **A multilingual code-switched used-market dataset is a
publishable novelty**, not just a course project.

### 3b. Ratio and interaction features

Raw specs are less informative than *value ratios*:

```python
gpu_to_cpu_ratio  = gpu_g3d_mark / cpu_mark        # gaming vs productivity build
storage_per_ram   = ssd_size / ram_size            # balanced vs lopsided config
pixels            = width * height                 # instead of an ordinal tier
ppi               = sqrt(w**2 + h**2) / screen_size # what people actually see
total_storage     = ssd + hdd
has_hdd, is_dual_drive, is_ssd_only
tdp_class         = f(cpu_tdp + gpu_tdp)           # ultrabook / mainstream / desktop-replacement
is_gaming         = dedicated_gpu and gpu_tdp > 45
is_workstation    = quadro/rtx-a gpu or xeon cpu or model in {ZBook, Precision, ThinkPad P}
is_ultrabook      = screen <= 14 and tdp <= 15 and no dGPU
perf_per_expected_dinar = cpu_mark / estimated_component_cost
```

### 3c. Temporal features (currently zero)

```python
listing_year, listing_month, listing_quarter
days_since_epoch                  # monotonic drift
month_sin, month_cos              # seasonality
is_ramadan, is_back_to_school     # Algeria-specific demand spikes
cpu_age_at_listing = listing_year - cpu_release_year
model_age_at_listing
fx_rate_that_month, cpi_that_month
```

Critically: **switch to a time-based split.** Train on ≤2024, test on 2025. Random splits
on time-series-ish data flatter the model. Our 0.82 is a random-split number; the "can we
price a laptop listed tomorrow" number will be lower — and that's the number that matters
for a product.

### 3d. Geographic features

```python
is_alger, is_grand_alger, wilaya (city→wilaya map), region (nord/hauts-plateaux/sud)
city_median_price_ratio    # target-encoded, OUT-OF-FOLD ONLY
distance_to_alger_km       # geocode the 470 cities
city_listing_density       # market thickness -> liquidity -> price
```

⚠️ Target encoding on `city` must be **out-of-fold** (K-fold or leave-one-out with
smoothing) or it leaks straight into the score. Classic footgun; we'd trip it on the first
try if we're not careful.

### 3e. Better categorical encoding (the brief asked for this explicitly)

We used ordinal encoding almost everywhere, including for things with no natural order
(brand), plus one-hot in `clustering_optimized_encoding.ipynb`. Neither is right for
47-brand / 470-city / 643-CPU cardinality. Worth benchmarking head-to-head:

| Encoding | Best for | Notes |
|---|---|---|
| **CatBoost ordered target encoding** | high-cardinality, everything | built-in, leak-safe by construction — probably our single best move |
| **Out-of-fold target/mean encoding** | city, brand, CPU model | needs careful CV; smoothing by group size |
| **Weight of Evidence** | after binning price | interpretable, good for the report |
| **Hashing trick** | CPU/GPU names (643/151 levels) | fixed width, handles unseen categories at inference |
| **Entity embeddings** | brand, city, CPU | learn a dense vector per category in a small NN; **the embeddings become a deliverable** — plot them in 2D and you get a learned brand map |
| **Frequency / count encoding** | everything, as a cheap extra | "how common is this config" is itself predictive |
| **Learned ordinal** | condition, resolution | fit the spacing from data instead of hardcoding 1/2/3 |
| **Native categorical** | LightGBM/CatBoost | declare them categorical, let the tree split on subsets |

**Proposal: make encoding an experiment, not an assumption.** One notebook, one model,
seven encoders, one table. Genuinely interesting result, and it directly answers "other
ways to do feature encoding" from the brief.

### 3f. Automated feature discovery

- **`featuretools`** deep feature synthesis over the listing/seller/city relational structure.
- **`tsfresh`** on per-model price series once we have listing history.
- **Symbolic regression** (`gplearn`, PySR) to *discover* the ratio features above rather
  than guessing them — and it outputs a human-readable formula, which makes a great slide.

## 4. 🤖 Modelling: past "which gradient booster wins"

### 4a. Models to add

- **CatBoost** — almost certainly beats our XGBoost here; our data is categorical-heavy and
  CatBoost's ordered target statistics are built for exactly this.
- **Explainable Boosting Machine (InterpretML)** — a GA²M. Accuracy near GBM, but you can
  *plot the exact price contribution of every feature*, including pairwise interactions.
  For a pricing product that's worth more than 0.005 R², because you can show the user *why*.
- **TabPFN v2** — a pre-trained transformer for tabular data. At our size it's frequently
  competitive with heavily tuned GBMs, out of the box, in seconds. Worth ten minutes.
- **FT-Transformer / TabNet** — deep tabular baselines; interesting as comparison and
  because they hand us the entity embeddings from §3e for free.
- **Stacked ensemble** — LightGBM + CatBoost + EBM + Ridge-on-embeddings with a
  meta-learner. Where the last 1–2% lives.
- **Optuna** instead of GridSearchCV. Our LightGBM grid was 576 candidates × 5 folds =
  2,880 fits to explore a space Optuna covers better in ~100 trials.

### 4b. Predict a *range*, not a point — the real product upgrade

Given §A4 (identical laptops sell 2–9× apart), **a single number is the wrong output.**
A user asking "what's my laptop worth" should get:

> **Estimated fair range: 78,000 – 96,000 DZD** (most likely ~86,000)
> Based on 34 similar listings. Prices for this config vary a lot — condition and included
> accessories matter more than usual here.

Ways to produce that:
- **LightGBM quantile regression** at q = 0.1 / 0.5 / 0.9 — three models, trivial to train.
- **NGBoost** — predicts a full distribution, not a point.
- **Conformal prediction** (MAPIE) — *statistically guaranteed* coverage with no
  distributional assumptions. Clean, easy to explain in a report, rare in student projects.
- Then report **prediction-interval coverage and width** as first-class metrics.

### 4c. Model the *right target*

- Train on **log price** (we do) but also try **price per performance-unit** or
  **price ÷ MSRP retention ratio** — often much easier to learn.
- **Two-stage:** classify price tier (budget/mid/high/premium), then regress within tier.
  Specialist models beat one generalist when price formation differs by segment — and it
  does here: gaming vs business vs bureautique.
- **Monotonic constraints**: more RAM should never *decrease* the predicted price.
  XGBoost/LightGBM support this directly. Costs nothing, makes the sliders in our UI behave
  sensibly, and prevents embarrassing demos.

### 4d. Honest evaluation

- Time-based split (§3c) as the *headline* number.
- Per-segment error tables: by price decile, brand, city, condition, year. Aggregate R²
  hides that we're probably terrible on premium/rare machines.
- **Baselines we never included**: global median; group median for the exact spec config;
  the component-cost estimate. A model that can't beat "median of identical listings" isn't
  earning its complexity.
- Learning curves — are we data-limited or model-limited? With 8,810 unique configs that's
  a real question and we never asked it.
- **SHAP** for global and per-prediction explanations. Feeds straight into the UI (§7).

## 5. 🧩 Clustering, redone as market segmentation

Scrap the current approach and rebuild around a question: **what natural segments exist in
the Algerian laptop market, and how does each one price?**

- **Gower distance** so we can cluster mixed numeric + categorical data without butchering
  it into numbers first.
- **UMAP → HDBSCAN** instead of PCA → K-Means. HDBSCAN finds variable-density clusters and
  refuses to force outliers into a group — a much better fit for a long-tailed marketplace.
- **Gaussian Mixture Models** for soft assignment. A laptop can be 70% "student
  bureautique", 30% "budget gaming". More honest than a hard label.
- **Cluster in a supervised-relevant space**: use SHAP values or supervised UMAP as the
  embedding, so clusters group by *what drives their price*, not raw specs.
- **Name the segments.** A cluster is worthless until you can say "this is the ~90,000 DZD
  refurbished-ThinkPad-for-students segment, 12% of the market, listings sit 4 days."
  Auto-generate the label with an LLM from the cluster's centroid stats.
- **Report a full metric panel** — silhouette *in the clustering space*, Davies-Bouldin,
  Calinski-Harabasz, and **cluster stability under bootstrap resampling**. Stability is what
  separates real structure from artefacts, and it would have flagged the 0.98-silhouette
  problem immediately.
- **Consensus clustering** across algorithms and seeds.
- **Per-cluster pricing models** — this closes the loop: segmentation that measurably
  improves regression is segmentation that means something.

## 6. 🚨 Anomaly detection — the missing deliverable, and the best product idea

This was in the brief, we never built it, and it has the clearest real-world value. Four
distinct products fall out of one model family:

### 6a. Scam / bait detection
An RTX 4090 machine listed at 60,000 DZD is not a bargain, it's a scam or a typo.
`IsolationForest` + `LocalOutlierFactor` + `OneClassSVM` on the (specs, price) joint
distribution, plus the residual approach below.

### 6b. Underpriced deal finder — *the killer feature*
Flip the residual: listings where **actual ≪ predicted** are genuine bargains. Rank by
`(predicted − actual) / predicted`, filter out the scam-shaped ones from 6a, and you have a
**daily "best deals on Ouedkniss right now" feed**. That's a product people would actually
use, and it's ~40 lines on top of what we already have.

### 6c. Spec-inconsistency detection
Association rules (which we already built!) are perfect for this. If `{RTX 4060} → {16GB
RAM}` holds at 82% confidence, a listing claiming *RTX 4060 + 4GB DDR3* is probably
mistyped or fraudulent. **This is the natural bridge between our association-rules
deliverable and our anomaly-detection deliverable, and it makes the project hang together
as one story instead of three separate assignments.**

### 6d. Duplicate / relist detection
43% of our rows are spec-duplicates (§A4); some are the *same physical laptop* relisted.
Fuzzy-match on (specs + seller + city + price) to collapse them, and use relist history as
a demand signal — a laptop relisted 5× is overpriced by revealed preference.

**Methods to benchmark:** IsolationForest, LOF, One-Class SVM, Elliptic Envelope,
**autoencoder reconstruction error**, **COPOD/ECOD** (PyOD — fast, parameter-free,
underused), and **quantile-residual outliers** from §4b. Evaluating unsupervised anomaly
detection is genuinely hard, so: hand-label ~200 listings as scam/normal/bargain and report
precision@k. That small labelling effort turns an unfalsifiable section into a real result.

## 7. 🖥️ The platform

### 7a. "What's my laptop worth?" — the core flow

- **Three input modes:**
  1. **Spec form** — dropdowns/sliders for CPU, GPU, RAM, storage, screen, condition, city.
  2. **Paste a Ouedkniss URL** — scrape and parse it live. Much lower friction, and it
     shows off the whole pipeline.
  3. *(stretch)* **Upload a photo of the laptop or a screenshot of the spec sheet** → a
     vision model extracts the specs. Demo gold.
- **Output:** a price *range* (§4b), not a number, plus:
  - **A waterfall chart** showing how each feature moved the price from the market baseline
    to this estimate — straight from SHAP. "Base 95,000 → RTX 4060 +48,000 → only 8GB RAM
    −12,000 → Bab Ezzouar +6,000 → **137,000 DZD**."
  - A **percentile badge**: "your asking price is in the 78th percentile for this config —
    expect it to sit on the market ~3 weeks."
  - **Similar listings** side by side (§7b).
  - A **"what would raise the value" panel**: "upgrading 8→16GB adds ~14,000 DZD to the
    estimate — an upgrade that costs ~9,000. Do it before listing." Genuinely actionable,
    and it falls straight out of the model.

### 7b. Similar-laptop recommender

- **k-NN in the learned embedding space** (entity embeddings from §3e, or SHAP space), not
  raw Euclidean on unscaled specs.
- Three tabs, because "similar" means three different things:
  - **Same machine, cheaper elsewhere** (exact spec match, sorted by price)
  - **Better value at this price** (same budget, higher performance-per-dinar)
  - **Cheaper alternatives that still meet your needs** ("you said CAD — here are 4
    machines 30% cheaper that still clear the bar")
- Powered by **FAISS** or `pgvector` so it stays fast at 100k+ listings.

### 7c. External listing links

- Deep-link to the live Ouedkniss search for that exact config (query-string construction —
  no scraping needed for the link itself).
- **Amazon / eBay / AliExpress new-price comparison**: "this used machine is 68% of the
  price of a new equivalent." That ratio is the most useful single number for a buyer, and
  we can compute it as soon as we have MSRP data from §2.
- Affiliate links are the obvious monetisation path if this ever goes public.

### 7d. Other surfaces

- **Buyer mode**: "I have 120,000 DZD and I need it for gaming" → ranked shortlist with
  reasoning.
- **Seller mode**: optimal asking price *and* expected time-to-sale at each price point.
  A price/time-on-market curve is a compelling chart.
- **Market dashboard**: price trends by segment, brand market share, cheapest cities, a GPU
  price index. Public-facing, updates weekly, gets shared.
- **Alerts**: "notify me when a ThinkPad with 16GB under 90,000 DZD appears in Algiers."
  The feature that gives people a reason to come back.
- **A public API** so others can build on it.

### 7e. Stack

Frontend **Next.js + Tailwind + Recharts/visx**, or **Streamlit** if we want it working
this week. Backend **FastAPI**, **PostgreSQL + pgvector**, model served as a single pipeline
artifact. Vercel + Railway/Fly.io. Redis for prediction caching. Nightly scrape on a
scheduled job. Realistically: **Streamlit first to prove the flows, Next.js once they're
settled.**

## 8. 📊 Visualisation — 2D and 3D

### 2D
- **Interactive price-vs-performance scatter** (Plotly/Altair): colour = brand, size = RAM,
  hover = full spec card + listing link. Brush-select to filter the table below.
- **Choropleth of Algeria** by wilaya: median price, listing count, "best place to buy".
- **Correlation heatmap done right** — dendrogram-ordered, not alphabetical.
- **Parallel-coordinates plot** across all specs, brushable — the single best way to *see*
  what defines a cluster.
- **Ridgeline plot** of price distributions per brand or per segment.
- **Residual maps**: where is the model wrong, and does the error correlate with city,
  brand, or time? Ours almost certainly does, and we've never looked.
- **Animated time-lapse** of the price/performance frontier 2019→2025. The GPU shortage
  should be visible as a bulge — a beautiful, self-explanatory chart.
- **SHAP beeswarm + per-prediction waterfall** (§7a).

### 3D
- **UMAP/t-SNE 3D embedding of the whole market**, coloured by cluster, price or brand,
  rotatable in the browser via `plotly` or `three.js`. This is the "wow" visual — and
  because the embedding is *learned from price behaviour* (§5) it isn't decoration:
  neighbourhoods in that space are genuinely comparable laptops.
- **3D price surface**: CPU mark × GPU mark × price as a mesh, with actual listings as
  points floating above and below it. Points far below the surface are §6b bargains. One
  picture that explains the entire product.
- **3D decision boundary** of the price-tier classifier.
- Stretch: **WebXR market walkthrough**. Silly, memorable, ~3 hours with `react-three-fiber`.

**Design rule:** every chart must answer a question someone actually asked. We currently
have a lot of charts that exist because the notebook had a plotting cell.

## 9. ⚙️ Engineering (turning notebooks into a system)

- **`src/` package, notebooks that only orchestrate.** Right now the preprocessing logic is
  145 cells with duplicated normalisation functions across three notebooks.
- **One `sklearn.Pipeline`** end-to-end: imputers → encoders → model. Serialise it whole.
  Fixes §A6 by construction and makes train/inference skew impossible.
- **Config-driven** (Hydra/YAML) — no more magic numbers scattered across cells.
- **MLflow or Weights & Biases**. We ran hundreds of fits and the only record is notebook
  output that gets overwritten on re-run.
- **pytest** on the preprocessing functions. `parse_ram_size("16GB")` deserves a test; it
  currently has a print statement.
- **Great Expectations** validation gates in the pipeline.
- **Prefect/Airflow** DAG: scrape → validate → preprocess → train → evaluate → deploy, weekly.
- **Drift monitoring** (Evidently) — laptop prices move; the model must know when it's stale.
- **Model card + data card**, documenting intended use, limitations, and the
  asking-price-≠-sale-price caveat.
- **Docker** so the whole thing runs anywhere.

## 10. 🌟 Ideas that would make this stand out

- **LLM listing parser.** Feed raw title + description to a small LLM with a structured
  output schema and get clean specs out. It would handle "i5 11ème gen 8go ram ssd 256
  clavier azerty rétroéclairé nickel" — which our regex pipeline mangles — and would
  probably recover a good chunk of the 875 rows we dropped for unmappable CPUs.
- **Negotiation-margin model.** With listing-lifecycle data (§2), predict the gap between
  asking and final price. "This seller will likely accept 82,000." Nobody has this.
- **Time-to-sell prediction** as a second target — survival analysis (Cox / Kaplan-Meier)
  on listing duration. Pairs with pricing to give a genuine price/speed tradeoff curve.
- **Counterfactual explanations** (DiCE): "if the condition were *jamais utilisé* instead of
  *bon état*, the estimate would be 118,000 instead of 89,000."
- **Fairness / consistency audit**: does the model systematically underprice listings from
  southern wilayas relative to comparable Algiers listings? If so, is that real market
  structure or our data being Algiers-heavy (5,757 of ~16k listings are Bab Ezzouar alone)?
  Genuinely interesting, and nobody in the course will have done it.
- **Cross-market arbitrage**: compare Algerian used prices to eBay/Amazon EU. Which models
  are unusually cheap or expensive here, and does that track import channels?
- **A "laptop passport"** — permalink per configuration with full price history, current
  listings, and a trend line. Shareable, SEO-friendly, the thing that makes a site grow.
- **Publish the dataset.** A cleaned, documented Algerian used-laptop marketplace dataset on
  Kaggle/HuggingFace would be the first of its kind and costs us nothing but a data card.
- **Write it up.** A multilingual, code-switched, developing-market used-goods pricing
  dataset with a full pipeline is a legitimate short paper. The Algerian-dialect angle and
  the parallel-FX-rate feature are both genuinely novel.

---

## Suggested order of work

| Phase | What | Why now |
|---|---|---|
| **0 — Repair** | §1 in full: restore dropped features, fix `spec_Etat`, scope the price-unit correction, one pipeline artifact, MAPE, group-aware CV | Measured −11% MAE for a weekend's work, and fixes the credibility problems in the reports |
| **1 — Deliver the missing brief** | §6 anomaly detection (a, b, c) | It was required, it's fast now, and 6b is the best demo in the project |
| **2 — Data** | §2 re-scrape with text, seller, photos, lifecycle | Everything downstream is capped by this; §A4 shows specs alone can't get past ~0.95 and realistically much less |
| **3 — Features & models** | §3 text + temporal + geo features, §3e encoding bake-off, §4 CatBoost/EBM/TabPFN + quantile intervals | Where the accuracy actually is |
| **4 — Product** | §7 platform, §8 visualisations | Now there's something worth wrapping a UI around |
| **5 — Rigour** | §5 clustering redo, §9 engineering, §10 write-up | Turns it from a demo into a defensible piece of work |

---

## The one-paragraph version

We built a competent course project with a missing deliverable (anomaly detection), a
degenerate clustering result we reported as a success, a price-unit correction that solves a
real problem but lets feature-derived estimates edit the target on 2.4% more rows than it
needs to, and three strong predictive columns deleted for no stated reason — restoring which
cuts MAE by 11% before we write a single new model. The
path from here isn't a fancier gradient booster: identical laptops in our own data sell 2–9×
apart, so the remaining error lives in listing text, seller context, timing and photos, none
of which we scraped. Fix the audit items, scrape richer data, predict a *range* instead of a
number, ship the underpriced-deal detector, and wrap it in a platform that tells someone what
their laptop is worth, why, and who's selling a better one down the street.
