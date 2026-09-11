# Architecture

## The shape of the problem

Scraped marketplace listings are messy in three specific ways that drive every design
decision here:

1. **Free-text specifications.** CPU, RAM and storage arrive as whatever the seller typed.
   RAM and storage are routinely swapped between columns.
2. **Three price conventions in one column.** Dinars, centimes (×100), and spoken shorthand
   (`"75"` = 75,000 DZD) all appear, sometimes in adjacent rows.
3. **A hard noise ceiling.** Identical configurations sell 2–9× apart, so there is a limit
   to what any model of the spec columns can achieve.

## Layers

```
┌──────────────────────────────────────────────────────────────────┐
│  notebooks/                                                       │
│  Narrative, charts, and the hand-curated CPU/GPU enrichment.      │
│  Import from src/; never define pipeline logic of their own.      │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  src/laptop_price/                                                │
│                                                                   │
│   cleaning/   price · memory · screen · cpu                       │
│               pure functions, no I/O, fully unit-tested           │
│                            │                                      │
│   features/   build_feature_matrix() + pandera contract           │
│                            │                                      │
│   models/     one sklearn Pipeline · training · registry          │
│                            │                                      │
│   evaluation/ metrics · baselines · three split strategies        │
│                            │                                      │
│   anomaly/    detectors · deal ranking · rules-based consistency  │
│                            │                                      │
│   serving.py  partial listing → prediction (shared by API + UI)   │
└───────────────────────────┬──────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌───────────────────┐                 ┌────────────────────┐
│  api/  FastAPI    │◀────────────────│  app/  Streamlit   │
│  :8000            │   (health only) │  :8501             │
└───────────────────┘                 └────────────────────┘
```

### Why `serving.py` is shared

The API and the UI answer the same question and must answer it identically. A partial
listing — the user knows RAM and CPU but not the GPU TDP — has to be turned into the exact
feature vector the pipeline was fitted on, with derived ratios computed the same way
training computed them. Duplicating that in two places is how train/serve skew starts, so
there is one implementation and both surfaces call it.

The Streamlit app talks to the model directly rather than through the API, so the UI still
works when the API container is down.

## The single artifact

Training emits **one** object:

```
models/v20260911T1241/
├── price_pipeline.joblib       ColumnTransformer → imputers → encoders → estimator
├── quantile_pipelines.joblib   the 10th/50th/90th percentile models
├── metadata.json               feature contract, metrics, git SHA, library versions
├── model_card.md               generated from the metadata
└── results.csv                 the full comparison table
```

This is a direct response to the previous release, which shipped three loose pickles:

```
saved_models/scaler.pkl      → expects 14 features
saved_models/best_model.pkl  → expects 10 features
selected_features.json       → lists 10
```

and the model had been trained on *unscaled* data. Anyone following the obvious path —
load the scaler, transform, predict — got silent garbage. With one `Pipeline` that failure
is unrepresentable: the preprocessing is inside the artifact, so the transformation applied
at inference is by construction the one fitted during training.

`metadata.json` publishes the feature contract, and `GET /schema` serves it. A caller can
check what the model expects instead of guessing.

## Data flow

| Stage | Owner | Input | Output |
|---|---|---|---|
| Scrape | *(external)* | Ouedkniss | `data/raw/data.csv` |
| CPU/GPU enrichment, cleaning | `notebooks/01` | `data/raw/`, `data/mappings/` | `data/processed/pre_processed_data.csv` |
| Original encoding | `notebooks/03` | `pre_processed_data.csv` | `model_ready_data.csv` |
| Repaired feature build | `laptop_price.data` | `pre_processed_data.csv` | `features.csv` |
| Training | `laptop_price.models.train` | `features.csv` | `models/<version>/` |
| Serving | `api/`, `app/` | `models/latest` | predictions |

Two model-ready tables exist on purpose. `model_ready_data.csv` reproduces the original
course artifact so the old results stay checkable; `features.csv` carries the repairs and is
what the service uses. They were previously the same file, and re-running notebook 03
silently clobbered the package's matrix — training then failed on a missing target column.

## Configuration

Every magic number that used to be scattered across notebook cells lives in
`src/laptop_price/config.yaml`: the plausible price band, the exchange rate, rare-level
thresholds, split parameters, anomaly thresholds, rule-mining parameters. Point
`$LAPTOP_PRICE_CONFIG` at another YAML file to override any subset.

Paths resolve through `src/laptop_price/paths.py`, which walks up to the directory holding
`pyproject.toml` (or honours `$LAPTOP_PRICE_ROOT`). Nothing depends on the working
directory, so notebooks run from `notebooks/` and the CLI runs from anywhere.

`data/raw/` is treated as immutable. Nothing in the project writes to it — the original
notebook's CPU price chain, which read and rewrote `cpus.csv` five times in sequence, now
writes to `data/interim/cpus_priced.csv` instead.
