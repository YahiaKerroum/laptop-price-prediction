# Deployment

## Why Docker here

This is not boilerplate. The notebooks are pandas-2 era Colab code, and the host this was
consolidated on runs **pandas 3.0.3** with `rapidfuzz` and `mlxtend` absent — the pipeline
simply cannot execute there. The shipped `.pkl` artifacts additionally emit version warnings
under XGBoost 3.x.

Pinning the stack is the only honest path to "reproducible". The image fixes Python 3.11,
pandas 2.2.3, scikit-learn 1.5.2, XGBoost 2.1.3 and LightGBM 4.5.0; see `requirements.txt`
for the full set, all exact-pinned.

## Image targets

One multi-stage `docker/Dockerfile`:

| Target | Contents | Serves |
|---|---|---|
| `base` | Pinned runtime deps + the package | — |
| `dev` | `base` + pytest, ruff, black, papermill, mkdocs | Builds, tests, notebooks |
| `lab` | `dev` + JupyterLab | :8888 |
| `api` | `base` + uvicorn entrypoint | :8000 |
| `app` | `base` + streamlit entrypoint | :8501 |

`libgomp1` is installed because XGBoost and LightGBM need it. Containers run as a non-root
`appuser` (uid 1000).

## Compose

```bash
make up       # everything: lab, api, app
make serve    # just api + app
make lab      # just JupyterLab
make down
```

Profiles (`lab`, `serve`, `all`) mean you start only what you need. `data/` and `models/` are
bind-mounted and shared, so a model trained in the lab is immediately servable by the API
without a rebuild — `POST /reload` picks it up without a restart.

The API and app mount `data/` and `models/` **read-only**. Only the lab and the CLI write.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LAPTOP_PRICE_ROOT` | auto-detected | Project root; set to `/app` in containers |
| `LAPTOP_PRICE_CONFIG` | unset | Path to a YAML file overriding any of `config.yaml` |
| `LAPTOP_PRICE_API_URL` | `http://api:8000` | Where the Streamlit app looks for the API |

To change a threshold without touching code:

```yaml
# my-config.yaml
anomaly:
  deal_threshold: 0.45
price:
  max_dzd: 1_500_000
```

```bash
docker run -e LAPTOP_PRICE_CONFIG=/app/my-config.yaml ...
```

## Health checks

Both service targets declare a `HEALTHCHECK`. `GET /health` distinguishes two states:

```json
{"status": "ok", "model_loaded": true, "model_version": "v20260911T0203"}
{"status": "degraded", "model_loaded": false, "model_version": null}
```

The API starts even with no model present, so `/health` can report *why* it is not ready
instead of crash-looping. Prediction endpoints return **503** with an actionable message.

## Model artifacts

```
models/
├── latest                   plain-text pointer to the current version
├── v20260911T0203/
│   ├── price_pipeline.joblib
│   ├── quantile_pipelines.joblib
│   ├── metadata.json
│   ├── model_card.md
│   └── results.csv
└── legacy_v0/               the original three pickles, kept for provenance
```

`legacy_v0/` is **not loadable as a unit** — its scaler expects 14 features and its model 10,
and the model was trained unscaled. It is retained so the original submission stays
inspectable, not because it works. Nothing loads it.

Rolling back is editing one file:

```bash
echo "v20260911T0159" > models/latest
curl -X POST localhost:8000/reload
```

## Deploying elsewhere

The compose file is a development convenience. For a real deployment:

1. Build and push the `api` target.
2. Ship `models/<version>/` alongside it — as a volume, an init-container fetch, or baked in.
3. Put TLS in front; the service speaks plain HTTP and has no auth.
4. Point the healthcheck at `/health` and treat `degraded` as not-ready.

CORS is currently `allow_origins=["*"]`. Narrow it before exposing the service publicly.

## Operations

**Retraining.** Nothing retrains automatically. `make pipeline && make train` writes a new
version and moves the pointer; `POST /reload` picks it up.

**Drift.** The model is trained on listings up to 2025. Algerian laptop prices move with the
parallel exchange rate, so it goes stale. `listing_month_index` is a feature, which means the
model extrapolates temporally — increasingly badly the further past its training window it
runs. Nothing currently monitors this; Evidently is the roadmap item.

**Logs.** Plain stdout, captured by `make logs`.

## Known operational limits

- **No authentication or rate limiting.** Fine behind a gateway, not on the open internet.
- **`/deals` scores the full dataset per request** — it fits the anomaly detectors each time,
  which takes seconds. Cache it or precompute before exposing it to real traffic.
- **Single-process uvicorn.** Add `--workers` behind a load balancer for concurrency.
- **The model is loaded once at startup** and cached per process. `POST /reload` is the only
  way to change it without a restart.
