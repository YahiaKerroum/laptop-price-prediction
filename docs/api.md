# API reference

Base URL `http://localhost:8000`. Interactive docs at `/docs`, OpenAPI at `/openapi.json`.

```bash
make serve
```

---

## `GET /health`

Liveness and readiness. Used by the container healthcheck.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v20260911T1241",
  "package_version": "1.0.0"
}
```

`status` is `degraded` when no model is loaded. The service still starts in that state so it
can report why; prediction endpoints then return 503.

---

## `GET /schema`

The feature contract the loaded pipeline was actually fitted on.

```json
{
  "model_version": "v20260911T1241",
  "target": "price_corrected",
  "numeric_features": ["RAM_SIZE", "SSD_SIZE", "..."],
  "categorical_features": ["brand", "city_grouped", "cpu_manufacturer", "cpu_family"],
  "trained_at": "2026-09-11T12:37:00+00:00",
  "git_sha": "a1b2c3d",
  "metrics": {"R2": 0.8165, "MAE": 21051.0, "MedAPE": 12.1}
}
```

This endpoint exists because the previous release shipped a scaler expecting 14 features next
to a model expecting 10, with nothing to reveal the mismatch. Publishing the contract means a
caller can check rather than guess.

---

## `POST /predict`

Estimate an asking price.

**Query parameters**

| Name | Default | Effect |
|---|---|---|
| `explain` | `false` | Include per-feature SHAP contributions |

**Body** — every field optional. Omitting what you do not know is *better* than guessing:
the model learned a split for "missing", and a fabricated zero reads as a real, very low
value.

| Field | Type | Notes |
|---|---|---|
| `ram_gb` | float | 1–128 |
| `ssd_gb`, `hdd_gb` | float | 0 means no such drive |
| `cpu_mark` | float | PassMark CPU score |
| `gpu_g3d_mark` | float | PassMark G3D; 0 for integrated |
| `tdp`, `gpu_tdp`, `cores` | float | |
| `screen_size` | float | 10–20 inches |
| `resolution` | float | Tier 1–9: 1=HD, 3=FHD, 5=QHD, 8=4K |
| `condition` | float | 1=MOYEN, 2=BON ÉTAT, 3=JAMAIS UTILISÉ — **omit if unknown** |
| `ram_type` | float | DDR generation, ordinal |
| `brand`, `city` | string | Free text; unseen values fall back safely |
| `listing_year`, `listing_month` | float | |

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"ram_gb":16,"ssd_gb":512,"cpu_mark":19776,"gpu_g3d_mark":16758,
       "screen_size":15.6,"resolution":3,"condition":2,
       "brand":"THINKPAD","city":"ALGER CENTRE","listing_year":2025}'
```

```json
{
  "estimate_dzd": 109100.0,
  "range_dzd": [105200.0, 147900.0],
  "quantiles": {"0.1": 105200.0, "0.5": 110900.0, "0.9": 147900.0},
  "model_version": "v20260911T1241",
  "currency": "DZD",
  "caveat": "Asking price, not sale price. Prices for a given configuration vary widely; treat the range as the real answer."
}
```

**`range_dzd` is the answer.** `estimate_dzd` alone overstates what the model knows —
identical configurations in the training data sell 2–9× apart. Observed interval coverage is
72.4% against a nominal 80%.

With `?explain=true`:

```json
"contributions": [
  {"feature": "cpu_mark", "contribution": -0.0960},
  {"feature": "estimated_component_cost", "contribution": -0.0642}
]
```

Contributions are in log space. Multiply through `expm1` for a percentage effect on price.

---

## `GET /deals`

Listings priced well below the model's estimate.

| Parameter | Default | Range |
|---|---|---|
| `limit` | 20 | 1–200 |
| `min_discount` | 35.0 | 0–95 (percent) |

```bash
curl 'localhost:8000/deals?limit=5&min_discount=40'
```

```json
[
  {
    "asking_price_dzd": 12000.0,
    "predicted_dzd": 29918.6,
    "discount_pct": 59.9,
    "verdict": "bargain",
    "brand": "DYNABOOK",
    "city": "DRARIA",
    "ram_gb": 6.0,
    "ssd_gb": 120.0,
    "cpu_mark": 2019.0
  }
]
```

Scam-shaped listings are filtered out first. A deep discount on an otherwise anomalous
(specs, price) pair is more likely bait than a bargain, and a feed that cannot tell them
apart is worse than no feed. `verdict` is one of `bargain`, `suspicious`, `overpriced`,
`normal`; only `bargain` is returned.

**This endpoint is expensive** — it fits the anomaly detectors over the full dataset on every
request. Cache or precompute before exposing it to real traffic.

---

## `POST /reload`

Re-read `models/latest` without restarting the container. Use after training a new version.

```json
{"reloaded": true, "model_version": "v20260911T0204", "error": null}
```

---

## Errors

| Code | Meaning |
|---|---|
| 422 | Request body failed validation — the response names the field |
| 503 | No model loaded, or `features.csv` missing for `/deals` |

There is no authentication or rate limiting. Put a gateway in front before exposing this
publicly, and narrow the CORS policy from its current `*`.
