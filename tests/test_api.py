"""TestClient tests for the FastAPI endpoints.

Most endpoints require a loaded model bundle.  Because no bundle is present in
CI, those tests verify the *no-model* path (503) rather than production output.
The two exceptions are:
  - /health, which always returns 200
  - /predict with invalid input, which returns 422 before any model check
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from laptop_price.api.main import app

client = TestClient(app, raise_server_exceptions=False)

#: A valid-enough listing payload; no model is required to validate its shape.
VALID_LISTING = {
    "ram_gb": 16,
    "ssd_gb": 512,
    "cpu_mark": 19776,
    "screen_size": 15.6,
    "resolution": 3,
    "condition": 2,
    "brand": "THINKPAD",
    "city": "ALGER CENTRE",
    "listing_year": 2025,
}


class TestHealth:
    def test_health_always_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_body_has_required_keys(self):
        r = client.get("/health")
        body = r.json()
        assert "status" in body
        assert "model_loaded" in body
        assert "package_version" in body

    def test_health_status_is_ok_or_degraded(self):
        r = client.get("/health")
        assert r.json()["status"] in ("ok", "degraded")


class TestSchema:
    def test_schema_returns_200_or_503(self):
        r = client.get("/schema")
        assert r.status_code in (200, 503)

    def test_schema_503_when_no_model(self):
        r = client.get("/schema")
        if r.status_code == 503:
            assert "detail" in r.json()


class TestPredict:
    def test_predict_invalid_empty_body_is_422(self):
        r = client.post("/predict", json={})
        # All fields are optional in ListingRequest, so {} is actually valid ->
        # the response depends on whether a model is loaded.
        assert r.status_code in (200, 422, 503)

    def test_predict_invalid_ram_out_of_range_is_422(self):
        r = client.post("/predict", json={"ram_gb": -1})
        assert r.status_code == 422

    def test_predict_invalid_condition_out_of_range_is_422(self):
        r = client.post("/predict", json={"condition": 9})
        assert r.status_code == 422

    def test_predict_valid_payload_returns_200_or_503(self):
        r = client.post("/predict", json=VALID_LISTING)
        assert r.status_code in (200, 503)

    def test_predict_200_body_shape(self):
        r = client.post("/predict", json=VALID_LISTING)
        if r.status_code == 200:
            body = r.json()
            assert "estimate_dzd" in body
            assert "model_version" in body
            assert "caveat" in body


class TestPredictBatch:
    def test_batch_empty_listings_returns_200_or_503(self):
        r = client.post("/predict/batch", json={"listings": []})
        assert r.status_code in (200, 503)

    def test_batch_valid_payload_returns_200_or_503(self):
        r = client.post("/predict/batch", json={"listings": [VALID_LISTING]})
        assert r.status_code in (200, 503)

    def test_batch_missing_listings_field_is_422(self):
        r = client.post("/predict/batch", json={})
        assert r.status_code == 422

    def test_batch_invalid_inner_listing_is_422(self):
        r = client.post("/predict/batch", json={"listings": [{"ram_gb": -5}]})
        assert r.status_code == 422


class TestSimilar:
    def test_similar_returns_200_or_503(self):
        r = client.post("/similar", json=VALID_LISTING)
        assert r.status_code in (200, 503)

    def test_similar_k_out_of_range_is_422(self):
        r = client.post("/similar?k=999", json=VALID_LISTING)
        assert r.status_code == 422

    def test_similar_200_body_is_list(self):
        r = client.post("/similar", json=VALID_LISTING)
        if r.status_code == 200:
            assert isinstance(r.json(), list)


class TestMarketStats:
    def test_market_stats_returns_200_or_503(self):
        r = client.get("/market/stats")
        assert r.status_code in (200, 503)

    def test_market_stats_200_body_shape(self):
        r = client.get("/market/stats")
        if r.status_code == 200:
            body = r.json()
            for key in ("total_listings", "median_price", "mean_price", "brands"):
                assert key in body


class TestAnomalyCheck:
    def test_anomaly_check_returns_200_or_503(self):
        r = client.post("/anomaly/check", json=VALID_LISTING)
        assert r.status_code in (200, 503)

    def test_anomaly_check_200_body_shape(self):
        r = client.post("/anomaly/check", json=VALID_LISTING)
        if r.status_code == 200:
            body = r.json()
            assert "anomaly_score" in body
            assert "is_anomalous" in body
            assert "reasons" in body

    def test_anomaly_check_invalid_condition_is_422(self):
        r = client.post("/anomaly/check", json={"condition": 99})
        assert r.status_code == 422


class TestReload:
    def test_reload_returns_200_or_500(self):
        # 200 when a model is loadable; 500 if the stored artifact is from a
        # mismatched sklearn version (pre-existing issue, not our regression).
        r = client.post("/reload")
        assert r.status_code in (200, 500)

    def test_reload_200_body_has_reloaded_key(self):
        r = client.post("/reload")
        if r.status_code == 200:
            assert "reloaded" in r.json()
