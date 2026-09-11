# syntax=docker/dockerfile:1
#
# Multi-stage image for the laptop-price project.
#
#   base  -> pinned scientific stack + the laptop_price package
#   lab   -> JupyterLab, for running the notebooks             (port 8888)
#   api   -> FastAPI prediction service                        (port 8000)
#   app   -> Streamlit "what is my laptop worth" UI            (port 8501)
#
# Python is pinned to 3.11 because the notebooks are pandas-2 era code; see
# docs/deployment.md for why this matters.

# ----------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LAPTOP_PRICE_ROOT=/app \
    MPLBACKEND=Agg

# libgomp1 is required by xgboost/lightgbm; the rest keeps wheels building.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

# Non-root by default; the compose file bind-mounts data/ and models/ which the
# entrypoints need to write, so keep ownership aligned.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/models /app/reports \
    && chown -R appuser:appuser /app

# ----------------------------------------------------------------------------
FROM base AS dev

COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
USER appuser

# ----------------------------------------------------------------------------
FROM dev AS lab

EXPOSE 8888
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.password=", \
     "--ServerApp.root_dir=/app"]

# ----------------------------------------------------------------------------
FROM base AS api

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "laptop_price.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ----------------------------------------------------------------------------
FROM base AS app

USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "/app/src/laptop_price/app/main.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
