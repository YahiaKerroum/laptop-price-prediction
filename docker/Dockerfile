# Multi-stage image for the laptop-price project.
#
#   base  -> pinned scientific stack + the laptop_price package
#   lab   -> JupyterLab, for running the notebooks             (port 8888)
#   api   -> FastAPI prediction service                        (port 8000)
#   app   -> Streamlit "what is my laptop worth" UI            (port 8501)
#
# Python is pinned to 3.11 because the notebooks are pandas-2 era code; see
# docs/deployment.md for why this matters.
#
# Deliberately no `# syntax=` directive: it makes every build fetch a frontend
# image from Docker Hub first, which fails on a slow or restricted network even
# though nothing here needs BuildKit-specific syntax.

# ----------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # The scientific wheels are large; on a slow link pip's 15s default read
    # timeout aborts the build partway through. Be patient instead.
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
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
# xgboost declares nvidia-nccl-cu12 on linux/x86_64: 342 MB of CUDA multi-GPU
# collectives that a CPU-only image never loads. Downloading it is the single
# most likely step of this build to fail on a slow link, and it adds ~350 MB to
# the image, so xgboost is installed without its dependency closure. Everything
# it actually needs (numpy, scipy) is pinned in requirements.txt and installed
# by the first command, which keeps requirements.txt the single source of truth.
RUN grep -v '^xgboost' requirements.txt > /tmp/requirements-no-xgb.txt \
    && pip install --no-cache-dir -r /tmp/requirements-no-xgb.txt \
    && pip install --no-cache-dir --no-deps "$(grep '^xgboost' requirements.txt)" \
    && rm /tmp/requirements-no-xgb.txt

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
# requirements-dev.txt begins with "-r requirements.txt", which would re-resolve
# xgboost *with* its dependency closure and pull the 342 MB CUDA package the
# base stage went out of its way to avoid. The base stage already installed all
# of requirements.txt, so drop that line and pass the file as a constraint
# instead: constraints pin versions without requesting installs, so a dev tool
# still cannot drag numpy or scipy off their pins.
# The sed strips extras ("uvicorn[standard]==..." -> "uvicorn==..."): pip
# rejects a constraints file containing them.
RUN grep -v '^-r requirements.txt' requirements-dev.txt > /tmp/requirements-dev-only.txt \
    && sed 's/\[[^]]*\]//' requirements.txt > /tmp/constraints.txt \
    && pip install --no-cache-dir -r /tmp/requirements-dev-only.txt -c /tmp/constraints.txt \
    && rm /tmp/requirements-dev-only.txt /tmp/constraints.txt
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

# ----------------------------------------------------------------------------
# Self-contained API for cloud hosts (Render, Railway, Fly): the model and the
# data it reads are baked in rather than bind-mounted. Last stage, so it is what
# a plain `docker build` produces. Hosts pass the port to listen on in $PORT.
FROM base AS deploy

COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser data/processed/features.csv data/processed/pre_processed_data.csv ./data/processed/
COPY --chown=appuser:appuser data/raw/cpu_prices.csv data/raw/gpu_prices.csv ./data/raw/
USER appuser
EXPOSE 8000
CMD uvicorn laptop_price.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
