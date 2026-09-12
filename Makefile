# Everything runs inside Docker, so the only host requirement is Docker itself.
# `make help` lists the targets.

SHELL := /bin/bash
COMPOSE := docker compose -f docker/docker-compose.yml
IMAGE := laptop-price:dev
DOCKER_RUN := docker run --rm \
	-v "$(CURDIR)/src:/app/src:ro" \
	-v "$(CURDIR)/data:/app/data" \
	-v "$(CURDIR)/models:/app/models" \
	-v "$(CURDIR)/reports:/app/reports" \
	-v "$(CURDIR)/notebooks:/app/notebooks" \
	-v "$(CURDIR)/docs:/app/docs" \
	-v "$(CURDIR)/tests:/app/tests:ro" \
	-e LAPTOP_PRICE_ROOT=/app \
	$(IMAGE)

.DEFAULT_GOAL := help
.PHONY: help build rebuild pipeline train predict deals test lint format notebooks docs \
        lab serve up down logs clean verify all

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build:  ## Build the Docker image (do this first; skipped if it already exists)
	@if docker image inspect $(IMAGE) >/dev/null 2>&1; then \
		echo "$(IMAGE) already exists - skipping. Use 'make rebuild' to force."; \
	else \
		docker build -f docker/Dockerfile --target dev -t $(IMAGE) .; \
	fi

rebuild:  ## Force a rebuild of the Docker image
	docker build --pull -f docker/Dockerfile --target dev -t $(IMAGE) .

pipeline:  ## Rebuild data/processed/model_ready_data.csv from the preprocessed listings
	$(DOCKER_RUN) python -m laptop_price pipeline --data-card

train:  ## Train the model and write models/<version>/
	$(DOCKER_RUN) python -m laptop_price train

deals:  ## Print the current top underpriced listings
	$(DOCKER_RUN) python -m laptop_price deals --top 20

predict:  ## Score one example listing (override with ARGS=...)
	$(DOCKER_RUN) python -m laptop_price predict \
		$(if $(ARGS),$(ARGS),--ram 16 --ssd 512 --cpu-mark 19776 --gpu-mark 16758 --brand THINKPAD)

test:  ## Run the test suite
	$(DOCKER_RUN) python -m pytest

lint:  ## Check formatting and lint rules
	$(DOCKER_RUN) bash -c "ruff check src tests && black --check src tests"

format:  ## Apply formatting
	docker run --rm -v "$(CURDIR):/app" -w /app $(IMAGE) \
		bash -c "ruff check --fix src tests && black src tests"

notebooks:  ## Execute every numbered notebook top-to-bottom (slow)
	$(DOCKER_RUN) bash scripts/run_notebooks.sh

verify:  ## Check the repo matches docs_for_claude/DataminingProject (1)
	python3 scripts/verify_sync.py

docs:  ## Build the documentation site into site/
	docker run --rm -v "$(CURDIR):/app" -w /app $(IMAGE) mkdocs build --strict

lab:  ## JupyterLab at http://localhost:8888
	$(COMPOSE) --profile lab up

serve:  ## API at :8000 and UI at :8501
	$(COMPOSE) --profile serve up

up:  ## Start everything
	$(COMPOSE) --profile all up -d
	@echo "lab  http://localhost:8888"
	@echo "api  http://localhost:8000/docs"
	@echo "app  http://localhost:8501"

down:  ## Stop everything
	$(COMPOSE) --profile all down

logs:  ## Tail the container logs
	$(COMPOSE) --profile all logs -f

clean:  ## Remove generated artifacts (keeps data/raw and the reports)
	rm -rf site .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

all: build pipeline train test  ## Build, run the pipeline, train, and test
