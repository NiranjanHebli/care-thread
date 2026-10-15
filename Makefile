.PHONY: up down health schema ingest seed-clinic test test-iso loadtest bench ui lint clean

PYTHON ?= uv run python
PYTEST ?= uv run pytest
RUFF ?= uv run ruff

up:
	docker compose up -d

down:
	docker compose down

health:
	@echo "Checking FalkorDB health..."
	@docker compose exec -T falkordb redis-cli ping || echo "FalkorDB not reachable"

schema:
	$(PYTHON) -m carethread.db.schema

seed-clinic:
	$(PYTHON) scripts/seed_clinic.py

ingest:
	$(PYTHON) scripts/ingest.py

lint:
	$(RUFF) check src tests
	$(RUFF) format --check src tests

format:
	$(RUFF) format src tests
	$(RUFF) check --fix src tests

test:
	CT_LLM_MODE=mock $(PYTEST) tests/unit tests/integration

test-iso:
	CT_LLM_MODE=mock $(PYTEST) tests/isolation

loadtest:
	$(PYTHON) loadtest/harness.py

bench:
	$(PYTHON) benchmark/run.py

ui:
	cd ui && npm run build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
