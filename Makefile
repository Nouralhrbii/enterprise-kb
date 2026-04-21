.PHONY: install test lint format serve ingest-all ingest-docs ingest-tickets ingest-changelog eval deploy clean

install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	cp -n .env.example .env || true

test:
	pytest -m "not integration" -v

test-cov:
	pytest -m "not integration" --cov=src --cov=api --cov-report=term-missing

lint:
	ruff check src/ api/ config/ tests/ scripts/

format:
	ruff format src/ api/ config/ tests/ scripts/

serve:
	uvicorn api.main:app --reload --port 8001

# Ingestion — run all three sources in correct order
ingest-all: ingest-docs ingest-tickets ingest-changelog

ingest-docs:
	python scripts/ingest_docs.py --folder data/raw/docs --overwrite

ingest-tickets:
	python scripts/ingest_tickets.py --folder data/raw/tickets

ingest-changelog:
	python scripts/ingest_changelog.py --folder data/raw/changelog

eval:
	python scripts/run_evaluation.py \
		--test-set data/eval_set.json \
		--output docs/eval_report.json

deploy:
	az webapp up \
		--name enterprise-kb \
		--resource-group rg-enterprise-kb \
		--runtime "PYTHON:3.11" \
		--sku B2

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache