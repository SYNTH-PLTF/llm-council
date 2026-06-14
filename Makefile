.PHONY: setup dev lint fmt types test test-cov check eval load up clean

setup:
	uv sync

dev:
	uv run uvicorn ai_council.api.app:app --reload --host 127.0.0.1 --port 8000

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

types:
	uv run pyright

test:
	uv run pytest

test-cov:
	uv run pytest --cov --cov-report=term-missing

check: lint types test

eval:
	uv run python -m ai_council.evals.run

load:
	uv run locust -f tests/load/locustfile.py --host http://localhost:8000

up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build
