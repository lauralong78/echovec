.PHONY: install lint format typecheck test cov check clean

install:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy

test:
	pytest

cov:
	pytest --cov=echovec --cov-report=term-missing

check: lint typecheck test

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
