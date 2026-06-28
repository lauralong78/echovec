#!/usr/bin/env bash
# Run the full local check suite: lint, format, type-check and tests.
set -euo pipefail

echo "==> ruff check"
ruff check .

echo "==> ruff format --check"
ruff format --check .

echo "==> mypy"
mypy

echo "==> pytest"
pytest --cov=echovec --cov-report=term-missing

echo "All checks passed."
