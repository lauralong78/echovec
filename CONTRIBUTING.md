# Contributing to echovec

Thanks for taking the time to contribute! echovec is a small, dependency-light
research toolkit, and the goal is to keep it readable and well-tested.

## Development setup

```bash
git clone https://github.com/lauralong78/echovec.git
cd echovec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running the checks

The same three checks run in CI:

```bash
ruff check .          # lint
ruff format --check . # formatting
mypy                  # type-checking
pytest                # tests (with coverage: pytest --cov=echovec)
```

`pre-commit run --all-files` runs lint, formatting and type-checking together.

## Guidelines

- **Keep the core free of heavy dependencies.** NumPy is the only required
  runtime dependency; please do not pull in a deep-learning framework.
- **New ops need gradient tests.** Anything added to `echovec.autograd` should
  come with a finite-difference check (see `echovec.testing.check_gradient`).
- **One concept per commit.** Small, focused commits are much easier to review.
- **Update the docs.** If you change behaviour, update the relevant file under
  `docs/` and add a `CHANGELOG.md` entry under *Unreleased*.

## Reporting bugs

Open an issue using the bug-report template and include a minimal reproduction.
A failing test is the most useful bug report of all.
