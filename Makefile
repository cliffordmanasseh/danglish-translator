.PHONY: install install-speech install-all test lint fmt fetch run serve talk bench doctor clean

PY ?= 3.12

install:
	uv sync --python $(PY) --extra dev --extra server

install-speech:
	uv sync --python $(PY) --extra dev --extra server --extra speech

install-all:
	uv sync --python $(PY) --all-extras

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

fetch:
	uv run --extra convert danglish fetch

run:
	uv run danglish repl

serve:
	uv run --extra server danglish serve

talk:
	uv run --extra speech danglish talk

bench:
	uv run danglish bench

doctor:
	uv run danglish doctor

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
