.PHONY: install test lint collect build

install:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests

collect:
	.venv/bin/volsurface collect

build:
	.venv/bin/volsurface build
