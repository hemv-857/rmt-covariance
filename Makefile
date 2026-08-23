.PHONY: setup test demo calibrate lint

setup:
	pip install -e ".[dev]"

test:
	pytest

demo:
	python -m rmtcov.cli demo --paths 40000

calibrate:
	python -m rmtcov.cli calibrate

lint:
	ruff check src tests
