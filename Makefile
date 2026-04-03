.PHONY: install test test-cov lint clean

install:
	pip install -e ".[all]"

test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --tb=short --cov=hla_analysis --cov-report=term-missing

lint:
	ruff check hla_analysis/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
