.PHONY: help install test smoke backtest seed perf retrain clean run lint format

PYTHON ?= python
PIP    ?= pip

help:
	@echo "Veilcrean — common targets:"
	@echo "  make install   - install Python dependencies"
	@echo "  make test      - run the test suite"
	@echo "  make smoke     - run end-to-end smoke test"
	@echo "  make backtest  - run a backtest"
	@echo "  make seed      - seed the trade journal with 200 synthetic trades"
	@echo "  make perf      - show performance summary"
	@echo "  make retrain   - force a retraining cycle"
	@echo "  make run       - launch the brain (requires ZMQ + MT5 EA)"
	@echo "  make lint      - run flake8"
	@echo "  make format    - run black"
	@echo "  make clean     - remove caches & generated artifacts"

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

smoke:
	$(PYTHON) scripts/run_smoke_test.py

backtest:
	$(PYTHON) scripts/backtest.py --bars 2000 --n-trades 50

seed:
	$(PYTHON) scripts/seed_journal.py --n 200

perf:
	$(PYTHON) scripts/show_performance.py

retrain:
	$(PYTHON) scripts/force_retrain.py

run:
	$(PYTHON) -m python_brain.main

lint:
	flake8 python_brain/ scripts/ --max-line-length=120 --ignore=E501,W503,E203

format:
	black python_brain/ scripts/ tests/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
