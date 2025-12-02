.PHONY: help install test train-ddp train-fsdp benchmark clean lint format

help: ## Show this help message
	@echo "Distributed Training Lab - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install dependencies including dev tools
	pip install -r requirements.txt
	pip install black flake8 mypy pytest pytest-cov

test: ## Run tests
	python -m pytest tests/ -v

test-cov: ## Run tests with coverage
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

train-ddp: ## Train with DDP (single GPU)
	python scripts/train_ddp.py

train-ddp-multi: ## Train with DDP (4 GPUs)
	torchrun --nproc_per_node=4 scripts/train_ddp.py

train-fsdp: ## Train with FSDP (single GPU)
	python scripts/train_fsdp.py

train-fsdp-multi: ## Train with FSDP (4 GPUs)
	torchrun --nproc_per_node=4 scripts/train_fsdp.py

benchmark: ## Run DDP vs FSDP benchmark (4 GPUs)
	torchrun --nproc_per_node=4 scripts/benchmark.py

benchmark-2: ## Run DDP vs FSDP benchmark (2 GPUs)
	torchrun --nproc_per_node=2 scripts/benchmark.py

lint: ## Run linter
	flake8 src scripts tests --max-line-length=100 --exclude=__pycache__

format: ## Format code with black
	black src scripts tests --line-length=100

type-check: ## Run type checker
	mypy src scripts --ignore-missing-imports

clean: ## Clean generated files
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf checkpoints
	rm -rf profiles
	rm -rf results
	rm -rf *.egg-info
	rm -rf dist build

check: lint type-check test ## Run all checks (lint, type-check, test)

all: clean install test ## Clean, install, and test

