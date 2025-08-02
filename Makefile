# Photonic Neural Network Foundry - Development Makefile
# Provides standardized commands for development, testing, and deployment

.PHONY: help install install-dev clean test test-all lint format security docker docs

# Default target
help: ## Show this help message
	@echo "Photonic Neural Network Foundry - Development Commands"
	@echo "======================================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment Variables:"
	@echo "  PYTHON_VERSION  Python version to use (default: 3.11)"
	@echo "  CONTAINER_TAG   Docker image tag (default: latest)"
	@echo "  PYTEST_ARGS    Additional pytest arguments"

# Python and environment detection
PYTHON_VERSION ?= 3.11
PYTHON := python$(PYTHON_VERSION)
VENV_NAME := venv
VENV_PATH := ./$(VENV_NAME)
VENV_PYTHON := $(VENV_PATH)/bin/python
VENV_PIP := $(VENV_PATH)/bin/pip

# Container settings
CONTAINER_TAG ?= latest
IMAGE_NAME := photonic-foundry
FULL_IMAGE_NAME := $(IMAGE_NAME):$(CONTAINER_TAG)

# Development commands
install: ## Install package in development mode
	$(PYTHON) -m pip install -e .

install-dev: ## Install package with development dependencies
	$(PYTHON) -m pip install -e .[dev,docs,test]

venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV_PATH)
	$(VENV_PIP) install --upgrade pip setuptools wheel
	$(VENV_PYTHON) -m pip install -e .[dev,docs,test]
	@echo "Virtual environment created at $(VENV_PATH)"
	@echo "Activate with: source $(VENV_PATH)/bin/activate"

clean: ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .tox/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf logs/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name ".DS_Store" -delete

# Testing commands
test: ## Run unit tests
	pytest tests/unit/ -v $(PYTEST_ARGS)

test-integration: ## Run integration tests
	pytest tests/integration/ -v $(PYTEST_ARGS)

test-e2e: ## Run end-to-end tests
	pytest tests/e2e/ -v $(PYTEST_ARGS)

test-performance: ## Run performance tests
	pytest tests/performance/ -v --benchmark-only $(PYTEST_ARGS)

test-security: ## Run security tests
	pytest tests/security/ -v $(PYTEST_ARGS)

test-all: ## Run all tests with coverage
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term $(PYTEST_ARGS)

test-parallel: ## Run tests in parallel
	pytest tests/ -n auto -v $(PYTEST_ARGS)

# Code quality commands
lint: ## Run all linting tools
	ruff check src/ tests/
	flake8 src/ tests/
	mypy src/
	bandit -r src/

lint-fix: ## Fix linting issues automatically
	ruff check --fix src/ tests/
	isort src/ tests/
	black src/ tests/

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

format-check: ## Check code formatting without making changes
	black --check src/ tests/
	isort --check-only src/ tests/

security: ## Run security scanning
	bandit -r src/
	safety check
	pip-audit

# Docker commands
docker-build: ## Build Docker image
	docker build -t $(FULL_IMAGE_NAME) .

docker-run: ## Run Docker container interactively
	docker run -it --rm -v $(PWD):/workspace $(FULL_IMAGE_NAME) /bin/bash

docker-jupyter: ## Start Jupyter Lab in Docker container
	docker run -it --rm -p 8888:8888 -v $(PWD):/workspace $(FULL_IMAGE_NAME) \
		jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

docker-test: ## Run tests in Docker container
	docker run --rm -v $(PWD):/workspace $(FULL_IMAGE_NAME) \
		pytest tests/ -v --cov=src

docker-clean: ## Clean Docker images and containers
	docker system prune -f
	docker rmi $(FULL_IMAGE_NAME) 2>/dev/null || true

# Documentation commands
docs: ## Build documentation
	cd docs && sphinx-build -b html . _build/html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

docs-clean: ## Clean documentation build artifacts
	cd docs && rm -rf _build/

# Development workflow commands
dev-setup: venv pre-commit ## Set up development environment
	@echo "Development environment setup complete!"
	@echo "Activate virtual environment: source $(VENV_PATH)/bin/activate"

pre-commit: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

pre-commit-all: ## Run pre-commit on all files
	pre-commit run --all-files

# Release commands
version-patch: ## Bump patch version
	bump2version patch

version-minor: ## Bump minor version
	bump2version minor

version-major: ## Bump major version
	bump2version major

build: ## Build distribution packages
	$(PYTHON) -m build

upload-test: ## Upload to TestPyPI
	twine upload --repository testpypi dist/*

upload: ## Upload to PyPI
	twine upload dist/*

# Utility commands
requirements: ## Generate requirements.txt from pyproject.toml
	pip-compile pyproject.toml
	pip-compile --extra dev -o requirements-dev.txt pyproject.toml

update-deps: ## Update all dependencies
	pip-compile --upgrade pyproject.toml
	pip-compile --upgrade --extra dev -o requirements-dev.txt pyproject.toml

check-deps: ## Check for dependency vulnerabilities
	safety check
	pip-audit

# Development server commands
dev: ## Start development server
	python -m photonic_foundry.cli --dev

jupyter: ## Start Jupyter Lab
	jupyter lab --ip=0.0.0.0 --port=8888

# Benchmarking and profiling
benchmark: ## Run performance benchmarks
	pytest tests/performance/ --benchmark-only --benchmark-sort=mean

profile: ## Profile application performance
	python -m cProfile -o profile.stats -m photonic_foundry.cli
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('tottime').print_stats(20)"

# Multi-environment testing
tox: ## Run tests across multiple Python versions
	tox

tox-parallel: ## Run tox tests in parallel
	tox -p auto

# Continuous integration simulation
ci: lint test-all security ## Run full CI pipeline locally
	@echo "✅ CI pipeline completed successfully!"

# Project initialization for new developers
init: clean dev-setup pre-commit ## Initialize project for new developer
	@echo "🎉 Project initialization complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Activate virtual environment: source $(VENV_PATH)/bin/activate"
	@echo "2. Run tests: make test"
	@echo "3. Start development: make dev"
	@echo "4. View available commands: make help"

# Environment information
info: ## Show environment information
	@echo "Environment Information:"
	@echo "======================="
	@echo "Python version: $(shell python --version)"
	@echo "Virtual env: $(VENV_PATH)"
	@echo "Current directory: $(PWD)"
	@echo "Git branch: $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'Not a git repository')"
	@echo "Git commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repository')"
	@echo "Docker available: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Container image: $(FULL_IMAGE_NAME)"