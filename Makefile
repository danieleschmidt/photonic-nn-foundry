# Photonic Neural Network Foundry - Development Makefile
# This Makefile provides convenient commands for development tasks

.PHONY: help install dev clean test lint format check docs docker security bench all

# Default target
.DEFAULT_GOAL := help

# Colors for terminal output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Project settings
PROJECT_NAME := photonic-nn-foundry
PYTHON := python3
PIP := pip3
DOCKER_IMAGE := photonic-foundry:latest

help: ## Show this help message
	@echo "$(BLUE)Photonic Neural Network Foundry - Development Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"; category=""} /^##.*:/ {category=substr($$0,4); print "$(YELLOW)" category "$(RESET)"} /^[a-zA-Z_-]+:.*?##/ { if (category) printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2; else printf "$(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^$$/ {category=""}' $(MAKEFILE_LIST)

## Installation & Setup:
install: ## Install the package and all dependencies
	@echo "$(BLUE)Installing photonic-foundry...$(RESET)"
	$(PIP) install -e .
	@echo "$(GREEN)✓ Installation complete$(RESET)"

dev: ## Install development dependencies and setup pre-commit hooks
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	$(PIP) install -e ".[dev,docs,test]"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)✓ Development environment ready$(RESET)"

clean: ## Clean build artifacts and temporary files
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".coverage" -delete
	find . -type d -name "htmlcov" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type d -name "*.egg-info" -delete
	rm -rf dist/ build/ .ruff_cache/
	@echo "$(GREEN)✓ Cleanup complete$(RESET)"

## Code Quality:
test: ## Run the test suite
	@echo "$(BLUE)Running tests...$(RESET)"
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	pytest tests/ --cov=src/photonic_foundry --cov-report=html --cov-report=term-missing

test-fast: ## Run tests in parallel (fast)
	@echo "$(BLUE)Running tests in parallel...$(RESET)"
	pytest tests/ -n auto -v

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(RESET)"
	flake8 src/ tests/
	mypy src/
	ruff check src/ tests/
	@echo "$(GREEN)✓ Linting complete$(RESET)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	black src/ tests/
	isort src/ tests/
	ruff format src/ tests/
	@echo "$(GREEN)✓ Code formatted$(RESET)"

check: ## Run all code quality checks
	@echo "$(BLUE)Running comprehensive code quality checks...$(RESET)"
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/
	ruff check src/ tests/
	@echo "$(GREEN)✓ All checks passed$(RESET)"

## Security & Safety:
security: ## Run security scans
	@echo "$(BLUE)Running security scans...$(RESET)"
	bandit -r src/
	safety check
	pip-audit
	@echo "$(GREEN)✓ Security scan complete$(RESET)"

## Performance:
bench: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	pytest tests/ --benchmark-only --benchmark-sort=mean

bench-save: ## Save benchmark results for comparison
	@echo "$(BLUE)Saving benchmark results...$(RESET)"
	pytest tests/ --benchmark-only --benchmark-save=baseline

bench-compare: ## Compare current benchmarks with saved baseline
	@echo "$(BLUE)Comparing benchmarks with baseline...$(RESET)"
	pytest tests/ --benchmark-only --benchmark-compare

## Documentation:
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	cd docs && make html
	@echo "$(GREEN)✓ Documentation built$(RESET)"

docs-serve: ## Build and serve documentation locally
	@echo "$(BLUE)Building and serving documentation...$(RESET)"
	cd docs && make html && python -m http.server 8000 --directory _build/html

docs-clean: ## Clean documentation build files
	@echo "$(BLUE)Cleaning documentation...$(RESET)"
	cd docs && make clean

## Docker:
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)✓ Docker image built$(RESET)"

docker-run: ## Run Docker container interactively
	@echo "$(BLUE)Running Docker container...$(RESET)"
	docker run -it -v $(PWD):/workspace $(DOCKER_IMAGE)

docker-test: ## Run tests in Docker container
	@echo "$(BLUE)Running tests in Docker...$(RESET)"
	docker run --rm -v $(PWD):/workspace $(DOCKER_IMAGE) make test

docker-clean: ## Clean Docker images and containers
	@echo "$(BLUE)Cleaning Docker artifacts...$(RESET)"
	docker system prune -f
	docker image prune -f

## Development Tools:
jupyter: ## Start Jupyter Lab server
	@echo "$(BLUE)Starting Jupyter Lab...$(RESET)"
	jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(RESET)"
	mypy src/ --strict

profile: ## Profile the application for performance bottlenecks
	@echo "$(BLUE)Profiling application...$(RESET)"
	python -m cProfile -o profile.stats -m photonic_foundry.cli --help
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

## Release & Publishing:
build: ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(RESET)"
	python -m build
	@echo "$(GREEN)✓ Build complete$(RESET)"

upload-test: ## Upload to Test PyPI
	@echo "$(BLUE)Uploading to Test PyPI...$(RESET)"
	python -m twine upload --repository testpypi dist/*

upload: ## Upload to PyPI (production)
	@echo "$(YELLOW)⚠️  Uploading to production PyPI...$(RESET)"
	python -m twine upload dist/*

version: ## Show current version
	@echo "$(BLUE)Current version:$(RESET)"
	python -c "import sys; sys.path.insert(0, 'src'); from photonic_foundry import __version__; print(__version__)"

## CI/CD Simulation:
ci: ## Simulate CI pipeline locally
	@echo "$(BLUE)Running CI pipeline simulation...$(RESET)"
	make clean
	make dev
	make check
	make test-cov
	make security
	make docs
	make build
	@echo "$(GREEN)✓ CI simulation complete$(RESET)"

## Comprehensive Targets:
all: clean dev check test-cov security docs build ## Run complete development workflow
	@echo "$(GREEN)✓ Complete development workflow finished$(RESET)"

quick: format lint test ## Quick development check (format, lint, test)
	@echo "$(GREEN)✓ Quick development check complete$(RESET)"

setup: install dev ## Initial setup for new contributors
	@echo "$(GREEN)✓ Setup complete! Run 'make test' to verify everything works$(RESET)"

## Environment Info:
info: ## Show development environment information
	@echo "$(BLUE)Development Environment Information:$(RESET)"
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Project name: $(PROJECT_NAME)"
	@echo "Docker image: $(DOCKER_IMAGE)"
	@echo "Current directory: $(PWD)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Git status: $$(git status --porcelain 2>/dev/null | wc -l || echo 'N/A') modified files"

validate: ## Validate development environment setup
	@echo "$(BLUE)Validating development environment...$(RESET)"
	@python -c "import sys; assert sys.version_info >= (3, 8), 'Python 3.8+ required'"
	@python -c "import photonic_foundry; print('✓ Package importable')"
	@python -c "import pytest; print('✓ Pytest available')"
	@python -c "import black; print('✓ Black available')"
	@python -c "import mypy; print('✓ MyPy available')"
	@which pre-commit > /dev/null && echo "✓ Pre-commit available" || echo "✗ Pre-commit not found"
	@which docker > /dev/null && echo "✓ Docker available" || echo "✗ Docker not found"
	@echo "$(GREEN)✓ Environment validation complete$(RESET)"

## Debugging:
debug: ## Run application in debug mode
	@echo "$(BLUE)Running in debug mode...$(RESET)"
	PHOTONIC_FOUNDRY_DEBUG=true python -m photonic_foundry.cli

debug-test: ## Run specific test in debug mode
	@echo "$(BLUE)Running test in debug mode...$(RESET)"
	@echo "Usage: make debug-test TEST=tests/test_specific.py::test_function"
	pytest $(TEST) -v -s --pdb

## Maintenance:
update-deps: ## Update all dependencies to latest versions
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	pip-compile --upgrade requirements.in
	pip-compile --upgrade requirements-dev.in
	@echo "$(GREEN)✓ Dependencies updated$(RESET)"

check-deps: ## Check for dependency vulnerabilities
	@echo "$(BLUE)Checking dependencies for vulnerabilities...$(RESET)"
	safety check
	pip-audit

outdated: ## Show outdated packages
	@echo "$(BLUE)Checking for outdated packages...$(RESET)"
	pip list --outdated

## Advanced Development:
complexity: ## Check code complexity
	@echo "$(BLUE)Checking code complexity...$(RESET)"
	radon cc src/ -a -nc

dead-code: ## Find dead code
	@echo "$(BLUE)Looking for dead code...$(RESET)"
	vulture src/

coverage-html: ## Generate HTML coverage report
	@echo "$(BLUE)Generating HTML coverage report...$(RESET)"
	pytest tests/ --cov=src/photonic_foundry --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(RESET)"

memory-profile: ## Profile memory usage
	@echo "$(BLUE)Profiling memory usage...$(RESET)"
	python -m memory_profiler examples/basic_usage.py

# Help formatting
define HELP_TEXT
$(BLUE)
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Photonic Neural Network Foundry                         ║
║                           Development Makefile                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
$(RESET)

This Makefile provides convenient commands for development, testing, and deployment.

$(YELLOW)Quick Start:$(RESET)
  1. make setup     # Initial setup for new contributors
  2. make test      # Run tests to verify everything works
  3. make quick     # Format, lint, and test (development cycle)

$(YELLOW)Common Workflows:$(RESET)
  • Development: make format lint test
  • Pre-commit: make check test-cov
  • Release prep: make all
  • CI simulation: make ci

$(YELLOW)For detailed help:$(RESET) make help

endef

welcome:
	$(info $(HELP_TEXT))