# Photonic Neural Network Foundry - Build System
# ==============================================

.PHONY: help build test lint format clean install dev docker security docs

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
DOCKER_IMAGE := photonic-foundry
DOCKER_TAG := latest
DOCKER_REGISTRY := ghcr.io/danieleschmidt
PROJECT_NAME := photonic-nn-foundry

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Photonic Neural Network Foundry - Build Commands$(NC)"
	@echo "=================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development Environment
install: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install

dev: ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	$(PYTHON) -m venv venv
	@echo "$(YELLOW)Activate with: source venv/bin/activate$(NC)"
	@echo "$(YELLOW)Then run: make install$(NC)"

# Code Quality
lint: ## Run all linting tools
	@echo "$(BLUE)Running linting checks...$(NC)"
	flake8 src/ tests/
	pylint src/
	mypy src/
	bandit -r src/

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	isort src/ tests/

format-check: ## Check code formatting without modifying
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check src/ tests/
	isort --check-only src/ tests/

# Testing
test: ## Run all tests
	@echo "$(BLUE)Running test suite...$(NC)"
	pytest tests/ -v --cov=src/photonic_foundry --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	pytest tests/e2e/ -v

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	pytest tests/performance/ -v --benchmark-only

test-security: ## Run security tests
	@echo "$(BLUE)Running security tests...$(NC)"
	pytest tests/security/ -v
	bandit -r src/

# Docker Operations
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(PROJECT_NAME):$(DOCKER_TAG)

docker-build-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(NC)"
	docker build -f Dockerfile.dev -t $(DOCKER_IMAGE):dev .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --rm -v $(PWD):/workspace $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-dev: ## Run development Docker container with volume mounts
	@echo "$(BLUE)Running development Docker container...$(NC)"
	docker run -it --rm -v $(PWD):/workspace -p 8888:8888 $(DOCKER_IMAGE):dev

docker-compose-up: ## Start all services with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	docker-compose up -d

docker-compose-down: ## Stop all services
	@echo "$(BLUE)Stopping services...$(NC)"
	docker-compose down

docker-push: ## Push Docker image to registry
	@echo "$(BLUE)Pushing Docker image to registry...$(NC)"
	docker push $(DOCKER_REGISTRY)/$(PROJECT_NAME):$(DOCKER_TAG)

# Security
security-scan: ## Run comprehensive security scans
	@echo "$(BLUE)Running security scans...$(NC)"
	safety check
	bandit -r src/
	docker run --rm -v $(PWD):/src -w /src aquasec/trivy fs .

security-scan-docker: ## Scan Docker image for vulnerabilities
	@echo "$(BLUE)Scanning Docker image for vulnerabilities...$(NC)"
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image $(DOCKER_IMAGE):$(DOCKER_TAG)

# Documentation
docs-build: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && make html

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	@echo "$(BLUE)Cleaning documentation build...$(NC)"
	cd docs && make clean

# Release Management
version: ## Show current version
	@echo "$(BLUE)Current version:$(NC)"
	@$(PYTHON) -c "import src.photonic_foundry; print(src.photonic_foundry.__version__)"

build: ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(NC)"
	$(PYTHON) -m build

release-check: ## Check if ready for release
	@echo "$(BLUE)Checking release readiness...$(NC)"
	$(PYTHON) -m twine check dist/*

# Database and Migrations (if applicable)
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	# Add database migration commands if needed

# Cleanup
clean: ## Clean build artifacts and cache
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	docker system prune -f

clean-all: clean ## Clean everything including Docker images
	@echo "$(BLUE)Cleaning Docker images...$(NC)"
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) || true
	docker rmi $(DOCKER_REGISTRY)/$(PROJECT_NAME):$(DOCKER_TAG) || true

# CI/CD helpers
ci-install: ## Install dependencies for CI
	@echo "$(BLUE)Installing CI dependencies...$(NC)"
	$(PIP) install -r requirements-dev.txt

ci-test: ## Run tests in CI environment
	@echo "$(BLUE)Running CI tests...$(NC)"
	pytest tests/ --cov=src/photonic_foundry --cov-report=xml --cov-fail-under=80

ci-build: ## Build for CI/CD
	@echo "$(BLUE)Building for CI/CD...$(NC)"
	$(MAKE) format-check lint test ci-security docker-build

ci-security: ## Run security checks in CI
	@echo "$(BLUE)Running CI security checks...$(NC)"
	safety check --json
	bandit -r src/ -f json

# Performance monitoring
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	pytest tests/performance/ --benchmark-only --benchmark-json=benchmark-results.json

profile: ## Profile application performance
	@echo "$(BLUE)Profiling application...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats src/photonic_foundry/cli.py --help
	@echo "$(YELLOW)View with: python -m pstats profile.stats$(NC)"

# Git hooks
pre-commit-all: ## Run pre-commit on all files
	@echo "$(BLUE)Running pre-commit on all files...$(NC)"
	pre-commit run --all-files

# Environment validation
validate-env: ## Validate development environment
	@echo "$(BLUE)Validating development environment...$(NC)"
	@$(PYTHON) --version
	@docker --version
	@docker-compose --version
	@$(PIP) check

# Quick development workflow
quick: format lint test ## Quick development check (format, lint, test)

# Production readiness check
prod-ready: clean format-check lint test security-scan docker-build ## Full production readiness check

# Show system info
info: ## Show system and project information
	@echo "$(BLUE)System Information:$(NC)"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Docker: $(shell docker --version)"
	@echo "Working Directory: $(PWD)"
	@echo "Git Branch: $(shell git branch --show-current)"
	@echo "Git Status: $(shell git status --porcelain | wc -l) changes"