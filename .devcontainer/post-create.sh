#!/bin/bash

# Photonic Neural Network Foundry - Dev Container Post-Create Script
# This script runs after the dev container is created to set up the development environment

set -e

echo "🚀 Setting up Photonic Neural Network Foundry development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional system dependencies for photonic simulation
echo "🔬 Installing photonic simulation dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    libopenmpi-dev \
    libhdf5-dev \
    libfftw3-dev \
    libgsl-dev \
    liblapack-dev \
    libblas-dev \
    libsuitesparse-dev \
    libnlopt-dev \
    octave \
    gnuplot \
    imagemagick \
    ffmpeg

# Install Verilog simulation tools
echo "⚡ Installing Verilog simulation tools..."
sudo apt-get install -y \
    iverilog \
    gtkwave \
    verilator

# Install Python development dependencies
echo "🐍 Installing Python development dependencies..."
pip install --upgrade pip setuptools wheel

# Install the package in development mode
echo "📦 Installing photonic-foundry in development mode..."
pip install -e ".[dev,docs,test]"

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Install additional development tools
echo "🛠️ Installing additional development tools..."
pip install \
    pytest-benchmark \
    pytest-mock \
    pytest-asyncio \
    pytest-timeout \
    pytest-sugar \
    coverage[toml] \
    bandit[toml] \
    safety \
    pip-audit \
    ruff

# Install Jupyter extensions
echo "📓 Setting up Jupyter environment..."
pip install \
    jupyterlab-git \
    jupyterlab-lsp \
    jupyter-lsp \
    jupyterlab-code-formatter \
    nbstripout

# Configure Jupyter
jupyter lab --generate-config
echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py

# Install nbstripout for git hooks
nbstripout --install

# Create development directories
echo "📁 Creating development directories..."
mkdir -p \
    ~/workspace/experiments \
    ~/workspace/notebooks \
    ~/workspace/models \
    ~/workspace/results \
    ~/.local/bin

# Set up shell aliases and functions
echo "🐚 Setting up shell configuration..."
cat >> ~/.zshrc << 'EOF'

# Photonic Foundry Development Aliases
alias pf='photonic-foundry'
alias pytest-fast='pytest -x -vs'
alias pytest-cov='pytest --cov=src/photonic_foundry --cov-report=html'
alias black-check='black --check src/ tests/'
alias black-fix='black src/ tests/'
alias lint='flake8 src/ tests/ && mypy src/'
alias format='black src/ tests/ && isort src/ tests/'
alias test-all='pytest tests/ --cov=src/photonic_foundry --cov-report=html'
alias serve-docs='cd docs && python -m http.server 8000'
alias clean='find . -type d -name __pycache__ -delete && find . -name "*.pyc" -delete'

# Development environment functions
pf-setup() {
    echo "🔧 Setting up photonic foundry development environment..."
    pip install -e ".[dev,docs,test]"
    pre-commit install
    echo "✅ Development environment ready!"
}

pf-test() {
    echo "🧪 Running photonic foundry test suite..."
    pytest tests/ --cov=src/photonic_foundry --cov-report=term-missing "$@"
}

pf-lint() {
    echo "🔍 Running code quality checks..."
    black --check src/ tests/
    isort --check-only src/ tests/
    flake8 src/ tests/
    mypy src/
    echo "✅ All checks passed!"
}

pf-format() {
    echo "🎨 Formatting code..."
    black src/ tests/
    isort src/ tests/
    echo "✅ Code formatted!"
}

pf-docs() {
    echo "📚 Building documentation..."
    cd docs && make html && python -m http.server 8000 --directory _build/html
}

pf-clean() {
    echo "🧹 Cleaning up temporary files..."
    find . -type d -name __pycache__ -delete
    find . -name "*.pyc" -delete
    find . -name "*.pyo" -delete
    find . -name ".coverage" -delete
    find . -type d -name "htmlcov" -delete
    find . -type d -name ".pytest_cache" -delete
    find . -type d -name ".mypy_cache" -delete
    find . -type d -name "*.egg-info" -delete
    echo "✅ Cleanup complete!"
}

# Environment variables
export PYTHONPATH="${PYTHONPATH}:/workspaces/photonic-nn-foundry/src"
export PHOTONIC_FOUNDRY_HOME="/workspaces/photonic-nn-foundry"
export PHOTONIC_FOUNDRY_DATA="~/workspace/models"
export PHOTONIC_FOUNDRY_RESULTS="~/workspace/results"

EOF

# Create a sample environment file
echo "🔧 Creating sample environment configuration..."
cat > .env.example << 'EOF'
# Photonic Neural Network Foundry - Environment Configuration

# Development Settings
PHOTONIC_FOUNDRY_DEBUG=true
PHOTONIC_FOUNDRY_LOG_LEVEL=INFO

# Simulation Settings
PHOTONIC_FOUNDRY_DEFAULT_PDK=skywater130
PHOTONIC_FOUNDRY_DEFAULT_WAVELENGTH=1550
PHOTONIC_FOUNDRY_SIMULATION_BACKEND=icarus

# Performance Settings
PHOTONIC_FOUNDRY_MAX_PARALLEL_JOBS=4
PHOTONIC_FOUNDRY_CACHE_DIR=~/.cache/photonic_foundry
PHOTONIC_FOUNDRY_TEMP_DIR=/tmp/photonic_foundry

# Integration Settings
PHOTONIC_FOUNDRY_ENABLE_METRICS=true
PHOTONIC_FOUNDRY_METRICS_ENDPOINT=http://localhost:9090

# Optional: Hardware Acceleration
# CUDA_VISIBLE_DEVICES=0
# PHOTONIC_FOUNDRY_USE_GPU=true

# Optional: Cloud Integration
# PHOTONIC_FOUNDRY_S3_BUCKET=my-photonic-models
# PHOTONIC_FOUNDRY_AWS_REGION=us-west-2

# Optional: Jupyter Configuration
JUPYTER_ENABLE_LAB=true
JUPYTER_PORT=8888
JUPYTER_TOKEN=
EOF

# Set up git configuration helpers
echo "🔧 Setting up git configuration..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input
git config --global core.filemode false

# Create initial development workspace
echo "📝 Creating development workspace structure..."
mkdir -p examples/{basic,advanced,benchmarks}
mkdir -p notebooks/{tutorials,experiments,analysis}
mkdir -p scripts/{build,deploy,analysis}

# Install additional development utilities
echo "🔧 Installing development utilities..."
npm install -g \
    markdownlint-cli \
    @commitlint/cli \
    @commitlint/config-conventional

# Create commitlint configuration
cat > .commitlintrc.json << 'EOF'
{
  "extends": ["@commitlint/config-conventional"],
  "rules": {
    "type-enum": [
      2,
      "always",
      [
        "feat",
        "fix",
        "docs",
        "style",
        "refactor",
        "perf",
        "test",
        "build",
        "ci",
        "chore",
        "revert"
      ]
    ],
    "scope-enum": [
      2,
      "always",
      [
        "core",
        "transpiler",
        "cli",
        "docs",
        "tests",
        "ci",
        "deps",
        "pdk",
        "simulation",
        "optimization"
      ]
    ],
    "scope-empty": [0, "never"],
    "subject-max-length": [2, "always", 72]
  }
}
EOF

# Final setup
echo "🎉 Development environment setup complete!"
echo ""
echo "Available commands:"
echo "  pf-setup    - Set up development environment"
echo "  pf-test     - Run test suite"
echo "  pf-lint     - Run code quality checks"
echo "  pf-format   - Format code"
echo "  pf-docs     - Build and serve documentation"
echo "  pf-clean    - Clean temporary files"
echo ""
echo "To get started:"
echo "  1. Copy .env.example to .env and customize"
echo "  2. Run 'pf-test' to verify everything works"
echo "  3. Start coding! 🚀"
echo ""
echo "Happy photonic computing! ✨"