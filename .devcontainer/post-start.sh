#!/bin/bash

# Post-start script for Photonic Neural Network Foundry Dev Container
# This script runs every time the container starts

set -e

echo "🔄 Starting Photonic Neural Network Foundry development services..."

# Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    echo "⚠️  Docker daemon not accessible. Some features may not work."
fi

# Update pip and core packages
echo "📦 Updating Python packages..."
pip install --upgrade pip setuptools wheel --quiet

# Sync pre-commit hooks
echo "🔗 Syncing pre-commit hooks..."
pre-commit autoupdate --quiet || echo "⚠️  Pre-commit autoupdate failed"

# Clean up any stale Python cache
echo "🧹 Cleaning Python cache..."
find /workspace -name "*.pyc" -delete 2>/dev/null || true
find /workspace -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Ensure logs directory exists and is writable
mkdir -p /workspace/logs
chmod 755 /workspace/logs

# Check package installation
echo "🔍 Verifying package installation..."
python -c "
import sys
sys.path.insert(0, '/workspace/src')
try:
    import photonic_foundry
    print('✅ Photonic Foundry package ready')
except ImportError as e:
    print(f'⚠️  Package import issue: {e}')
    print('   Run: pip install -e .')
"

# Start background services if needed
echo "🚀 Starting background services..."

# Start SSH agent for git operations
eval "$(ssh-agent -s)" >/dev/null 2>&1 || true

# Set up git user if not configured
if [ -z "$(git config --global user.name)" ]; then
    echo "⚙️  Setting up git configuration..."
    git config --global user.name "Developer"
    git config --global user.email "developer@photonic-foundry.dev"
fi

# Check disk space
DISK_USAGE=$(df /workspace | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    echo "⚠️  Warning: Disk usage is at ${DISK_USAGE}%. Consider cleaning up."
fi

# Display environment info
echo ""
echo "🌟 Photonic Neural Network Foundry Development Environment"
echo "================================================================"
echo "📍 Workspace: /workspace"
echo "🐍 Python: $(python --version)"
echo "🔧 Git: $(git --version)"
echo "🐳 Docker: $(docker --version 2>/dev/null || echo 'Not available')"
echo "📊 Jupyter Lab: http://localhost:8888 (token: devtoken)"
echo "📈 Prometheus: http://localhost:9090"
echo "📊 Grafana: http://localhost:3000 (admin/admin)"
echo ""

# Show current branch and status
if [ -d /workspace/.git ]; then
    echo "🌿 Git Branch: $(git branch --show-current 2>/dev/null || echo 'Not in git repo')"
    CHANGES=$(git status --porcelain 2>/dev/null | wc -l)
    if [ "$CHANGES" -gt 0 ]; then
        echo "📝 Uncommitted changes: $CHANGES files"
    fi
fi

# Show available make targets
echo ""
echo "🛠️  Available commands (run 'make help' for full list):"
echo "   make test     - Run tests"
echo "   make lint     - Check code quality"
echo "   make format   - Format code"
echo "   make dev      - Development workflow"
echo ""

echo "✅ Development environment ready!"