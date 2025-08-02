#!/bin/bash

# Post-create script for Photonic Neural Network Foundry Dev Container
# This script runs after the container is created

set -e

echo "🚀 Setting up Photonic Neural Network Foundry development environment..."

# Install Python package in development mode
echo "📦 Installing Python package in development mode..."
pip install -e .

# Install pre-commit hooks
echo "🔗 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p cache
mkdir -p models
mkdir -p data
mkdir -p outputs
mkdir -p profiling

# Set up git hooks if not already present
if [ ! -f .git/hooks/commit-msg ]; then
    echo "🎯 Setting up git commit message template..."
    cat > .git/hooks/commit-msg << 'EOF'
#!/bin/bash
# Validate commit message format
commit_regex='^(feat|fix|docs|style|refactor|test|chore|perf|ci|build)(\(.+\))?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "❌ Invalid commit message format!"
    echo "Format: type(scope): description"
    echo "Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build"
    echo "Example: feat(transpiler): add PyTorch to Verilog conversion"
    exit 1
fi
EOF
    chmod +x .git/hooks/commit-msg
fi

# Create .env file from example if it doesn't exist
if [ ! -f .env ]; then
    echo "🔧 Creating .env file from template..."
    cp .env.example .env
    echo "✏️  Please edit .env file with your specific configuration"
fi

# Set up Jupyter Lab extensions and configuration
echo "🔬 Configuring Jupyter Lab..."
jupyter lab --generate-config || true

# Create Jupyter config if it doesn't exist
mkdir -p ~/.jupyter
cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = 'devtoken'
c.ServerApp.notebook_dir = '/workspace'
EOF

# Install additional development utilities
echo "🛠️  Installing additional development utilities..."
npm install -g markdownlint-cli
pip install --user --upgrade pip-tools

# Generate requirements lock files
echo "🔒 Generating requirements lock files..."
pip-compile requirements.in --output-file requirements.txt || echo "⚠️  requirements.in not found, skipping lock file generation"
pip-compile requirements-dev.in --output-file requirements-dev.txt || echo "⚠️  requirements-dev.in not found, skipping lock file generation"

# Run initial tests to verify setup
echo "🧪 Running initial tests to verify setup..."
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check if our package imports correctly
python -c "
try:
    import src.photonic_foundry
    print('✅ Photonic Foundry package imports successfully')
except ImportError as e:
    print(f'⚠️  Package import issue: {e}')
"

# Set up VS Code workspace settings if not present
if [ ! -f .vscode/settings.json ]; then
    echo "⚙️  Setting up VS Code workspace settings..."
    mkdir -p .vscode
    cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "/usr/local/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.watcherExclude": {
        "**/.git/objects/**": true,
        "**/.git/subtree-cache/**": true,
        "**/node_modules/*/**": true,
        "**/__pycache__/**": true,
        "**/htmlcov/**": true,
        "**/.coverage": true,
        "**/.pytest_cache/**": true
    }
}
EOF
fi

# Create launch configuration for debugging
if [ ! -f .vscode/launch.json ]; then
    echo "🐛 Setting up VS Code debugging configuration..."
    cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Python: CLI",
            "type": "python",
            "request": "launch",
            "module": "photonic_foundry.cli",
            "args": ["--help"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Pytest: Current File",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}", "-v"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
EOF
fi

echo "✅ Development environment setup complete!"
echo ""
echo "🎉 Welcome to Photonic Neural Network Foundry!"
echo "📝 Next steps:"
echo "   1. Edit .env file with your configuration"
echo "   2. Run 'make test' to verify everything works"
echo "   3. Run 'make dev' for development workflow"
echo "   4. Open Jupyter Lab at http://localhost:8888 (token: devtoken)"
echo ""
echo "🔗 Useful commands:"
echo "   make help    - Show all available commands"
echo "   make test    - Run test suite"
echo "   make lint    - Check code quality"
echo "   make format  - Format code"
echo ""