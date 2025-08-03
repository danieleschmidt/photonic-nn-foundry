#!/bin/bash

# Photonic Neural Network Foundry - Development Environment Setup
set -e

echo "🔧 Setting up Photonic Neural Network Foundry development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    htop \
    jq \
    tree \
    vim \
    wget \
    zsh \
    postgresql-client \
    redis-tools

# Install Oh My Zsh for better shell experience
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    echo "🐚 Installing Oh My Zsh..."
    sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
    chsh -s $(which zsh)
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p /workspace/data
mkdir -p /workspace/logs
mkdir -p /workspace/tmp
mkdir -p /workspace/exports

# Set up Python virtual environment
echo "🐍 Setting up Python virtual environment..."
if [ ! -d "/workspace/.venv" ]; then
    python3 -m venv /workspace/.venv
fi

# Activate virtual environment
source /workspace/.venv/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools

# Install project dependencies
echo "📚 Installing Python dependencies..."
cd /workspace
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install project in development mode
pip install -e .

# Install additional development tools
echo "🔨 Installing development tools..."
pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly \
    streamlit \
    gradio

# Configure Jupyter Lab
echo "📊 Configuring Jupyter Lab..."
jupyter lab --generate-config
cat >> ~/.jupyter/jupyter_lab_config.py << EOF
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
EOF

# Set up pre-commit hooks
echo "🔍 Setting up pre-commit hooks..."
pre-commit install

# Initialize database
echo "🗄️ Initializing development database..."
python -c "
from src.photonic_foundry.database import get_database
db = get_database()
print('Database initialized successfully')
"

# Create .env file from example if it doesn't exist
if [ ! -f "/workspace/.env" ]; then
    echo "⚙️ Creating environment configuration..."
    cp /workspace/.env.example /workspace/.env
fi

# Set up git configuration if not already set
if [ -z "$(git config --global user.name)" ]; then
    echo "🔧 Setting up git configuration..."
    git config --global user.name "Development User"
    git config --global user.email "dev@photonicfoundry.com"
    git config --global init.defaultBranch main
fi

# Create helpful aliases
echo "🔗 Setting up useful aliases..."
cat >> ~/.zshrc << 'EOF'

# Photonic Foundry aliases
alias pf-serve="cd /workspace && python -m src.photonic_foundry.api.server"
alias pf-test="cd /workspace && python -m pytest tests/ -v"
alias pf-lint="cd /workspace && flake8 src/ tests/"
alias pf-format="cd /workspace && black src/ tests/ && isort src/ tests/"
alias pf-jupyter="cd /workspace && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
alias pf-db-reset="cd /workspace && python scripts/reset_database.py"
alias pf-logs="tail -f /workspace/logs/app.log"

# Navigation aliases
alias ll="ls -la"
alias la="ls -A"
alias l="ls -CF"

# Python environment
export PYTHONPATH="/workspace/src:$PYTHONPATH"
source /workspace/.venv/bin/activate
EOF

# Create database reset script
echo "🔄 Creating database management scripts..."
cat > /workspace/scripts/reset_database.py << 'EOF'
#!/usr/bin/env python3
"""
Reset development database with fresh seed data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from photonic_foundry.database import get_database
from photonic_foundry.database.seeds import seed_development_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Reset and seed development database."""
    try:
        logger.info("Resetting development database...")
        
        # Initialize database
        db = get_database()
        logger.info("Database connection established")
        
        # Clear existing data
        db.clear_all_data()
        logger.info("Existing data cleared")
        
        # Seed with development data
        seed_development_data(db)
        logger.info("Development data seeded successfully")
        
        logger.info("Database reset complete!")
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x /workspace/scripts/reset_database.py

# Create log directory and initial log file
touch /workspace/logs/app.log

# Set ownership and permissions
sudo chown -R vscode:vscode /workspace
chmod +x /workspace/.devcontainer/setup.sh

echo "✅ Development environment setup complete!"
echo ""
echo "🚀 Quick start commands:"
echo "  pf-serve     # Start API server"
echo "  pf-jupyter   # Start Jupyter Lab"
echo "  pf-test      # Run tests"
echo "  pf-format    # Format code"
echo "  pf-db-reset  # Reset database"
echo ""
echo "📖 Documentation: http://localhost:8000/docs"
echo "📊 Jupyter Lab: http://localhost:8888"
echo ""
echo "Happy coding! 🧪✨"