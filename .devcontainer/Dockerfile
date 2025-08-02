# Development Container for Photonic Neural Network Foundry
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    vim \
    zsh \
    htop \
    tree \
    jq \
    make \
    cmake \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://get.docker.com | sh

# Install Docker Compose
RUN curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose \
    && chmod +x /usr/local/bin/docker-compose

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install Oh My Zsh for better terminal experience
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true

# Set up Python environment
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements*.txt ./
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install -r requirements-dev.txt

# Install additional development tools
RUN pip install \
    ipython \
    jupyterlab \
    notebook \
    black \
    isort \
    flake8 \
    pylint \
    mypy \
    bandit \
    safety \
    pre-commit \
    pytest-cov \
    pytest-benchmark \
    pytest-xdist \
    sphinx \
    sphinx-rtd-theme

# Install scientific computing libraries
RUN pip install \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    pandas \
    scikit-learn \
    torch \
    torchvision \
    torchaudio

# Set up git configuration
RUN git config --global init.defaultBranch main
RUN git config --global pull.rebase false

# Create non-root user
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid vscode --shell /bin/zsh --create-home vscode \
    && echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers.d/vscode

# Switch to non-root user
USER vscode

# Set up Oh My Zsh for vscode user
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true

# Configure shell
RUN echo 'export PYTHONPATH="/workspace/src:$PYTHONPATH"' >> ~/.zshrc \
    && echo 'export PATH="/workspace/scripts:$PATH"' >> ~/.zshrc \
    && echo 'alias ll="ls -la"' >> ~/.zshrc \
    && echo 'alias la="ls -la"' >> ~/.zshrc \
    && echo 'alias ..="cd .."' >> ~/.zshrc \
    && echo 'alias ...="cd ../.."' >> ~/.zshrc

# Set default shell
SHELL ["/bin/zsh", "-c"]

# Set working directory
WORKDIR /workspace

# Default command
CMD ["zsh"]