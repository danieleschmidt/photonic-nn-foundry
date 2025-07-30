# Multi-stage build for photonic-nn-foundry
FROM python:3.11-slim as base

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python environment setup
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Install pre-commit hooks
RUN git init && pre-commit install

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install -e .

# Production stage
FROM base as production

# Copy only necessary files
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY pyproject.toml .
RUN pip install .

# Create non-root user
RUN useradd --create-home --shell /bin/bash photonic
USER photonic

# Default command
CMD ["photonic-foundry", "--help"]

# Jupyter stage for development
FROM development as jupyter

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]