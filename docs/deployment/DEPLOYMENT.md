# Deployment Guide

This guide covers deployment strategies and configurations for photonic-nn-foundry across different environments.

## Deployment Overview

Photonic-nn-foundry supports multiple deployment modes:
- **Local Development**: Direct Python installation
- **Container Deployment**: Docker-based deployment
- **Cloud Deployment**: Container orchestration platforms
- **CI/CD Integration**: Automated deployment pipelines

## Local Development Deployment

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional, for containerized development)

### Installation Methods

#### Method 1: pip Installation

```bash
# Install from PyPI (when available)
pip install photonic-nn-foundry

# Or install development version
pip install git+https://github.com/terragon-labs/photonic-nn-foundry.git
```

#### Method 2: Development Installation

```bash
# Clone repository
git clone https://github.com/terragon-labs/photonic-nn-foundry.git
cd photonic-nn-foundry

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Verify installation
photonic-foundry --version
```

### Configuration

Create a configuration file at `~/.photonic-foundry/config.yml`:

```yaml
# Default configuration
default:
  output_directory: "./output"
  log_level: "INFO"
  max_model_size_mb: 100
  
  # Hardware constraints
  hardware:
    max_mzi_count: 1000
    max_waveguides: 10000
    precision: "float32"
  
  # Optimization settings
  optimization:
    enable_circuit_optimization: true
    optimization_level: 2
    parallel_processing: true

# Development overrides
development:
  log_level: "DEBUG"
  enable_profiling: true
  
# Production overrides  
production:
  log_level: "WARNING"
  enable_profiling: false
```

## Container Deployment

### Docker Images

The project provides multi-stage Docker builds:

```bash
# Build development image
docker build --target development -t photonic-foundry:dev .

# Build production image
docker build --target production -t photonic-foundry:prod .

# Build Jupyter notebook image
docker build --target jupyter -t photonic-foundry:jupyter .
```

### Docker Compose Deployment

For local development with supporting services:

```yaml
# docker-compose.yml
version: '3.8'

services:
  photonic-foundry:
    build:
      context: .
      target: development
    volumes:
      - .:/workspace
      - ./data:/data
      - ./output:/output
    environment:
      - PYTHONPATH=/workspace/src
      - LOG_LEVEL=DEBUG
    ports:
      - "8888:8888"  # Jupyter
    command: jupyter lab --ip=0.0.0.0 --no-browser --allow-root

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
```

### Container Security

```dockerfile
# Security best practices in Dockerfile
FROM python:3.10-slim as base

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set security-focused defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

USER appuser
WORKDIR /app
```

## Cloud Deployment

### Kubernetes Deployment

#### Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: photonic-foundry
  labels:
    name: photonic-foundry

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: photonic-foundry-config
  namespace: photonic-foundry
data:
  config.yml: |
    default:
      log_level: "INFO"
      output_directory: "/data/output"
      hardware:
        max_mzi_count: 1000
        precision: "float32"
```

#### Deployment Configuration

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: photonic-foundry
  namespace: photonic-foundry
spec:
  replicas: 3
  selector:
    matchLabels:
      app: photonic-foundry
  template:
    metadata:
      labels:
        app: photonic-foundry
    spec:
      containers:
      - name: photonic-foundry
        image: photonic-foundry:latest
        ports:
        - containerPort: 8080
        env:
        - name: CONFIG_PATH
          value: "/config/config.yml"
        volumeMounts:
        - name: config-volume
          mountPath: /config
        - name: data-volume
          mountPath: /data
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config-volume
        configMap:
          name: photonic-foundry-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: photonic-foundry-pvc
```

#### Service and Ingress

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: photonic-foundry-service
  namespace: photonic-foundry
spec:
  selector:
    app: photonic-foundry
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: photonic-foundry-ingress
  namespace: photonic-foundry
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - photonic-foundry.example.com
    secretName: photonic-foundry-tls
  rules:
  - host: photonic-foundry.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: photonic-foundry-service
            port:
              number: 80
```

### AWS Deployment

#### ECS Task Definition

```json
{
  "family": "photonic-foundry",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/photonic-foundry-task-role",
  "containerDefinitions": [
    {
      "name": "photonic-foundry",
      "image": "your-registry/photonic-foundry:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "AWS_REGION",
          "value": "us-west-2"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/photonic-foundry",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

## CI/CD Integration

### GitHub Actions Deployment

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          registry.example.com/photonic-foundry:staging
          registry.example.com/photonic-foundry:${{ github.sha }}
    
    - name: Deploy to staging
      run: |
        kubectl set image deployment/photonic-foundry \
          photonic-foundry=registry.example.com/photonic-foundry:${{ github.sha }} \
          -n photonic-foundry-staging

  deploy-production:
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    environment: production
    needs: [deploy-staging]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build and push production image
      uses: docker/build-push-action@v4
      with:
        context: .
        target: production
        push: true
        tags: |
          registry.example.com/photonic-foundry:latest
          registry.example.com/photonic-foundry:${{ github.ref_name }}
    
    - name: Deploy to production
      run: |
        kubectl set image deployment/photonic-foundry \
          photonic-foundry=registry.example.com/photonic-foundry:${{ github.ref_name }} \
          -n photonic-foundry-production
```

## Environment Configuration

### Environment Variables

```bash
# Core configuration
PHOTONIC_FOUNDRY_CONFIG_PATH=/etc/photonic-foundry/config.yml
PHOTONIC_FOUNDRY_LOG_LEVEL=INFO
PHOTONIC_FOUNDRY_OUTPUT_DIR=/data/output

# Database configuration
DATABASE_URL=postgresql://user:pass@host:5432/photonic_foundry
REDIS_URL=redis://localhost:6379/0

# Security configuration
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# Performance configuration
WORKER_PROCESSES=4
MAX_MEMORY_MB=1024
ENABLE_PROFILING=false

# Cloud configuration
AWS_REGION=us-west-2
AWS_S3_BUCKET=photonic-foundry-data
KUBERNETES_NAMESPACE=photonic-foundry
```

### Configuration Validation

```python
# config/validation.py
from pydantic import BaseSettings, validator

class DeploymentConfig(BaseSettings):
    log_level: str = "INFO"
    max_memory_mb: int = 1024
    worker_processes: int = 4
    
    @validator('log_level')
    def validate_log_level(cls, v):
        if v not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            raise ValueError('Invalid log level')
        return v
    
    @validator('max_memory_mb')
    def validate_memory(cls, v):
        if v < 256 or v > 8192:
            raise ValueError('Memory must be between 256MB and 8GB')
        return v
    
    class Config:
        env_prefix = 'PHOTONIC_FOUNDRY_'
```

## Monitoring and Observability

### Health Checks

```python
# health.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.get("/ready")
async def readiness_check():
    # Check dependencies
    try:
        # Database connection check
        # Redis connection check
        # Other dependency checks
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not ready", "error": str(e)}
```

### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Application metrics
requests_total = Counter('photonic_foundry_requests_total', 'Total requests')
request_duration = Histogram('photonic_foundry_request_duration_seconds', 'Request duration')
active_models = Gauge('photonic_foundry_active_models', 'Number of active models')

def record_request():
    requests_total.inc()

def record_processing_time(duration):
    request_duration.observe(duration)
```

## Security Considerations

### Production Security Checklist

- [ ] Use non-root containers
- [ ] Enable resource limits
- [ ] Implement health checks
- [ ] Use secrets management
- [ ] Enable network policies
- [ ] Regular security scanning
- [ ] Monitor for vulnerabilities
- [ ] Implement proper logging
- [ ] Use TLS for all communications
- [ ] Regular backup procedures

### Secrets Management

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: photonic-foundry-secrets
  namespace: photonic-foundry
type: Opaque
data:
  database-url: <base64-encoded-url>
  api-key: <base64-encoded-key>
  jwt-secret: <base64-encoded-secret>
```

## Troubleshooting

### Common Deployment Issues

1. **Container Startup Failures**
   - Check resource limits
   - Verify configuration files
   - Review application logs

2. **Network Connectivity Issues**
   - Verify service configurations
   - Check ingress rules
   - Test DNS resolution

3. **Performance Issues**
   - Monitor resource usage
   - Check application metrics
   - Review scaling policies

4. **Configuration Problems**
   - Validate environment variables
   - Check config map contents
   - Verify secret references

### Deployment Validation

```bash
# Deployment health check script
#!/bin/bash

echo "Checking deployment health..."

# Check pod status
kubectl get pods -n photonic-foundry

# Check service endpoints
kubectl get endpoints -n photonic-foundry

# Test application health
curl -f http://photonic-foundry.example.com/health

# Check resource usage
kubectl top pods -n photonic-foundry

echo "Deployment health check complete"
```

This deployment guide provides comprehensive coverage for deploying photonic-nn-foundry in various environments with proper security, monitoring, and troubleshooting procedures.