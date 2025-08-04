# Production Deployment Guide

## Photonic Foundry Production Deployment

This guide covers the complete production deployment of the Photonic Foundry system using Docker Compose.

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 16GB RAM
- 100GB available disk space
- SSL certificates for HTTPS

### Quick Start

1. **Clone the repository:**
```bash
git clone <repository-url>
cd photonic-foundry
```

2. **Set up environment variables:**
```bash
cp deployment/.env.example deployment/.env.production
# Edit the file with your production values
```

3. **Create required directories:**
```bash
sudo mkdir -p /opt/photonic-foundry/{data,cache,logs}
sudo chown -R $USER:$USER /opt/photonic-foundry
```

4. **Generate SSL certificates:**
```bash
# Place your SSL certificates in deployment/nginx/ssl/
# Required files: photonic-foundry.crt, photonic-foundry.key
```

5. **Deploy the stack:**
```bash
cd deployment
docker-compose -f docker-compose.production.yml up -d
```

### Architecture Overview

The production deployment consists of:

- **photonic-api**: Main FastAPI application (2 CPU, 4GB RAM)
- **photonic-worker**: Background processing workers (4 CPU, 8GB RAM, 2 replicas)
- **redis**: Caching and task queue (0.5 CPU, 1GB RAM)
- **prometheus**: Metrics collection (1 CPU, 2GB RAM)
- **grafana**: Monitoring dashboards (0.5 CPU, 1GB RAM)
- **nginx**: Reverse proxy with SSL termination (0.5 CPU, 512MB RAM)

### Configuration

#### Environment Variables

Create `/deployment/.env.production`:

```bash
# Database
CIRCUIT_DB_PATH=/app/data/circuits.db

# Cache
CIRCUIT_CACHE_DIR=/app/cache/circuits
CIRCUIT_CACHE_ENABLED=true
MAX_CACHED_CIRCUITS=5000

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
MAX_REQUEST_SIZE=50
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60

# Security
CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
ENABLE_API_DOCS=false

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO

# Grafana
GRAFANA_PASSWORD=secure_password_here
```

#### SSL Configuration

Place your SSL certificates in `deployment/nginx/ssl/`:
- `photonic-foundry.crt` - SSL certificate
- `photonic-foundry.key` - Private key

#### Nginx Configuration

The Nginx configuration in `deployment/nginx/nginx.conf` provides:
- SSL termination
- Rate limiting
- Compression
- Security headers
- Load balancing

### Health Checks

The deployment includes comprehensive health checks:

- API health endpoint: `https://yourdomain.com/health`
- Prometheus metrics: `https://yourdomain.com:9090`
- Grafana dashboards: `https://yourdomain.com:3000`

### Monitoring

#### Prometheus Metrics

Available at `http://localhost:9090`, collecting:
- API request metrics
- Circuit processing times
- Resource utilization
- Error rates

#### Grafana Dashboards

Available at `http://localhost:3000` (admin/your_password):
- System Overview
- API Performance
- Circuit Processing
- Resource Usage

### Security

#### Container Security
- Read-only root filesystems
- No new privileges
- Non-root user execution
- Minimal attack surface

#### Network Security
- Custom bridge network
- Service isolation
- Rate limiting
- CORS protection

### Scaling

#### Horizontal Scaling

Scale workers:
```bash
docker-compose -f docker-compose.production.yml up -d --scale photonic-worker=4
```

#### Vertical Scaling

Update resource limits in `docker-compose.production.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
```

### Backup and Recovery

#### Database Backup
```bash
# Daily backup
docker exec photonic-api-prod sqlite3 /app/data/circuits.db ".backup '/app/data/backup_$(date +%Y%m%d).db'"
```

#### Volume Backup
```bash
# Backup persistent data
tar -czf photonic-backup-$(date +%Y%m%d).tar.gz /opt/photonic-foundry/
```

### Troubleshooting

#### Common Issues

1. **Container won't start:**
   - Check logs: `docker-compose logs photonic-api`
   - Verify environment variables
   - Check disk space

2. **SSL certificate errors:**
   - Verify certificate files exist
   - Check certificate validity
   - Ensure proper file permissions

3. **Performance issues:**
   - Monitor resource usage in Grafana
   - Scale workers if needed
   - Check cache hit rates

#### Log Access

```bash
# API logs
docker-compose logs -f photonic-api

# Worker logs
docker-compose logs -f photonic-worker

# System logs
docker-compose logs -f
```

### Maintenance

#### Updates

1. Pull latest images:
```bash
docker-compose pull
```

2. Rolling update:
```bash
docker-compose up -d --no-deps photonic-api
```

#### Cleanup

```bash
# Remove unused containers and images
docker system prune -f

# Clean up old backups (keep last 30 days)
find /opt/photonic-foundry/data -name "backup_*.db" -mtime +30 -delete
```

### Performance Tuning

#### Database Optimization
- Regular VACUUM operations
- Index optimization
- Connection pooling

#### Cache Optimization
- Monitor cache hit rates
- Adjust cache size based on usage
- Implement cache warming

#### Worker Optimization
- Monitor queue lengths
- Scale workers based on load
- Optimize task distribution

### Support

For production support:
- Check monitoring dashboards first
- Review application logs
- Consult troubleshooting guide
- Contact support team if needed