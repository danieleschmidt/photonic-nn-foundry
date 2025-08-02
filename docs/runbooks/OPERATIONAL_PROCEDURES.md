# Operational Procedures for Photonic Neural Network Foundry

## Overview

This document provides comprehensive operational procedures for managing, monitoring, and maintaining the photonic-nn-foundry system in development and production environments.

## Table of Contents

1. [System Startup and Shutdown](#system-startup-and-shutdown)
2. [Health Monitoring](#health-monitoring)
3. [Performance Monitoring](#performance-monitoring)
4. [Troubleshooting](#troubleshooting)
5. [Backup and Recovery](#backup-and-recovery)
6. [Security Operations](#security-operations)
7. [Maintenance Procedures](#maintenance-procedures)

---

## System Startup and Shutdown

### 🚀 System Startup

#### 1. Pre-startup Checks

```bash
# Check system requirements
./scripts/health-check.sh --system-only

# Verify Docker is running
docker info

# Check available disk space (minimum 10GB recommended)
df -h .

# Verify network connectivity
ping -c 3 8.8.8.8
```

#### 2. Development Environment Startup

```bash
# Option A: Using Docker Compose (Recommended)
docker-compose up -d

# Option B: Using individual services
docker-compose up -d postgres redis
docker-compose up -d photonic-foundry
docker-compose up -d jupyter

# Option C: Local development
source venv/bin/activate
python -m photonic_foundry.cli --version
```

#### 3. Production Environment Startup

```bash
# Build production images
make docker-build

# Deploy with production configuration
docker-compose -f docker-compose.prod.yml up -d

# Verify all services are healthy
./scripts/health-check.sh
```

#### 4. Post-startup Verification

```bash
# Check service status
docker-compose ps

# Verify health endpoints
curl http://localhost:8000/health
curl http://localhost:8888/api/status
curl http://localhost:9090/-/healthy

# Check logs for errors
docker-compose logs --tail=50
```

### 🛑 System Shutdown

#### 1. Graceful Shutdown

```bash
# Save any ongoing work
# - Export Jupyter notebooks
# - Complete running simulations
# - Save configuration changes

# Stop services gracefully
docker-compose down

# For production, use longer timeout
docker-compose down --timeout 60
```

#### 2. Emergency Shutdown

```bash
# Force stop all containers
docker stop $(docker ps -q)

# Clean up if needed
docker system prune -f
```

#### 3. Post-shutdown Cleanup

```bash
# Check for persistent data
docker volume ls

# Backup important data if needed
docker run --rm -v photonic_foundry_postgres_data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres-backup.tar.gz /data
```

---

## Health Monitoring

### 📊 Automated Health Checks

#### 1. Continuous Monitoring

```bash
# Run comprehensive health check
./scripts/health-check.sh

# Monitor specific components
./scripts/health-check.sh --services-only
./scripts/health-check.sh --docker-only
./scripts/health-check.sh --system-only
```

#### 2. Health Check Endpoints

| Service | Endpoint | Expected Response |
|---------|----------|-------------------|
| Main App | `GET /health` | `{"status": "healthy"}` |
| Jupyter | `GET /api/status` | `{"started": true}` |
| Prometheus | `GET /-/healthy` | `Prometheus is Healthy.` |
| Grafana | `GET /api/health` | `{"database": "ok"}` |

#### 3. Critical Metrics to Monitor

**System Metrics:**
- CPU usage < 80%
- Memory usage < 85%
- Disk usage < 90%
- Network connectivity available

**Application Metrics:**
- Model conversion success rate > 95%
- Simulation accuracy > 99%
- Response time < 30 seconds
- Error rate < 1%

**Infrastructure Metrics:**
- Docker container health status
- Database connection pool utilization
- Redis memory usage
- Log file sizes

### 🚨 Alerting Thresholds

#### Critical Alerts (Immediate Response Required)

```yaml
# System
- CPU usage > 95% for 5 minutes
- Memory usage > 95% for 2 minutes
- Disk usage > 95%
- Service down for > 1 minute

# Application
- Error rate > 10% for 5 minutes
- Response time > 60 seconds
- Model conversion failure rate > 20%
- Database connection failures
```

#### Warning Alerts (Response Within 1 Hour)

```yaml
# System
- CPU usage > 80% for 15 minutes
- Memory usage > 85% for 10 minutes
- Disk usage > 85%

# Application
- Error rate > 5% for 10 minutes
- Response time > 30 seconds
- Unusual number of retries
```

---

## Performance Monitoring

### 📈 Key Performance Indicators (KPIs)

#### 1. Business Metrics

- **Model Conversion Success Rate**: Target > 95%
- **Simulation Accuracy**: Target > 99.9%
- **Energy Efficiency Improvement**: Target > 50x vs GPU
- **Latency Reduction**: Target > 100x vs CPU

#### 2. Technical Metrics

- **Response Time**: P95 < 10 seconds
- **Throughput**: > 100 models/hour
- **Resource Utilization**: CPU < 70%, Memory < 80%
- **Error Rate**: < 0.1%

#### 3. Monitoring Dashboards

**Grafana Dashboards:**
- System Overview: CPU, memory, disk, network
- Application Performance: Response times, throughput, errors
- Business Metrics: Conversion rates, simulation results
- Infrastructure: Docker containers, database, cache

**Prometheus Queries:**

```promql
# Average response time
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# CPU usage
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100
```

### 🔍 Performance Optimization

#### 1. Regular Performance Reviews

**Weekly:**
- Review performance dashboards
- Analyze slow queries and operations
- Check resource utilization trends
- Identify optimization opportunities

**Monthly:**
- Capacity planning review
- Performance baseline updates
- Infrastructure scaling decisions
- Technology stack evaluation

#### 2. Performance Tuning

**Application Level:**
```python
# Enable performance tracking
from photonic_foundry.observability import track_performance

@track_performance("model_conversion")
def convert_model(model):
    # Implementation
    pass
```

**System Level:**
```bash
# Database optimization
docker exec postgres psql -c "VACUUM ANALYZE;"

# Redis memory optimization
docker exec redis redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Container resource limits
docker update --memory 4g --cpus 2 photonic-foundry
```

---

## Troubleshooting

### 🔧 Common Issues and Solutions

#### 1. Service Startup Issues

**Problem**: Container fails to start
```bash
# Diagnosis
docker logs <container-name>
docker inspect <container-name>

# Common solutions
docker system prune -f  # Clean up resources
docker-compose down && docker-compose up -d  # Restart services
./scripts/setup.sh  # Reinitialize environment
```

**Problem**: Port conflicts
```bash
# Check port usage
netstat -tulpn | grep :8000

# Kill process using port
sudo fuser -k 8000/tcp

# Use different ports
export JUPYTER_PORT=8889
docker-compose up -d
```

#### 2. Performance Issues

**Problem**: High memory usage
```bash
# Check memory usage
docker stats

# Identify memory leaks
docker exec photonic-foundry python -c "
import gc
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Objects: {len(gc.get_objects())}')
"

# Restart services if needed
docker-compose restart photonic-foundry
```

**Problem**: Slow response times
```bash
# Check system load
uptime

# Analyze slow operations
docker logs photonic-foundry | grep "duration_seconds"

# Profile application
docker exec photonic-foundry python -m cProfile -o profile.stats app.py
```

#### 3. Network Issues

**Problem**: Service connectivity issues
```bash
# Check network configuration
docker network ls
docker network inspect photonic-network

# Test service connectivity
docker exec photonic-foundry ping redis
docker exec photonic-foundry nc -zv postgres 5432

# Recreate network if needed
docker-compose down
docker network prune
docker-compose up -d
```

#### 4. Storage Issues

**Problem**: Disk space full
```bash
# Check disk usage
df -h

# Clean up Docker resources
docker system prune -af
docker volume prune

# Remove old logs
find /var/log -name "*.log" -mtime +7 -delete
```

### 📞 Escalation Procedures

#### Level 1: Self-Service (0-15 minutes)
- Check health endpoints
- Review recent logs
- Restart affected services
- Consult troubleshooting guide

#### Level 2: Team Lead (15-60 minutes)
- Contact team lead if issue persists
- Provide error logs and system status
- Implement temporary workarounds
- Escalate to Level 3 if critical

#### Level 3: Senior Engineer (1-4 hours)
- Complex system issues
- Performance degradation analysis
- Security incident response
- Infrastructure changes

#### Level 4: Emergency Response (Immediate)
- Complete system outage
- Security breach
- Data corruption
- Contact: [Emergency contact information]

---

## Backup and Recovery

### 💾 Backup Procedures

#### 1. Automated Backups

**Database Backup (Daily):**
```bash
#!/bin/bash
# Database backup script
DATE=$(date +%Y%m%d_%H%M%S)
docker exec postgres pg_dump -U photonic photonic_foundry > backup_${DATE}.sql
aws s3 cp backup_${DATE}.sql s3://photonic-foundry-backups/
```

**Configuration Backup (Daily):**
```bash
#!/bin/bash
# Configuration backup
tar czf config_backup_$(date +%Y%m%d).tar.gz \
  .env docker-compose.yml monitoring/ scripts/
```

**Model Artifacts Backup (Weekly):**
```bash
#!/bin/bash
# Model artifacts backup
docker run --rm -v photonic_foundry_models:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/models_$(date +%Y%m%d).tar.gz /data
```

#### 2. Backup Verification

```bash
# Test backup restore
docker run --rm -v $(pwd):/backup postgres:15 \
  psql postgresql://test:test@testdb/test -f /backup/backup_latest.sql

# Verify backup integrity
tar -tzf models_backup.tar.gz > /dev/null && echo "Backup OK" || echo "Backup corrupted"
```

### 🔄 Recovery Procedures

#### 1. Database Recovery

```bash
# Stop services
docker-compose down

# Restore database
docker-compose up -d postgres
sleep 10
docker exec -i postgres psql -U photonic photonic_foundry < backup_latest.sql

# Start remaining services
docker-compose up -d
```

#### 2. Complete System Recovery

```bash
# Clean environment
docker-compose down -v
docker system prune -af

# Restore configuration
tar xzf config_backup_latest.tar.gz

# Restore data volumes
docker volume create photonic_foundry_postgres_data
docker run --rm -v photonic_foundry_postgres_data:/data -v $(pwd):/backup \
  ubuntu tar xzf /backup/postgres_backup_latest.tar.gz -C /data

# Start system
docker-compose up -d

# Verify recovery
./scripts/health-check.sh
```

#### 3. Disaster Recovery

**RTO (Recovery Time Objective)**: 4 hours
**RPO (Recovery Point Objective)**: 24 hours

**Recovery Steps:**
1. Assess damage and data loss
2. Provision new infrastructure if needed
3. Restore from latest backups
4. Verify system functionality
5. Resume operations
6. Conduct post-incident review

---

## Security Operations

### 🛡️ Security Monitoring

#### 1. Security Health Checks

```bash
# Run security scans
make security-scan

# Check for vulnerabilities
docker run --rm -v $(pwd):/workspace aquasec/trivy fs /workspace

# Verify SSL certificates
openssl s_client -connect your-domain.com:443 -servername your-domain.com
```

#### 2. Access Control

**Container Security:**
```bash
# Check running processes
docker exec photonic-foundry ps aux

# Verify user permissions
docker exec photonic-foundry id

# Check file permissions
docker exec photonic-foundry ls -la /app
```

**Network Security:**
```bash
# Check open ports
docker exec photonic-foundry netstat -tulpn

# Verify firewall rules
sudo iptables -L

# Check SSL/TLS configuration
nmap --script ssl-enum-ciphers -p 443 your-domain.com
```

#### 3. Security Incident Response

**Detection:**
- Monitor security alerts
- Review access logs
- Check for unusual activity
- Verify system integrity

**Response:**
1. Isolate affected systems
2. Preserve evidence
3. Assess impact
4. Implement containment
5. Eradicate threats
6. Recover services
7. Document incident

### 🔐 Security Maintenance

#### 1. Regular Security Updates

**Weekly:**
```bash
# Update base images
docker pull python:3.11-slim
docker pull postgres:15-alpine
docker pull redis:7-alpine

# Rebuild with updates
make docker-build
```

**Monthly:**
```bash
# Update dependencies
pip-audit --fix
npm audit fix

# Security scan
bandit -r src/
safety check
```

#### 2. Access Review

**Quarterly:**
- Review user access permissions
- Audit API keys and tokens
- Verify SSL certificate expiration
- Update security documentation

---

## Maintenance Procedures

### 🔧 Routine Maintenance

#### 1. Daily Tasks (Automated)

```bash
#!/bin/bash
# Daily maintenance script

# Check service health
./scripts/health-check.sh

# Clean up old logs
find logs/ -name "*.log" -mtime +7 -delete

# Database maintenance
docker exec postgres psql -c "VACUUM;"

# Update metrics
./scripts/collect_metrics.py

# Backup data
./scripts/backup.sh
```

#### 2. Weekly Tasks

```bash
#!/bin/bash
# Weekly maintenance script

# Full database vacuum
docker exec postgres psql -c "VACUUM FULL;"

# Clean up Docker resources
docker system prune -f

# Update dependencies check
./scripts/dependency_health_check.py

# Performance review
./scripts/performance_report.py

# Security scan
make security-scan
```

#### 3. Monthly Tasks

**System Updates:**
```bash
# Update base system
sudo apt update && sudo apt upgrade

# Update Docker
sudo apt install docker-ce docker-ce-cli containerd.io

# Update Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
```

**Capacity Planning:**
```bash
# Generate capacity report
./scripts/capacity_report.py

# Review resource usage trends
# Plan for scaling requirements
# Update infrastructure sizing
```

#### 4. Quarterly Tasks

**Infrastructure Review:**
- Evaluate performance trends
- Plan technology upgrades
- Review disaster recovery procedures
- Update documentation
- Conduct security audit

**Configuration Review:**
- Update monitoring thresholds
- Optimize resource allocations
- Review backup procedures
- Update operational procedures

### 📋 Maintenance Checklist

#### Pre-Maintenance
- [ ] Schedule maintenance window
- [ ] Notify stakeholders
- [ ] Create backup
- [ ] Prepare rollback plan
- [ ] Test maintenance procedures

#### During Maintenance
- [ ] Follow procedures exactly
- [ ] Monitor system status
- [ ] Document any issues
- [ ] Verify each step
- [ ] Test functionality

#### Post-Maintenance
- [ ] Verify system health
- [ ] Check performance metrics
- [ ] Update documentation
- [ ] Notify completion
- [ ] Schedule follow-up review

---

## Emergency Contacts

### 📞 Contact Information

**Primary Contacts:**
- Technical Lead: [Name] - [Phone] - [Email]
- DevOps Engineer: [Name] - [Phone] - [Email]
- Security Officer: [Name] - [Phone] - [Email]

**Escalation:**
- Level 2: Team Lead - [Contact]
- Level 3: Engineering Manager - [Contact]
- Level 4: CTO - [Contact]

**External:**
- Cloud Provider Support: [Contact]
- Security Vendor: [Contact]
- Network Provider: [Contact]

### 🚨 Emergency Response

**Critical Outage:**
1. Assess impact and severity
2. Implement immediate mitigation
3. Notify stakeholders
4. Begin recovery procedures
5. Provide regular updates
6. Conduct post-incident review

**Security Incident:**
1. Isolate affected systems
2. Contact security team
3. Preserve evidence
4. Notify legal/compliance
5. Implement remediation
6. Document incident

---

## Documentation Updates

This document should be reviewed and updated:
- **Monthly**: Operational procedures
- **Quarterly**: Contact information
- **Annually**: Complete review and update

**Version**: 1.0  
**Last Updated**: January 2025  
**Next Review**: April 2025  
**Owner**: DevOps Team