# Maintenance and Operations Runbook

This runbook covers routine maintenance, backup procedures, and operational tasks for the Photonic Neural Network Foundry.

## Overview

Regular maintenance ensures system reliability, performance, and security. This document provides procedures for:

- Routine maintenance tasks
- Backup and restore procedures
- Database maintenance
- Security updates
- Performance optimization
- Capacity planning

## Maintenance Schedule

### Daily Tasks (Automated)

- Health check verification
- Log rotation and cleanup
- Metrics collection validation
- Backup verification
- Security scan execution

### Weekly Tasks

- Performance trend analysis
- Capacity utilization review
- Security patch assessment
- Documentation updates
- Monitoring threshold review

### Monthly Tasks

- Full system backup verification
- Disaster recovery testing
- Performance baseline updates
- Security audit review
- Infrastructure cost analysis

### Quarterly Tasks

- Major version updates
- Architecture review
- Business continuity testing
- Security penetration testing
- Capacity expansion planning

## Backup and Restore Procedures

### Database Backup

#### PostgreSQL Backup

**Daily Backup (Automated)**:
```bash
#!/bin/bash
# Daily PostgreSQL backup script

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgres"
DB_NAME="photonic_foundry"

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup
pg_dump -h postgres -U photonic -d $DB_NAME > $BACKUP_DIR/backup_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/backup_$DATE.sql

# Remove backups older than 30 days
find $BACKUP_DIR -name "backup_*.sql.gz" -type f -mtime +30 -delete

# Verify backup integrity
gunzip -t $BACKUP_DIR/backup_$DATE.sql.gz
if [ $? -eq 0 ]; then
    echo "Backup completed successfully: backup_$DATE.sql.gz"
else
    echo "Backup verification failed!" | mail -s "Backup Failed" admin@company.com
fi
```

**Manual Backup**:
```bash
# Full database backup
kubectl exec deployment/postgres -- pg_dump -U photonic photonic_foundry > backup_$(date +%Y%m%d).sql

# Backup specific table
kubectl exec deployment/postgres -- pg_dump -U photonic -t simulations photonic_foundry > simulations_backup.sql

# Backup with custom format (for faster restore)
kubectl exec deployment/postgres -- pg_dump -U photonic -Fc photonic_foundry > backup_$(date +%Y%m%d).dump
```

#### Database Restore

**Full Restore**:
```bash
# Stop application to prevent connections
kubectl scale deployment photonic-foundry --replicas=0

# Drop and recreate database
kubectl exec -it deployment/postgres -- psql -U postgres -c "DROP DATABASE IF EXISTS photonic_foundry;"
kubectl exec -it deployment/postgres -- psql -U postgres -c "CREATE DATABASE photonic_foundry OWNER photonic;"

# Restore from SQL dump
kubectl exec -i deployment/postgres -- psql -U photonic photonic_foundry < backup_20250802.sql

# Restore from custom format
kubectl exec deployment/postgres -- pg_restore -U photonic -d photonic_foundry backup_20250802.dump

# Restart application
kubectl scale deployment photonic-foundry --replicas=3
```

**Point-in-Time Recovery**:
```bash
# Restore to specific timestamp
kubectl exec deployment/postgres -- pg_restore -U photonic -d photonic_foundry -t "2025-08-02 10:30:00" backup.dump
```

### Redis Backup

**Manual Backup**:
```bash
# Create Redis backup
kubectl exec deployment/redis -- redis-cli BGSAVE

# Copy RDB file
kubectl cp deployment/redis:/data/dump.rdb ./redis_backup_$(date +%Y%m%d).rdb
```

**Restore Redis**:
```bash
# Stop Redis
kubectl scale deployment redis --replicas=0

# Copy backup file
kubectl cp redis_backup_20250802.rdb deployment/redis:/data/dump.rdb

# Start Redis
kubectl scale deployment redis --replicas=1
```

### Application Data Backup

**Configuration Backup**:
```bash
# Backup Kubernetes configurations
kubectl get configmaps -o yaml > configmaps_backup.yaml
kubectl get secrets -o yaml > secrets_backup.yaml
kubectl get deployments -o yaml > deployments_backup.yaml

# Backup environment files
tar -czf env_backup_$(date +%Y%m%d).tar.gz .env* docker-compose.yml Dockerfile
```

**Model and Results Backup**:
```bash
# Backup simulation results
kubectl exec deployment/photonic-foundry -- tar -czf /tmp/results_backup.tar.gz /app/results/
kubectl cp deployment/photonic-foundry:/tmp/results_backup.tar.gz ./results_backup_$(date +%Y%m%d).tar.gz

# Backup trained models
kubectl exec deployment/photonic-foundry -- tar -czf /tmp/models_backup.tar.gz /app/models/
kubectl cp deployment/photonic-foundry:/tmp/models_backup.tar.gz ./models_backup_$(date +%Y%m%d).tar.gz
```

## Database Maintenance

### PostgreSQL Maintenance

**Routine Maintenance**:
```sql
-- Update table statistics
ANALYZE;

-- Vacuum to reclaim space
VACUUM;

-- Full vacuum (requires downtime)
VACUUM FULL;

-- Reindex for performance
REINDEX DATABASE photonic_foundry;
```

**Performance Monitoring**:
```sql
-- Check database size
SELECT pg_size_pretty(pg_database_size('photonic_foundry'));

-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
```

**Index Maintenance**:
```sql
-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
ORDER BY idx_scan DESC;

-- Find unused indexes
SELECT 
    schemaname,
    tablename,
    indexname
FROM pg_stat_user_indexes 
WHERE idx_scan = 0;

-- Check index bloat
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;
```

### Redis Maintenance

**Memory Management**:
```bash
# Check memory usage
kubectl exec deployment/redis -- redis-cli info memory

# Check key expiration
kubectl exec deployment/redis -- redis-cli info keyspace

# Clean expired keys
kubectl exec deployment/redis -- redis-cli eval "return redis.call('del', unpack(redis.call('keys', 'expired:*')))" 0
```

**Performance Monitoring**:
```bash
# Monitor Redis performance
kubectl exec deployment/redis -- redis-cli --latency-history -i 1

# Check slow log
kubectl exec deployment/redis -- redis-cli slowlog get 10

# Monitor connections
kubectl exec deployment/redis -- redis-cli info clients
```

## System Updates and Patches

### Operating System Updates

**Container Base Image Updates**:
```bash
# Update base images in Dockerfile
FROM python:3.11-slim  # Update to latest patch version

# Rebuild and test
docker build -t photonic-foundry:test .
docker run --rm photonic-foundry:test python -c "import photonic_foundry; print('OK')"

# Deploy with rolling update
kubectl set image deployment/photonic-foundry photonic-foundry=photonic-foundry:test
kubectl rollout status deployment/photonic-foundry
```

### Python Dependencies

**Security Updates**:
```bash
# Check for security vulnerabilities
safety check

# Update requirements
pip-compile --upgrade requirements.in
pip-compile --upgrade requirements-dev.in

# Test updates
make test

# Update production
docker build -t photonic-foundry:updated .
kubectl set image deployment/photonic-foundry photonic-foundry=photonic-foundry:updated
```

### Application Updates

**Rolling Deployment**:
```bash
# Deploy new version
kubectl set image deployment/photonic-foundry photonic-foundry=photonic-foundry:v1.2.0

# Monitor rollout
kubectl rollout status deployment/photonic-foundry

# Verify health
kubectl get pods -l app=photonic-foundry
curl -f http://localhost:8000/health

# Rollback if needed
kubectl rollout undo deployment/photonic-foundry
```

**Blue-Green Deployment**:
```bash
# Create green environment
kubectl apply -f deployment-green.yaml

# Test green environment
kubectl port-forward service/photonic-foundry-green 8001:8000 &
curl -f http://localhost:8001/health

# Switch traffic
kubectl patch service photonic-foundry -p '{"spec":{"selector":{"version":"green"}}}'

# Remove blue environment
kubectl delete deployment photonic-foundry-blue
```

## Performance Optimization

### Application Performance

**Memory Optimization**:
```bash
# Monitor memory usage
kubectl top pods

# Check for memory leaks
kubectl exec deployment/photonic-foundry -- python -c "
import gc
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'Objects in memory: {len(gc.get_objects())}')
"

# Tune garbage collection
kubectl set env deployment/photonic-foundry PYTHONOPTIMIZE=1
```

**CPU Optimization**:
```bash
# Check CPU usage patterns
kubectl exec deployment/photonic-foundry -- top -b -n 1

# Profile application
kubectl exec deployment/photonic-foundry -- python -m cProfile -s cumulative -m photonic_foundry.cli --help

# Adjust resource limits
kubectl patch deployment photonic-foundry -p '{"spec":{"template":{"spec":{"containers":[{"name":"photonic-foundry","resources":{"limits":{"cpu":"2","memory":"4Gi"}}}]}}}}'
```

### Database Performance

**Query Optimization**:
```sql
-- Analyze slow queries
EXPLAIN ANALYZE SELECT * FROM simulations WHERE created_at > NOW() - INTERVAL '1 day';

-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_simulations_user_id_created_at ON simulations(user_id, created_at);

-- Update table statistics
ANALYZE simulations;
```

**Connection Pool Tuning**:
```bash
# Check current connections
kubectl exec deployment/postgres -- psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# Adjust connection limits
kubectl set env deployment/photonic-foundry DATABASE_POOL_SIZE=20
kubectl set env deployment/photonic-foundry DATABASE_MAX_OVERFLOW=10
```

## Capacity Planning

### Resource Monitoring

**Current Usage Analysis**:
```bash
# Check resource usage trends
# Use Prometheus queries in Grafana

# CPU usage over time
rate(container_cpu_usage_seconds_total[5m])

# Memory usage trend
container_memory_usage_bytes

# Disk usage growth
(node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes
```

**Capacity Forecasting**:
```bash
# Calculate growth rate
# Based on 30-day metrics trend
growth_rate = (current_usage - usage_30_days_ago) / usage_30_days_ago

# Estimate capacity needs
# Project 90 days forward
projected_usage = current_usage * (1 + growth_rate * 3)
```

### Scaling Decisions

**Horizontal Scaling**:
```bash
# Scale based on CPU utilization
kubectl autoscale deployment photonic-foundry --cpu-percent=70 --min=2 --max=10

# Scale based on custom metrics
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: photonic-foundry-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: photonic-foundry
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: simulation_queue_size
      target:
        type: AverageValue
        averageValue: "10"
EOF
```

**Vertical Scaling**:
```bash
# Increase resource limits
kubectl patch deployment photonic-foundry -p '{"spec":{"template":{"spec":{"containers":[{"name":"photonic-foundry","resources":{"requests":{"cpu":"1","memory":"2Gi"},"limits":{"cpu":"4","memory":"8Gi"}}}]}}}}'
```

## Security Maintenance

### Security Scanning

**Vulnerability Scanning**:
```bash
# Scan container images
trivy image photonic-foundry:latest

# Scan dependencies
safety check

# Scan code for security issues
bandit -r src/

# Generate security report
{
    echo "# Security Scan Report - $(date)"
    echo "## Container Vulnerabilities"
    trivy image photonic-foundry:latest
    echo "## Python Dependencies"
    safety check --json
    echo "## Code Security Issues"
    bandit -r src/ -f json
} > security_report_$(date +%Y%m%d).md
```

**Certificate Management**:
```bash
# Check certificate expiration
kubectl get certificates

# Renew certificates (cert-manager)
kubectl annotate certificate photonic-foundry-tls cert-manager.io/force-renewal=true

# Manual certificate renewal
openssl x509 -in cert.pem -text -noout | grep "Not After"
```

### Access Control Review

**User Access Audit**:
```bash
# Review service account permissions
kubectl get serviceaccounts
kubectl describe serviceaccount photonic-foundry

# Check RBAC policies
kubectl get roles,rolebindings,clusterroles,clusterrolebindings

# Audit API access
kubectl logs -n kube-system kube-apiserver | grep photonic-foundry
```

**Security Configuration Review**:
```bash
# Check Pod Security Standards
kubectl get pods -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.securityContext}{"\n"}{end}'

# Review network policies
kubectl get networkpolicies

# Check secrets and configmaps
kubectl get secrets,configmaps
```

## Disaster Recovery Testing

### Recovery Procedures

**Full System Recovery**:
```bash
# 1. Restore infrastructure
kubectl apply -f infrastructure/

# 2. Restore database
kubectl cp backup_20250802.sql.gz deployment/postgres:/tmp/
kubectl exec deployment/postgres -- gunzip /tmp/backup_20250802.sql.gz
kubectl exec deployment/postgres -- psql -U photonic photonic_foundry < /tmp/backup_20250802.sql

# 3. Restore application
kubectl apply -f deployment/
kubectl rollout status deployment/photonic-foundry

# 4. Verify functionality
curl -f http://localhost:8000/health
python scripts/test_basic_functionality.py
```

**Testing Schedule**:
- **Monthly**: Backup restore testing
- **Quarterly**: Full disaster recovery simulation
- **Annually**: Cross-region failover testing

### Recovery Time Objectives (RTO)

| Component | RTO | RPO | Recovery Procedure |
|-----------|-----|-----|-------------------|
| Application | 15 minutes | 5 minutes | Rolling restart |
| Database | 30 minutes | 1 hour | Backup restore |
| Full System | 2 hours | 4 hours | Complete rebuild |

## Monitoring and Alerting Maintenance

### Alert Tuning

**Review Alert Thresholds**:
```bash
# Analyze alert frequency
kubectl logs -n monitoring alertmanager | grep "alert firing"

# Adjust thresholds based on trends
# Edit prometheus-rules.yml
- alert: HighMemoryUsage
  expr: process_resident_memory_bytes > 4GB  # Adjusted from 2GB
```

**Alert Fatigue Prevention**:
```yaml
# Group related alerts
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 24h

# Implement alert suppression
inhibit_rules:
- source_match:
    severity: 'critical'
  target_match:
    severity: 'warning'
  equal: ['alertname', 'cluster', 'service']
```

This maintenance runbook should be reviewed and updated quarterly to ensure procedures remain current and effective.