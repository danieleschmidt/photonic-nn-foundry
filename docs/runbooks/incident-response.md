# Incident Response Runbook

## Overview

This runbook provides step-by-step procedures for responding to incidents in the Photonic Neural Network Foundry system. It covers common issues, escalation procedures, and recovery strategies.

## Incident Classification

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P0 - Critical | Service completely down | 15 minutes | Total service outage, data loss |
| P1 - High | Major functionality impacted | 1 hour | API errors >10%, significant performance degradation |
| P2 - Medium | Minor functionality impacted | 4 hours | Non-critical features down, minor performance issues |
| P3 - Low | Minimal impact | 24 hours | Cosmetic issues, enhancement requests |

## Common Incidents and Procedures

### Service Outages

#### Photonic Foundry Service Down (Alert: PhotonicFoundryDown)

**Symptoms:**
- Service health checks failing
- Users unable to access the application
- No response from application endpoints

**Immediate Response:**
1. **Verify the alert** - Check Grafana dashboard for service status
2. **Check container status:**
   ```bash
   docker ps | grep photonic-foundry
   docker logs photonic-foundry-prod --tail 50
   ```
3. **Restart the service:**
   ```bash
   docker-compose restart photonic-foundry
   ```
4. **Monitor recovery** - Watch health checks and error logs

**If restart doesn't work:**
1. Check resource availability:
   ```bash
   docker stats
   df -h
   free -m
   ```
2. Check for configuration issues:
   ```bash
   docker-compose config
   ```
3. Rebuild and redeploy if necessary:
   ```bash
   ./scripts/build.sh production
   docker-compose up -d photonic-foundry
   ```

**Escalation:** If service doesn't recover within 30 minutes, escalate to on-call engineer.

#### API Service Down (Alert: PhotonicAPIDown)

**Symptoms:**
- API endpoints returning 500 errors
- Unable to connect to API service
- Transpilation requests failing

**Immediate Response:**
1. **Check API service status:**
   ```bash
   curl -f http://localhost:5000/health
   docker logs photonic-api-prod --tail 50
   ```
2. **Restart API service:**
   ```bash
   docker-compose restart photonic-api
   ```
3. **Verify database connectivity:**
   ```bash
   docker exec photonic-redis-prod redis-cli ping
   ```

### Performance Issues

#### High Response Time (Alert: HighResponseTime)

**Symptoms:**
- 95th percentile response time > 1 second
- Users experiencing slow application performance
- Timeouts in client applications

**Investigation Steps:**
1. **Check current load:**
   ```bash
   # Monitor active requests
   curl http://localhost:8080/metrics | grep http_requests_total
   
   # Check CPU and memory usage
   docker stats photonic-foundry-prod
   ```

2. **Identify bottlenecks:**
   - Check database performance (Redis)
   - Review application logs for slow operations
   - Monitor transpilation queue size

3. **Immediate mitigation:**
   - Scale up if using orchestration platform
   - Restart service to clear memory leaks
   - Enable request rate limiting if not already active

**Root Cause Analysis:**
- Review slow query logs
- Analyze profiling data
- Check for memory leaks or resource contention

#### High Error Rate (Alert: HighErrorRate)

**Symptoms:**
- Error rate > 5% for sustained period
- Increased 500 status codes
- User reports of application failures

**Investigation Steps:**
1. **Analyze error patterns:**
   ```bash
   # Check error types
   docker logs photonic-foundry-prod | grep ERROR | tail -20
   
   # Monitor error metrics
   curl http://localhost:8080/metrics | grep http_requests_total | grep "status=\"5"
   ```

2. **Check dependencies:**
   - Verify Redis connectivity
   - Test external API endpoints
   - Check file system permissions

3. **Immediate actions:**
   - Identify and fix obvious configuration issues
   - Restart affected services
   - Route traffic away from failing instances if load balanced

### Resource Issues

#### High Memory Usage (Alert: HighMemoryUsage)

**Symptoms:**
- Memory usage > 90% of container limit
- Application slowness
- Possible out-of-memory kills

**Immediate Response:**
1. **Check memory usage:**
   ```bash
   docker stats photonic-foundry-prod
   cat /proc/meminfo
   ```

2. **Identify memory consumers:**
   ```bash
   # Check for memory leaks in application
   docker exec photonic-foundry-prod ps aux --sort=-%mem | head -10
   ```

3. **Immediate mitigation:**
   - Restart service to reclaim memory
   - Scale up memory limits if possible
   - Clear caches if safe to do so

**Long-term actions:**
- Profile application for memory leaks
- Optimize memory usage in code
- Implement memory monitoring and alerts

#### Low Disk Space (Alert: LowDiskSpace)

**Symptoms:**
- Available disk space < 10%
- Application unable to write logs or temporary files
- Database write failures

**Immediate Response:**
1. **Check disk usage:**
   ```bash
   df -h
   du -sh /var/log/* | sort -hr | head -10
   du -sh /tmp/* | sort -hr | head -10
   ```

2. **Clean up space:**
   ```bash
   # Clean docker artifacts
   docker system prune -f
   
   # Clean old logs
   find /var/log -name "*.log" -mtime +7 -delete
   
   # Clean temporary files
   rm -rf /tmp/photonic_*
   ```

3. **Monitor recovery:**
   ```bash
   watch df -h
   ```

### Security Incidents

#### High Rate of Unauthorized Access (Alert: UnauthorizedAccess)

**Symptoms:**
- Spike in 401 HTTP status codes
- Multiple failed authentication attempts
- Possible brute force attack

**Immediate Response:**
1. **Analyze access patterns:**
   ```bash
   # Check access logs
   docker logs nginx | grep "401" | tail -20
   
   # Identify source IPs
   docker logs nginx | grep "401" | awk '{print $1}' | sort | uniq -c | sort -nr
   ```

2. **Implement rate limiting:**
   ```bash
   # Update nginx configuration to add rate limiting
   # Apply IP-based blocking if needed
   ```

3. **Monitor and alert:**
   - Set up additional monitoring for suspicious patterns
   - Alert security team if attack persists

## Escalation Procedures

### On-Call Escalation

1. **Primary On-Call:** 15 minutes response time for P0/P1 incidents
2. **Secondary On-Call:** 30 minutes response time if primary doesn't respond
3. **Manager Escalation:** For incidents lasting > 2 hours

### External Escalation

- **Cloud Provider:** For infrastructure-related issues
- **Security Team:** For security incidents
- **Legal/Compliance:** For data breach scenarios

## Communication Procedures

### Internal Communication

1. **Create incident channel:** #incident-YYYY-MM-DD-HHMM
2. **Post initial status:** Within 15 minutes of incident detection
3. **Regular updates:** Every 30 minutes for P0/P1, hourly for P2
4. **Post-incident review:** Within 24 hours of resolution

### External Communication

- **Status page updates:** For customer-facing incidents
- **Customer notifications:** For P0/P1 incidents affecting users
- **Stakeholder briefings:** For business-critical incidents

## Recovery Procedures

### Service Recovery

1. **Verify service health:**
   ```bash
   curl -f http://localhost:8080/health
   curl -f http://localhost:5000/health
   ```

2. **Run smoke tests:**
   ```bash
   # Test basic functionality
   python scripts/smoke_test.py
   ```

3. **Monitor key metrics:**
   - Response times return to normal
   - Error rates below 1%
   - All alerts cleared

### Data Recovery

1. **Assess data integrity:**
   ```bash
   # Check Redis data consistency
   docker exec photonic-redis-prod redis-cli --scan --pattern "*" | wc -l
   ```

2. **Restore from backup if needed:**
   ```bash
   # Restore Redis from backup
   docker exec photonic-redis-prod redis-cli FLUSHALL
   docker exec photonic-redis-prod redis-cli --rdb /backup/redis-backup.rdb
   ```

## Post-Incident Actions

### Immediate (Within 1 hour)

- [ ] Verify complete service recovery
- [ ] Update stakeholders on resolution
- [ ] Create incident timeline
- [ ] Identify immediate preventive measures

### Short-term (Within 24 hours)

- [ ] Conduct post-incident review meeting
- [ ] Document root cause analysis
- [ ] Create action items for prevention
- [ ] Update runbooks and procedures

### Long-term (Within 1 week)

- [ ] Implement preventive measures
- [ ] Update monitoring and alerting
- [ ] Conduct team training if needed
- [ ] Review and update incident response procedures

## Tools and Resources

### Monitoring and Observability

- **Grafana:** http://localhost:3000
- **Prometheus:** http://localhost:9090
- **Application logs:** `docker logs photonic-foundry-prod`

### Commands Quick Reference

```bash
# Service status
docker-compose ps
docker-compose logs SERVICE_NAME

# Resource monitoring
docker stats
htop
df -h

# Application health
curl http://localhost:8080/health
curl http://localhost:5000/health

# Database health
docker exec photonic-redis-prod redis-cli ping

# Build and deploy
./scripts/build.sh production
docker-compose up -d
```

### Contact Information

- **Primary On-Call:** [Your phone/email]
- **Secondary On-Call:** [Backup contact]
- **Security Team:** [Security contact]
- **Infrastructure Team:** [Infrastructure contact]

## Appendix

### Incident Template

```
**Incident Summary**
- Start Time: YYYY-MM-DD HH:MM UTC
- Detection Method: [Alert/User Report/Monitoring]
- Severity: [P0/P1/P2/P3]
- Affected Services: [List services]

**Timeline**
- HH:MM - Incident detected
- HH:MM - Initial response started
- HH:MM - Root cause identified
- HH:MM - Fix applied
- HH:MM - Service recovered

**Root Cause**
[Detailed explanation of what caused the incident]

**Resolution**
[Steps taken to resolve the incident]

**Action Items**
- [ ] Action 1 (Owner: Name, Due: Date)
- [ ] Action 2 (Owner: Name, Due: Date)
```

### Reference Links

- [Monitoring Dashboard](http://localhost:3000)
- [Application Documentation](../README.md)
- [Architecture Documentation](../development/ARCHITECTURE.md)
- [Security Guidelines](../security/SECURITY_GUIDELINES.md)