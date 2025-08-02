# Incident Response Runbook

This runbook provides step-by-step procedures for responding to incidents in the Photonic Neural Network Foundry system.

## Overview

### Incident Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| **P0 - Critical** | Complete service outage, data loss | 5 minutes | Immediate |
| **P1 - High** | Major feature unavailable, significant performance degradation | 15 minutes | 30 minutes |
| **P2 - Medium** | Minor feature impact, moderate performance issues | 1 hour | 2 hours |
| **P3 - Low** | Minor issues, enhancement requests | 4 hours | Next business day |

### Contact Information

- **On-Call Engineer**: Check PagerDuty rotation
- **Team Lead**: [Team Lead Contact]
- **Infrastructure Team**: [Infrastructure Contact]
- **Security Team**: [Security Contact]

## General Incident Response Process

### 1. Initial Response (First 5 minutes)

1. **Acknowledge the incident** in monitoring system
2. **Assess severity** using the criteria above
3. **Notify stakeholders** based on severity level
4. **Create incident channel** (#incident-YYYY-MM-DD-NNN)
5. **Start incident timeline** documentation

### 2. Investigation Phase

1. **Gather information**:
   - Check Grafana dashboards
   - Review recent deployments
   - Examine application logs
   - Check infrastructure status

2. **Identify root cause**:
   - Use distributed tracing (Jaeger)
   - Analyze metrics patterns
   - Review error logs
   - Check external dependencies

3. **Document findings** in incident channel

### 3. Mitigation Phase

1. **Implement immediate fix** (if available)
2. **Apply workaround** (if fix not immediately available)
3. **Monitor system recovery**
4. **Update stakeholders** on progress

### 4. Resolution Phase

1. **Verify service restoration**
2. **Monitor for regression**
3. **Update stakeholders**
4. **Schedule post-mortem** (for P0/P1 incidents)

## Common Incident Scenarios

## Scenario 1: Application Down (P0)

### Symptoms
- Health check endpoints returning 5xx errors
- Prometheus alert: `ApplicationDown`
- Users unable to access the service

### Investigation Steps

1. **Check application logs**:
   ```bash
   kubectl logs -l app=photonic-foundry --tail=100
   # or for Docker
   docker logs photonic-foundry
   ```

2. **Verify infrastructure**:
   ```bash
   # Check container status
   kubectl get pods -l app=photonic-foundry
   
   # Check resource usage
   kubectl top pods
   ```

3. **Check dependencies**:
   - Database connectivity
   - Redis availability
   - External API status

### Resolution Steps

1. **If container crashed**:
   ```bash
   # Restart the application
   kubectl rollout restart deployment/photonic-foundry
   
   # Check pod status
   kubectl get pods -w
   ```

2. **If resource exhaustion**:
   ```bash
   # Scale up replicas
   kubectl scale deployment photonic-foundry --replicas=3
   
   # Increase resource limits (if needed)
   kubectl patch deployment photonic-foundry -p '{"spec":{"template":{"spec":{"containers":[{"name":"photonic-foundry","resources":{"limits":{"memory":"2Gi"}}}]}}}}'
   ```

3. **If database issues**:
   ```bash
   # Check database status
   kubectl get pods -l app=postgres
   
   # Verify database connectivity
   kubectl exec -it deployment/photonic-foundry -- python -c "
   import psycopg2
   conn = psycopg2.connect('postgresql://user:pass@postgres:5432/db')
   print('Database connected successfully')
   "
   ```

## Scenario 2: High Error Rate (P1)

### Symptoms
- Prometheus alert: `HighErrorRate`
- 5xx response rate above 10%
- User reports of failed operations

### Investigation Steps

1. **Analyze error patterns**:
   ```bash
   # Check error logs
   kubectl logs -l app=photonic-foundry | grep -i error | tail -50
   
   # Group errors by type
   kubectl logs -l app=photonic-foundry | grep ERROR | awk '{print $5}' | sort | uniq -c
   ```

2. **Check recent changes**:
   - Review recent deployments
   - Check configuration changes
   - Verify feature flag status

3. **Analyze traffic patterns**:
   - Check Grafana for unusual request patterns
   - Review user behavior analytics
   - Check for potential DDoS

### Resolution Steps

1. **If recent deployment caused issues**:
   ```bash
   # Rollback to previous version
   kubectl rollout undo deployment/photonic-foundry
   
   # Monitor rollback progress
   kubectl rollout status deployment/photonic-foundry
   ```

2. **If database performance issues**:
   ```sql
   -- Check for long-running queries
   SELECT pid, now() - pg_stat_activity.query_start AS duration, query
   FROM pg_stat_activity
   WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';
   
   -- Check for locks
   SELECT * FROM pg_locks WHERE NOT granted;
   ```

3. **If external dependency issues**:
   - Check third-party service status
   - Implement circuit breaker patterns
   - Enable fallback mechanisms

## Scenario 3: High Response Time (P1)

### Symptoms
- Prometheus alert: `HighResponseTime`
- 95th percentile response time > 2 seconds
- User complaints about slow performance

### Investigation Steps

1. **Analyze performance metrics**:
   - Check Grafana performance dashboard
   - Review distributed traces in Jaeger
   - Identify bottleneck components

2. **Check resource utilization**:
   ```bash
   # CPU and memory usage
   kubectl top pods
   
   # Check for resource limits
   kubectl describe pods -l app=photonic-foundry
   ```

3. **Database performance**:
   ```sql
   -- Check slow queries
   SELECT query, mean_time, calls
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   ```

### Resolution Steps

1. **Scale horizontally**:
   ```bash
   # Increase replica count
   kubectl scale deployment photonic-foundry --replicas=5
   ```

2. **Optimize database**:
   ```sql
   -- Add missing indexes
   CREATE INDEX CONCURRENTLY idx_simulation_created_at 
   ON simulations(created_at);
   
   -- Update table statistics
   ANALYZE;
   ```

3. **Enable caching**:
   ```bash
   # Check Redis status
   kubectl exec -it deployment/redis -- redis-cli info
   
   # Clear cache if needed
   kubectl exec -it deployment/redis -- redis-cli flushall
   ```

## Scenario 4: Simulation Failures (P2)

### Symptoms
- Prometheus alert: `SimulationFailureRate`
- High rate of photonic simulation errors
- Users reporting failed transpilation

### Investigation Steps

1. **Check simulation logs**:
   ```bash
   # Filter simulation-specific logs
   kubectl logs -l app=photonic-foundry | grep "simulation" | grep -i error
   ```

2. **Analyze failure patterns**:
   - Check failure by simulation type
   - Review input validation errors
   - Check for resource constraints

3. **Verify photonic libraries**:
   ```bash
   # Check library versions
   kubectl exec -it deployment/photonic-foundry -- python -c "
   import torch
   import numpy as np
   print(f'PyTorch: {torch.__version__}')
   print(f'NumPy: {np.__version__}')
   "
   ```

### Resolution Steps

1. **If input validation issues**:
   - Review and update validation rules
   - Implement better error handling
   - Add input sanitization

2. **If library compatibility issues**:
   ```bash
   # Update dependencies
   kubectl set image deployment/photonic-foundry photonic-foundry=photonic-foundry:latest
   ```

3. **If resource constraints**:
   ```bash
   # Increase simulation worker resources
   kubectl patch deployment photonic-foundry -p '{"spec":{"template":{"spec":{"containers":[{"name":"photonic-foundry","resources":{"limits":{"cpu":"2","memory":"4Gi"}}}]}}}}'
   ```

## Scenario 5: Security Incident (P0/P1)

### Symptoms
- Security alerts from monitoring
- Unusual authentication patterns
- Suspected data breach

### Immediate Actions

1. **Isolate affected systems**:
   ```bash
   # Scale down to minimal replicas
   kubectl scale deployment photonic-foundry --replicas=1
   
   # Block suspicious IPs (example)
   kubectl apply -f - <<EOF
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: block-suspicious-ips
   spec:
     podSelector:
       matchLabels:
         app: photonic-foundry
     policyTypes:
     - Ingress
     ingress:
     - from: []
   EOF
   ```

2. **Preserve evidence**:
   ```bash
   # Capture logs
   kubectl logs -l app=photonic-foundry --since=24h > incident-logs.txt
   
   # Export metrics
   # Use Prometheus API to export relevant metrics
   ```

3. **Notify security team** immediately

### Investigation Steps

1. **Analyze access logs**:
   ```bash
   # Check authentication failures
   kubectl logs -l app=photonic-foundry | grep "authentication.*failed"
   
   # Check authorization violations
   kubectl logs -l app=photonic-foundry | grep "unauthorized"
   ```

2. **Review user activity**:
   - Check unusual API usage patterns
   - Review admin access logs
   - Analyze database access patterns

3. **Check for data exfiltration**:
   - Monitor outbound network traffic
   - Review file access logs
   - Check for unusual data queries

## Monitoring and Alerting

### Key Metrics to Monitor During Incidents

1. **Application Health**:
   ```prometheus
   up{job="photonic-foundry"}
   rate(http_requests_total[5m])
   histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
   ```

2. **Resource Usage**:
   ```prometheus
   process_resident_memory_bytes
   rate(process_cpu_seconds_total[5m])
   container_memory_usage_bytes
   ```

3. **Business Metrics**:
   ```prometheus
   rate(photonic_simulation_total[5m])
   rate(photonic_simulation_errors_total[5m])
   transpiler_queue_size
   ```

### Setting Up Alerts

```yaml
# Add to prometheus-rules.yml
- alert: IncidentInProgress
  expr: up{job="photonic-foundry"} == 0 or rate(http_requests_total{status=~"5.."}[5m]) > 0.1
  for: 0m
  labels:
    severity: critical
  annotations:
    summary: "Incident detected - immediate attention required"
```

## Communication Templates

### Initial Incident Notification

```
🚨 INCIDENT ALERT - P[SEVERITY]

Service: Photonic Neural Network Foundry
Issue: [Brief description]
Impact: [User impact description]
Status: Investigating
ETA: [Estimated resolution time]
Channel: #incident-[date]-[number]

Updates will be provided every 15 minutes.
```

### Status Update

```
📊 INCIDENT UPDATE - P[SEVERITY]

Status: [Investigating/Mitigating/Resolved]
Progress: [Current actions and findings]
Next Steps: [Planned actions]
ETA: [Updated estimate]

Next update in 15 minutes or upon significant change.
```

### Resolution Notification

```
✅ INCIDENT RESOLVED - P[SEVERITY]

Service: Photonic Neural Network Foundry
Resolution: [Brief description of fix]
Root Cause: [Summary - detailed post-mortem to follow]
Monitoring: Continuing to monitor for 2 hours

Post-mortem scheduled for [date/time].
```

## Post-Incident Activities

### Immediate (Within 24 hours)

1. **Conduct post-mortem** for P0/P1 incidents
2. **Document lessons learned**
3. **Create action items** for improvements
4. **Update runbooks** based on learnings

### Follow-up (Within 1 week)

1. **Implement preventive measures**
2. **Update monitoring and alerting**
3. **Conduct team review**
4. **Share learnings** with broader team

### Post-Mortem Template

```markdown
# Post-Mortem: [Incident Title]

## Summary
- **Date**: [Date]
- **Duration**: [Total duration]
- **Impact**: [User impact description]
- **Root Cause**: [Primary cause]

## Timeline
- [Time] - [Event description]
- [Time] - [Event description]

## Root Cause Analysis
[Detailed analysis of what went wrong]

## Resolution
[What was done to resolve the issue]

## Action Items
- [ ] [Action item 1] - Owner: [Name] - Due: [Date]
- [ ] [Action item 2] - Owner: [Name] - Due: [Date]

## Lessons Learned
[Key takeaways and process improvements]
```

This runbook should be regularly updated based on new learnings and system changes. All team members should be familiar with these procedures and practice them during regular drills.