# Observability and Monitoring

This document describes the comprehensive observability strategy for the Photonic Neural Network Foundry.

## Overview

The observability stack provides comprehensive monitoring, alerting, and debugging capabilities for:

- Application performance and health
- Business metrics and SLAs
- Infrastructure resource usage
- Security and compliance monitoring
- Photonic simulation performance
- Distributed system tracing

## Architecture

### Monitoring Stack

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│   Prometheus    │───▶│     Grafana     │
│                 │    │   (Metrics)     │    │  (Visualization)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Jaeger      │    │   AlertManager  │    │    Dashboards   │
│   (Tracing)     │    │   (Alerting)    │    │   & Reports     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Metrics Collection**: Applications expose metrics via `/metrics` endpoints
2. **Metric Storage**: Prometheus scrapes and stores time-series data
3. **Visualization**: Grafana creates dashboards and alerts
4. **Tracing**: Jaeger collects distributed traces
5. **Alerting**: AlertManager handles alert routing and notifications

## Components

### Prometheus

**Purpose**: Time-series metrics collection and storage

**Configuration**: `/monitoring/prometheus.yml`

**Key Features**:
- 15-second scrape interval for real-time monitoring
- 30-day retention policy
- Multi-target service discovery
- Custom photonic simulation metrics

**Metrics Endpoints**:
```
http://localhost:9090/metrics     # Prometheus self-metrics
http://localhost:8000/metrics     # Application metrics
http://localhost:8000/simulation/metrics  # Simulation-specific metrics
```

### Grafana

**Purpose**: Metrics visualization and dashboards

**Access**: http://localhost:3000 (admin/admin)

**Dashboards**:
- **Overview Dashboard**: Application health, request rates, response times
- **Performance Dashboard**: Resource usage, simulation metrics
- **Business Dashboard**: Throughput, user activity, SLA compliance
- **Infrastructure Dashboard**: System resources, database performance

**Key Features**:
- Automated dashboard provisioning
- Real-time data visualization
- Custom alerting rules
- Multi-datasource support

### Jaeger

**Purpose**: Distributed tracing and request flow analysis

**Access**: http://localhost:16686

**Features**:
- End-to-end request tracing
- Performance bottleneck identification
- Service dependency mapping
- Error propagation tracking

## Metrics Categories

### Application Metrics

**HTTP Metrics**:
```prometheus
# Request rate by method and status
http_requests_total{method="GET", status="200"}

# Request duration percentiles
http_request_duration_seconds_bucket

# Active connections
http_active_connections
```

**Business Metrics**:
```prometheus
# Photonic simulations
photonic_simulation_total{type="linear_layer"}
photonic_simulation_duration_seconds
photonic_simulation_errors_total

# Transpiler operations
transpiler_operations_total{operation="pytorch_to_verilog"}
transpiler_queue_size
transpiler_compilation_time_seconds
```

### Infrastructure Metrics

**System Resources**:
```prometheus
# Memory usage
process_resident_memory_bytes
container_memory_usage_bytes

# CPU utilization
process_cpu_seconds_total
container_cpu_usage_seconds_total

# Disk I/O
container_fs_reads_total
container_fs_writes_total
```

**Database Metrics**:
```prometheus
# PostgreSQL
pg_up
pg_stat_database_tup_inserted
pg_stat_database_tup_updated

# Redis
redis_connected_clients
redis_used_memory_bytes
redis_keyspace_hits_total
```

### Custom Photonic Metrics

**Simulation Performance**:
```prometheus
# Optical circuit complexity
photonic_circuit_components_total{type="mzi"}
photonic_circuit_depth

# Power consumption estimates
photonic_power_consumption_watts
photonic_energy_per_operation_picojoules

# Accuracy metrics
photonic_simulation_accuracy_percentage
photonic_noise_floor_db
```

## Alerting

### Alert Categories

**Critical Alerts** (Immediate Response Required):
- Application completely down
- Database connectivity lost
- High error rates (>10%)
- Critical resource exhaustion

**Warning Alerts** (Investigation Required):
- High response times (>2s)
- Memory usage above 80%
- Unusual user activity patterns
- Simulation failure rates >5%

**Info Alerts** (Awareness):
- Low throughput periods
- SLA boundary approaches
- Planned maintenance windows

### Alert Rules

Located in `/monitoring/alerting/prometheus-rules.yml`:

```yaml
- alert: ApplicationDown
  expr: up{job="photonic-foundry"} == 0
  for: 1m
  severity: critical

- alert: HighSimulationLatency
  expr: histogram_quantile(0.95, photonic_simulation_duration_seconds_bucket) > 30
  for: 10m
  severity: warning
```

### Alert Routing

Configure AlertManager for notification routing:

```yaml
# Example alerting configuration
route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://slack-webhook/alerts'
```

## Dashboards

### Overview Dashboard

**Key Panels**:
- Application health status
- Request rate and error rate
- Response time percentiles
- Active user sessions

**Use Cases**:
- Quick health checks
- Incident response
- Performance monitoring

### Performance Dashboard

**Key Panels**:
- CPU and memory utilization
- Database connection pools
- Cache hit rates
- Garbage collection metrics

**Use Cases**:
- Capacity planning
- Performance optimization
- Resource allocation

### Business Dashboard

**Key Panels**:
- Simulation throughput
- User engagement metrics
- Feature usage statistics
- SLA compliance

**Use Cases**:
- Business intelligence
- Product decisions
- Customer success metrics

## Distributed Tracing

### Trace Collection

Jaeger collects traces from:
- HTTP request handlers
- Database queries
- External API calls
- Photonic simulation workflows

### Trace Analysis

**Performance Analysis**:
```
Request → Auth → Validation → Simulation → Response
   5ms     2ms        3ms        150ms      2ms
```

**Error Investigation**:
- Trace error propagation
- Identify failure points
- Analyze retry patterns

### Custom Spans

Add custom spans for photonic operations:

```python
with tracer.start_span('photonic_simulation') as span:
    span.set_tag('circuit_type', 'linear_layer')
    span.set_tag('num_components', len(components))
    result = simulate_photonic_circuit(components)
    span.set_tag('simulation_time', time.time() - start)
```

## Health Checks

### Application Health

**Endpoint**: `/health`

**Checks**:
- Database connectivity
- Cache availability
- External service status
- Simulation engine status

**Response Format**:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-02T10:00:00Z",
  "checks": {
    "database": "healthy",
    "cache": "healthy", 
    "simulation_engine": "healthy"
  },
  "version": "1.0.0"
}
```

### Deep Health Checks

**Endpoint**: `/health/detailed`

**Additional Checks**:
- Photonic library versions
- Hardware acceleration status
- Model validation
- Configuration validation

## Log Management

### Structured Logging

Use JSON format for logs:

```json
{
  "timestamp": "2025-08-02T10:00:00Z",
  "level": "INFO",
  "logger": "photonic_foundry.simulation",
  "message": "Simulation completed successfully",
  "trace_id": "abc123",
  "span_id": "def456",
  "user_id": "user123",
  "simulation_id": "sim789",
  "duration_ms": 150,
  "circuit_components": 42
}
```

### Log Levels

- **DEBUG**: Detailed development information
- **INFO**: General application flow
- **WARN**: Potentially harmful situations
- **ERROR**: Error events that don't stop execution
- **CRITICAL**: Serious errors that may cause termination

### Log Aggregation

For production deployments, consider:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Fluentd for log collection
- Centralized log storage
- Log retention policies

## Performance Monitoring

### Key Performance Indicators (KPIs)

**Availability**: 99.9% uptime target
```prometheus
(sum(rate(http_requests_total{status!~"5.."}[5m])) / 
 sum(rate(http_requests_total[5m]))) * 100
```

**Response Time**: 95th percentile under 1 second
```prometheus
histogram_quantile(0.95, 
  rate(http_request_duration_seconds_bucket[5m]))
```

**Throughput**: Minimum 100 simulations per hour
```prometheus
rate(photonic_simulation_total[1h])
```

**Error Rate**: Less than 1% error rate
```prometheus
(sum(rate(http_requests_total{status=~"5.."}[5m])) / 
 sum(rate(http_requests_total[5m]))) * 100
```

### Simulation-Specific KPIs

**Accuracy**: Simulation accuracy tracking
```prometheus
avg(photonic_simulation_accuracy_percentage)
```

**Efficiency**: Energy per operation
```prometheus
avg(photonic_energy_per_operation_picojoules)
```

**Complexity**: Circuit complexity trends
```prometheus
avg(photonic_circuit_components_total)
```

## Troubleshooting

### Common Issues

**High Memory Usage**:
1. Check for memory leaks in simulations
2. Review garbage collection metrics
3. Analyze object allocation patterns
4. Consider increasing memory limits

**Slow Response Times**:
1. Identify bottlenecks using tracing
2. Check database query performance
3. Review cache hit rates
4. Analyze simulation complexity

**Alert Fatigue**:
1. Review alert thresholds
2. Implement alert suppression
3. Group related alerts
4. Improve signal-to-noise ratio

### Monitoring Checklist

**Daily**:
- [ ] Review error rates and trends
- [ ] Check resource utilization
- [ ] Verify backup completion
- [ ] Monitor security events

**Weekly**:
- [ ] Analyze performance trends
- [ ] Review capacity planning
- [ ] Update alert thresholds
- [ ] Test disaster recovery

**Monthly**:
- [ ] Dashboard maintenance
- [ ] Metric retention review
- [ ] Performance baseline update
- [ ] Monitoring infrastructure health

## Security Monitoring

### Security Metrics

**Authentication**:
```prometheus
authentication_failures_total
authentication_success_total
session_duration_seconds
```

**Authorization**:
```prometheus
authorization_denied_total{resource="simulation"}
privilege_escalation_attempts_total
```

**Network Security**:
```prometheus
http_requests_total{status="403"}
suspicious_ip_requests_total
rate_limit_exceeded_total
```

### Security Alerts

- Failed authentication attempts (>5/min)
- Privilege escalation attempts
- Unusual API usage patterns
- Security scanner detections

## Production Considerations

### Scalability

**Horizontal Scaling**:
- Multiple Prometheus instances
- Grafana clustering
- Distributed tracing storage
- Load balancing for metrics

**Data Management**:
- Metric retention policies
- Downsampling strategies
- Archive storage
- Backup procedures

### High Availability

**Redundancy**:
- Multiple monitoring instances
- Cross-region replication
- Failover procedures
- Health check monitoring

**Disaster Recovery**:
- Configuration backups
- Data replication
- Recovery procedures
- Testing protocols

### Cost Optimization

**Storage Management**:
- Appropriate retention periods
- Metric sampling rates
- Data compression
- Archive strategies

**Resource Optimization**:
- Right-sizing instances
- Query optimization
- Cache utilization
- Network efficiency

This comprehensive observability setup ensures complete visibility into the Photonic Neural Network Foundry's operation, performance, and health.