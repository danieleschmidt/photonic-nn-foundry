# Terragon Autonomous SDLC System

This directory contains the autonomous value discovery and execution system for photonic-nn-foundry.

## Overview

The Terragon Autonomous SDLC system continuously discovers, prioritizes, and executes the highest-value work items to improve repository maturity and deliver ongoing value.

## Components

### 1. Value Discovery Engine (`value-discovery.py`)
Discovers potential work items from multiple sources:
- Code comments (TODO, FIXME, HACK markers)
- Test gaps (unimplemented test methods)
- Static analysis results
- Security vulnerabilities
- Performance gaps

### 2. Autonomous Executor (`autonomous-executor.py`)
Executes high-value work items autonomously:
- Creates feature branches
- Implements changes
- Runs quality gates
- Commits results or rolls back

### 3. Continuous Monitor (`continuous-monitor.py`)
Tracks long-term trends and anomalies:
- Historical value metrics
- Backlog growth patterns
- Technical debt trends
- Anomaly detection

### 4. Configuration (`config.json`)
Adaptive configuration based on repository maturity:
- Scoring weights (WSJF, ICE, Technical Debt)
- Execution thresholds
- Quality gate requirements

## Usage

### Manual Discovery
```bash
python3 .terragon/value-discovery.py
```

### Manual Execution
```bash
python3 .terragon/autonomous-executor.py
```

### Monitoring
```bash
python3 .terragon/continuous-monitor.py
```

### View Current Backlog
```bash
cat BACKLOG.md
```

## Automated Execution

### GitHub Actions
Copy the workflow template to enable automated discovery:
```bash
cp docs/workflows/autonomous-value.yml .github/workflows/
```

### Cron Schedule (Linux/macOS)
Add to crontab for regular monitoring:
```bash
# Every 6 hours - value discovery
0 */6 * * * cd /path/to/repo && python3 .terragon/value-discovery.py

# Daily - continuous monitoring  
0 2 * * * cd /path/to/repo && python3 .terragon/continuous-monitor.py

# Weekly - deep analysis and reporting
0 3 * * 1 cd /path/to/repo && python3 .terragon/value-discovery.py && python3 .terragon/autonomous-executor.py
```

## Scoring Model

### Repository Maturity: MATURING (75%)
- **WSJF Weight**: 60% (Weighted Shortest Job First)
- **Technical Debt**: 20% (Maintenance cost reduction)
- **ICE Weight**: 10% (Impact × Confidence × Ease)
- **Security**: 10% (Risk mitigation boost)

### Execution Thresholds
- **Minimum Score**: 15 (composite)
- **Maximum Effort**: 4 hours
- **Quality Gates**: 80% test coverage, linting pass
- **Autonomous Types**: test_debt, dependency_update, technical_debt

## Files Generated

### Core System Files
- `.terragon/config.json` - System configuration
- `.terragon/value-metrics.json` - Current discovered items
- `.terragon/value-trends.json` - Historical trend data
- `.terragon/execution-history.json` - Autonomous execution log

### Reports
- `BACKLOG.md` - Current prioritized backlog
- `.terragon/monitoring-report.md` - Trend analysis report

## Integration Points

### CI/CD Integration
- Triggers on PR merge to main branch
- Scheduled runs every 6 hours
- Manual dispatch capability
- Automatic backlog updates

### Quality Gates
- Test suite execution
- Code coverage verification (≥80%)
- Linting and style checks
- Security scan validation

### Safety Mechanisms
- Automatic rollback on failures
- Branch-based execution (no direct main commits)
- Human review requirements for high-risk changes
- Execution logging for continuous learning

## Metrics Tracked

### Value Delivery
- Total items discovered
- Items completed per cycle
- Average composite scores
- Value delivered (score × completion)

### Quality Indicators
- Test coverage trends
- Technical debt ratio
- Security vulnerability count
- Code quality metrics

### Process Efficiency
- Discovery cycle time
- Execution success rate
- False positive rates
- Human intervention frequency

## Customization

### Scoring Weights
Modify `.terragon/config.json` to adjust prioritization:
```json
{
  "scoring": {
    "weights": {
      "maturing": {
        "wsjf": 0.6,
        "technicalDebt": 0.2,
        "ice": 0.1,
        "security": 0.1
      }
    }
  }
}
```

### Execution Control
Configure autonomous execution behavior:
```json
{
  "execution": {
    "maxConcurrentTasks": 1,
    "testRequirements": {
      "minCoverage": 80
    },
    "rollbackTriggers": [
      "testFailure",
      "buildFailure"
    ]
  }
}
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   chmod +x .terragon/*.py
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Git Configuration**
   ```bash
   git config user.name "Terragon Autonomous SDLC"
   git config user.email "terragon@noreply.github.com"
   ```

4. **Quality Gate Failures**
   - Check test coverage: `pytest --cov=src`
   - Run linting: `flake8 src/`
   - Fix failing tests before autonomous execution

### Logs and Debugging
- Execution history: `.terragon/execution-history.json`
- Trend data: `.terragon/value-trends.json`
- System logs: Check GitHub Actions output

## Advanced Usage

### Custom Discovery Sources
Extend `ValueDiscoveryEngine` to add new discovery sources:
```python
def _discover_from_custom_source(self) -> List[Dict[str, Any]]:
    # Implement custom discovery logic
    return discovered_items
```

### Custom Executors
Add new execution types by extending `AutonomousExecutor`:
```python
def execute_custom_type(self, item: dict) -> bool:
    # Implement custom execution logic
    return success
```

### Monitoring Extensions
Add custom metrics to `ContinuousMonitor`:
```python
def calculate_custom_metrics(self, items: List[Dict]) -> Dict[str, float]:
    # Calculate domain-specific metrics
    return metrics
```

---

**🤖 Terragon Autonomous SDLC v1.0**  
**Continuously discovering and delivering maximum value**