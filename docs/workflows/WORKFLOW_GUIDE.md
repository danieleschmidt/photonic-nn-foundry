# GitHub Workflows Guide

This document provides a comprehensive guide to the GitHub Actions workflows implemented for the Photonic Neural Network Foundry project.

## Overview

The SDLC implementation includes three primary workflows that provide comprehensive automation for development, security, and maintenance:

1. **CI/CD Pipeline** (`ci.yml`) - Core development workflow
2. **Security Scanning** (`security-scan.yml`) - Comprehensive security automation
3. **Dependency Management** (`dependency-update.yml`) - Automated dependency maintenance

## Workflow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Developer     │    │   Pull Request  │    │   Main Branch   │
│   Push/PR       │────▶│   Validation    │────▶│   Deployment    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Code Quality   │    │  Security Scan  │    │   Release       │
│  & Testing      │    │  & Compliance   │    │   Automation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 1. CI/CD Pipeline Workflow

### Trigger Conditions

```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]
```

### Workflow Jobs

#### Job Flow
```
Quality Check ──┐
                ├─▶ Container Security ──▶ Build & Push ──▶ Deploy Staging ──▶ Release
Test Matrix ────┤
                └─▶ Performance Tests ─────────────────────────────────────────▶ Notify
```

#### 1. Code Quality & Security (`quality`)

**Purpose**: Validate code quality and security standards

**Matrix Strategy**: Parallel execution of linting, security, and type checking

**Key Steps**:
- **Linting**: flake8 + pylint with configurable thresholds
- **Security**: bandit + safety for vulnerability scanning
- **Type Checking**: mypy with strict mode

**Quality Gates**:
- Pylint score ≥ 8.0
- Zero critical security issues
- Type checking compliance

**Outputs**: Security reports uploaded as artifacts

#### 2. Test Suite (`test`)

**Purpose**: Comprehensive testing across multiple environments

**Matrix Strategy**:
```yaml
matrix:
  os: [ubuntu-latest, windows-latest, macos-latest]
  python-version: ['3.8', '3.9', '3.10', '3.11']
```

**Test Categories**:
- **Unit Tests**: Core functionality validation
- **Integration Tests**: Component interaction testing
- **Coverage Analysis**: Minimum 80% coverage requirement

**Platform-Specific Setup**:
- Ubuntu: OpenBLAS/LAPACK installation
- macOS: Homebrew dependencies
- Windows: Native toolchain setup

#### 3. Performance Testing (`performance`)

**Purpose**: Validate performance characteristics and prevent regressions

**Trigger**: Main branch pushes only

**Features**:
- Benchmark execution with pytest-benchmark
- Historical trend analysis
- Regression detection with 200% threshold
- Automated alerts on performance degradation

#### 4. Container Security (`container-security`)

**Purpose**: Comprehensive container security validation

**Security Tools**:
- **Trivy**: Multi-layer vulnerability scanning
- **Container Structure Tests**: Image validation
- **Grype**: Alternative vulnerability detection

**Scan Types**:
- Image vulnerabilities
- Configuration issues
- Filesystem security
- SARIF output for GitHub Security tab

#### 5. Build & Push (`build-push`)

**Purpose**: Multi-architecture container build and registry push

**Features**:
- Multi-platform builds (AMD64, ARM64)
- Layer caching optimization
- Semantic versioning tags
- SBOM generation for compliance

**Registry Integration**:
- GitHub Container Registry (GHCR)
- Automated metadata extraction
- Digest-based image references

#### 6. Documentation (`docs`)

**Purpose**: Automated documentation generation and deployment

**Process**:
- Sphinx documentation build
- GitHub Pages deployment
- API documentation generation
- Version-controlled documentation

#### 7. Semantic Release (`release`)

**Purpose**: Automated versioning and release management

**Process Flow**:
1. Conventional commit analysis
2. Version calculation
3. Changelog generation
4. Asset creation (wheels, SBOM)
5. GitHub release publication
6. PyPI package publishing

**Version Bump Rules**:
- `feat:` → Minor version
- `fix:` → Patch version
- `BREAKING CHANGE:` → Major version

#### 8. CodeQL Analysis (`codeql`)

**Purpose**: Advanced static application security testing

**Features**:
- Security-extended query pack
- Automated vulnerability detection
- GitHub Security tab integration
- Continuous security monitoring

## 2. Security Scanning Workflow

### Comprehensive Security Pipeline

```
Python Security ──┐
                  ├─▶ Security Report ──▶ Compliance Check
Container Scan ───┤
                  ├─▶ Aggregation
License Check ────┤
                  ├─▶ Analysis
Secrets Scan ─────┤
                  └─▶ Remediation
IaC Security ─────┘
```

#### Security Scan Categories

##### 1. Python Dependency Security

**Tools**:
- **Safety**: Known vulnerability database
- **Bandit**: Static security analysis
- **pip-audit**: Supply chain security

**Process**:
```bash
safety check --json --output safety-report.json
bandit -r src/ -f json -o bandit-report.json
pip-audit --format=cyclonedx --output=sbom.json
```

##### 2. Container Security

**Multi-Tool Approach**:
- **Trivy**: Comprehensive scanning
- **Grype**: Anchore vulnerability detection
- **Syft**: SBOM generation

**Scan Coverage**:
- OS packages
- Language libraries
- Configuration files
- Container layers

##### 3. License Compliance

**License Analysis**:
- Dependency license extraction
- Compatibility checking
- GPL/copyleft detection
- Commercial license validation

##### 4. Secrets Detection

**Tools**:
- **TruffleHog**: High-accuracy secret detection
- **GitLeaks**: Git history scanning
- **detect-secrets**: Baseline management

**Coverage**:
- Source code
- Git history
- Configuration files
- Documentation

##### 5. Infrastructure as Code Security

**IaC Scanning**:
- **Checkov**: Multi-framework analysis
- **kube-score**: Kubernetes security
- **Docker best practices**: Container hardening

#### 6. Dynamic Application Security Testing (DAST)

**Process**:
1. Application deployment
2. Service availability validation
3. OWASP ZAP scanning
4. Vulnerability assessment

**OWASP ZAP Integration**:
- Baseline passive scanning
- Full active security testing
- Custom rule configuration
- Automated report generation

## 3. Dependency Management Workflow

### Automated Dependency Maintenance

#### Update Types

1. **Patch Updates**: Bug fixes and security patches
2. **Minor Updates**: New features, backward compatible
3. **Major Updates**: Breaking changes (manual review required)
4. **Security Updates**: Critical vulnerability fixes

#### Process Flow

```
Vulnerability Check ──▶ Dependency Update ──▶ Testing ──▶ PR Creation ──▶ Auto-merge
       │                      │                   │            │              │
       ▼                      ▼                   ▼            ▼              ▼
Security Analysis    Requirements Update    Full Test Suite   Review Process   Cleanup
```

#### Automation Features

##### 1. Smart Update Strategy

```python
# Update logic based on type
if update_type == "security":
    # Only update vulnerable packages
    update_vulnerable_packages()
elif update_type == "patch":
    # Patch-level updates only
    pip_compile("--upgrade-package '*'")
elif update_type in ["minor", "major"]:
    # Full dependency refresh
    pip_compile("--upgrade")
```

##### 2. Comprehensive Testing

**Test Coverage**:
- Unit test validation
- Integration test execution
- Security vulnerability re-scan
- Performance regression testing
- Container build validation

##### 3. Intelligent Auto-merge

**Auto-merge Conditions**:
- All tests pass
- No breaking changes detected
- Security improvements confirmed
- Update type is patch or security

**Safety Mechanisms**:
- Draft PR for failed tests
- Manual review requirement for major updates
- Rollback procedures for issues

#### Container Updates

**Base Image Management**:
- Automated base image updates
- Security patch detection
- Compatibility validation
- Multi-architecture support

**Update Process**:
1. Check for newer base images
2. Update Dockerfile
3. Build and test container
4. Create update PR

## Workflow Configuration

### Environment Variables

```yaml
env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
  PYTHON_VERSION: '3.11'
  FAIL_ON_CRITICAL: true
  MAX_HIGH_VULNS: 5
```

### Matrix Strategy Examples

#### Test Matrix
```yaml
strategy:
  fail-fast: false
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ['3.8', '3.9', '3.10', '3.11']
    exclude:
      - os: windows-latest
        python-version: '3.8'
```

#### Security Scan Matrix
```yaml
strategy:
  matrix:
    scan-type: [dependencies, container, secrets, iac]
    tool: [primary, secondary]
```

### Conditional Execution

#### Branch-Based Conditions
```yaml
if: github.ref == 'refs/heads/main'  # Main branch only
if: github.event_name == 'pull_request'  # PR only
if: contains(github.event.head_commit.message, '[deploy]')  # Message-based
```

#### Dependency Conditions
```yaml
needs: [quality, test]  # Sequential dependency
if: needs.test.result == 'success'  # Conditional on previous job
```

## Monitoring and Notifications

### Success Notifications

```yaml
- name: Notify Success
  uses: 8398a7/action-slack@v3
  with:
    status: success
    channel: '#deployments'
    text: '✅ Deployment successful'
```

### Failure Alerting

```yaml
- name: Alert on Failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    channel: '#alerts'
    text: '❌ Critical workflow failure'
```

### Security Alerts

```yaml
- name: Security Alert
  if: steps.security-scan.outputs.critical-issues > 0
  run: |
    curl -X POST $SECURITY_WEBHOOK \
      -d "Critical security issues found: ${{ steps.security-scan.outputs.critical-issues }}"
```

## Best Practices

### Performance Optimization

1. **Caching Strategy**:
   ```yaml
   - uses: actions/cache@v3
     with:
       path: ~/.cache/pip
       key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
   ```

2. **Parallel Execution**:
   - Matrix strategies for parallel testing
   - Job dependencies for optimal flow
   - Conditional execution to reduce unnecessary work

3. **Resource Management**:
   - Appropriate runner selection
   - Timeout configuration
   - Resource cleanup

### Security Considerations

1. **Secret Management**:
   - Use GitHub secrets for sensitive data
   - Rotate tokens regularly
   - Principle of least privilege

2. **Supply Chain Security**:
   - Pin action versions
   - Use verified actions
   - Generate SBOMs

3. **Code Signing**:
   - Sign container images
   - Verify signatures in deployment
   - Maintain signing key security

### Maintenance

1. **Regular Updates**:
   - Keep action versions current
   - Update security tools
   - Review and update thresholds

2. **Monitoring**:
   - Track workflow success rates
   - Monitor execution times
   - Review security findings

3. **Documentation**:
   - Keep workflow documentation current
   - Document configuration changes
   - Maintain troubleshooting guides

## Troubleshooting

### Common Issues

1. **Permission Errors**:
   - Verify token permissions
   - Check repository settings
   - Validate secret configuration

2. **Build Failures**:
   - Check dependency compatibility
   - Verify environment setup
   - Review error logs

3. **Security Scan False Positives**:
   - Review scan configuration
   - Add exclusions for known issues
   - Update security tools

### Debug Strategies

1. **Enable Debug Logging**:
   ```yaml
   env:
     ACTIONS_STEP_DEBUG: true
   ```

2. **Workflow Debugging**:
   - Use `continue-on-error` for investigation
   - Add debug output steps
   - Review action documentation

3. **Local Testing**:
   - Test workflows with act
   - Validate Docker builds locally
   - Run security scans manually

This comprehensive workflow system provides enterprise-grade CI/CD capabilities with security, compliance, and automation built-in from the start.