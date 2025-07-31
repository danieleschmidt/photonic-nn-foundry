# GitHub Actions Workflows Documentation

This directory contains documentation for recommended GitHub Actions workflows for photonic-nn-foundry.

## Required Workflows

### 1. CI Pipeline (`ci.yml`)

**Purpose:** Continuous integration testing, linting, and security scanning

**Triggers:**
- Pull requests to main branch
- Push to main branch
- Manual workflow dispatch

**Jobs:**
- **Test Matrix:** Python 3.8, 3.9, 3.10, 3.11 on Ubuntu, macOS, Windows
- **Code Quality:** Black, isort, flake8, mypy, bandit
- **Security Scanning:** Dependency vulnerability check, SAST
- **Coverage:** Generate and upload coverage reports
- **Container:** Build and test Docker images

### 2. Security Scanning (`security.yml`)

**Purpose:** Automated security vulnerability detection

**Triggers:**
- Schedule: Daily at 3 AM UTC
- Pull requests (for new dependencies)
- Manual dispatch

**Features:**
- CodeQL analysis for Python
- Dependency vulnerability scanning
- Container image security scanning
- SBOM generation
- Security advisory notifications

### 3. Release Management (`release.yml`)

**Purpose:** Automated versioning and package publishing

**Triggers:**
- Push to main branch with version tags
- Manual release creation

**Process:**
- Automated changelog generation
- Version bumping based on conventional commits
- PyPI package publishing
- Docker image publishing to registry
- GitHub release creation with artifacts

### 4. Dependency Updates (`dependabot.yml`)

**Purpose:** Automated dependency management

**Configuration:**
- Daily Python dependency checks
- Weekly Docker base image updates
- GitHub Actions updates monthly
- Auto-merge for patch version updates
- Security update prioritization

## Workflow Templates

### Basic CI Template

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", 3.11]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Run pre-commit
      run: pre-commit run --all-files
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Security Scanning Template

```yaml
name: Security Scan
on:
  schedule:
    - cron: '0 3 * * *'  # Daily at 3 AM UTC
  pull_request:
    paths: ['requirements*.txt', 'pyproject.toml']

jobs:
  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Bandit Security Scan
      run: |
        pip install bandit
        bandit -r src/ -f json -o bandit-report.json
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
```

## Setup Instructions

1. **Create `.github/workflows/` directory** in repository root
2. **Copy workflow templates** and customize for your needs
3. **Configure secrets** in repository settings:
   - `PYPI_API_TOKEN` for package publishing
   - `DOCKER_USERNAME` and `DOCKER_PASSWORD` for container registry
   - `CODECOV_TOKEN` for coverage reporting

4. **Enable Dependabot** by creating `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "daily"
    open-pull-requests-limit: 5
    
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
```

## Security Considerations

- **Secrets Management:** Never commit API tokens or passwords
- **Dependency Pinning:** Pin action versions to specific commits
- **Permission Minimization:** Use least-privilege principle for workflow permissions
- **Branch Protection:** Require status checks before merging
- **Code Review:** Mandate review for workflow changes

## Monitoring and Alerts

- **Failed Workflow Notifications:** Configure Slack/email alerts
- **Security Vulnerability Alerts:** Enable GitHub Security Advisories
- **Performance Monitoring:** Track build times and test execution
- **Coverage Tracking:** Monitor code coverage trends

## Troubleshooting

Common issues and solutions:

1. **Python Version Conflicts:** Ensure matrix versions match pyproject.toml
2. **Dependency Installation Failures:** Check requirements.txt compatibility
3. **Test Failures:** Verify test environment setup and fixtures
4. **Security Scan False Positives:** Configure bandit exclusions
5. **Docker Build Issues:** Check multi-stage build dependencies

For detailed workflow implementation, see individual template files and adjust based on project-specific requirements.