# Build System Documentation

This document describes the comprehensive build system for the Photonic Neural Network Foundry project.

## Overview

The build system is designed to provide:
- Consistent development environment across all platforms
- Automated testing and quality assurance
- Containerized deployment with security best practices
- Semantic versioning and automated release management
- Software Bill of Materials (SBOM) generation for compliance

## Build Tools

### Core Tools

- **Make**: Primary build orchestration via `Makefile`
- **Docker**: Containerization and deployment
- **Docker Compose**: Multi-service development environment
- **Python Build**: Package distribution creation
- **Semantic Release**: Automated versioning and releases

### Quality Assurance

- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting
- **pylint**: Advanced static analysis
- **mypy**: Type checking
- **bandit**: Security scanning
- **safety**: Dependency vulnerability scanning

## Quick Start

```bash
# Setup development environment
make dev
source venv/bin/activate
make install

# Development workflow
make quick    # format + lint + test

# Full production check
make prod-ready

# Container operations
make docker-build
make docker-run
```

## Make Targets

### Development

| Target | Description |
|--------|-------------|
| `make help` | Show all available commands |
| `make dev` | Create Python virtual environment |
| `make install` | Install dependencies and pre-commit hooks |
| `make quick` | Fast development check (format + lint + test) |

### Code Quality

| Target | Description |
|--------|-------------|
| `make format` | Format code with black and isort |
| `make format-check` | Check formatting without changes |
| `make lint` | Run all linting tools |
| `make pre-commit-all` | Run pre-commit on all files |

### Testing

| Target | Description |
|--------|-------------|
| `make test` | Run full test suite with coverage |
| `make test-unit` | Run unit tests only |
| `make test-integration` | Run integration tests |
| `make test-e2e` | Run end-to-end tests |
| `make test-performance` | Run performance benchmarks |
| `make test-security` | Run security tests |

### Docker Operations

| Target | Description |
|--------|-------------|
| `make docker-build` | Build production Docker image |
| `make docker-build-dev` | Build development image |
| `make docker-run` | Run container interactively |
| `make docker-compose-up` | Start all services |
| `make docker-compose-down` | Stop all services |
| `make docker-push` | Push to registry |

### Security

| Target | Description |
|--------|-------------|
| `make security-scan` | Comprehensive security scan |
| `make security-scan-docker` | Scan Docker image |

### Documentation

| Target | Description |
|--------|-------------|
| `make docs-build` | Build documentation |
| `make docs-serve` | Serve docs locally |
| `make docs-clean` | Clean build artifacts |

### Release Management

| Target | Description |
|--------|-------------|
| `make build` | Build distribution packages |
| `make version` | Show current version |
| `make release-check` | Validate release readiness |

### Utilities

| Target | Description |
|--------|-------------|
| `make clean` | Clean build artifacts |
| `make clean-all` | Clean everything including containers |
| `make validate-env` | Check environment setup |
| `make info` | Show system information |

## Container Architecture

### Development Container

The development environment uses VS Code Dev Containers with:

- **Base Image**: Python 3.11 with development tools
- **Features**: Docker-in-Docker, Git, Node.js
- **Extensions**: Python tools, Docker support, GitHub integration
- **Services**: Redis, PostgreSQL, Prometheus, Grafana, Jaeger

### Production Container

Multi-stage Dockerfile with:

1. **Base Stage**: Minimal Python runtime
2. **Dependencies Stage**: Install and compile dependencies
3. **Development Stage**: Add development tools
4. **Production Stage**: Optimized runtime image
5. **Jupyter Stage**: Interactive development

### Security Features

- Non-root user execution
- Minimal attack surface
- Security scanning with Trivy
- Dependency vulnerability checking
- SBOM generation for compliance

## Development Workflow

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/danieleschmidt/photonic-nn-foundry.git
cd photonic-nn-foundry

# Setup development environment
make dev
source venv/bin/activate
make install
```

### 2. Development Container (Recommended)

```bash
# Open in VS Code
code .

# VS Code will prompt to reopen in container
# Or use Command Palette: "Dev Containers: Reopen in Container"
```

### 3. Development Cycle

```bash
# Make changes to code
# ...

# Quick validation
make quick

# Run specific tests
make test-unit

# Full validation before commit
make prod-ready
```

### 4. Commit Process

Pre-commit hooks automatically run:
- Code formatting (black, isort)
- Linting (flake8, pylint)
- Type checking (mypy)
- Security scanning (bandit)
- Commit message validation

## Docker Compose Services

### Core Services

- **photonic-foundry**: Main application container
- **jupyter**: Jupyter Lab for interactive development
- **tests**: Automated test runner
- **docs**: Documentation server

### Infrastructure Services

- **redis**: Caching and session storage
- **postgres**: Database for metadata and results
- **prometheus**: Metrics collection
- **grafana**: Metrics visualization
- **jaeger**: Distributed tracing
- **security-scanner**: Vulnerability scanning

### Port Mapping

| Service | Port | Description |
|---------|------|-------------|
| Application | 8000 | Main API server |
| Jupyter | 8888 | Interactive development |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Metrics dashboard |
| Jaeger UI | 16686 | Tracing interface |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache |

## Release Process

### Semantic Versioning

Automated versioning based on conventional commits:

- `feat:` → Minor version bump
- `fix:` → Patch version bump
- `BREAKING CHANGE:` → Major version bump
- `docs:`, `style:`, `refactor:` → Patch version bump

### Release Pipeline

1. **Version Calculation**: Based on commit messages
2. **Changelog Generation**: Automated from commits
3. **Version Updates**: Update all version references
4. **SBOM Generation**: Create software bill of materials
5. **Asset Creation**: Build distributions and containers
6. **GitHub Release**: Create release with assets
7. **Registry Push**: Push containers to registry

### Manual Release

```bash
# Check release readiness
make release-check

# Build distribution
make build

# Generate SBOM
python scripts/generate_sbom.py

# Create GitHub release (automated via semantic-release)
```

## Environment Variables

### Development

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your settings
```

Key development variables:
- `APP_ENV=development`
- `DEBUG=true`
- `LOG_LEVEL=DEBUG`
- `PHOTONIC_PDK=skywater130`

### Production

Required production variables:
- `APP_ENV=production`
- `DEBUG=false`
- `LOG_LEVEL=INFO`
- Database connection strings
- API keys and secrets
- Registry credentials

## Monitoring and Observability

### Metrics

Prometheus metrics available at `/metrics`:
- Application performance metrics
- Business logic metrics
- Infrastructure metrics
- Custom photonic simulation metrics

### Logging

Structured JSON logging with:
- Request/response logging
- Performance metrics
- Error tracking
- Security events

### Tracing

Distributed tracing with Jaeger:
- Request flow tracking
- Performance bottleneck identification
- Service dependency mapping

## Security Considerations

### Scanning

Automated security scanning includes:
- **Bandit**: Python security issues
- **Safety**: Known vulnerabilities in dependencies
- **Trivy**: Container vulnerabilities
- **CodeQL**: Static analysis (in CI/CD)

### Best Practices

- Secrets managed via environment variables
- Non-root container execution
- Minimal container images
- Regular dependency updates
- SBOM generation for compliance

## Troubleshooting

### Common Issues

**Container build fails**:
```bash
# Clean Docker cache
docker system prune -f
make docker-build
```

**Permission issues in container**:
```bash
# Check user mapping in devcontainer
# Ensure UID/GID match host system
```

**Test failures**:
```bash
# Run specific test with verbose output
pytest tests/unit/test_specific.py -v -s

# Check test dependencies
make validate-env
```

**Development container issues**:
```bash
# Rebuild container
# Command Palette: "Dev Containers: Rebuild Container"

# Check logs
# Command Palette: "Dev Containers: Show Container Log"
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
export DEVELOPMENT_MODE=true
```

### Performance Profiling

```bash
# Run with profiling
make profile

# View results
python -m pstats profile.stats
```

## Contributing

### Build System Changes

When modifying the build system:

1. Update relevant documentation
2. Test changes in clean environment
3. Verify CI/CD compatibility
4. Update version if breaking changes

### Adding Dependencies

```bash
# Add to requirements
echo "new-package>=1.0.0" >> requirements.txt

# For development dependencies
echo "dev-package>=1.0.0" >> requirements-dev.txt

# Rebuild containers
make docker-build
```

### Custom Make Targets

Add custom targets to `Makefile`:

```makefile
my-custom-target: ## Description of target
	@echo "Running custom command"
	# Add commands here
```

## Performance Optimization

### Build Performance

- Multi-stage Docker builds for layer caching
- .dockerignore to reduce build context
- Parallel test execution with pytest-xdist
- Dependency caching in CI/CD

### Runtime Performance

- Optimized Python package installation
- Container resource limits
- Connection pooling for databases
- Caching strategies

## Compliance and Auditing

### SBOM Generation

Automated SBOM (Software Bill of Materials) generation includes:
- Python dependencies with versions
- System dependencies
- Container base images
- Build environment information

### Audit Trail

All builds include:
- Git commit information
- Build timestamp
- Environment details
- Dependency checksums
- Security scan results

This comprehensive build system ensures consistent, secure, and maintainable software delivery for the Photonic Neural Network Foundry project.