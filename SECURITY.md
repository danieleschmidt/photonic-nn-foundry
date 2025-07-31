# Security Policy

## Supported Versions

We actively maintain security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### Reporting Process

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Send a detailed report to: security@terragonlabs.com
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Any suggested fixes (if available)

### Response Timeline

- **Initial Response**: Within 48 hours of report
- **Status Update**: Weekly updates until resolution
- **Resolution Timeline**: 90 days maximum for critical issues

### Security Best Practices

When using photonic-nn-foundry:

- Keep dependencies updated to latest versions
- Use container security scanning for Docker images
- Validate all input data and model files
- Implement proper authentication for CLI access
- Monitor for unusual computational resource usage

## Security Features

- Input validation via Pydantic models
- Container security with non-root user execution
- Dependency vulnerability scanning via Bandit
- Pre-commit security hooks enabled

## Disclosure Policy

We follow coordinated disclosure principles:
- 90-day disclosure timeline for critical vulnerabilities
- Public acknowledgment of security researchers (with permission)
- Security advisories published via GitHub Security Advisories

For non-security related issues, please use our standard [issue tracker](../../issues).