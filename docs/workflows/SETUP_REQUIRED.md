# Required GitHub Setup for SDLC Workflows

This document describes the manual setup required to activate the SDLC workflows for the Photonic Neural Network Foundry project.

## Overview

Due to GitHub App permission limitations, the following workflows and configurations must be manually copied and configured by repository maintainers.

## Quick Setup Checklist

- [ ] Copy workflow files to `.github/workflows/`
- [ ] Configure repository secrets
- [ ] Enable GitHub features
- [ ] Set up branch protection rules
- [ ] Configure Dependabot
- [ ] Enable GitHub Pages
- [ ] Set up environments

## 1. Copy Workflow Files

Copy the following workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

## 2. Required Repository Secrets

Configure the following secrets in **Settings → Secrets and variables → Actions**:

### Core Secrets

| Secret Name | Description | Required For |
|-------------|-------------|--------------|
| `SEMANTIC_RELEASE_TOKEN` | GitHub Personal Access Token with repo permissions | Automated releases |
| `PYPI_TOKEN` | PyPI API token for package publishing | Package publishing |
| `CODECOV_TOKEN` | Codecov integration token | Code coverage reporting |
| `DEPENDENCY_UPDATE_TOKEN` | GitHub PAT for dependency PRs | Automated dependency updates |

### Optional Secrets

| Secret Name | Description | Required For |
|-------------|-------------|--------------|
| `SLACK_WEBHOOK` | Slack webhook URL for notifications | Team notifications |
| `DOCKER_REGISTRY_TOKEN` | Container registry authentication | Private registry push |
| `SONAR_TOKEN` | SonarCloud integration token | Advanced code analysis |

### Creating GitHub Personal Access Tokens

1. Go to **Settings → Developer settings → Personal access tokens**
2. Click **Generate new token (classic)**
3. Select the following scopes:
   - `repo` (Full repository access)
   - `workflow` (Update GitHub Action workflows)
   - `write:packages` (Upload packages)
4. Copy the token and add it to repository secrets

## 3. Enable GitHub Features

### Security Features

1. **Code scanning alerts**:
   - Go to **Settings → Security & analysis**
   - Enable **Dependency graph**
   - Enable **Dependabot alerts**
   - Enable **Dependabot security updates**
   - Enable **Secret scanning**

2. **Advanced security** (GitHub Enterprise):
   - Enable **Code scanning**
   - Enable **Secret scanning push protection**

### GitHub Pages

1. Go to **Settings → Pages**
2. Set **Source** to "GitHub Actions"
3. Documentation will be published automatically on main branch updates

## 4. Branch Protection Rules

Set up branch protection for the `main` branch:

1. Go to **Settings → Branches**
2. Click **Add rule**
3. Configure the following:

```yaml
Branch name pattern: main
Protection rules:
  ☑ Require a pull request before merging
    ☑ Require approvals: 1
    ☑ Dismiss stale reviews when new commits are pushed
    ☑ Require review from code owners
  ☑ Require status checks to pass before merging
    ☑ Require branches to be up to date before merging
    Required status checks:
      - Code Quality & Security (lint)
      - Code Quality & Security (security)
      - Code Quality & Security (type-check)
      - Test Suite (ubuntu-latest, 3.11)
      - Container Security Scan
  ☑ Require conversation resolution before merging
  ☑ Restrict pushes that create files larger than 100 MB
  ☑ Do not allow bypassing the above settings
```

## 5. Configure Dependabot

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 10
    reviewers:
      - "team-leads"
    assignees:
      - "maintainer"
    commit-message:
      prefix: "chore"
      include: "scope"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5

  # Docker
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

## 6. Set Up Environments

Create the following environments in **Settings → Environments**:

### Staging Environment

```yaml
Name: staging
Protection rules:
  ☑ Required reviewers: 1
  ☑ Wait timer: 0 minutes
Environment secrets:
  STAGING_DATABASE_URL: [staging database connection]
  STAGING_API_KEY: [staging API credentials]
```

### Production Environment

```yaml
Name: production
Protection rules:
  ☑ Required reviewers: 2
  ☑ Wait timer: 5 minutes
  ☑ Deployment branches: Selected branches (main)
Environment secrets:
  PRODUCTION_DATABASE_URL: [production database connection]
  PRODUCTION_API_KEY: [production API credentials]
```

## 7. Repository Settings

Configure additional repository settings:

### General Settings

1. **Features**:
   - ☑ Issues
   - ☑ Projects
   - ☑ Wiki
   - ☑ Discussions

2. **Pull Requests**:
   - ☑ Allow merge commits
   - ☑ Allow squash merging (default)
   - ☑ Allow rebase merging
   - ☑ Always suggest updating pull request branches
   - ☑ Allow auto-merge
   - ☑ Automatically delete head branches

### Security Settings

1. **Vulnerability reporting**:
   - ☑ Private vulnerability reporting

2. **Token permissions**:
   - Set **Actions permissions** to "Permissive"
   - Set **Workflow permissions** to "Read and write permissions"

## 8. Issue and PR Templates

The repository already includes issue and PR templates. Verify they are working:

- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/pull_request_template.md`

## 9. Container Registry Setup

### GitHub Container Registry (GHCR)

1. **Enable GitHub Packages**:
   - The workflows are configured to use `ghcr.io`
   - No additional setup required

### Alternative Registry (Docker Hub, AWS ECR, etc.)

Update workflows if using a different registry:

```yaml
env:
  REGISTRY: docker.io  # or your registry
  IMAGE_NAME: username/photonic-foundry
```

Add registry credentials to secrets:
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

## 10. Third-Party Integrations

### Codecov

1. Sign up at [codecov.io](https://codecov.io)
2. Connect your GitHub repository
3. Copy the repository token
4. Add `CODECOV_TOKEN` to repository secrets

### SonarCloud (Optional)

1. Sign up at [sonarcloud.io](https://sonarcloud.io)
2. Import your repository
3. Copy the project token
4. Add `SONAR_TOKEN` to repository secrets
5. Add SonarCloud workflow step:

```yaml
- name: SonarCloud Scan
  uses: SonarSource/sonarcloud-github-action@master
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

## 11. Notifications Setup

### Slack Integration

1. Create a Slack webhook:
   - Go to your Slack workspace
   - Create a new app
   - Add webhook URL to secrets as `SLACK_WEBHOOK`

### Email Notifications

Configure in **Settings → Notifications**:
- ☑ Actions
- ☑ Dependabot alerts
- ☑ Security alerts

## 12. Validation Checklist

After setup, verify the following:

### Workflow Validation

- [ ] Workflows appear in **Actions** tab
- [ ] No syntax errors in workflow files
- [ ] All required secrets are configured
- [ ] Branch protection rules are active

### Security Validation

- [ ] Dependabot alerts are enabled
- [ ] Secret scanning is active
- [ ] Code scanning is configured
- [ ] Security advisories are enabled

### Integration Validation

- [ ] Codecov integration working
- [ ] Container registry push successful
- [ ] Notifications are delivered
- [ ] Documentation builds correctly

## 13. Troubleshooting

### Common Issues

**Workflow fails with "Secret not found"**:
- Verify secret names match exactly
- Check secret is available to the workflow

**Permission denied errors**:
- Ensure PAT has sufficient permissions
- Check repository settings allow workflow writes

**Container push fails**:
- Verify registry credentials
- Check image name format
- Ensure registry permissions

**Dependabot not creating PRs**:
- Check `dependabot.yml` syntax
- Verify branch protection doesn't block automation
- Ensure reviewers/assignees exist

### Debug Workflows

Enable debug logging by adding repository secret:
```
ACTIONS_STEP_DEBUG = true
```

### Contact Support

For additional help:
- GitHub Discussions in this repository
- GitHub Support (for GitHub Enterprise)
- Team lead or maintainer contact

## 14. Maintenance

### Regular Tasks

**Monthly**:
- Review secret expiration dates
- Update PAT tokens before expiry
- Review security alerts and advisories
- Validate backup procedures

**Quarterly**:
- Review and update branch protection rules
- Audit user permissions and access
- Test disaster recovery procedures
- Update integration configurations

This setup provides enterprise-grade CI/CD capabilities with comprehensive security, testing, and automation for the Photonic Neural Network Foundry project.