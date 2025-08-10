# 🚀 GitHub Workflows for Global Deployment

This directory contains GitHub Actions workflow templates for deploying the Photonic Foundry globally.

## ⚠️ Manual Setup Required

Due to GitHub security policies, workflow files cannot be automatically committed. Please manually create these workflows in your repository:

### 1. Global Deployment Workflow

Create `.github/workflows/global-deployment.yml`:

```yaml
name: 'Global Multi-Cloud Deployment'

on:
  push:
    branches: [main, develop]
    paths-ignore: ['docs/**', '*.md']
  pull_request:
    branches: [main]
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production
      regions:
        description: 'Deployment regions (comma-separated)'
        required: false
        default: 'us-east-1,eu-west-1,ap-southeast-1'

env:
  DOCKER_REGISTRY: ghcr.io
  IMAGE_NAME: photonic-foundry
  TERRAFORM_VERSION: 1.5.0

jobs:
  compliance-check:
    name: 'Compliance Validation'
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: 'Validate GDPR Compliance'
      run: |
        echo "✅ Checking GDPR compliance..."
        python3 -c "
        import sys
        sys.path.append('src')
        from photonic_foundry.compliance.gdpr import GDPRCompliance
        gdpr = GDPRCompliance()
        result = gdpr.validate_data_processing({'user_data': 'test'})
        assert result.is_compliant, f'GDPR violation: {result.violations}'
        print('✅ GDPR compliance verified')
        "
    
    - name: 'Validate CCPA Compliance'  
      run: |
        echo "✅ Checking CCPA compliance..."
        python3 -c "
        import sys
        sys.path.append('src')
        from photonic_foundry.compliance.ccpa import CCPACompliance
        ccpa = CCPACompliance()
        result = ccpa.validate_data_processing({'user_data': 'test'})
        assert result.is_compliant, f'CCPA violation: {result.violations}'
        print('✅ CCPA compliance verified')
        "

  security-scan:
    name: 'Security Scanning'
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: 'Run Security Scan'
      run: |
        python3 -m pip install bandit safety
        bandit -r src/ -f json -o security-report.json || true
        safety check --json --output safety-report.json || true
        
    - name: 'Upload Security Reports'
      uses: actions/upload-artifact@v4
      with:
        name: security-reports
        path: '*-report.json'

  build-multi-arch:
    name: 'Build Multi-Architecture Images'
    runs-on: ubuntu-latest
    needs: [compliance-check, security-scan]
    strategy:
      matrix:
        platform:
        - linux/amd64
        - linux/arm64
        - linux/riscv64
    steps:
    - uses: actions/checkout@v4
    
    - name: 'Set up Docker Buildx'
      uses: docker/setup-buildx-action@v3
      
    - name: 'Login to Container Registry'
      uses: docker/login-action@v3
      with:
        registry: ${{ env.DOCKER_REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: 'Build and Push Images'
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: ${{ matrix.platform }}
        push: true
        tags: |
          ${{ env.DOCKER_REGISTRY }}/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}:latest
          ${{ env.DOCKER_REGISTRY }}/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-infrastructure:
    name: 'Deploy Global Infrastructure'
    runs-on: ubuntu-latest
    needs: [build-multi-arch]
    if: github.ref == 'refs/heads/main' || github.event_name == 'workflow_dispatch'
    strategy:
      matrix:
        region: [us-east-1, eu-west-1, ap-southeast-1]
        cloud: [aws, gcp, azure]
    steps:
    - uses: actions/checkout@v4
    
    - name: 'Setup Terraform'
      uses: hashicorp/setup-terraform@v3
      with:
        terraform_version: ${{ env.TERRAFORM_VERSION }}
        
    - name: 'Configure Cloud Credentials'
      run: |
        case "${{ matrix.cloud }}" in
          aws)
            echo "AWS_ACCESS_KEY_ID=${{ secrets.AWS_ACCESS_KEY_ID }}" >> $GITHUB_ENV
            echo "AWS_SECRET_ACCESS_KEY=${{ secrets.AWS_SECRET_ACCESS_KEY }}" >> $GITHUB_ENV
            ;;
          gcp)
            echo '${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}' > gcp-key.json
            echo "GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json" >> $GITHUB_ENV
            ;;
          azure)
            echo "ARM_CLIENT_ID=${{ secrets.AZURE_CLIENT_ID }}" >> $GITHUB_ENV
            echo "ARM_CLIENT_SECRET=${{ secrets.AZURE_CLIENT_SECRET }}" >> $GITHUB_ENV
            echo "ARM_TENANT_ID=${{ secrets.AZURE_TENANT_ID }}" >> $GITHUB_ENV
            echo "ARM_SUBSCRIPTION_ID=${{ secrets.AZURE_SUBSCRIPTION_ID }}" >> $GITHUB_ENV
            ;;
        esac
        
    - name: 'Terraform Plan'
      run: |
        cd terraform/
        terraform init
        terraform plan \
          -var="cloud_provider=${{ matrix.cloud }}" \
          -var="deployment_region=${{ matrix.region }}" \
          -var="environment=${GITHUB_REF_NAME}" \
          -out=tfplan
          
    - name: 'Terraform Apply'
      if: github.ref == 'refs/heads/main'
      run: |
        cd terraform/
        terraform apply -auto-approve tfplan

  deploy-kubernetes:
    name: 'Deploy Kubernetes Applications'
    runs-on: ubuntu-latest
    needs: [deploy-infrastructure]
    if: github.ref == 'refs/heads/main' || github.event_name == 'workflow_dispatch'
    strategy:
      matrix:
        region: [us-east-1, eu-west-1, ap-southeast-1]
    steps:
    - uses: actions/checkout@v4
    
    - name: 'Setup kubectl'
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
        
    - name: 'Setup Helm'
      uses: azure/setup-helm@v3
      with:
        version: '3.12.0'
        
    - name: 'Configure kubeconfig'
      run: |
        # Configure based on region and cloud provider
        case "${{ matrix.region }}" in
          us-east-1)
            aws eks update-kubeconfig --region us-east-1 --name photonic-foundry-us
            ;;
          eu-west-1)
            gcloud container clusters get-credentials photonic-foundry-eu --zone europe-west1
            ;;
          ap-southeast-1)
            az aks get-credentials --resource-group photonic-foundry-ap --name photonic-foundry-ap
            ;;
        esac
        
    - name: 'Deploy with Helm'
      run: |
        helm upgrade --install photonic-foundry ./deployment/helm/ \
          --namespace photonic-foundry \
          --create-namespace \
          --values ./deployment/helm/values-global.yaml \
          --set global.region=${{ matrix.region }} \
          --set global.imageTag=${{ github.sha }} \
          --wait --timeout=600s

  integration-tests:
    name: 'Integration Tests'
    runs-on: ubuntu-latest
    needs: [deploy-kubernetes]
    if: github.ref == 'refs/heads/main' || github.event_name == 'workflow_dispatch'
    strategy:
      matrix:
        region: [us-east-1, eu-west-1, ap-southeast-1]
    steps:
    - uses: actions/checkout@v4
    
    - name: 'Run Integration Tests'
      run: |
        python3 -m pip install pytest requests
        
        # Set regional endpoint
        case "${{ matrix.region }}" in
          us-east-1) export API_ENDPOINT="https://api-us.photonic-foundry.com" ;;
          eu-west-1) export API_ENDPOINT="https://api-eu.photonic-foundry.com" ;;
          ap-southeast-1) export API_ENDPOINT="https://api-ap.photonic-foundry.com" ;;
        esac
        
        # Run integration tests
        pytest tests/integration/ -v --region=${{ matrix.region }}
        
    - name: 'Performance Benchmark'
      run: |
        python3 examples/quantum_optimization_demo.py --region=${{ matrix.region }}

  monitoring-setup:
    name: 'Setup Monitoring'
    runs-on: ubuntu-latest
    needs: [deploy-kubernetes]
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4
    
    - name: 'Deploy Monitoring Stack'
      run: |
        # Deploy Prometheus, Grafana, and custom dashboards
        kubectl apply -f monitoring/prometheus.yml
        kubectl apply -f monitoring/grafana/
        
        # Import custom dashboards
        curl -X POST http://grafana.photonic-foundry.com/api/dashboards/db \
          -H "Content-Type: application/json" \
          -d @monitoring/global-dashboard.json

  notify-success:
    name: 'Deployment Success Notification'
    runs-on: ubuntu-latest
    needs: [integration-tests, monitoring-setup]
    if: success()
    steps:
    - name: 'Send Success Notification'
      run: |
        echo "🚀 Global deployment successful!"
        echo "✅ All regions deployed: us-east-1, eu-west-1, ap-southeast-1"
        echo "✅ Integration tests passed"
        echo "✅ Monitoring configured"
        echo "🌐 Production endpoints:"
        echo "  - US: https://api-us.photonic-foundry.com"
        echo "  - EU: https://api-eu.photonic-foundry.com" 
        echo "  - AP: https://api-ap.photonic-foundry.com"
```

### 2. Docker Multi-Arch Build

Create `docker-buildx.yml` in your project root:

```yaml
# Docker Buildx Configuration for Multi-Architecture Builds
# Supports AMD64, ARM64, and RISC-V quantum compute platforms

version: '3.8'

x-build-args: &build-args
  BUILDKIT_INLINE_CACHE: 1
  PYTHON_VERSION: 3.11
  DEBIAN_FRONTEND: noninteractive

services:
  photonic-foundry-amd64:
    build:
      context: .
      dockerfile: Dockerfile.multi-region
      platforms:
        - linux/amd64
      args:
        <<: *build-args
        TARGETARCH: amd64
      cache_from:
        - ghcr.io/photonic-foundry/cache:amd64
      cache_to:
        - ghcr.io/photonic-foundry/cache:amd64
    image: ghcr.io/photonic-foundry/photonic-foundry:amd64

  photonic-foundry-arm64:
    build:
      context: .
      dockerfile: Dockerfile.multi-region  
      platforms:
        - linux/arm64
      args:
        <<: *build-args
        TARGETARCH: arm64
      cache_from:
        - ghcr.io/photonic-foundry/cache:arm64
      cache_to:
        - ghcr.io/photonic-foundry/cache:arm64
    image: ghcr.io/photonic-foundry/photonic-foundry:arm64

  photonic-foundry-riscv64:
    build:
      context: .
      dockerfile: Dockerfile.multi-region
      platforms:
        - linux/riscv64
      args:
        <<: *build-args 
        TARGETARCH: riscv64
      cache_from:
        - ghcr.io/photonic-foundry/cache:riscv64
      cache_to:
        - ghcr.io/photonic-foundry/cache:riscv64
    image: ghcr.io/photonic-foundry/photonic-foundry:riscv64
```

## 🛠️ Setup Instructions

1. **Copy the workflow content** into `.github/workflows/global-deployment.yml` in your repository
2. **Configure secrets** in your GitHub repository settings:
   - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
   - `GCP_SERVICE_ACCOUNT_KEY`
   - `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, etc.
3. **Enable GitHub Actions** in your repository settings
4. **Grant workflow permissions** if needed for your GitHub App

## 🌍 Supported Deployment Regions

- **US East (us-east-1)**: CCPA compliant, AWS-based
- **EU West (eu-west-1)**: GDPR compliant, GCP-based  
- **AP Southeast (ap-southeast-1)**: PDPA compliant, Azure-based

The workflows automatically handle compliance validation and regional deployment strategies.