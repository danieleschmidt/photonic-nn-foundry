#!/bin/bash
set -e

# Production Deployment Script for Photonic Neural Network Foundry

echo "🚀 Starting production deployment..."

# Configuration
NAMESPACE="photonic-foundry"
IMAGE_NAME="photonic-foundry"
IMAGE_TAG="${1:-latest}"
REGISTRY="${REGISTRY:-docker.io/photonicfoundry}"

# Functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log "ERROR: kubectl is not installed"
        exit 1
    fi
    
    # Check if helm is available
    if ! command -v helm &> /dev/null; then
        log "ERROR: helm is not installed" 
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log "ERROR: Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log "Prerequisites check passed ✅"
}

build_and_push_image() {
    log "Building and pushing Docker image..."
    
    # Build image
    docker build -f Dockerfile.production.optimized -t "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" .
    
    # Push to registry
    docker push "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    log "Image built and pushed ✅"
}

deploy_kubernetes_resources() {
    log "Deploying Kubernetes resources..."
    
    # Create namespace
    kubectl apply -f deployment/k8s/production/namespace.yaml
    
    # Apply ConfigMap and Secrets
    kubectl apply -f deployment/k8s/production/configmap.yaml
    
    # Note: Secrets need to be populated with actual values
    log "⚠️  Remember to update secrets with actual values before applying"
    # kubectl apply -f deployment/k8s/production/secret.yaml
    
    # Apply other resources
    kubectl apply -f deployment/k8s/production/deployment.yaml
    kubectl apply -f deployment/k8s/production/service.yaml
    kubectl apply -f deployment/k8s/production/ingress.yaml
    kubectl apply -f deployment/k8s/production/hpa.yaml
    
    log "Kubernetes resources deployed ✅"
}

wait_for_deployment() {
    log "Waiting for deployment to be ready..."
    
    kubectl rollout status deployment/photonic-foundry-app -n ${NAMESPACE} --timeout=300s
    
    log "Deployment ready ✅"
}

run_smoke_tests() {
    log "Running smoke tests..."
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get service photonic-foundry-service -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
    
    # Port forward for testing (in background)
    kubectl port-forward service/photonic-foundry-service 8080:80 -n ${NAMESPACE} &
    PORT_FORWARD_PID=$!
    
    sleep 5
    
    # Test health endpoint
    if curl -f http://localhost:8080/health &> /dev/null; then
        log "Health check passed ✅"
    else
        log "Health check failed ❌"
        kill $PORT_FORWARD_PID
        exit 1
    fi
    
    # Cleanup
    kill $PORT_FORWARD_PID
    
    log "Smoke tests passed ✅"
}

# Main execution
main() {
    log "Starting deployment process..."
    
    check_prerequisites
    build_and_push_image
    deploy_kubernetes_resources
    wait_for_deployment
    run_smoke_tests
    
    log "🎉 Production deployment completed successfully!"
    log "Access the application at: https://api.photonic-foundry.com"
    log "Monitor at: https://grafana.photonic-foundry.com"
}

# Execute main function
main "$@"
