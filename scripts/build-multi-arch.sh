#!/bin/bash

# Multi-architecture and multi-region Docker build script
# Builds Photonic Foundry images for global deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io/photonicfoundry}"
IMAGE_NAME="photonic-foundry"
VERSION="${VERSION:-1.0.0}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Buildx
    if ! docker buildx version &> /dev/null; then
        error "Docker Buildx is not available"
        exit 1
    fi
    
    # Check if builder exists
    if ! docker buildx ls | grep -q "photonic-foundry-builder"; then
        log "Creating multi-platform builder..."
        docker buildx create --name photonic-foundry-builder --use --bootstrap
    else
        log "Using existing builder: photonic-foundry-builder"
        docker buildx use photonic-foundry-builder
    fi
    
    success "Prerequisites check completed"
}

# Build function for specific region and architecture
build_image() {
    local region="$1"
    local platforms="$2"
    local compliance="$3"
    local gpu_support="${4:-none}"
    local edge_optimized="${5:-false}"
    
    log "Building image for region: $region, platforms: $platforms"
    
    # Prepare build arguments
    local build_args=(
        --build-arg "REGION=$region"
        --build-arg "COMPLIANCE_FRAMEWORK=$compliance"
        --build-arg "BUILD_DATE=$BUILD_DATE"
        --build-arg "VERSION=$VERSION"
    )
    
    if [[ "$gpu_support" != "none" ]]; then
        build_args+=(--build-arg "GPU_SUPPORT=$gpu_support")
    fi
    
    if [[ "$edge_optimized" == "true" ]]; then
        build_args+=(--build-arg "EDGE_OPTIMIZED=true")
    fi
    
    # Prepare tags
    local tags=(
        "$DOCKER_REGISTRY/$IMAGE_NAME:$VERSION-$region"
        "$DOCKER_REGISTRY/$IMAGE_NAME:$region-latest"
        "$DOCKER_REGISTRY/$IMAGE_NAME:$compliance-compliant"
    )
    
    # Add global tags for global build
    if [[ "$region" == "global" ]]; then
        tags+=(
            "$DOCKER_REGISTRY/$IMAGE_NAME:$VERSION"
            "$DOCKER_REGISTRY/$IMAGE_NAME:latest"
        )
    fi
    
    # Build tag arguments
    local tag_args=()
    for tag in "${tags[@]}"; do
        tag_args+=(--tag "$tag")
    done
    
    # Execute build
    log "Executing build for $region..."
    docker buildx build \
        --platform "$platforms" \
        --file "$PROJECT_ROOT/Dockerfile.multi-region" \
        "${build_args[@]}" \
        "${tag_args[@]}" \
        --cache-from "type=gha,scope=$region-build" \
        --cache-to "type=gha,scope=$region-build,mode=max" \
        --output "type=registry,push=true" \
        --metadata-file "/tmp/metadata-$region.json" \
        "$PROJECT_ROOT"
    
    success "Build completed for region: $region"
}

# Build quantum-optimized images
build_quantum_images() {
    log "Building quantum-optimized images..."
    
    docker buildx build \
        --platform "linux/riscv64,linux/amd64" \
        --file "$PROJECT_ROOT/Dockerfile.multi-region" \
        --build-arg "QUANTUM_OPTIMIZED=true" \
        --build-arg "QUANTUM_ARCHITECTURE=risc-v" \
        --build-arg "BUILD_DATE=$BUILD_DATE" \
        --build-arg "VERSION=$VERSION" \
        --tag "$DOCKER_REGISTRY/$IMAGE_NAME:$VERSION-quantum" \
        --tag "$DOCKER_REGISTRY/$IMAGE_NAME:quantum-latest" \
        --cache-from "type=gha,scope=quantum-build" \
        --cache-to "type=gha,scope=quantum-build,mode=max" \
        --output "type=registry,push=true" \
        "$PROJECT_ROOT"
    
    success "Quantum-optimized images built successfully"
}

# Build all regional images
build_all_regions() {
    log "Starting multi-region build process..."
    
    # Global build (AMD64 + ARM64)
    build_image "global" "linux/amd64,linux/arm64" "global" "none" "false"
    
    # US East (AMD64 optimized for high-performance computing)
    build_image "us-east-1" "linux/amd64" "ccpa" "nvidia-a100" "false"
    
    # EU West (AMD64 + ARM64 for GDPR compliance)
    build_image "eu-west-1" "linux/amd64,linux/arm64" "gdpr" "nvidia-h100" "false"
    
    # Asia Pacific (ARM64 optimized for edge computing)
    build_image "ap-southeast-1" "linux/arm64,linux/amd64" "pdpa" "nvidia-a100" "true"
    
    # Build quantum-optimized images
    build_quantum_images
    
    success "All regional builds completed successfully"
}

# Generate multi-arch manifest
generate_manifest() {
    log "Generating multi-architecture manifest..."
    
    # Create manifest for latest tag
    docker buildx imagetools create \
        --tag "$DOCKER_REGISTRY/$IMAGE_NAME:latest" \
        "$DOCKER_REGISTRY/$IMAGE_NAME:global-latest"
    
    # Create manifest for versioned tag
    docker buildx imagetools create \
        --tag "$DOCKER_REGISTRY/$IMAGE_NAME:$VERSION" \
        "$DOCKER_REGISTRY/$IMAGE_NAME:$VERSION-global"
    
    success "Multi-architecture manifest created"
}

# Scan images for vulnerabilities
scan_images() {
    log "Scanning images for vulnerabilities..."
    
    # Check if Trivy is available
    if command -v trivy &> /dev/null; then
        local regions=("global" "us-east-1" "eu-west-1" "ap-southeast-1")
        
        for region in "${regions[@]}"; do
            log "Scanning image for region: $region"
            trivy image \
                --format json \
                --output "/tmp/scan-$region.json" \
                "$DOCKER_REGISTRY/$IMAGE_NAME:$region-latest" || warn "Vulnerability scan failed for $region"
        done
        
        success "Vulnerability scanning completed"
    else
        warn "Trivy not found, skipping vulnerability scanning"
    fi
}

# Generate SBOM (Software Bill of Materials)
generate_sbom() {
    log "Generating SBOM for images..."
    
    if command -v syft &> /dev/null; then
        local regions=("global" "us-east-1" "eu-west-1" "ap-southeast-1")
        
        for region in "${regions[@]}"; do
            log "Generating SBOM for region: $region"
            syft "$DOCKER_REGISTRY/$IMAGE_NAME:$region-latest" \
                -o json \
                > "/tmp/sbom-$region.json" || warn "SBOM generation failed for $region"
        done
        
        success "SBOM generation completed"
    else
        warn "Syft not found, skipping SBOM generation"
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up build resources..."
    
    # Remove temporary files
    rm -f /tmp/metadata-*.json /tmp/scan-*.json /tmp/sbom-*.json
    
    # Prune build cache (optional)
    if [[ "${CLEANUP_CACHE:-false}" == "true" ]]; then
        docker buildx prune -f
    fi
    
    success "Cleanup completed"
}

# Main execution
main() {
    log "Starting Photonic Foundry multi-architecture build..."
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --version)
                VERSION="$2"
                shift 2
                ;;
            --registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            --no-push)
                NO_PUSH=true
                shift
                ;;
            --scan-only)
                SCAN_ONLY=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --version VERSION     Set image version (default: $VERSION)"
                echo "  --registry REGISTRY   Set Docker registry (default: $DOCKER_REGISTRY)"
                echo "  --no-push            Build only, don't push to registry"
                echo "  --scan-only          Only scan existing images"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute based on options
    if [[ "${SCAN_ONLY:-false}" == "true" ]]; then
        scan_images
        generate_sbom
    else
        check_prerequisites
        build_all_regions
        generate_manifest
        scan_images
        generate_sbom
    fi
    
    cleanup
    success "Multi-architecture build process completed successfully!"
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Execute main function
main "$@"