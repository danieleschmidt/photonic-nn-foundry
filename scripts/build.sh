#!/bin/bash
set -euo pipefail

# Build script for photonic-nn-foundry
# Supports multi-architecture builds and different targets

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="photonic-foundry"
REGISTRY="${REGISTRY:-ghcr.io/danieleschmidt}"
BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
VERSION="${VERSION:-$(git describe --tags --always 2>/dev/null || echo 'dev')}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] [TARGET]

Build Docker images for photonic-nn-foundry

TARGETS:
    development     Build development image (default)
    production      Build production image
    jupyter         Build Jupyter development image
    testing         Build testing image
    docs            Build documentation image
    benchmark       Build benchmarking image
    security        Build security scanning image
    all             Build all targets

OPTIONS:
    -h, --help              Show this help message
    -t, --tag TAG           Tag for the image (default: latest)
    -r, --registry REG      Container registry (default: $REGISTRY)
    --platform PLATFORM    Target platform (e.g., linux/amd64,linux/arm64)
    --no-cache              Build without cache
    --push                  Push to registry after build
    --multi-arch            Build for multiple architectures
    --scan                  Scan image for vulnerabilities after build
    --test                  Run tests after build
    --clean                 Clean up build artifacts
    -v, --verbose           Verbose output

EXAMPLES:
    $0                      # Build development image
    $0 production           # Build production image
    $0 --multi-arch all     # Build all targets for multiple architectures
    $0 --push production    # Build and push production image
    $0 --scan --test prod   # Build, scan, and test production image

EOF
}

# Parse command line arguments
TARGET="development"
TAG="latest"
PLATFORM=""
NO_CACHE=""
PUSH=false
MULTI_ARCH=false
SCAN=false
TEST=false
CLEAN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="--platform $2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --multi-arch)
            MULTI_ARCH=true
            PLATFORM="--platform linux/amd64,linux/arm64"
            shift
            ;;
        --scan)
            SCAN=true
            shift
            ;;
        --test)
            TEST=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -*)
            log_error "Unknown option $1"
            show_help
            exit 1
            ;;
        *)
            TARGET="$1"
            shift
            ;;
    esac
done

# Validate target
valid_targets=("development" "production" "jupyter" "testing" "docs" "benchmark" "security" "all")
if [[ ! " ${valid_targets[@]} " =~ " ${TARGET} " ]]; then
    log_error "Invalid target: $TARGET"
    log_info "Valid targets: ${valid_targets[*]}"
    exit 1
fi

# Enable verbose output if requested
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Clean function
clean_build_artifacts() {
    log_info "Cleaning build artifacts..."
    docker system prune -f --filter "label=org.opencontainers.image.title=Photonic Neural Network Foundry"
    docker builder prune -f
    log_success "Build artifacts cleaned"
}

# Build function
build_image() {
    local target="$1"
    local image_tag="${REGISTRY}/${IMAGE_NAME}:${target}-${TAG}"
    
    log_info "Building ${target} image: ${image_tag}"
    
    # Prepare build arguments
    local build_args=(
        "--file" "Dockerfile.multi-arch"
        "--target" "$target"
        "--tag" "$image_tag"
        "--label" "org.opencontainers.image.created=${BUILD_DATE}"
        "--label" "org.opencontainers.image.revision=${VCS_REF}"
        "--label" "org.opencontainers.image.version=${VERSION}"
        "--label" "org.opencontainers.image.title=Photonic Neural Network Foundry"
        "--label" "org.opencontainers.image.description=Silicon-photonic AI accelerator software stack"
        "--label" "org.opencontainers.image.source=https://github.com/danieleschmidt/photonic-nn-foundry"
    )
    
    # Add optional flags
    if [[ -n "$PLATFORM" ]]; then
        build_args+=($PLATFORM)
    fi
    
    if [[ -n "$NO_CACHE" ]]; then
        build_args+=("$NO_CACHE")
    fi
    
    if [[ "$MULTI_ARCH" == "true" ]]; then
        # Use buildx for multi-architecture builds
        docker buildx build "${build_args[@]}" "$PROJECT_ROOT"
    else
        docker build "${build_args[@]}" "$PROJECT_ROOT"
    fi
    
    log_success "Built ${target} image: ${image_tag}"
    
    # Push if requested
    if [[ "$PUSH" == "true" ]]; then
        log_info "Pushing ${image_tag}..."
        docker push "$image_tag"
        log_success "Pushed ${image_tag}"
    fi
    
    # Scan if requested
    if [[ "$SCAN" == "true" ]]; then
        scan_image "$image_tag"
    fi
    
    # Test if requested
    if [[ "$TEST" == "true" ]]; then
        test_image "$image_tag" "$target"
    fi
}

# Security scanning function
scan_image() {
    local image="$1"
    log_info "Scanning image for vulnerabilities: $image"
    
    # Use Trivy for vulnerability scanning
    if command -v trivy >/dev/null 2>&1; then
        trivy image --exit-code 1 --severity HIGH,CRITICAL "$image"
        log_success "Security scan completed for $image"
    else
        log_warning "Trivy not found, skipping security scan"
    fi
}

# Test function
test_image() {
    local image="$1"
    local target="$2"
    
    log_info "Testing image: $image"
    
    case "$target" in
        "production")
            docker run --rm "$image" photonic-foundry --version
            ;;
        "development"|"testing")
            docker run --rm "$image" pytest --version
            ;;
        "jupyter")
            docker run --rm "$image" jupyter --version
            ;;
        "docs")
            docker run --rm "$image" sphinx-build --version
            ;;
        "benchmark")
            docker run --rm "$image" pytest --version
            ;;
        "security")
            docker run --rm "$image" bandit --version
            ;;
    esac
    
    log_success "Image test completed for $image"
}

# Buildx setup for multi-architecture builds
setup_buildx() {
    if [[ "$MULTI_ARCH" == "true" ]]; then
        log_info "Setting up Docker Buildx for multi-architecture builds..."
        
        # Create and use buildx builder
        docker buildx create --name photonic-builder --use --bootstrap 2>/dev/null || true
        docker buildx inspect --bootstrap
        
        log_success "Docker Buildx setup complete"
    fi
}

# Main function
main() {
    log_info "Starting build process..."
    log_info "Target: $TARGET"
    log_info "Tag: $TAG"
    log_info "Registry: $REGISTRY"
    log_info "Version: $VERSION"
    log_info "Build Date: $BUILD_DATE"
    log_info "VCS Ref: $VCS_REF"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Clean if requested
    if [[ "$CLEAN" == "true" ]]; then
        clean_build_artifacts
    fi
    
    # Setup buildx for multi-arch builds
    setup_buildx
    
    # Build target(s)
    if [[ "$TARGET" == "all" ]]; then
        for target in "development" "production" "jupyter" "testing" "docs" "benchmark" "security"; do
            build_image "$target"
        done
    else
        build_image "$TARGET"
    fi
    
    log_success "Build process completed successfully!"
}

# Trap for cleanup
trap 'log_error "Build process interrupted"' INT TERM

# Run main function
main "$@"