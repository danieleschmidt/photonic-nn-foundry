#!/bin/bash
# Docker build script for photonic-nn-foundry

set -euo pipefail

# Configuration
PROJECT_NAME="photonic-nn-foundry"
REGISTRY="ghcr.io/danieleschmidt"
DOCKERFILE="Dockerfile"
BUILD_CONTEXT="."

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
Docker Build Script for Photonic Neural Network Foundry

Usage: $0 [OPTIONS] [TARGET]

OPTIONS:
    -h, --help          Show this help message
    -v, --version       Specify version tag (default: latest)
    -r, --registry      Specify registry (default: ${REGISTRY})
    -p, --push          Push image to registry after build
    -c, --cache         Use cache for build
    --no-cache          Disable cache for build
    --platform          Specify platform (e.g., linux/amd64,linux/arm64)
    --build-arg         Pass build arguments (can be used multiple times)

TARGETS:
    production          Build production image (default)
    development         Build development image
    jupyter             Build Jupyter image
    all                 Build all images

EXAMPLES:
    $0                              # Build production image with latest tag
    $0 -v 1.0.0 -p production      # Build and push production v1.0.0
    $0 development                  # Build development image
    $0 --platform linux/amd64,linux/arm64 --push all  # Multi-arch build and push
    $0 --build-arg PYTHON_VERSION=3.11 production      # Custom build args

EOF
}

# Parse command line arguments
VERSION="latest"
REGISTRY_URL="${REGISTRY}"
PUSH=false
USE_CACHE=true
PLATFORM=""
BUILD_ARGS=()
TARGET="production"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY_URL="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -c|--cache)
            USE_CACHE=true
            shift
            ;;
        --no-cache)
            USE_CACHE=false
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --build-arg)
            BUILD_ARGS+=("--build-arg" "$2")
            shift 2
            ;;
        production|development|jupyter|all)
            TARGET="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate Docker availability
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    log_error "Docker daemon is not running"
    exit 1
fi

# Build function
build_image() {
    local target=$1
    local image_name="${REGISTRY_URL}/${PROJECT_NAME}"
    local tag="${VERSION}"
    local full_tag="${image_name}:${tag}"
    
    if [[ "${target}" != "production" ]]; then
        full_tag="${image_name}:${tag}-${target}"
    fi
    
    log_info "Building ${target} image: ${full_tag}"
    
    # Prepare build command
    local build_cmd="docker build"
    
    # Add platform if specified
    if [[ -n "${PLATFORM}" ]]; then
        build_cmd+=" --platform ${PLATFORM}"
    fi
    
    # Add cache option
    if [[ "${USE_CACHE}" == "false" ]]; then
        build_cmd+=" --no-cache"
    fi
    
    # Add build args
    if [[ ${#BUILD_ARGS[@]} -gt 0 ]]; then
        build_cmd+=" ${BUILD_ARGS[*]}"
    fi
    
    # Add target, tag, and context
    build_cmd+=" --target ${target} -t ${full_tag} ${BUILD_CONTEXT}"
    
    # Add latest tag for production
    if [[ "${target}" == "production" && "${VERSION}" != "latest" ]]; then
        build_cmd+=" -t ${image_name}:latest"
    fi
    
    log_info "Executing: ${build_cmd}"
    
    # Execute build
    if eval "${build_cmd}"; then
        log_success "Successfully built ${full_tag}"
        
        # Push if requested
        if [[ "${PUSH}" == "true" ]]; then
            push_image "${full_tag}"
            
            # Push latest tag for production
            if [[ "${target}" == "production" && "${VERSION}" != "latest" ]]; then
                push_image "${image_name}:latest"
            fi
        fi
    else
        log_error "Failed to build ${full_tag}"
        return 1
    fi
}

# Push function
push_image() {
    local image=$1
    
    log_info "Pushing image: ${image}"
    
    if docker push "${image}"; then
        log_success "Successfully pushed ${image}"
    else
        log_error "Failed to push ${image}"
        return 1
    fi
}

# Build security scan
security_scan() {
    local image=$1
    
    log_info "Running security scan on ${image}"
    
    # Check if Trivy is available
    if command -v trivy &> /dev/null; then
        if trivy image --exit-code 1 --severity HIGH,CRITICAL "${image}"; then
            log_success "Security scan passed for ${image}"
        else
            log_warning "Security vulnerabilities found in ${image}"
        fi
    else
        log_warning "Trivy not available for security scanning"
    fi
}

# Image size optimization check
check_image_size() {
    local image=$1
    
    log_info "Checking image size for ${image}"
    
    local size=$(docker images --format "table {{.Size}}" "${image}" | tail -n +2)
    log_info "Image size: ${size}"
    
    # Warn if image is too large (>2GB)
    local size_bytes=$(docker inspect "${image}" --format='{{.Size}}')
    local size_gb=$((size_bytes / 1024 / 1024 / 1024))
    
    if [[ ${size_gb} -gt 2 ]]; then
        log_warning "Image size is quite large (${size_gb}GB). Consider optimization."
    fi
}

# Multi-architecture build
build_multiarch() {
    local target=$1
    local image_name="${REGISTRY_URL}/${PROJECT_NAME}"
    local tag="${VERSION}"
    local full_tag="${image_name}:${tag}"
    
    if [[ "${target}" != "production" ]]; then
        full_tag="${image_name}:${tag}-${target}"
    fi
    
    log_info "Building multi-architecture image: ${full_tag}"
    
    # Create and use buildx builder
    docker buildx create --name multiarch-builder --use 2>/dev/null || docker buildx use multiarch-builder
    
    # Build and push multi-arch image
    local buildx_cmd="docker buildx build --platform ${PLATFORM} --target ${target}"
    
    if [[ "${USE_CACHE}" == "false" ]]; then
        buildx_cmd+=" --no-cache"
    fi
    
    if [[ ${#BUILD_ARGS[@]} -gt 0 ]]; then
        buildx_cmd+=" ${BUILD_ARGS[*]}"
    fi
    
    buildx_cmd+=" -t ${full_tag}"
    
    if [[ "${PUSH}" == "true" ]]; then
        buildx_cmd+=" --push"
    fi
    
    buildx_cmd+=" ${BUILD_CONTEXT}"
    
    log_info "Executing: ${buildx_cmd}"
    
    if eval "${buildx_cmd}"; then
        log_success "Successfully built multi-arch image ${full_tag}"
    else
        log_error "Failed to build multi-arch image ${full_tag}"
        return 1
    fi
}

# Main execution
main() {
    log_info "Starting Docker build for ${PROJECT_NAME}"
    log_info "Target: ${TARGET}, Version: ${VERSION}, Push: ${PUSH}"
    
    case "${TARGET}" in
        "production"|"development"|"jupyter")
            if [[ -n "${PLATFORM}" && "${PLATFORM}" == *","* ]]; then
                build_multiarch "${TARGET}"
            else
                build_image "${TARGET}"
                
                # Additional checks for built image
                local image_name="${REGISTRY_URL}/${PROJECT_NAME}"
                local full_tag="${image_name}:${VERSION}"
                
                if [[ "${TARGET}" != "production" ]]; then
                    full_tag="${image_name}:${VERSION}-${TARGET}"
                fi
                
                check_image_size "${full_tag}"
                
                if [[ "${TARGET}" == "production" ]]; then
                    security_scan "${full_tag}"
                fi
            fi
            ;;
        "all")
            for target in production development jupyter; do
                if [[ -n "${PLATFORM}" && "${PLATFORM}" == *","* ]]; then
                    build_multiarch "${target}"
                else
                    build_image "${target}"
                fi
            done
            ;;
        *)
            log_error "Unknown target: ${TARGET}"
            show_help
            exit 1
            ;;
    esac
    
    log_success "Build process completed successfully!"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Remove dangling images
    docker image prune -f >/dev/null 2>&1 || true
}

# Set up trap for cleanup
trap cleanup EXIT

# Run main function
main "$@"