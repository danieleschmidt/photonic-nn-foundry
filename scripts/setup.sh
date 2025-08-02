#!/bin/bash
# Setup script for photonic-nn-foundry development environment

set -euo pipefail

# Configuration
PROJECT_NAME="photonic-nn-foundry"
PYTHON_MIN_VERSION="3.8"
NODE_MIN_VERSION="16.0.0"

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
Setup Script for Photonic Neural Network Foundry

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -f, --force         Force reinstallation of dependencies
    --skip-python       Skip Python environment setup
    --skip-docker       Skip Docker setup verification
    --skip-git          Skip Git hooks setup
    --minimal           Minimal setup (skip optional dependencies)

DESCRIPTION:
    This script sets up the development environment for the photonic-nn-foundry
    project. It will:
    
    1. Check system requirements
    2. Set up Python virtual environment
    3. Install Python dependencies
    4. Configure Git hooks
    5. Verify Docker setup
    6. Run basic tests to ensure everything works

EOF
}

# Parse command line arguments
FORCE=false
SKIP_PYTHON=false
SKIP_DOCKER=false
SKIP_GIT=false
MINIMAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --skip-python)
            SKIP_PYTHON=true
            shift
            ;;
        --skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --skip-git)
            SKIP_GIT=true
            shift
            ;;
        --minimal)
            MINIMAL=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Version comparison function
version_compare() {
    if [[ "$1" == "$2" ]]; then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
        ver1[i]=0
    done
    for ((i=${#ver2[@]}; i<${#ver1[@]}; i++)); do
        ver2[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ -z ${ver2[i]} ]]; then
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]})); then
            return 1
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]})); then
            return 2
        fi
    done
    return 0
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_success "Operating System: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_success "Operating System: macOS"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        log_success "Operating System: Windows"
    else
        log_warning "Unknown operating system: $OSTYPE"
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version | cut -d' ' -f2)
        version_compare "$python_version" "$PYTHON_MIN_VERSION"
        local result=$?
        
        if [[ $result -eq 1 ]] || [[ $result -eq 0 ]]; then
            log_success "Python: $python_version (>= $PYTHON_MIN_VERSION required)"
        else
            log_error "Python version $python_version is too old. Minimum required: $PYTHON_MIN_VERSION"
            return 1
        fi
    else
        log_error "Python 3 is not installed"
        return 1
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
        log_success "pip is available"
    else
        log_error "pip is not installed"
        return 1
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        local git_version=$(git --version | cut -d' ' -f3)
        log_success "Git: $git_version"
    else
        log_error "Git is not installed"
        return 1
    fi
    
    # Check Docker (optional)
    if [[ "$SKIP_DOCKER" == "false" ]]; then
        if command -v docker &> /dev/null; then
            local docker_version=$(docker --version | cut -d' ' -f3 | sed 's/,//')
            log_success "Docker: $docker_version"
            
            # Check if Docker daemon is running
            if docker info &> /dev/null; then
                log_success "Docker daemon is running"
            else
                log_warning "Docker is installed but daemon is not running"
            fi
        else
            log_warning "Docker is not installed (optional for development)"
        fi
        
        # Check Docker Compose
        if command -v docker-compose &> /dev/null; then
            local compose_version=$(docker-compose --version | cut -d' ' -f3 | sed 's/,//')
            log_success "Docker Compose: $compose_version"
        else
            log_warning "Docker Compose is not installed (optional for development)"
        fi
    fi
    
    # Check Node.js (for some tools)
    if command -v node &> /dev/null; then
        local node_version=$(node --version | sed 's/v//')
        version_compare "$node_version" "$NODE_MIN_VERSION"
        local result=$?
        
        if [[ $result -eq 1 ]] || [[ $result -eq 0 ]]; then
            log_success "Node.js: $node_version (optional)"
        else
            log_warning "Node.js version $node_version is old. Some tools may not work properly."
        fi
    else
        log_info "Node.js is not installed (optional for some development tools)"
    fi
}

# Setup Python environment
setup_python_environment() {
    if [[ "$SKIP_PYTHON" == "true" ]]; then
        log_info "Skipping Python environment setup"
        return 0
    fi
    
    log_info "Setting up Python environment..."
    
    # Check if virtual environment already exists
    if [[ -d "venv" && "$FORCE" == "false" ]]; then
        log_info "Virtual environment already exists. Use --force to recreate."
    else
        if [[ -d "venv" ]]; then
            log_warning "Removing existing virtual environment"
            rm -rf venv
        fi
        
        log_info "Creating virtual environment..."
        python3 -m venv venv
        log_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Install development dependencies
    log_info "Installing development dependencies..."
    if [[ -f "requirements-dev.txt" ]]; then
        pip install -r requirements-dev.txt
    else
        log_warning "requirements-dev.txt not found, installing from pyproject.toml"
        pip install -e ".[dev,test,docs]"
    fi
    
    # Install the package in editable mode
    log_info "Installing package in editable mode..."
    pip install -e .
    
    log_success "Python environment setup completed"
}

# Setup Git hooks
setup_git_hooks() {
    if [[ "$SKIP_GIT" == "true" ]]; then
        log_info "Skipping Git hooks setup"
        return 0
    fi
    
    log_info "Setting up Git hooks..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_warning "Not in a Git repository. Skipping Git hooks setup."
        return 0
    fi
    
    # Install pre-commit hooks
    if command -v pre-commit &> /dev/null; then
        log_info "Installing pre-commit hooks..."
        pre-commit install --install-hooks
        log_success "Pre-commit hooks installed"
        
        # Test pre-commit
        log_info "Testing pre-commit configuration..."
        if pre-commit run --all-files > /dev/null 2>&1; then
            log_success "Pre-commit configuration is valid"
        else
            log_warning "Pre-commit found some issues (this is normal for first run)"
        fi
    else
        log_warning "pre-commit not available. Install with: pip install pre-commit"
    fi
    
    # Set up Git configuration
    log_info "Checking Git configuration..."
    
    if ! git config user.name > /dev/null 2>&1; then
        log_warning "Git user.name not set. Please configure with: git config --global user.name 'Your Name'"
    fi
    
    if ! git config user.email > /dev/null 2>&1; then
        log_warning "Git user.email not set. Please configure with: git config --global user.email 'your.email@example.com'"
    fi
}

# Verify Docker setup
verify_docker_setup() {
    if [[ "$SKIP_DOCKER" == "true" ]]; then
        log_info "Skipping Docker setup verification"
        return 0
    fi
    
    if ! command -v docker &> /dev/null; then
        log_info "Docker not available. Skipping Docker verification."
        return 0
    fi
    
    log_info "Verifying Docker setup..."
    
    # Test Docker build
    if [[ -f "Dockerfile" ]]; then
        log_info "Testing Docker build (dry run)..."
        if docker build --dry-run . > /dev/null 2>&1; then
            log_success "Dockerfile syntax is valid"
        else
            log_warning "Dockerfile may have syntax issues"
        fi
    fi
    
    # Test Docker Compose
    if [[ -f "docker-compose.yml" ]] && command -v docker-compose &> /dev/null; then
        log_info "Testing Docker Compose configuration..."
        if docker-compose config > /dev/null 2>&1; then
            log_success "Docker Compose configuration is valid"
        else
            log_warning "Docker Compose configuration may have issues"
        fi
    fi
}

# Run basic tests
run_basic_tests() {
    log_info "Running basic tests to verify setup..."
    
    # Test Python import
    if python -c "import photonic_foundry; print('Package import successful')" 2>/dev/null; then
        log_success "Package can be imported successfully"
    else
        log_warning "Package import failed. This may be expected if the package is not fully implemented."
    fi
    
    # Test CLI
    if python -m photonic_foundry.cli --help > /dev/null 2>&1; then
        log_success "CLI is working"
    else
        log_warning "CLI test failed. This may be expected if CLI is not fully implemented."
    fi
    
    # Test basic pytest
    if command -v pytest &> /dev/null; then
        log_info "Running a quick test suite..."
        if pytest tests/ -x -q --tb=no > /dev/null 2>&1; then
            log_success "Basic tests are passing"
        else
            log_warning "Some tests are failing. This may be expected during development."
        fi
    fi
}

# Create development files
create_development_files() {
    log_info "Creating development configuration files..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f ".env" && -f ".env.example" ]]; then
        log_info "Creating .env file from template..."
        cp .env.example .env
        log_success ".env file created. Please customize it for your environment."
    fi
    
    # Create local directories
    for dir in logs data models outputs; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
}

# Display post-setup instructions
show_post_setup_instructions() {
    log_success "Setup completed successfully!"
    echo
    log_info "Next steps:"
    echo "  1. Activate the virtual environment: source venv/bin/activate"
    echo "  2. Review and customize .env file for your environment"
    echo "  3. Run tests: make test (or pytest tests/)"
    echo "  4. Start development: make dev"
    echo "  5. Read the documentation in docs/"
    echo
    log_info "Useful commands:"
    echo "  make help           - Show available make targets"
    echo "  make test           - Run test suite"
    echo "  make lint           - Run code linting"
    echo "  make format         - Format code"
    echo "  make docker-build   - Build Docker images"
    echo
    log_info "For Docker development:"
    echo "  docker-compose up -d     - Start all services"
    echo "  docker-compose logs -f   - View logs"
    echo "  docker-compose down      - Stop all services"
    echo
}

# Main execution
main() {
    log_info "Starting setup for ${PROJECT_NAME}"
    echo
    
    # Run setup steps
    check_system_requirements || exit 1
    echo
    
    setup_python_environment
    echo
    
    setup_git_hooks
    echo
    
    verify_docker_setup
    echo
    
    create_development_files
    echo
    
    if [[ "$MINIMAL" == "false" ]]; then
        run_basic_tests
        echo
    fi
    
    show_post_setup_instructions
}

# Cleanup function
cleanup() {
    # Deactivate virtual environment if it was activated
    if [[ "${VIRTUAL_ENV:-}" ]]; then
        deactivate 2>/dev/null || true
    fi
}

# Set up trap for cleanup
trap cleanup EXIT

# Run main function
main "$@"