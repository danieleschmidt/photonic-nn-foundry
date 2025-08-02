#!/bin/bash
# Health check script for photonic-nn-foundry services

set -euo pipefail

# Configuration
SERVICES=(
    "photonic-foundry:8000:/health"
    "jupyter:8888:/api/status"
    "redis:6379:/ping"
    "postgres:5432:pg_isready"
    "prometheus:9090:/-/healthy"
    "grafana:3000:/api/health"
)

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

# Health check function
check_service_health() {
    local service_info=$1
    local service_name=$(echo "$service_info" | cut -d':' -f1)
    local port=$(echo "$service_info" | cut -d':' -f2)
    local endpoint=$(echo "$service_info" | cut -d':' -f3)
    
    log_info "Checking $service_name..."
    
    # Special handling for different services
    case $service_name in
        "redis")
            if command -v redis-cli &> /dev/null; then
                if redis-cli -h localhost -p "$port" ping &> /dev/null; then
                    log_success "$service_name is healthy"
                    return 0
                fi
            else
                # Fallback to basic connection test
                if timeout 5 bash -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; then
                    log_success "$service_name is responding"
                    return 0
                fi
            fi
            ;;
        "postgres")
            if command -v pg_isready &> /dev/null; then
                if pg_isready -h localhost -p "$port" &> /dev/null; then
                    log_success "$service_name is healthy"
                    return 0
                fi
            else
                # Fallback to basic connection test
                if timeout 5 bash -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; then
                    log_success "$service_name is responding"
                    return 0
                fi
            fi
            ;;
        *)
            # HTTP-based health checks
            if command -v curl &> /dev/null; then
                local url="http://localhost:$port$endpoint"
                if curl -f -s --max-time 10 "$url" > /dev/null 2>&1; then
                    log_success "$service_name is healthy"
                    return 0
                elif curl -s --max-time 10 "$url" > /dev/null 2>&1; then
                    log_warning "$service_name is responding but may have issues"
                    return 1
                fi
            elif command -v wget &> /dev/null; then
                local url="http://localhost:$port$endpoint"
                if wget -q --timeout=10 --spider "$url" 2>/dev/null; then
                    log_success "$service_name is healthy"
                    return 0
                fi
            else
                # Basic port check
                if timeout 5 bash -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; then
                    log_success "$service_name port is open"
                    return 0
                fi
            fi
            ;;
    esac
    
    log_error "$service_name is not healthy"
    return 1
}

# Docker service check
check_docker_services() {
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not available, skipping container checks"
        return 0
    fi
    
    log_info "Checking Docker services..."
    
    # Check if docker-compose is running
    if command -v docker-compose &> /dev/null && [[ -f "docker-compose.yml" ]]; then
        local running_services=$(docker-compose ps --services --filter status=running 2>/dev/null || echo "")
        
        if [[ -n "$running_services" ]]; then
            log_success "Docker Compose services are running:"
            echo "$running_services" | while read -r service; do
                echo "  ✓ $service"
            done
        else
            log_warning "No Docker Compose services are running"
        fi
    fi
    
    # Check individual containers
    local containers=$(docker ps --format "{{.Names}}" 2>/dev/null || echo "")
    
    if [[ -n "$containers" ]]; then
        log_info "Running containers:"
        echo "$containers" | while read -r container; do
            local status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "unknown")
            if [[ "$status" == "healthy" ]]; then
                echo "  ✓ $container (healthy)"
            elif [[ "$status" == "unhealthy" ]]; then
                echo "  ✗ $container (unhealthy)"
            else
                echo "  ? $container (no health check)"
            fi
        done
    else
        log_info "No containers are running"
    fi
}

# System resource check
check_system_resources() {
    log_info "Checking system resources..."
    
    # Check disk space
    local disk_usage=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ $disk_usage -gt 90 ]]; then
        log_error "Disk usage is critical: ${disk_usage}%"
    elif [[ $disk_usage -gt 80 ]]; then
        log_warning "Disk usage is high: ${disk_usage}%"
    else
        log_success "Disk usage is normal: ${disk_usage}%"
    fi
    
    # Check memory usage (if available)
    if command -v free &> /dev/null; then
        local memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
        if [[ $memory_usage -gt 90 ]]; then
            log_error "Memory usage is critical: ${memory_usage}%"
        elif [[ $memory_usage -gt 80 ]]; then
            log_warning "Memory usage is high: ${memory_usage}%"
        else
            log_success "Memory usage is normal: ${memory_usage}%"
        fi
    fi
    
    # Check load average (if available)
    if [[ -f "/proc/loadavg" ]]; then
        local load_avg=$(cat /proc/loadavg | cut -d' ' -f1)
        local cpu_count=$(nproc 2>/dev/null || echo "1")
        local load_percentage=$(echo "$load_avg $cpu_count" | awk '{printf "%.0f", $1/$2 * 100}')
        
        if [[ $load_percentage -gt 100 ]]; then
            log_warning "System load is high: $load_avg (${load_percentage}% of capacity)"
        else
            log_success "System load is normal: $load_avg"
        fi
    fi
}

# Application-specific health checks
check_application_health() {
    log_info "Checking application-specific health..."
    
    # Check Python environment
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
        
        # Test package import
        if python -c "import photonic_foundry" 2>/dev/null; then
            log_success "Python package can be imported"
        else
            log_warning "Python package import failed"
        fi
        
        # Check dependencies
        if command -v pip &> /dev/null; then
            local broken_deps=$(pip check 2>&1 | grep -c "broken" || echo "0")
            if [[ $broken_deps -eq 0 ]]; then
                log_success "Python dependencies are consistent"
            else
                log_warning "Found $broken_deps broken Python dependencies"
            fi
        fi
    else
        log_info "Virtual environment not found, skipping Python checks"
    fi
    
    # Check configuration files
    local config_files=(".env" "pyproject.toml" "requirements.txt")
    for config_file in "${config_files[@]}"; do
        if [[ -f "$config_file" ]]; then
            log_success "Configuration file exists: $config_file"
        else
            log_warning "Configuration file missing: $config_file"
        fi
    done
}

# Network connectivity check
check_network_connectivity() {
    log_info "Checking network connectivity..."
    
    # Check internet connectivity
    if ping -c 1 8.8.8.8 &> /dev/null; then
        log_success "Internet connectivity is available"
    else
        log_warning "Internet connectivity may be limited"
    fi
    
    # Check DNS resolution
    if nslookup google.com &> /dev/null; then
        log_success "DNS resolution is working"
    else
        log_warning "DNS resolution may have issues"
    fi
}

# Main health check function
run_health_checks() {
    local failed_checks=0
    local total_checks=0
    
    echo "=================================="
    echo "Photonic NN Foundry Health Check"
    echo "=================================="
    echo
    
    # System checks
    check_system_resources
    echo
    
    check_network_connectivity
    echo
    
    check_docker_services
    echo
    
    check_application_health
    echo
    
    # Service checks
    log_info "Checking individual services..."
    for service in "${SERVICES[@]}"; do
        ((total_checks++))
        if ! check_service_health "$service"; then
            ((failed_checks++))
        fi
    done
    echo
    
    # Summary
    local success_count=$((total_checks - failed_checks))
    log_info "Health check summary: $success_count/$total_checks services healthy"
    
    if [[ $failed_checks -eq 0 ]]; then
        log_success "All health checks passed!"
        return 0
    elif [[ $failed_checks -lt $((total_checks / 2)) ]]; then
        log_warning "Some health checks failed, but system is mostly healthy"
        return 1
    else
        log_error "Multiple health checks failed, system may have issues"
        return 2
    fi
}

# Usage information
show_usage() {
    cat << EOF
Health Check Script for Photonic Neural Network Foundry

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -q, --quiet         Quiet mode (minimal output)
    -v, --verbose       Verbose mode (detailed output)
    --services-only     Check only service endpoints
    --system-only       Check only system resources
    --docker-only       Check only Docker services

DESCRIPTION:
    This script performs comprehensive health checks on the photonic-nn-foundry
    system, including:
    
    - System resource utilization
    - Network connectivity
    - Docker services status
    - Application health
    - Individual service endpoints

EXIT CODES:
    0 - All checks passed
    1 - Some checks failed
    2 - Many checks failed

EOF
}

# Parse command line arguments
QUIET=false
VERBOSE=false
SERVICES_ONLY=false
SYSTEM_ONLY=false
DOCKER_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --services-only)
            SERVICES_ONLY=true
            shift
            ;;
        --system-only)
            SYSTEM_ONLY=true
            shift
            ;;
        --docker-only)
            DOCKER_ONLY=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Adjust output based on flags
if [[ "$QUIET" == "true" ]]; then
    exec 1>/dev/null
fi

# Run appropriate checks based on flags
if [[ "$SERVICES_ONLY" == "true" ]]; then
    for service in "${SERVICES[@]}"; do
        check_service_health "$service"
    done
elif [[ "$SYSTEM_ONLY" == "true" ]]; then
    check_system_resources
    check_network_connectivity
    check_application_health
elif [[ "$DOCKER_ONLY" == "true" ]]; then
    check_docker_services
else
    run_health_checks
fi