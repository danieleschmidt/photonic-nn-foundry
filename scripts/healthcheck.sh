#!/bin/bash
set -euo pipefail

# Health check script for photonic-nn-foundry containers
# Verifies that the application is running correctly

# Configuration
TIMEOUT=10
LOG_FILE="/tmp/healthcheck.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [HEALTHCHECK] $1" | tee -a "$LOG_FILE"
}

# Check if Python is working
check_python() {
    log "Checking Python environment..."
    if ! timeout "$TIMEOUT" python3 -c "import sys; print(f'Python {sys.version} OK')" 2>/dev/null; then
        log "ERROR: Python check failed"
        return 1
    fi
    log "Python environment OK"
    return 0
}

# Check if photonic-foundry package is importable
check_package_import() {
    log "Checking package import..."
    if ! timeout "$TIMEOUT" python3 -c "
try:
    import photonic_foundry
    print('Package import OK')
except ImportError as e:
    print(f'Package import failed: {e}')
    exit(1)
" 2>/dev/null; then
        log "ERROR: Package import failed"
        return 1
    fi
    log "Package import OK"
    return 0
}

# Check if CLI is responsive
check_cli() {
    log "Checking CLI responsiveness..."
    if command -v photonic-foundry >/dev/null 2>&1; then
        if ! timeout "$TIMEOUT" photonic-foundry --version >/dev/null 2>&1; then
            log "ERROR: CLI not responsive"
            return 1
        fi
        log "CLI responsive"
    else
        log "WARNING: CLI not found (might be development environment)"
    fi
    return 0
}

# Check disk space
check_disk_space() {
    log "Checking disk space..."
    local available_space
    available_space=$(df /tmp | awk 'NR==2 {print $4}')
    local min_space=1048576  # 1GB in KB
    
    if [ "$available_space" -lt "$min_space" ]; then
        log "ERROR: Insufficient disk space (${available_space}KB available, ${min_space}KB required)"
        return 1
    fi
    log "Disk space OK (${available_space}KB available)"
    return 0
}

# Check memory usage
check_memory() {
    log "Checking memory usage..."
    local memory_usage
    memory_usage=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    local min_memory=262144  # 256MB in KB
    
    if [ "$memory_usage" -lt "$min_memory" ]; then
        log "WARNING: Low memory (${memory_usage}KB available)"
    else
        log "Memory OK (${memory_usage}KB available)"
    fi
    return 0
}

# Check for Jupyter if running
check_jupyter() {
    if pgrep -f "jupyter" >/dev/null 2>&1; then
        log "Checking Jupyter Lab..."
        if ! timeout "$TIMEOUT" curl -f http://localhost:8888/ >/dev/null 2>&1; then
            log "ERROR: Jupyter Lab not responsive"
            return 1
        fi
        log "Jupyter Lab OK"
    fi
    return 0
}

# Check for development dependencies
check_dev_dependencies() {
    log "Checking development dependencies..."
    local dev_tools=("pytest" "black" "isort" "mypy")
    
    for tool in "${dev_tools[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            log "Development tool $tool found"
        fi
    done
    return 0
}

# Check file permissions
check_permissions() {
    log "Checking file permissions..."
    local test_file="/tmp/permission_test_$$"
    
    if ! touch "$test_file" 2>/dev/null; then
        log "ERROR: Cannot write to /tmp"
        return 1
    fi
    
    rm -f "$test_file"
    log "File permissions OK"
    return 0
}

# Main health check function
main() {
    log "Starting health check..."
    local exit_code=0
    
    # Initialize log file
    : > "$LOG_FILE"
    
    # Run checks
    check_python || exit_code=$?
    check_package_import || exit_code=$?
    check_cli || exit_code=$?
    check_disk_space || exit_code=$?
    check_memory || exit_code=$?
    check_jupyter || exit_code=$?
    check_dev_dependencies || exit_code=$?
    check_permissions || exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "Health check PASSED"
    else
        log "Health check FAILED with exit code $exit_code"
        # Show recent logs for debugging
        echo "Recent log entries:"
        tail -10 "$LOG_FILE"
    fi
    
    return $exit_code
}

# Trap signals for graceful shutdown
trap 'log "Health check interrupted"; exit 130' INT TERM

# Run main function
main "$@"