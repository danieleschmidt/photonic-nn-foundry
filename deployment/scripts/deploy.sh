#!/bin/bash

# Photonic Foundry Production Deployment Script
# Usage: ./deploy.sh [staging|production]

set -e

ENVIRONMENT=${1:-production}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_DIR="$PROJECT_ROOT"

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

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root. Consider using a non-root user with sudo privileges."
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check available memory
    AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [[ $AVAILABLE_MEM -lt 4000 ]]; then
        log_warning "Available memory is less than 4GB. Performance may be impacted."
    fi
    
    # Check disk space
    AVAILABLE_DISK=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [[ $AVAILABLE_DISK -lt 50 ]]; then
        log_warning "Available disk space is less than 50GB. Consider freeing up space."
    fi
    
    log_success "System requirements check completed"
}

# Prepare environment
prepare_environment() {
    log_info "Preparing environment for $ENVIRONMENT deployment..."
    
    # Create required directories
    sudo mkdir -p /opt/photonic-foundry/{data,cache,logs,backups}
    sudo chown -R $USER:$USER /opt/photonic-foundry
    
    # Copy environment file if it doesn't exist
    if [[ ! -f "$DEPLOYMENT_DIR/.env.$ENVIRONMENT" ]]; then
        if [[ -f "$DEPLOYMENT_DIR/.env.example" ]]; then
            cp "$DEPLOYMENT_DIR/.env.example" "$DEPLOYMENT_DIR/.env.$ENVIRONMENT"
            log_warning "Created .env.$ENVIRONMENT from example. Please edit it with your configuration."
        else
            log_error ".env.example not found. Cannot create environment file."
            exit 1
        fi
    fi
    
    log_success "Environment preparation completed"
}

# Check SSL certificates
check_ssl() {
    log_info "Checking SSL certificates..."
    
    SSL_DIR="$DEPLOYMENT_DIR/nginx/ssl"
    CERT_FILE="$SSL_DIR/photonic-foundry.crt"
    KEY_FILE="$SSL_DIR/photonic-foundry.key"
    
    if [[ ! -f "$CERT_FILE" ]] || [[ ! -f "$KEY_FILE" ]]; then
        log_warning "SSL certificates not found. Creating self-signed certificates for testing..."
        
        mkdir -p "$SSL_DIR"
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$KEY_FILE" \
            -out "$CERT_FILE" \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        log_warning "Self-signed certificates created. Replace with proper certificates for production."
    else
        # Check certificate validity
        if openssl x509 -in "$CERT_FILE" -checkend 86400 -noout; then
            log_success "SSL certificates are valid"
        else
            log_warning "SSL certificate expires within 24 hours"
        fi
    fi
}

# Build images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker build -f Dockerfile.multi-arch -t photonic-foundry:production \
            --build-arg BUILD_ENV=production .
    else
        docker build -f Dockerfile -t photonic-foundry:staging .
    fi
    
    log_success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    cd "$DEPLOYMENT_DIR"
    
    # Stop existing services
    if docker-compose -f "docker-compose.$ENVIRONMENT.yml" ps -q | grep -q .; then
        log_info "Stopping existing services..."
        docker-compose -f "docker-compose.$ENVIRONMENT.yml" down
    fi
    
    # Start services
    docker-compose -f "docker-compose.$ENVIRONMENT.yml" up -d
    
    log_success "Services deployed successfully"
}

# Health checks
perform_health_checks() {
    log_info "Performing health checks..."
    
    # Wait for services to start
    sleep 30
    
    # Check API health
    if curl -sf http://localhost:8000/health > /dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Check Prometheus
    if curl -sf http://localhost:9090/-/healthy > /dev/null; then
        log_success "Prometheus health check passed"
    else
        log_warning "Prometheus health check failed"
    fi
    
    # Check Grafana
    if curl -sf http://localhost:3000/api/health > /dev/null; then
        log_success "Grafana health check passed"
    else
        log_warning "Grafana health check failed"
    fi
    
    log_success "Health checks completed"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create Grafana dashboards directory if it doesn't exist
    mkdir -p "$DEPLOYMENT_DIR/../monitoring/grafana/dashboards"
    
    # Copy dashboard configurations
    if [[ -d "$PROJECT_ROOT/monitoring/grafana/dashboards" ]]; then
        cp -r "$PROJECT_ROOT/monitoring/grafana/dashboards/"* \
            "$DEPLOYMENT_DIR/../monitoring/grafana/dashboards/" 2>/dev/null || true
    fi
    
    log_success "Monitoring setup completed"
}

# Setup backups
setup_backups() {
    log_info "Setting up backup system..."
    
    # Create backup script
    cat > /opt/photonic-foundry/backup.sh << 'EOF'
#!/bin/bash

BACKUP_DIR=/opt/photonic-foundry/backups
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
docker exec photonic-api-prod sqlite3 /app/data/circuits.db \
    ".backup '/app/data/backup_${DATE}.db'" || exit 1

# Copy backup to backup directory
docker cp photonic-api-prod:/app/data/backup_${DATE}.db \
    $BACKUP_DIR/circuits_${DATE}.db

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -name "circuits_*.db" -mtime +30 -delete

echo "Backup completed: circuits_${DATE}.db"
EOF
    
    chmod +x /opt/photonic-foundry/backup.sh
    
    # Setup cron job for daily backups
    (crontab -l 2>/dev/null; echo "0 2 * * * /opt/photonic-foundry/backup.sh") | crontab -
    
    log_success "Backup system configured"
}

# Display deployment information
display_info() {
    log_success "Deployment completed successfully!"
    echo
    echo "=== Deployment Information ==="
    echo "Environment: $ENVIRONMENT"
    echo "API URL: https://localhost"
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3000"
    echo
    echo "=== Service Status ==="
    docker-compose -f "$DEPLOYMENT_DIR/docker-compose.$ENVIRONMENT.yml" ps
    echo
    echo "=== Next Steps ==="
    echo "1. Update DNS records to point to this server"
    echo "2. Replace self-signed certificates with proper SSL certificates"
    echo "3. Configure monitoring alerts"
    echo "4. Set up log rotation"
    echo "5. Test backup and recovery procedures"
    echo
    echo "=== Useful Commands ==="
    echo "View logs: docker-compose -f docker-compose.$ENVIRONMENT.yml logs -f"
    echo "Restart services: docker-compose -f docker-compose.$ENVIRONMENT.yml restart"
    echo "Update services: docker-compose -f docker-compose.$ENVIRONMENT.yml pull && docker-compose -f docker-compose.$ENVIRONMENT.yml up -d"
}

# Cleanup on failure
cleanup_on_failure() {
    log_error "Deployment failed. Cleaning up..."
    docker-compose -f "$DEPLOYMENT_DIR/docker-compose.$ENVIRONMENT.yml" down 2>/dev/null || true
}

# Main deployment function
main() {
    log_info "Starting Photonic Foundry deployment..."
    
    # Set trap for cleanup on failure
    trap cleanup_on_failure ERR
    
    check_root
    check_requirements
    prepare_environment
    check_ssl
    build_images
    setup_monitoring
    deploy_services
    
    if perform_health_checks; then
        setup_backups
        display_info
    else
        log_error "Health checks failed. Please check the logs."
        exit 1
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi