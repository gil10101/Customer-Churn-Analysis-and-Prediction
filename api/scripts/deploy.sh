#!/bin/bash

# Deployment Script for Churn Prediction API
# Supports multiple environments: development, staging, production

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
API_DIR="${PROJECT_ROOT}/api"

# Default values
ENVIRONMENT="development"
BUILD_ARGS=""
COMPOSE_FILE="docker-compose.yml"
HEALTH_CHECK_TIMEOUT=300
VALIDATION_TIMEOUT=60

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
Deployment Script for Churn Prediction API

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV    Target environment (development|staging|production) [default: development]
    -f, --compose-file FILE  Docker compose file to use [default: docker-compose.yml]
    -b, --build-args ARGS    Additional build arguments
    -t, --timeout SECONDS    Health check timeout in seconds [default: 300]
    -v, --validate           Run deployment validation after deployment
    -c, --cleanup            Cleanup old containers and images
    -h, --help              Show this help message

Examples:
    $0 --environment development
    $0 --environment staging --validate
    $0 --environment production --cleanup --validate
    $0 --compose-file docker-compose.prod.yml --build-args "VERSION=1.2.3"

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -f|--compose-file)
                COMPOSE_FILE="$2"
                shift 2
                ;;
            -b|--build-args)
                BUILD_ARGS="$2"
                shift 2
                ;;
            -t|--timeout)
                HEALTH_CHECK_TIMEOUT="$2"
                shift 2
                ;;
            -v|--validate)
                RUN_VALIDATION=true
                shift
                ;;
            -c|--cleanup)
                RUN_CLEANUP=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        development|staging|production)
            log_info "Deploying to environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed"
        exit 1
    fi
    
    # Check if compose file exists
    if [[ ! -f "${API_DIR}/${COMPOSE_FILE}" ]]; then
        log_error "Compose file not found: ${API_DIR}/${COMPOSE_FILE}"
        exit 1
    fi
    
    # Check if models directory exists
    if [[ ! -d "${PROJECT_ROOT}/models" ]]; then
        log_warning "Models directory not found. Creating empty directory..."
        mkdir -p "${PROJECT_ROOT}/models"
    fi
    
    log_success "Prerequisites check passed"
}

# Set environment variables
set_environment_variables() {
    log_info "Setting environment variables for $ENVIRONMENT..."
    
    # Common variables
    export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    export VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    
    case $ENVIRONMENT in
        development)
            export VERSION="dev-${VCS_REF}"
            export LOG_LEVEL="debug"
            export API_PORT="8001"
            export WORKERS="1"
            ;;
        staging)
            export VERSION="staging-${VCS_REF}"
            export LOG_LEVEL="info"
            export API_PORT="8000"
            export WORKERS="2"
            ;;
        production)
            export VERSION="${VERSION:-prod-${VCS_REF}}"
            export LOG_LEVEL="warning"
            export API_PORT="8000"
            export WORKERS="4"
            export MAX_WORKERS="8"
            ;;
    esac
    
    log_success "Environment variables set"
}

# Cleanup old containers and images
cleanup_old_resources() {
    if [[ "${RUN_CLEANUP:-false}" == "true" ]]; then
        log_info "Cleaning up old containers and images..."
        
        cd "${API_DIR}"
        
        # Stop and remove containers
        docker-compose -f "${COMPOSE_FILE}" down --remove-orphans || true
        
        # Remove unused images
        docker image prune -f || true
        
        # Remove unused volumes (be careful in production)
        if [[ "$ENVIRONMENT" != "production" ]]; then
            docker volume prune -f || true
        fi
        
        log_success "Cleanup completed"
    fi
}

# Build and deploy services
deploy_services() {
    log_info "Building and deploying services..."
    
    cd "${API_DIR}"
    
    # Build services
    log_info "Building Docker images..."
    if [[ -n "$BUILD_ARGS" ]]; then
        docker-compose -f "${COMPOSE_FILE}" build --build-arg $BUILD_ARGS
    else
        docker-compose -f "${COMPOSE_FILE}" build
    fi
    
    # Deploy based on environment
    case $ENVIRONMENT in
        development)
            log_info "Starting development services..."
            docker-compose -f "${COMPOSE_FILE}" --profile development up -d
            ;;
        staging)
            log_info "Starting staging services..."
            docker-compose -f "${COMPOSE_FILE}" --profile api --profile cache up -d
            ;;
        production)
            log_info "Starting production services..."
            docker-compose -f "${COMPOSE_FILE}" --profile production --profile full up -d
            ;;
    esac
    
    log_success "Services deployed"
}

# Wait for services to be healthy
wait_for_health() {
    log_info "Waiting for services to be healthy..."
    
    local timeout=$HEALTH_CHECK_TIMEOUT
    local elapsed=0
    local interval=5
    
    while [[ $elapsed -lt $timeout ]]; do
        if check_service_health; then
            log_success "All services are healthy"
            return 0
        fi
        
        log_info "Waiting for services... (${elapsed}s/${timeout}s)"
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    
    log_error "Services did not become healthy within ${timeout} seconds"
    show_service_logs
    return 1
}

# Check service health
check_service_health() {
    cd "${API_DIR}"
    
    # Get API port based on environment
    local api_port
    case $ENVIRONMENT in
        development)
            api_port="8001"
            ;;
        *)
            api_port="8000"
            ;;
    esac
    
    # Check API health
    if curl -f -s "http://localhost:${api_port}/model/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Show service logs for debugging
show_service_logs() {
    log_info "Showing service logs for debugging..."
    cd "${API_DIR}"
    docker-compose -f "${COMPOSE_FILE}" logs --tail=50
}

# Run deployment validation
run_validation() {
    if [[ "${RUN_VALIDATION:-false}" == "true" ]]; then
        log_info "Running deployment validation..."
        
        # Get API URL based on environment
        local api_url
        case $ENVIRONMENT in
            development)
                api_url="http://localhost:8001"
                ;;
            *)
                api_url="http://localhost:8000"
                ;;
        esac
        
        # Run validation script
        if [[ -f "${API_DIR}/scripts/validate-deployment.py" ]]; then
            python3 "${API_DIR}/scripts/validate-deployment.py" \
                --url "$api_url" \
                --timeout $VALIDATION_TIMEOUT \
                --output "${API_DIR}/validation-results.json" \
                --verbose
            
            if [[ $? -eq 0 ]]; then
                log_success "Deployment validation passed"
            else
                log_error "Deployment validation failed"
                return 1
            fi
        else
            log_warning "Validation script not found, skipping validation"
        fi
    fi
}

# Show deployment summary
show_deployment_summary() {
    log_info "Deployment Summary"
    echo "===================="
    echo "Environment: $ENVIRONMENT"
    echo "Version: ${VERSION:-unknown}"
    echo "Build Date: ${BUILD_DATE:-unknown}"
    echo "VCS Ref: ${VCS_REF:-unknown}"
    echo "Compose File: $COMPOSE_FILE"
    
    # Show running services
    cd "${API_DIR}"
    echo ""
    echo "Running Services:"
    docker-compose -f "${COMPOSE_FILE}" ps
    
    # Show API endpoints
    local api_port
    case $ENVIRONMENT in
        development)
            api_port="8001"
            ;;
        *)
            api_port="8000"
            ;;
    esac
    
    echo ""
    echo "API Endpoints:"
    echo "  Health Check: http://localhost:${api_port}/model/health"
    echo "  API Docs: http://localhost:${api_port}/docs"
    echo "  Prediction: http://localhost:${api_port}/predict"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo "  Nginx: http://localhost:80"
        echo "  Monitoring: http://localhost:3000 (Grafana)"
    fi
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    cd "${API_DIR}"
    docker-compose -f "${COMPOSE_FILE}" down
    
    # Restore previous version if available
    # This would typically involve pulling a previous image tag
    # or restoring from a backup
    
    log_info "Rollback completed"
}

# Signal handlers for graceful shutdown
cleanup_on_exit() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
        if [[ "${ROLLBACK_ON_FAILURE:-false}" == "true" ]]; then
            rollback_deployment
        fi
    fi
    exit $exit_code
}

# Main deployment function
main() {
    log_info "Starting deployment of Churn Prediction API"
    
    # Parse arguments
    parse_args "$@"
    
    # Validate environment
    validate_environment
    
    # Check prerequisites
    check_prerequisites
    
    # Set environment variables
    set_environment_variables
    
    # Cleanup if requested
    cleanup_old_resources
    
    # Deploy services
    deploy_services
    
    # Wait for services to be healthy
    if ! wait_for_health; then
        log_error "Deployment failed - services are not healthy"
        exit 1
    fi
    
    # Run validation if requested
    if ! run_validation; then
        log_error "Deployment validation failed"
        exit 1
    fi
    
    # Show deployment summary
    show_deployment_summary
    
    log_success "Deployment completed successfully!"
}

# Set up signal handlers
trap cleanup_on_exit EXIT INT TERM

# Run main function
main "$@"