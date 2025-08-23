# Deployment Guide for Churn Prediction API

This guide covers the deployment infrastructure for the Customer Churn Prediction API, including Docker containerization, orchestration, monitoring, and CI/CD pipelines.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Deployment Environments](#deployment-environments)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [CI/CD Pipeline](#cicd-pipeline)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## Overview

The deployment infrastructure provides:

- **Multi-stage Docker builds** for optimized container images
- **Docker Compose** configurations for different environments
- **Nginx reverse proxy** with load balancing and SSL termination
- **Redis caching** for improved performance
- **Prometheus monitoring** with Grafana dashboards
- **Automated CI/CD pipeline** with GitHub Actions
- **Health checks and graceful shutdown** handling
- **Security hardening** and best practices

## Prerequisites

### Required Software

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.9+ (for validation scripts)
- Git (for version control)

### Optional Tools

- k6 (for load testing)
- curl (for API testing)
- jq (for JSON processing)

### System Requirements

#### Development
- CPU: 2 cores
- RAM: 4GB
- Disk: 10GB free space

#### Production
- CPU: 4+ cores
- RAM: 8GB+
- Disk: 50GB+ free space
- Network: Stable internet connection

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd churn-prediction

# Copy environment configuration
cp api/.env.example api/.env

# Edit configuration as needed
nano api/.env
```

### 2. Development Deployment

```bash
# Start development environment
cd api
docker-compose --profile development up --build

# API will be available at http://localhost:8001
```

### 3. Production Deployment

```bash
# Deploy to production
./api/scripts/deploy.sh --environment production --validate

# API will be available at http://localhost:80 (via Nginx)
```

## Deployment Environments

### Development

**Purpose**: Local development and testing

**Configuration**:
- Single API container with hot reload
- Debug logging enabled
- Port 8001 exposed
- Volume mounts for code changes

**Start Command**:
```bash
docker-compose --profile development up --build
```

### Staging

**Purpose**: Pre-production testing and validation

**Configuration**:
- API container with Redis caching
- Info-level logging
- Port 8000 exposed
- Resource limits applied

**Start Command**:
```bash
docker-compose --profile api --profile cache up --build
```

### Production

**Purpose**: Live production environment

**Configuration**:
- Full stack with Nginx, Redis, monitoring
- Warning-level logging
- SSL/TLS termination
- Health checks and auto-restart
- Resource limits and monitoring

**Start Command**:
```bash
docker-compose --profile production --profile full up --build
```

## Configuration

### Environment Variables

Key configuration options (see `.env.example` for complete list):

```bash
# Environment
ENVIRONMENT=production
VERSION=1.0.0

# API Settings
API_PORT=8000
WORKERS=4
LOG_LEVEL=info

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### Docker Compose Profiles

The system uses Docker Compose profiles for different deployment scenarios:

- `development`: Development with hot reload
- `api`: Core API service only
- `cache`: API with Redis caching
- `production`: Full production stack with Nginx
- `monitoring`: Prometheus and Grafana
- `full`: All services enabled

### Service Configuration

#### API Service
```yaml
churn-prediction-api:
  build: 
    context: ..
    dockerfile: api/Dockerfile
  ports:
    - "8000:8000"
  environment:
    - LOG_LEVEL=info
    - WORKERS=4
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/model/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

#### Nginx Proxy
```yaml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf:ro
  depends_on:
    churn-prediction-api:
      condition: service_healthy
```

## Monitoring

### Health Checks

The system includes comprehensive health monitoring:

#### Application Health
- **Endpoint**: `/model/health`
- **Checks**: Model loading, feature count, service status
- **Frequency**: Every 30 seconds
- **Timeout**: 10 seconds

#### Container Health
- Docker health checks for all services
- Automatic restart on failure
- Health status in Docker Compose

#### Infrastructure Health
- Nginx status endpoint
- Redis connectivity
- System resource monitoring

### Metrics and Monitoring

#### Prometheus Metrics

The API exposes metrics for monitoring:

```
# API Metrics
http_requests_total
http_request_duration_seconds
prediction_requests_total
prediction_errors_total
model_loaded

# System Metrics
process_cpu_seconds_total
process_memory_bytes
process_open_fds
```

#### Grafana Dashboards

Pre-configured dashboards for:
- API performance and errors
- Model prediction metrics
- System resource usage
- Business metrics (predictions per hour, error rates)

#### Alerting Rules

Prometheus alerts for:
- API downtime
- High error rates
- High latency
- Resource exhaustion
- Model loading failures

### Accessing Monitoring

```bash
# Prometheus
http://localhost:9090

# Grafana (admin/admin)
http://localhost:3000

# API Metrics
http://localhost:8000/metrics
```

## CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline includes:

1. **Code Quality**
   - Black formatting
   - Flake8 linting
   - MyPy type checking
   - Bandit security scanning

2. **Testing**
   - Unit tests with pytest
   - Integration tests
   - API endpoint testing
   - Load testing with k6

3. **Security**
   - Dependency vulnerability scanning
   - Container image scanning with Trivy
   - OWASP ZAP security testing

4. **Build and Deploy**
   - Multi-stage Docker builds
   - Container registry push
   - Automated deployment to staging
   - Production deployment on release

### Pipeline Triggers

- **Push to main**: Full pipeline with staging deployment
- **Push to develop**: Testing and staging deployment
- **Pull requests**: Code quality and testing only
- **Releases**: Full pipeline with production deployment

### Deployment Validation

Automated validation includes:
- Health check verification
- API endpoint testing
- Performance benchmarking
- Security scanning
- Smoke tests

## Security

### Container Security

- **Non-root user**: Containers run as `appuser` (UID 1001)
- **Minimal base image**: Python slim image
- **Read-only filesystems**: Where possible
- **Resource limits**: CPU and memory constraints
- **Security scanning**: Trivy vulnerability scanning

### Network Security

- **Reverse proxy**: Nginx handles external traffic
- **Rate limiting**: API rate limits configured
- **Security headers**: HSTS, CSP, X-Frame-Options
- **SSL/TLS**: HTTPS termination at Nginx
- **Network isolation**: Docker networks

### Application Security

- **Input validation**: Pydantic models
- **Error handling**: Structured error responses
- **Logging**: Security event logging
- **Authentication**: API key support (configurable)
- **CORS**: Configurable CORS policies

### Secrets Management

- Environment variables for configuration
- Docker secrets for sensitive data
- External secret management integration ready

## Troubleshooting

### Common Issues

#### API Not Starting

```bash
# Check container logs
docker-compose logs churn-prediction-api

# Check health status
curl http://localhost:8000/model/health

# Verify model files
ls -la models/
```

#### High Memory Usage

```bash
# Check container stats
docker stats

# Adjust worker count
export WORKERS=2
docker-compose up --build
```

#### Slow Response Times

```bash
# Check API metrics
curl http://localhost:8000/metrics

# Monitor with Grafana
# Check database connections
# Review model complexity
```

### Debugging Commands

```bash
# View all container logs
docker-compose logs -f

# Execute shell in container
docker-compose exec churn-prediction-api bash

# Check service health
docker-compose ps

# View resource usage
docker stats

# Test API endpoints
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_customer.json
```

### Log Analysis

```bash
# API logs
docker-compose logs churn-prediction-api | grep ERROR

# Nginx access logs
docker-compose logs nginx | grep "POST /predict"

# System logs
journalctl -u docker
```

## Maintenance

### Regular Tasks

#### Daily
- Monitor health dashboards
- Check error rates and alerts
- Review resource usage

#### Weekly
- Update dependencies
- Review security alerts
- Backup model files and configurations

#### Monthly
- Update base images
- Review and rotate secrets
- Performance optimization review

### Updates and Upgrades

#### Application Updates
```bash
# Pull latest code
git pull origin main

# Rebuild and deploy
./api/scripts/deploy.sh --environment production --validate
```

#### System Updates
```bash
# Update base images
docker-compose pull

# Rebuild containers
docker-compose build --no-cache

# Deploy with validation
./api/scripts/deploy.sh --environment production --validate
```

### Backup and Recovery

#### Model Backup
```bash
# Backup models directory
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Upload to cloud storage
aws s3 cp models-backup-*.tar.gz s3://backup-bucket/
```

#### Configuration Backup
```bash
# Backup configuration
tar -czf config-backup-$(date +%Y%m%d).tar.gz api/*.conf api/.env
```

#### Recovery Procedure
1. Stop services: `docker-compose down`
2. Restore model files from backup
3. Restore configuration files
4. Start services: `docker-compose up -d`
5. Validate deployment: `./api/scripts/validate-deployment.py`

### Performance Optimization

#### Scaling Up
```bash
# Increase workers
export WORKERS=8
export MAX_WORKERS=16

# Add resource limits
export CPU_LIMIT=4.0
export MEMORY_LIMIT=4G

# Redeploy
docker-compose up --build
```

#### Horizontal Scaling
- Use Docker Swarm or Kubernetes
- Load balancer configuration
- Shared storage for models
- Database clustering

## Support and Documentation

### Additional Resources

- [API Documentation](README.md)
- [Model Training Guide](../notebooks/README.md)
- [Development Setup](../README.md)

### Getting Help

1. Check this deployment guide
2. Review container logs
3. Check monitoring dashboards
4. Run validation scripts
5. Contact the development team

### Contributing

1. Follow the deployment testing procedures
2. Update documentation for changes
3. Test in staging before production
4. Follow security best practices