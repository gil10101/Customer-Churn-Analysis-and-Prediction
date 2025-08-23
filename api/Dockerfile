# Multi-stage Docker build for Customer Churn Prediction API
# Optimized for production deployment with security and performance best practices

# Build stage - Install dependencies and compile packages
FROM python:3.9-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add metadata labels
LABEL maintainer="ML Engineering Team" \
      org.opencontainers.image.title="Churn Prediction API Builder" \
      org.opencontainers.image.description="Build stage for Customer Churn Prediction API" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}"

# Set working directory
WORKDIR /build

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Compile Python files to bytecode for faster startup
RUN python -m compileall /root/.local/lib/python3.9/site-packages/

# Production stage - Minimal runtime image
FROM python:3.9-slim as production

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add metadata labels
LABEL maintainer="ML Engineering Team" \
      org.opencontainers.image.title="Churn Prediction API" \
      org.opencontainers.image.description="Production Customer Churn Prediction API" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/company/churn-prediction" \
      org.opencontainers.image.documentation="https://github.com/company/churn-prediction/blob/main/api/README.md"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with specific UID/GID for security
RUN groupadd -r -g 1001 appuser && \
    useradd -r -u 1001 -g appuser -d /app -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Create necessary directories with proper permissions
RUN mkdir -p models logs data cache && \
    chown -R appuser:appuser /app

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Set environment variables for production
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WORKERS=1 \
    MAX_WORKERS=4 \
    TIMEOUT=30 \
    KEEP_ALIVE=2 \
    LOG_LEVEL=info \
    ACCESS_LOG=true

# Expose port
EXPOSE 8000

# Add health check with improved configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/model/health || exit 1

# Add signal handling for graceful shutdown
STOPSIGNAL SIGTERM

# Default command with production settings
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "2", \
     "--access-log", \
     "--no-server-header"]