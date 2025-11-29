# TaskFlow AI Dockerfile
# Multi-stage build for optimized production deployment

# Stage 1: Builder stage for dependencies
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/* && \
    # Create non-root user
    useradd --create-home --shell /bin/bash app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose ports for API and dashboard
EXPOSE 5000 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Default command - run both API server and dashboard
CMD ["python", "-c", "
import subprocess
import time
import sys

print('🚀 Starting TaskFlow AI Container')
print('📊 Services: API Server (port 5000) + Dashboard (port 5001)')

# Start API server in background
api_process = subprocess.Popen([
    sys.executable, 'api.py',
    '--host', '0.0.0.0',
    '--port', '5000'
])

# Wait a moment for API to start
time.sleep(2)

# Start dashboard server
dashboard_process = subprocess.Popen([
    sys.executable, 'dashboard.py',
    '--host', '0.0.0.0',
    '--port', '5001'
])

print('✅ Services started successfully!')
print('🌐 API Server: http://localhost:5000')
print('📊 Dashboard: http://localhost:5001')

try:
    # Wait for either process to exit
    api_process.wait()
    dashboard_process.wait()
except KeyboardInterrupt:
    print('🛑 Shutting down services...')
    api_process.terminate()
    dashboard_process.terminate()
    api_process.wait()
    dashboard_process.wait()
"]