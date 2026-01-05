#### single stage

# Single-Stage Dockerfile
# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Install uv for faster dependency management
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     UV_SYSTEM_PYTHON=1 \
#     UV_HTTP_TIMEOUT=300

# # Copy dependency files first (for better caching)
# COPY pyproject.toml README.md uv.lock ./

# # Install dependencies using uv (no venv needed with UV_SYSTEM_PYTHON=1)
# RUN uv sync --frozen --no-dev

# # Copy the entire project
# COPY . .

# # Install the package in editable mode to make imports work
# RUN uv pip install -e .

# # Expose FastAPI port
# EXPOSE 8000

# # Run the application
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


#### 2 stage builder-runtime

# Multi-Stage Dockerfile - Optimized for minimal size
# Stage 1: Builder
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install uv for faster dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300

# Copy dependency files
COPY pyproject.toml README.md uv.lock ./

# Create virtual environment and install dependencies
RUN uv sync --frozen --no-dev

# Copy the entire project
COPY . .

# Install the package in the virtual environment
RUN uv pip install --no-deps -e .

# Remove unnecessary files to reduce size
RUN find /app/.venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -name "*.pyc" -delete 2>/dev/null || true && \
    find /app/.venv -name "*.pyo" -delete 2>/dev/null || true && \
    find /app/.venv -type d -name "*.dist-info" -exec sh -c 'rm -rf "$1"/RECORD "$1"/WHEEL "$1"/INSTALLER' _ {} \; 2>/dev/null || true

# Stage 2: Runtime - Minimal image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only essential runtime libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

# Copy only the virtual environment and application code from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs
COPY --from=builder /app/tools /app/tools
COPY --from=builder /app/api.py /app/api.py
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]