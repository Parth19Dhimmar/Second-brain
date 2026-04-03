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
# FROM python:3.11-slim AS builder

# # # Set working directory
# WORKDIR /app

# # Install uv for faster dependency management
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# # Set environment variables for build
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     UV_COMPILE_BYTECODE=1 \
#     UV_LINK_MODE=copy \
#     UV_HTTP_TIMEOUT=300

# # Copy dependency files
# COPY pyproject.toml README.md uv.lock ./

# # Create virtual environment and install dependencies
# RUN uv sync --frozen --no-dev

# # Copy the entire project
# COPY . .

# # Install the package in the virtual environment
# RUN uv pip install --no-deps -e .

# # Remove unnecessary files to reduce size
# RUN find /app/.venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
#     find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
#     find /app/.venv -name "*.pyc" -delete 2>/dev/null || true && \
#     find /app/.venv -name "*.pyo" -delete 2>/dev/null || true && \
#     find /app/.venv -type d -name "*.dist-info" -exec sh -c 'rm -rf "$1"/RECORD "$1"/WHEEL "$1"/INSTALLER' _ {} \; 2>/dev/null || true

# # Stage 2: Runtime - Minimal image
# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Install only essential runtime libraries
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PATH="/app/.venv/bin:$PATH" \
#     PYTHONPATH="/app:$PYTHONPATH"

# # Copy only the virtual environment and application code from builder
# COPY --from=builder /app/.venv /app/.venv
# COPY --from=builder /app/src /app/src
# COPY --from=builder /app/configs /app/configs
# COPY --from=builder /app/tools /app/tools
# COPY --from=builder /app/api.py /app/api.py
# COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# # Create non-root user for security
# RUN useradd -m -u 1000 appuser && \
#     chown -R appuser:appuser /app

# # Switch to non-root user
# USER appuser

# # Expose FastAPI port
# EXPOSE 8000

# # Run the application
# CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


# Dockerfile (in your backend root) - working 6 GB +

# FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# WORKDIR /app

# # Install system dependencies for Python packages
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Copy dependency files
# COPY pyproject.toml uv.lock* ./

# # Install dependencies with uv
# RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application code
# COPY . .

# # Expose port
# EXPOSE 8000

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# # Run the application
# CMD ["uvicorn", "second_brain_online.api:app", "--host", "0.0.0.0", "--port", "8000"]




# Dockerfile - Optimized Two-Stage Build
# Dockerfile - FINAL WORKING & OPTIMIZED

# ==================== Builder Stage ====================
# FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# ENV UV_NO_CACHE=1 \
#     PYTHONDONTWRITEBYTECODE=1

# WORKDIR /build

# # Copy lock files
# COPY pyproject.toml uv.lock ./
# RUN touch README.md

# # Export WITHOUT hashes (--no-hashes) to avoid hash mismatch
# RUN uv export --frozen --no-dev --no-hashes --no-emit-project > requirements-all.txt

# # Filter out NVIDIA/GPU packages and torch (we'll install CPU version)
# RUN grep -v "nvidia" requirements-all.txt | \
#     grep -v "triton" | \
#     grep -v "^torch==" | \
#     grep -v "^torchvision==" > requirements.txt

# # ==================== Runtime Stage ====================
# FROM python:3.11-slim-bookworm AS runtime

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PYTHONPATH="/app"

# WORKDIR /app

# # Install minimal system deps
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends curl ca-certificates && \
#     rm -rf /var/lib/apt/lists/*

# # Copy filtered requirements
# COPY --from=builder /build/requirements.txt .

# # Install packages in correct order
# RUN pip install --no-cache-dir \
#         torch==2.5.1+cpu \
#         torchvision==0.20.1+cpu \
#         --index-url https://download.pytorch.org/whl/cpu && \
#     pip install --no-cache-dir --no-deps -r requirements.txt && \
#     pip install --no-cache-dir uvicorn[standard] && \
#     rm requirements.txt

# # Aggressive cleanup
# RUN find /usr/local/lib/python3.11/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
#     find /usr/local/lib/python3.11/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
#     find /usr/local/lib/python3.11/site-packages -type f -name "*.pyc" -delete && \
#     find /usr/local/lib/python3.11/site-packages -type f -name "*.pyo" -delete

# # Copy application
# COPY api.py exceptions.py response.py ./
# COPY src/second_brain_online ./second_brain_online
# COPY configs ./configs
# COPY tools ./tools

# # Non-root user
# RUN useradd -m -u 1000 appuser && \
#     chown -R appuser:appuser /app

# USER appuser

# EXPOSE 8000

# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

####### new working compose 2-GB

# ==================== Builder Stage ====================
FROM ghcr.io/astral-sh/uv:python3.11-slim AS builder

ENV UV_NO_CACHE=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY pyproject.toml uv.lock ./
RUN touch README.md

# Export without hashes
RUN uv export --frozen --no-dev --no-hashes --no-emit-project > requirements-all.txt

# Strip ALL nvidia/cuda/gpu packages aggressively.
# These are pulled as transitive deps of torch even on CPU installs.
# Each nvidia-* package is 200-800MB — they are the reason for 12GB image.
# We install CPU-only torch explicitly in the next stage, so none of these are needed.
RUN grep -viE \
    "torch|torchvision|torchaudio|triton|nvidia|cuda|cudnn|nccl|gradio|boto3|botocore" \
    requirements-all.txt > requirements.txt

# Sanity check — print what torch-related lines remain (should be none)
RUN echo "=== Remaining torch/nvidia lines (should be empty) ===" && \
    grep -iE "torch|nvidia|cuda|triton" requirements.txt || echo "CLEAN - no gpu packages"

# ==================== Runtime Stage ====================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    # Tell HuggingFace to cache models in a writable directory inside /app
    # Without this models download to ~/.cache which may not persist
    HF_HOME="/app/.cache/huggingface" \
    TRANSFORMERS_CACHE="/app/.cache/huggingface"

WORKDIR /app

# curl       -> HEALTHCHECK
# libgomp1   -> Flashrank + torch OpenMP (missing this causes silent crash)
# ca-certificates -> HTTPS downloads for model weights
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/requirements.txt .

# WHY THIS ORDER MATTERS:
# 1. torch CPU first with its own index — if torch appears in requirements.txt
#    pip would try to install CUDA version from PyPI. Installing CPU version
#    first with --no-deps prevents that.
# 2. Project deps with --no-deps — avoids pip pulling in nvidia packages
#    as "missing" deps of torch since we already have the CPU version.
# 3. slowapi + tenacity — rate limiting + retries (not in pyproject yet)
# 4. uvicorn last — runtime server
RUN pip install --no-cache-dir \
    onnxruntime \
    optimum \
    && pip install --no-cache-dir --no-deps -r requirements.txt \
    && pip install --no-cache-dir slowapi tenacity "uvicorn[standard]" \
    && rm -rf /root/.cache/pip \
    && rm requirements.txt

# Verify no CUDA packages snuck in — build fails if they did
RUN pip list | grep -iE "nvidia|cuda|triton" && \
    echo "ERROR: GPU packages found in image!" && exit 1 || \
    echo "VERIFIED: No GPU packages in image"

# Cleanup — removes ~300MB of test files, pyc, dist-info metadata
RUN find /usr/local/lib/python3.11/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -type f -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11/site-packages -type f -name "*.pyo" -delete && \
    find /usr/local/lib/python3.11/site-packages -type d -name "*.dist-info" \
        -exec sh -c 'rm -rf "$1/RECORD" "$1/WHEEL" "$1/INSTALLER"' _ {} \; 2>/dev/null || true

# Copy application
# Structure: PYTHONPATH="/app" so second_brain_online must be at /app/second_brain_online
# COPY api.py exceptions.py response.py ./
COPY api.py ./
COPY src/second_brain_online ./second_brain_online
COPY configs ./configs
COPY tools ./tools

# Non-root user + fix HuggingFace cache permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/.cache/huggingface && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# start-period=120s — gte-large (670MB) downloads from HuggingFace Hub on first start
# This is expected and only happens once (cached in HF_HOME after that)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]