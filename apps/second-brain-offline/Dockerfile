# FROM python:3.12.8-slim

# # Install uv
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# WORKDIR /app

# # Copy everything
# COPY . .

# # Install dependencies AND install the current package in editable mode
# RUN uv sync --frozen --no-dev && uv pip install -e .

# # Run the pipeline
# CMD ["uv", "run", "python", "pipelines/compute_rag_vector_index.py"]


# 2. 

# FROM python:3.12.8-slim AS builder

# # Install uv
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# WORKDIR /app

# # Copy dependency files
# COPY pyproject.toml uv.lock README.md ./
# COPY src/ ./src/

# # Check if we should install CUDA or CPU version
# ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
# ARG TORCH_EXTRA=""

# # Install with optional CUDA
# RUN if [ "$TORCH_INDEX_URL" != "https://download.pytorch.org/whl/cpu" ]; then \
#         apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#         && rm -rf /var/lib/apt/lists/*; \
#     fi && \
#     echo "torch==2.9.1 --index-url $TORCH_INDEX_URL $TORCH_EXTRA" > constraints.txt && \
#     uv sync --frozen --no-dev --constraint constraints.txt && \
#     uv pip install -e . && \
#     uv cache clean

# # Runtime stage
# FROM python:3.12.8-slim

# WORKDIR /app

# # Copy from builder
# COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
# COPY --from=builder /usr/local/bin /usr/local/bin
# COPY --from=builder /app/src /app/src

# # Copy application code
# COPY pipelines/ ./pipelines/
# COPY steps/ ./steps/
# COPY configs/ ./configs/

# # Run the pipeline
# CMD ["python", "pipelines/compute_rag_vector_index.py"]

# 3. 
FROM python:3.12.8-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy everything
COPY . .

# Install CPU-only torch
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1

# Install your package
RUN uv pip install -e .

# Clean cache
RUN uv cache clean

CMD ["python", "pipelines/compute_rag_vector_index.py"]