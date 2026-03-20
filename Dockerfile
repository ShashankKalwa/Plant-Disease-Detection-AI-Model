# ==============================================================================
# Dockerfile — Multi-stage build for PlantDoctor AI API
# ==============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Create non-root user
RUN groupadd -r plantdoctor && useradd -r -g plantdoctor -d /app plantdoctor

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY config.py dataset.py model.py train.py evaluate.py predict.py utils.py main.py ./
COPY api/ ./api/
COPY export/ ./export/
COPY scripts/ ./scripts/
COPY plant_disease_training_dataset_optionB.csv ./

# Create directories for mounted volumes
RUN mkdir -p /app/data /app/outputs/models /app/outputs/plots && \
    chown -R plantdoctor:plantdoctor /app

# Switch to non-root user
USER plantdoctor

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command: start the API server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
