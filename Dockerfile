# =============================================================================
# PETR4 Price Predictor — Dockerfile
# Multi-stage build: keeps the final image lean by separating build from runtime.
# =============================================================================

# --- Stage 1: build dependencies --------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools required by some packages (e.g. prophet, shap)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements-api.txt .
RUN pip install --no-cache-dir --user -r requirements-api.txt


# --- Stage 2: runtime -------------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home appuser
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application source
COPY api/        ./api/
COPY src/        ./src/
COPY config.yaml ./config.yaml

# Models and artifacts are mounted at runtime (via docker-compose volume)
RUN mkdir -p models/trained models/artifacts logs

# Make Python packages from builder available
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Health-check so orchestrators know when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
