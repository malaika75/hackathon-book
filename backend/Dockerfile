# ============================================
# Dockerfile for Hugging Face Spaces
# Optimized for Python 3.11 FastAPI
# ============================================

# -------- Base Stage --------
FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some Python packages + healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------- Requirements Stage (cache optimization) --------
# Copy ONLY requirements.txt first for layer caching
COPY requirements.txt .

# Install Python dependencies with no cache
RUN pip install --no-cache-dir --break-system-packages \
    -r requirements.txt

# -------- Application Stage --------
# Copy remaining application code
COPY . .

# Create app user for security (run as non-root)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (HF Spaces: 7860, Default app: 8000)
EXPOSE 8000

# Health check (app uses port 8000)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
# Note: Module path is "backend.src.main:app" (src/main.py)
# Note: For HF Spaces, change port to 7860
CMD ["uvicorn", "backend.src.main:app", "--host", "0.0.0.0", "--port", "8000"]