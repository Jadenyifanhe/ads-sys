# Multi-stage Docker build for Generative Ads System

# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM base as app

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Create directories for data and checkpoints
RUN mkdir -p /app/data /app/checkpoints /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV RQVAE_PATH=/app/checkpoints/rqvae/best_model.pt
ENV GENERATIVE_PATH=/app/checkpoints/generative/best_model.pt
ENV RANKING_PATH=/app/checkpoints/ranking/best_model.pt
ENV TRIE_PATH=/app/data/semantic_trie.pkl

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the ad server
CMD ["python", "src/serving/ad_server.py"]
