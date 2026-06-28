# ──────────────────────────────────────────────────────────────
#  TrustFin Bank – Loan Default Risk System
#  Multi-stage Docker build
# ──────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="TrustFin ML Engineering"
LABEL description="Loan Default Risk Analysis System"

# libgomp1 is required at runtime by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/       ./src/
COPY app/       ./app/
COPY reports/   ./reports/
COPY models/    ./models/

# Create directories that may be absent in the repo
RUN mkdir -p data models reports mlruns

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Add src to Python path so Streamlit can import from it
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

EXPOSE 8501

# Default: run the Streamlit dashboard
# Override at `docker run` time to execute other commands:
#   docker run trustfin-risk python src/train.py
CMD ["streamlit", "run", "app/streamlit_app.py"]