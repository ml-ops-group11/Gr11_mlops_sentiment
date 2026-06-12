# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim — chosen reasons:
#   • "slim" variant is ~45 MB vs ~900 MB for the full image → faster pull.
#   • Python 3.11 matches the version used in GitHub Actions workflows.
#   • Debian-based, so apt-get works for any OS-level deps if needed.
FROM python:3.11-slim

# ── Build argument — override at docker build time ────────────────────────────
# Usage: docker build --build-arg HF_MODEL_NAME=your-user/your-repo .
ARG HF_MODEL_NAME="rohit-2145/roberta-sst2-v4"
ENV HF_MODEL=${HF_MODEL_NAME}

# ── System dependencies ───────────────────────────────────────────────────────
# git is needed by huggingface_hub for some model downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Copy & install Python dependencies ───────────────────────────────────────
# Copy requirements first (Docker layer cache: only re-installs on changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY src/inference.py .

# ── Run inference ─────────────────────────────────────────────────────────────
# INPUT_TEXT and HF_TOKEN are passed via  docker run -e  at runtime
CMD ["python", "inference.py"]
