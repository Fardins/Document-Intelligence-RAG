FROM python:3.11-slim

WORKDIR /app

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
# build-essential is useful for Python packages that may need
# compilation during installation.
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# ------------------------------------------------------------
# Application
# ------------------------------------------------------------

COPY app ./app


# ------------------------------------------------------------
# ChromaDB storage
# ------------------------------------------------------------

RUN mkdir -p /app/chroma_store


# ------------------------------------------------------------
# Hugging Face embedding model
# ------------------------------------------------------------
# The model is downloaded during image build so the first
# API request does not need to download it.
#
# This model runs locally inside the container.
# ------------------------------------------------------------

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

ENV PYTHONUNBUFFERED=1

ENV CHROMA_PATH=/app/chroma_store


# ------------------------------------------------------------
# API port
# ------------------------------------------------------------

EXPOSE 8000


# ------------------------------------------------------------
# Start FastAPI
# ------------------------------------------------------------

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]