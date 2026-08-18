"""
config.py
---------
Centralized configuration for the Document Intelligence RAG system.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Base directory / .env
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE_WORDS = int(
    os.getenv("CHUNK_SIZE_WORDS", "220")
)

CHUNK_OVERLAP_WORDS = int(
    os.getenv("CHUNK_OVERLAP_WORDS", "40")
)

MIN_CHUNK_WORDS = int(
    os.getenv("MIN_CHUNK_WORDS", "25")
)


# ---------------------------------------------------------------------------
# Hugging Face Embeddings
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# all-MiniLM-L6-v2 = 384 dimensions
EMBEDDING_DIM = 384

EMBEDDING_BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", "32")
)


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    str(STORAGE_DIR / "chroma_store"),
)

CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "book_chunks",
)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

TOP_K = int(
    os.getenv("TOP_K", "4")
)

RETRIEVAL_CANDIDATE_MULTIPLIER = int(
    os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "4")
)

MAX_RETRIEVAL_CANDIDATES = int(
    os.getenv("MAX_RETRIEVAL_CANDIDATES", "20")
)

SEMANTIC_WEIGHT = float(
    os.getenv("SEMANTIC_WEIGHT", "0.80")
)

LEXICAL_WEIGHT = float(
    os.getenv("LEXICAL_WEIGHT", "0.20")
)

DUPLICATE_SIMILARITY_THRESHOLD = float(
    os.getenv("DUPLICATE_SIMILARITY_THRESHOLD", "0.92")
)


# ---------------------------------------------------------------------------
# Conversation Memory
# ---------------------------------------------------------------------------

# Maximum number of previous user/assistant exchanges retained
# for a conversation.
MAX_CONVERSATION_TURNS = int(
    os.getenv("MAX_CONVERSATION_TURNS", "8")
)

# Maximum characters used when sending conversation history
# to the LLM.
MAX_CONVERSATION_HISTORY_CHARS = int(
    os.getenv(
        "MAX_CONVERSATION_HISTORY_CHARS",
        "10000",
    )
)


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "",
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "groq/compound",
)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_MB = int(
    os.getenv("MAX_FILE_SIZE_MB", "25")
)

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if not HF_TOKEN:
    print("Warning: HF_TOKEN is not set.")

if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY is not set.")