"""
main.py
-------

FastAPI entry point for the Document Intelligence RAG system.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from pydantic import (
    BaseModel,
    Field,
)

from . import config

from .conversation_memory import (
    conversation_memory,
)

from .document_processor import (
    UnsupportedFileType,
    process_document,
)

from .rag_pipeline import (
    RagResult,
    answer_question,
)

from .vector_store import (
    vector_store,
)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document Intelligence RAG API",
    description=(
        "PDF/TXT Document Question Answering "
        "using ChromaDB, Hugging Face embeddings, "
        "Groq and conversation memory."
    ),
    version="2.1.0",
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

cors_origins = (
    ["*"]
    if config.CORS_ORIGINS.strip() == "*"
    else [
        origin.strip()
        for origin in config.CORS_ORIGINS.split(",")
        if origin.strip()
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):

    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int


class DocumentSummary(BaseModel):

    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int


class AskRequest(BaseModel):

    doc_id: str

    question: str = Field(
        ...,
        min_length=1,
        description="Question about the document.",
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Number of final chunks to retrieve."
        ),
    )

    # -----------------------------------------------------------------------
    # NEW
    # -----------------------------------------------------------------------

    conversation_id: str | None = Field(
        default=None,
        description=(
            "Conversation identifier used to maintain "
            "follow-up question context."
        ),
    )


class SourceModel(BaseModel):

    page_number: int
    chunk_id: str
    chunk_index: int
    snippet: str
    relevance_score: float


class AskResponse(BaseModel):

    answer: str
    sources: list[SourceModel]

    # -----------------------------------------------------------------------
    # NEW
    # -----------------------------------------------------------------------

    conversation_id: str

    # Useful for debugging / frontend display.
    # This is the standalone query actually sent to retrieval.
    retrieval_query: str


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Document Intelligence RAG API",
        "message": "Backend is running.",
        "endpoints": [
            "/health",
            "/documents",
            "/documents/upload",
            "/questions/ask",
        ],
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "embedding_model": (
            config.EMBEDDING_MODEL_NAME
        ),
        "embedding_dimension": (
            config.EMBEDDING_DIM
        ),
        "vector_database": "ChromaDB",
        "llm_provider": "Groq",
        "llm_model": config.GROQ_MODEL,
        "conversation_memory": "enabled",
        "max_conversation_turns": (
            config.MAX_CONVERSATION_TURNS
        ),
    }


# ---------------------------------------------------------------------------# Upload helpers
# ---------------------------------------------------------------------------


def create_temp_upload_path(filename: str) -> Path:
    """Create a temp file in the OS temp directory, not Render's project disk."""

    suffix = Path(filename).suffix.lower()
    fd, temp_name = tempfile.mkstemp(
        prefix="doc_upload_",
        suffix=suffix,
        dir=tempfile.gettempdir(),
    )
    os.close(fd)
    return Path(temp_name)


# ---------------------------------------------------------------------------# Upload
# ---------------------------------------------------------------------------

@app.post(
    "/documents/upload",
    response_model=UploadResponse,
)
async def upload_document(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    suffix = Path(
        file.filename
    ).suffix.lower()

    if (
        suffix
        not in config.ALLOWED_EXTENSIONS
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type "
                f"'{suffix}'. "
                f"Allowed: "
                f"{sorted(config.ALLOWED_EXTENSIONS)}"
            ),
        )

    # -----------------------------------------------------------------------
    # Temporary upload
    # -----------------------------------------------------------------------

    tmp_path = create_temp_upload_path(file.filename)

    size = 0

    try:

        with tmp_path.open(
            "wb"
        ) as out_file:

            while True:

                data = await file.read(
                    1024 * 1024
                )

                if not data:
                    break

                size += len(data)

                if (
                    size
                    > config.MAX_FILE_SIZE_MB
                    * 1024
                    * 1024
                ):

                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "File exceeds maximum "
                            f"allowed size of "
                            f"{config.MAX_FILE_SIZE_MB}MB."
                        ),
                    )

                out_file.write(data)

        # -------------------------------------------------------------------
        # Process
        # -------------------------------------------------------------------

        try:

            processed = process_document(
                tmp_path,
                file.filename,
            )

        except UnsupportedFileType as exc:

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        except Exception as exc:

            raise HTTPException(
                status_code=422,
                detail=(
                    "Failed to process document: "
                    f"{exc}"
                ),
            )

        # -------------------------------------------------------------------
        # Validate
        # -------------------------------------------------------------------

        if not processed.chunks:

            raise HTTPException(
                status_code=422,
                detail=(
                    "No extractable text was found. "
                    "The PDF may be scanned/image-only."
                ),
            )

        # -------------------------------------------------------------------
        # Embed + ChromaDB
        # -------------------------------------------------------------------

        try:

            vector_store.index_document(
                processed
            )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to index document: "
                    f"{exc}"
                ),
            )

        return UploadResponse(
            doc_id=processed.doc_id,
            filename=processed.filename,
            num_pages=processed.num_pages,
            num_chunks=len(
                processed.chunks
            ),
        )

    finally:

        tmp_path.unlink(
            missing_ok=True
        )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@app.get(
    "/documents",
    response_model=list[
        DocumentSummary
    ],
)
def list_documents():

    return vector_store.list_documents()


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

@app.post(
    "/questions/ask",
    response_model=AskResponse,
)
def ask_question(
    payload: AskRequest,
):

    question = (
        payload.question.strip()
    )

    if not question:

        raise HTTPException(
            status_code=400,
            detail=(
                "Question must not be empty."
            ),
        )

    top_k = (
        payload.top_k
        if payload.top_k is not None
        else config.TOP_K
    )

    try:

        result: RagResult = (
            answer_question(
                doc_id=payload.doc_id,
                question=question,
                top_k=top_k,
                conversation_id=(
                    payload.conversation_id
                ),
            )
        )

    except KeyError:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown doc_id: "
                f"{payload.doc_id}"
            ),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate answer: "
                f"{exc}"
            ),
        )

    return result


# ---------------------------------------------------------------------------
# Clear conversation
# ---------------------------------------------------------------------------

@app.delete(
    "/conversations/{conversation_id}"
)
def clear_conversation(
    conversation_id: str,
):

    deleted = conversation_memory.clear(
        conversation_id
    )

    return {
        "conversation_id": conversation_id,
        "cleared": deleted,
    }
