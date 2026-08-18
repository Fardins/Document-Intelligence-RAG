"""
vector_store.py
---------------

ChromaDB vector storage + semantic retrieval + lightweight reranking.

Embedding model:
    sentence-transformers/all-MiniLM-L6-v2

Vector database:
    ChromaDB
"""

from __future__ import annotations

import re
import threading

from dataclasses import dataclass
from typing import Dict, List

import chromadb

from . import config
from .document_processor import (
    Chunk,
    ProcessedDocument,
)
from .embeddings import (
    embed_texts,
    embed_query,
)


_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

def _get_chroma_client():

    return chromadb.PersistentClient(
        path=config.CHROMA_PATH
    )


def _get_collection():

    client = _get_chroma_client()

    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={
            "description": (
                "Document Intelligence RAG chunks"
            ),
            "hnsw:space": "cosine",
        },
    )


_collection = _get_collection()


# ---------------------------------------------------------------------------
# Document index
# ---------------------------------------------------------------------------

class DocumentIndex:

    def __init__(
        self,
        doc_id: str,
        filename: str,
        num_pages: int,
    ):

        self.doc_id = doc_id
        self.filename = filename
        self.num_pages = num_pages

    # -----------------------------------------------------------------------
    # Index chunks
    # -----------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[Chunk],
    ) -> None:

        if not chunks:
            return

        texts = [
            chunk.text
            for chunk in chunks
        ]

        vectors = embed_texts(
            texts
        )

        ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        metadatas = [

            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "filename": self.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }

            for chunk in chunks
        ]

        _collection.add(
            ids=ids,
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int,
    ) -> List[dict]:

        if not query.strip():
            return []

        # ---------------------------------------------------------------
        # Retrieve more candidates than we finally return.
        # ---------------------------------------------------------------

        candidate_k = min(
            max(
                top_k
                * config.RETRIEVAL_CANDIDATE_MULTIPLIER,
                top_k,
            ),
            config.MAX_RETRIEVAL_CANDIDATES,
        )

        query_vector = embed_query(
            query
        )

        results = _collection.query(
            query_embeddings=[
                query_vector.tolist()
            ],
            n_results=candidate_k,
            where={
                "doc_id": self.doc_id
            },
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = (
            results.get("ids", [[]])[0]
        )

        if not ids:
            return []

        documents = (
            results.get(
                "documents",
                [[]],
            )[0]
        )

        metadatas = (
            results.get(
                "metadatas",
                [[]],
            )[0]
        )

        distances = (
            results.get(
                "distances",
                [[]],
            )[0]
        )

        candidates = []

        for i in range(len(ids)):

            metadata = metadatas[i]

            chunk = Chunk(
                chunk_id=metadata["chunk_id"],
                doc_id=metadata["doc_id"],
                text=documents[i],
                page_number=int(
                    metadata.get(
                        "page_number",
                        0,
                    )
                ),
                chunk_index=int(
                    metadata.get(
                        "chunk_index",
                        0,
                    )
                ),
                char_start=int(
                    metadata.get(
                        "char_start",
                        0,
                    )
                ),
                char_end=int(
                    metadata.get(
                        "char_end",
                        0,
                    )
                ),
            )

            cosine_distance = float(
                distances[i]
            )

            semantic_score = (
                1.0 - cosine_distance
            )

            candidates.append(
                {
                    "chunk": chunk,
                    "score": semantic_score,
                    "semantic_score": semantic_score,
                }
            )

        # ---------------------------------------------------------------
        # Lexical reranking.
        # ---------------------------------------------------------------

        for candidate in candidates:

            lexical_score = _lexical_score(
                query,
                candidate["chunk"].text,
            )

            candidate["lexical_score"] = (
                lexical_score
            )

            # Normalize semantic score into [0, 1].
            semantic = max(
                0.0,
                min(
                    1.0,
                    (
                        candidate[
                            "semantic_score"
                        ]
                        + 1.0
                    )
                    / 2.0,
                ),
            )

            candidate["rerank_score"] = (
                config.SEMANTIC_WEIGHT
                * semantic
                +
                config.LEXICAL_WEIGHT
                * lexical_score
            )

        candidates.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        # ---------------------------------------------------------------
        # Diversity filtering.
        #
        # Overlapping chunks can be almost identical. Do not send
        # four copies of the same evidence to the LLM.
        # ---------------------------------------------------------------

        selected = []

        for candidate in candidates:

            if len(selected) >= top_k:
                break

            is_duplicate = False

            for existing in selected:

                similarity = _text_similarity(
                    candidate["chunk"].text,
                    existing["chunk"].text,
                )

                if (
                    similarity
                    >= config.DUPLICATE_SIMILARITY_THRESHOLD
                ):
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            candidate["score"] = (
                candidate["rerank_score"]
            )

            selected.append(
                candidate
            )

        return selected


# ---------------------------------------------------------------------------
# Lexical scoring
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:

    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE,
    )


def _lexical_score(
    query: str,
    document: str,
) -> float:
    """
    Lightweight lexical relevance.

    This is NOT a replacement for semantic search.
    It provides additional signal for important query terms.
    """

    query_tokens = set(
        _tokenize(query)
    )

    document_tokens = set(
        _tokenize(document)
    )

    if not query_tokens:
        return 0.0

    matched = (
        query_tokens
        & document_tokens
    )

    coverage = (
        len(matched)
        / len(query_tokens)
    )

    # Phrase-level bonus.
    query_normalized = " ".join(
        _tokenize(query)
    )

    document_normalized = " ".join(
        _tokenize(document)
    )

    phrase_bonus = (
        0.15
        if (
            query_normalized
            and query_normalized
            in document_normalized
        )
        else 0.0
    )

    return min(
        coverage + phrase_bonus,
        1.0,
    )


# ---------------------------------------------------------------------------
# Text similarity
# ---------------------------------------------------------------------------

def _text_similarity(
    text_a: str,
    text_b: str,
) -> float:

    tokens_a = set(
        _tokenize(text_a)
    )

    tokens_b = set(
        _tokenize(text_b)
    )

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = (
        tokens_a & tokens_b
    )

    union = (
        tokens_a | tokens_b
    )

    return (
        len(intersection)
        / len(union)
    )


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class VectorStore:

    def __init__(self):

        self._docs: Dict[
            str,
            DocumentIndex,
        ] = {}

        self._load_existing()

    # -----------------------------------------------------------------------
    # Load existing documents
    # -----------------------------------------------------------------------

    def _load_existing(self):

        try:

            result = _collection.get(
                include=["metadatas"]
            )

            metadatas = (
                result.get(
                    "metadatas",
                    [],
                )
                or []
            )

            for metadata in metadatas:

                if not metadata:
                    continue

                doc_id = metadata.get(
                    "doc_id"
                )

                if not doc_id:
                    continue

                if doc_id not in self._docs:

                    self._docs[doc_id] = (
                        DocumentIndex(
                            doc_id=doc_id,
                            filename=metadata.get(
                                "filename",
                                "Unknown",
                            ),
                            num_pages=0,
                        )
                    )

        except Exception as exc:

            print(
                "Warning: Could not load "
                f"documents from ChromaDB: {exc}"
            )

    # -----------------------------------------------------------------------
    # Index document
    # -----------------------------------------------------------------------

    def index_document(
        self,
        processed: ProcessedDocument,
    ) -> None:

        if not processed.chunks:
            return

        with _LOCK:

            # Remove existing chunks for same doc ID.
            try:

                _collection.delete(
                    where={
                        "doc_id": processed.doc_id
                    }
                )

            except Exception:
                pass

            doc_index = DocumentIndex(
                doc_id=processed.doc_id,
                filename=processed.filename,
                num_pages=processed.num_pages,
            )

            doc_index.add_chunks(
                processed.chunks
            )

            self._docs[
                processed.doc_id
            ] = doc_index

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    def search(
        self,
        doc_id: str,
        query: str,
        top_k: int = config.TOP_K,
    ) -> List[dict]:

        doc_index = self._docs.get(
            doc_id
        )

        if doc_index is None:

            raise KeyError(
                f"Unknown doc_id: {doc_id}"
            )

        return doc_index.search(
            query,
            top_k,
        )

    # -----------------------------------------------------------------------
    # List documents
    # -----------------------------------------------------------------------

    def list_documents(self) -> List[dict]:

        documents = []

        for doc in self._docs.values():

            try:

                result = _collection.get(
                    where={
                        "doc_id": doc.doc_id
                    },
                    include=[
                        "metadatas"
                    ],
                )

                metadatas = (
                    result.get(
                        "metadatas",
                        [],
                    )
                    or []
                )

                ids = (
                    result.get(
                        "ids",
                        [],
                    )
                    or []
                )

                pages = {
                    metadata.get(
                        "page_number"
                    )
                    for metadata in metadatas
                    if metadata
                    and metadata.get(
                        "page_number"
                    )
                }

                num_pages = (
                    max(pages)
                    if pages
                    else doc.num_pages
                )

                num_chunks = len(ids)

            except Exception:

                num_chunks = 0
                num_pages = doc.num_pages

            documents.append(
                {
                    "doc_id": doc.doc_id,
                    "filename": doc.filename,
                    "num_pages": num_pages,
                    "num_chunks": num_chunks,
                }
            )

        return documents

    # -----------------------------------------------------------------------
    # Get document
    # -----------------------------------------------------------------------

    def get_document(
        self,
        doc_id: str,
    ) -> DocumentIndex | None:

        return self._docs.get(
            doc_id
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

vector_store = VectorStore()