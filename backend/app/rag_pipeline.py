"""
rag_pipeline.py
---------------

Conversation-aware RAG pipeline.

Question
   ↓
Conversation history
   ↓
Standalone question rewriting
   ↓
Embedding
   ↓
ChromaDB retrieval
   ↓
Semantic + lexical reranking
   ↓
Top relevant chunks
   ↓
Conversation history + document context
   ↓
Groq LLM
   ↓
Answer + sources
   ↓
Save conversation turn
"""

from __future__ import annotations

from typing import List, TypedDict

from . import config
from .conversation_memory import (
    conversation_memory,
)
from .llm import (
    generate_answer,
    rewrite_question,
)
from .vector_store import vector_store


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

class Source(TypedDict):

    page_number: int
    chunk_id: str
    chunk_index: int
    snippet: str
    relevance_score: float


class RagResult(TypedDict):

    answer: str
    sources: List[Source]
    conversation_id: str
    retrieval_query: str


# ---------------------------------------------------------------------------
# Snippet
# ---------------------------------------------------------------------------

def _make_snippet(
    text: str,
    max_len: int = 320,
) -> str:

    text = text.strip()

    if len(text) <= max_len:
        return text

    snippet = (
        text[:max_len]
        .rsplit(" ", 1)[0]
    )

    return snippet + "..."


# ---------------------------------------------------------------------------
# Main RAG pipeline
# ---------------------------------------------------------------------------

def answer_question(
    doc_id: str,
    question: str,
    top_k: int = config.TOP_K,
    conversation_id: str | None = None,
) -> RagResult:

    question = question.strip()

    if not question:

        return {
            "answer": (
                "Please provide a question."
            ),
            "sources": [],
            "conversation_id": (
                conversation_id or doc_id
            ),
            "retrieval_query": "",
        }

    # -----------------------------------------------------------------------
    # Conversation ID
    #
    # If the frontend does not provide one, use doc_id.
    #
    # This means the current frontend can gain basic memory without
    # immediately requiring frontend changes.
    # -----------------------------------------------------------------------

    active_conversation_id = (
        conversation_id.strip()
        if conversation_id
        and conversation_id.strip()
        else doc_id
    )

    # -----------------------------------------------------------------------
    # Get previous conversation
    # -----------------------------------------------------------------------

    conversation_history = (
        conversation_memory.format_history(
            active_conversation_id
        )
    )

    # -----------------------------------------------------------------------
    # Rewrite follow-up question
    #
    # Example:
    #
    # Previous:
    # "What is the research about?"
    #
    # Current:
    # "Who proposed this approach?"
    #
    # Becomes:
    #
    # "Who proposed the approach described in the research?"
    # -----------------------------------------------------------------------

    retrieval_query = rewrite_question(
        question=question,
        conversation_history=conversation_history,
    )

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    retrieved = vector_store.search(
        doc_id,
        retrieval_query,
        top_k=top_k,
    )

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------

    answer_text = generate_answer(
        question=question,
        chunks_with_scores=retrieved,
        conversation_history=conversation_history,
    )

    # -----------------------------------------------------------------------
    # Save conversation turn
    #
    # IMPORTANT:
    # Save the original user question rather than the rewritten
    # retrieval query.
    # -----------------------------------------------------------------------

    conversation_memory.add_turn(
        conversation_id=active_conversation_id,
        question=question,
        answer=answer_text,
    )

    # -----------------------------------------------------------------------
    # Sources
    # -----------------------------------------------------------------------

    sources: List[Source] = []

    for item in retrieved:

        chunk = item["chunk"]

        sources.append(
            {
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "snippet": _make_snippet(
                    chunk.text
                ),
                "relevance_score": round(
                    float(
                        item["score"]
                    ),
                    4,
                ),
            }
        )

    return {
        "answer": answer_text,
        "sources": sources,
        "conversation_id": active_conversation_id,
        "retrieval_query": retrieval_query,
    }