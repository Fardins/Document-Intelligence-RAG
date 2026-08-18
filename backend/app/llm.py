"""
llm.py
------

Grounded document QA with conversation memory.

Responsibilities:
    1. Rewrite follow-up questions into standalone questions.
    2. Generate grounded answers.
    3. Use conversation history for contextual follow-ups.
    4. Never falsely attribute general knowledge to the document.
"""

from __future__ import annotations

from typing import List

from . import config


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an intelligent document question-answering assistant.

Your job is to answer the user's question accurately while checking
the provided document context and conversation history.

IMPORTANT RULES:

1. Answer the user's question directly and naturally.

2. Use the conversation history to understand references such as:
   - "this"
   - "that"
   - "the approach"
   - "the method"
   - "he"
   - "she"
   - "they"
   - "it"
   - "the previous study"

3. Use the retrieved document context as evidence.

4. You may use general knowledge when the document does not contain
   enough information.

5. NEVER claim that general knowledge came from the document.

6. After the answer, provide a short "Document Evidence" section.

7. The Document Evidence section must explain what the uploaded
   document actually supports.

8. If the document only mentions the topic but does not explain it,
   explicitly say so.

9. If the document provides a detailed explanation, use that
   information and state that it is supported by the document.

10. Never invent document evidence.

11. Do not discuss the retrieval process.

12. Do not include source numbers such as [Source 1].
    Sources are displayed separately by the application.

13. Keep answers concise and useful.

Response structure:

Answer:
<direct answer>

Document Evidence:
<what the uploaded document actually supports>
"""


# ---------------------------------------------------------------------------
# Conversation-aware question rewriting
# ---------------------------------------------------------------------------

def rewrite_question(
    question: str,
    conversation_history: str,
) -> str:

    question = question.strip()

    if not question:
        return question

    # No previous conversation means there is nothing to rewrite.
    if not conversation_history.strip():
        return question

    # Without an API key we use a lightweight fallback.
    if not config.GROQ_API_KEY:
        return _fallback_contextual_query(
            question,
            conversation_history,
        )

    from groq import Groq

    client = Groq(
        api_key=config.GROQ_API_KEY
    )

    prompt = f"""
You are a query rewriting component for a document RAG system.

Your job is NOT to answer the question.

Rewrite the user's latest question into a standalone search query
that contains all necessary context from the conversation.

Rules:

1. Resolve references such as:
   "this", "that", "it", "the approach", "the method", "he", "she",
   "they", etc.

2. Preserve the user's actual intent.

3. Do not add facts that are not present in the conversation.

4. Do not answer the question.

5. Return ONLY the rewritten standalone question.

Conversation history:

{conversation_history}

Latest user question:

{question}

Standalone retrieval question:
"""

    models_to_try = []

    configured_model = (
        config.GROQ_MODEL or ""
    ).strip()

    if configured_model:
        models_to_try.append(
            configured_model
        )

    for fallback_model in (
        "groq/compound",
        "openai/gpt-oss-20b",
    ):
        if fallback_model not in models_to_try:
            models_to_try.append(
                fallback_model
            )

    last_error = None

    for model_name in models_to_try:

        try:

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite follow-up questions "
                            "into standalone search queries."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
                max_tokens=200,
            )

            rewritten = (
                response
                .choices[0]
                .message
                .content
            )

            if rewritten:
                return rewritten.strip()

        except Exception as exc:

            last_error = exc

            print(
                f"Question rewrite model "
                f"'{model_name}' failed: {exc}"
            )

    print(
        f"Question rewriting failed: {last_error}"
    )

    return _fallback_contextual_query(
        question,
        conversation_history,
    )


# ---------------------------------------------------------------------------
# Fallback contextual query
# ---------------------------------------------------------------------------

def _fallback_contextual_query(
    question: str,
    conversation_history: str,
) -> str:

    if not conversation_history:
        return question

    # Use the most recent user question as retrieval context.
    lines = conversation_history.splitlines()

    previous_question = ""

    for line in reversed(lines):

        if line.startswith("User:"):

            previous_question = (
                line[len("User:"):].strip()
            )

            break

    if not previous_question:
        return question

    return (
        f"Previous question: "
        f"{previous_question}\n"
        f"Current question: "
        f"{question}"
    )


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------

def _build_context_block(
    chunks_with_scores: List[dict],
) -> str:

    blocks = []

    for rank, item in enumerate(
        chunks_with_scores,
        start=1,
    ):

        chunk = item["chunk"]

        page_label = (
            f"Page {chunk.page_number}"
            if chunk.page_number
            else "Document"
        )

        blocks.append(
            f"[Retrieved Source {rank}]\n"
            f"{page_label}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Relevance: {item['score']:.4f}\n"
            f"Content:\n{chunk.text}"
        )

    return "\n\n---\n\n".join(
        blocks
    )


# ---------------------------------------------------------------------------
# Main answer generation
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    chunks_with_scores: List[dict],
    conversation_history: str = "",
) -> str:

    if not chunks_with_scores:

        return (
            "Answer:\n"
            "I could not find relevant information in the "
            "provided document.\n\n"
            "Document Evidence:\n"
            "No relevant document evidence was retrieved."
        )

    context_block = _build_context_block(
        chunks_with_scores
    )

    if config.GROQ_API_KEY:

        try:

            return _generate_with_groq(
                question=question,
                context_block=context_block,
                conversation_history=conversation_history,
            )

        except Exception as exc:

            print(
                f"Groq API error: {exc}"
            )

            return _generate_extractive(
                chunks_with_scores,
                fallback_reason=(
                    "Groq API request failed"
                ),
            )

    return _generate_extractive(
        chunks_with_scores,
        fallback_reason=(
            "GROQ_API_KEY is not configured"
        ),
    )


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

def _generate_with_groq(
    question: str,
    context_block: str,
    conversation_history: str,
) -> str:

    from groq import Groq

    client = Groq(
        api_key=config.GROQ_API_KEY
    )

    history_block = (
        conversation_history
        if conversation_history.strip()
        else "No previous conversation."
    )

    user_prompt = f"""
Conversation history:

{history_block}

---

Latest user question:

{question}

---

Retrieved document context:

{context_block}

---

Answer the latest user question.

Use the conversation history only to understand context and
references.

Use the retrieved document context as document evidence.

If the answer is supported by the document, explain it naturally.

If the document does not contain enough information, you may use
general knowledge, but clearly distinguish that from document
evidence.

Do not include a Sources section.
The application will generate sources separately.
"""

    models_to_try = []

    configured_model = (
        config.GROQ_MODEL or ""
    ).strip()

    if configured_model:
        models_to_try.append(
            configured_model
        )

    for fallback_model in (
        "groq/compound",
        "openai/gpt-oss-20b",
    ):
        if fallback_model not in models_to_try:
            models_to_try.append(
                fallback_model
            )

    last_error = None

    for model_name in models_to_try:

        try:

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.1,
                max_tokens=600,
            )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

            if not answer:
                raise RuntimeError(
                    "Groq returned an empty response."
                )

            return answer.strip()

        except Exception as exc:

            last_error = exc

            print(
                f"Groq model "
                f"'{model_name}' failed: {exc}"
            )

    raise RuntimeError(
        "Groq request failed for all "
        "configured/fallback models. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Extractive fallback
# ---------------------------------------------------------------------------

def _generate_extractive(
    chunks_with_scores: List[dict],
    fallback_reason: str,
) -> str:

    top = chunks_with_scores[0]["chunk"]

    page_label = (
        f"page {top.page_number}"
        if top.page_number
        else "the document"
    )

    snippet = top.text.strip()

    if len(snippet) > 600:

        snippet = (
            snippet[:600]
            .rsplit(" ", 1)[0]
            + "..."
        )

    return (
        "Answer:\n"
        "LLM-generated explanation is unavailable.\n\n"
        "Document Evidence:\n"
        f"The most relevant evidence from {page_label} "
        "is:\n\n"
        f"\"{snippet}\"\n\n"
        f"Reason: {fallback_reason}"
    )