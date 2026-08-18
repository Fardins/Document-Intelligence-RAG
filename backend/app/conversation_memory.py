"""
conversation_memory.py
----------------------

In-memory conversation history for document Q&A.

The memory is intentionally lightweight.

Each conversation contains:
    - user question
    - assistant answer

The memory is keyed by conversation_id.

Example:

conversation_id
      ↓
[
    {
        "question": "...",
        "answer": "..."
    },
    {
        "question": "...",
        "answer": "..."
    }
]
"""

from __future__ import annotations

import threading

from dataclasses import dataclass
from typing import Dict, List

from . import config


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    question: str
    answer: str


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

class ConversationMemory:

    def __init__(self):
        self._conversations: Dict[
            str,
            List[ConversationTurn],
        ] = {}

        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Get history
    # -----------------------------------------------------------------------

    def get_history(
        self,
        conversation_id: str,
    ) -> List[ConversationTurn]:

        with self._lock:

            history = self._conversations.get(
                conversation_id,
                [],
            )

            return list(history)

    # -----------------------------------------------------------------------
    # Add turn
    # -----------------------------------------------------------------------

    def add_turn(
        self,
        conversation_id: str,
        question: str,
        answer: str,
    ) -> None:

        if not conversation_id:
            return

        if not question.strip():
            return

        with self._lock:

            history = self._conversations.setdefault(
                conversation_id,
                [],
            )

            history.append(
                ConversationTurn(
                    question=question.strip(),
                    answer=answer.strip(),
                )
            )

            # Keep only the latest N turns.
            if len(history) > config.MAX_CONVERSATION_TURNS:

                del history[
                    :-config.MAX_CONVERSATION_TURNS
                ]

    # -----------------------------------------------------------------------
    # Clear conversation
    # -----------------------------------------------------------------------

    def clear(
        self,
        conversation_id: str,
    ) -> bool:

        with self._lock:

            existed = (
                conversation_id
                in self._conversations
            )

            self._conversations.pop(
                conversation_id,
                None,
            )

            return existed

    # -----------------------------------------------------------------------
    # Format history for LLM
    # -----------------------------------------------------------------------

    def format_history(
        self,
        conversation_id: str,
    ) -> str:

        history = self.get_history(
            conversation_id
        )

        if not history:
            return ""

        blocks = []

        for index, turn in enumerate(
            history,
            start=1,
        ):

            blocks.append(
                f"Turn {index}\n"
                f"User: {turn.question}\n"
                f"Assistant: {turn.answer}"
            )

        formatted = "\n\n".join(
            blocks
        )

        # Prevent excessively large prompts.
        if (
            len(formatted)
            > config.MAX_CONVERSATION_HISTORY_CHARS
        ):

            formatted = formatted[
                -config.MAX_CONVERSATION_HISTORY_CHARS:
            ]

        return formatted


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

conversation_memory = ConversationMemory()