"""
document_processor.py
---------------------

Document extraction, cleaning and intelligent chunking.

Supported:
    PDF
    TXT

The module has no knowledge of:
    - embeddings
    - ChromaDB
    - LLMs
"""

from __future__ import annotations

import re
import uuid
import unicodedata

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from . import config


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str

    # 1-indexed PDF page.
    # 0 for TXT.
    page_number: int

    chunk_index: int

    char_start: int
    char_end: int

    
    


@dataclass
class ProcessedDocument:
    doc_id: str
    filename: str
    num_pages: int
    chunks: List[Chunk] = field(default_factory=list)


class UnsupportedFileType(Exception):
    pass


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _extract_pdf_with_pymupdf(file_path: Path) -> List[str]:
    """
    Extract PDF text using PyMuPDF.

    PyMuPDF usually provides better extraction quality for
    real-world PDFs than basic PDF text extraction.
    """

    import fitz

    document = fitz.open(str(file_path))

    pages: List[str] = []

    try:
        for page in document:
            text = page.get_text("text") or ""
            pages.append(text)
    finally:
        document.close()

    return pages


def _extract_pdf_with_pypdf(file_path: Path) -> List[str]:
    """
    Fallback PDF extractor.
    """

    from pypdf import PdfReader

    reader = PdfReader(str(file_path))

    return [
        page.extract_text() or ""
        for page in reader.pages
    ]


def extract_pages(file_path: Path) -> List[str]:
    """
    Extract text while preserving PDF page boundaries.
    """

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":

        try:
            return _extract_pdf_with_pymupdf(file_path)

        except Exception as exc:

            print(
                f"PyMuPDF extraction failed: {exc}. "
                "Falling back to pypdf."
            )

            return _extract_pdf_with_pypdf(file_path)

    if suffix == ".txt":

        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        return [text]

    raise UnsupportedFileType(
        f"Unsupported file type: {suffix}"
    )


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(raw_text: str) -> str:
    """
    Clean extracted text while preserving Unicode.

    Important:
    Do NOT remove non-ASCII characters because that would destroy
    Bangla and other languages.
    """

    if not raw_text:
        return ""

    text = raw_text.replace("\x00", " ")

    # Unicode normalization.
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    # Join words broken by PDF line wrapping.
    #
    # Example:
    # informa-
    # tion
    #
    # becomes:
    # information
    text = re.sub(
        r"(?<=\w)-\s*\n\s*(?=\w)",
        "",
        text,
        flags=re.UNICODE,
    )

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Preserve paragraph boundaries.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # Normalize spaces/tabs.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Remove spaces surrounding newlines.
    text = re.sub(
        r" *\n *",
        "\n",
        text,
    )

    return text.strip()


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_PATTERN = re.compile(
    r"""
    (?<=[.!?。！？])      # sentence-ending punctuation
    \s+
    |
    \n+
    """,
    re.VERBOSE,
)


def _split_sentences(text: str) -> List[str]:
    """
    Split text into reasonably meaningful sentence units.
    """

    parts = _SENTENCE_PATTERN.split(text)

    sentences = []

    for part in parts:

        part = re.sub(
            r"\s+",
            " ",
            part,
        ).strip()

        if part:
            sentences.append(part)

    return sentences


# ---------------------------------------------------------------------------
# Word counting
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    """
    Unicode-aware approximate word count.
    """

    return len(
        re.findall(
            r"\S+",
            text,
            flags=re.UNICODE,
        )
    )


# ---------------------------------------------------------------------------
# Intelligent chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size_words: int = config.CHUNK_SIZE_WORDS,
    overlap_words: int = config.CHUNK_OVERLAP_WORDS,
) -> List[tuple[str, int, int]]:
    """
    Sentence/paragraph-aware overlapping chunker.

    Returns:
        (chunk_text, char_start, char_end)
    """

    if not text.strip():
        return []

    if chunk_size_words <= 0:
        raise ValueError(
            "chunk_size_words must be greater than zero."
        )

    if overlap_words < 0:
        raise ValueError(
            "overlap_words cannot be negative."
        )

    if overlap_words >= chunk_size_words:
        overlap_words = chunk_size_words // 4

    # -----------------------------------------------------------------------
    # Split into sentences.
    # -----------------------------------------------------------------------

    sentences = _split_sentences(text)

    if not sentences:
        return []

    # -----------------------------------------------------------------------
    # Build chunks from sentences.
    # -----------------------------------------------------------------------

    chunks: List[tuple[str, int, int]] = []

    current_sentences: List[str] = []
    current_words = 0

    def flush_current():

        nonlocal current_sentences
        nonlocal current_words

        if not current_sentences:
            return

        chunk = " ".join(
            current_sentences
        ).strip()

        if chunk:
            start = text.find(chunk)

            if start < 0:
                start = 0

            end = start + len(chunk)

            chunks.append(
                (
                    chunk,
                    start,
                    end,
                )
            )

        # ---------------------------------------------------------------
        # Keep overlapping sentences.
        # ---------------------------------------------------------------

        overlap_sentences: List[str] = []
        overlap_count = 0

        for sentence in reversed(current_sentences):

            sentence_words = _word_count(sentence)

            if (
                overlap_count + sentence_words
                > overlap_words
                and overlap_sentences
            ):
                break

            overlap_sentences.insert(
                0,
                sentence,
            )

            overlap_count += sentence_words

            if overlap_count >= overlap_words:
                break

        current_sentences = overlap_sentences
        current_words = overlap_count

    # -----------------------------------------------------------------------
    # Add sentences.
    # -----------------------------------------------------------------------

    for sentence in sentences:

        sentence_words = _word_count(sentence)

        # A single sentence larger than chunk size.
        if (
            sentence_words > chunk_size_words
            and not current_sentences
        ):

            words = sentence.split()

            start_index = 0

            while start_index < len(words):

                end_index = min(
                    start_index + chunk_size_words,
                    len(words),
                )

                piece = " ".join(
                    words[start_index:end_index]
                ).strip()

                if piece:
                    position = text.find(piece)

                    chunks.append(
                        (
                            piece,
                            max(position, 0),
                            max(position, 0) + len(piece),
                        )
                    )

                if end_index >= len(words):
                    break

                start_index += max(
                    chunk_size_words - overlap_words,
                    1,
                )

            continue

        # Normal sentence.
        if (
            current_sentences
            and current_words + sentence_words
            > chunk_size_words
        ):
            flush_current()

        current_sentences.append(sentence)
        current_words += sentence_words

    flush_current()

    # -----------------------------------------------------------------------
    # Remove tiny chunks where possible.
    # -----------------------------------------------------------------------

    filtered = []

    for chunk in chunks:

        chunk_str = chunk[0]

        if (
            _word_count(chunk_str)
            >= config.MIN_CHUNK_WORDS
            or len(chunks) == 1
        ):
            filtered.append(chunk)

    return filtered


# ---------------------------------------------------------------------------
# Process document
# ---------------------------------------------------------------------------

def process_document(
    file_path: Path,
    filename: str,
) -> ProcessedDocument:
    """
    Complete processing:

        extraction
             ↓
        cleaning
             ↓
        chunking
             ↓
        metadata
    """

    pages = extract_pages(file_path)

    doc_id = str(uuid.uuid4())

    all_chunks: List[Chunk] = []

    chunk_index = 0

    for page_num, raw_page in enumerate(
        pages,
        start=1,
    ):

        cleaned = clean_text(
            raw_page
        )

        if not cleaned:
            continue

        page_number = (
            page_num
            if file_path.suffix.lower() == ".pdf"
            else 0
        )

        page_chunks = chunk_text(
            cleaned,
            chunk_size_words=config.CHUNK_SIZE_WORDS,
            overlap_words=config.CHUNK_OVERLAP_WORDS,
        )

        for (
            chunk_str,
            char_start,
            char_end,
        ) in page_chunks:

            all_chunks.append(
                Chunk(
                    chunk_id=(
                        f"{doc_id}_{chunk_index}"
                    ),
                    doc_id=doc_id,
                    text=chunk_str,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    char_start=char_start,
                    char_end=char_end,
                )
            )

            chunk_index += 1

    return ProcessedDocument(
        doc_id=doc_id,
        filename=filename,
        num_pages=len(pages),
        chunks=all_chunks,
    )