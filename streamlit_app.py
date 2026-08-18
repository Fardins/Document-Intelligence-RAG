"""
Streamlit Frontend
------------------
Document Intelligence RAG System

Features:
    - PDF/TXT upload
    - Document indexing
    - Document selection
    - Chat-based question answering
    - Exactly 3 retrieved sources
    - Expandable full-page source viewer
    - Relevance scores
    - Chat history
    - Backend health status
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_API_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
)

MAX_UPLOAD_SIZE_MB = 25

ALLOWED_EXTENSIONS = [
    "pdf",
    "txt",
]


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Document Intelligence RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Custom CSS
# ============================================================================

st.markdown(
    """
    <style>

    /* ---------------------------------------------------------
       Global
    --------------------------------------------------------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* ---------------------------------------------------------
       Header
       --------------------------------------------------------- */

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* ---------------------------------------------------------
       Source cards
       --------------------------------------------------------- */

    .source-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        background: #ffffff;
    }

    .source-header {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 6px;
    }

    .source-meta {
        color: #6b7280;
        font-size: 0.82rem;
        margin-bottom: 10px;
    }

    .source-snippet {
        font-size: 0.92rem;
        line-height: 1.6;
        color: #374151;
    }

    /* ---------------------------------------------------------
       Status
       --------------------------------------------------------- */

    .status-online {
        color: #15803d;
        font-weight: 600;
    }

    .status-offline {
        color: #dc2626;
        font-weight: 600;
    }

    /* ---------------------------------------------------------
       Chat
       --------------------------------------------------------- */

    .answer-label {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 8px;
    }

    /* ---------------------------------------------------------
       Metrics
       --------------------------------------------------------- */

    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px;
    }

    /* ---------------------------------------------------------
       File uploader
       --------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Session State
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_API_URL.rstrip("/")


# ============================================================================
# API Helpers
# ============================================================================

def api_url(path: str) -> str:
    """Build backend API URL."""

    return (
        st.session_state.backend_url.rstrip("/")
        + "/"
        + path.lstrip("/")
    )


def check_backend() -> dict[str, Any] | None:
    """Check backend health."""

    try:
        response = requests.get(
            api_url("/health"),
            timeout=5,
        )

        if response.status_code == 200:
            return response.json()

    except requests.RequestException:
        pass

    return None


def get_documents() -> list[dict[str, Any]]:
    """Get indexed documents from backend."""

    try:

        response = requests.get(
            api_url("/documents"),
            timeout=10,
        )

        if response.status_code != 200:
            return []

        data = response.json()

        if isinstance(data, list):
            return data

        return []

    except requests.RequestException:
        return []


def upload_document(
    uploaded_file,
) -> tuple[bool, str, dict[str, Any] | None]:
    """
    Upload a PDF/TXT document to the backend.
    """

    try:

        file_bytes = uploaded_file.getvalue()

        size_mb = len(file_bytes) / (
            1024 * 1024
        )

        if size_mb > MAX_UPLOAD_SIZE_MB:

            return (
                False,
                (
                    f"File is too large. Maximum size is "
                    f"{MAX_UPLOAD_SIZE_MB} MB."
                ),
                None,
            )

        files = {
            "file": (
                uploaded_file.name,
                file_bytes,
                uploaded_file.type
                or "application/octet-stream",
            )
        }

        response = requests.post(
            api_url("/documents/upload"),
            files=files,
            timeout=300,
        )

        if response.status_code == 200:

            data = response.json()

            return (
                True,
                "Document indexed successfully.",
                data,
            )

        try:
            error = response.json()
            detail = error.get(
                "detail",
                response.text,
            )
        except Exception:
            detail = response.text

        return (
            False,
            f"Upload failed: {detail}",
            None,
        )

    except requests.Timeout:

        return (
            False,
            "The backend took too long to process the document.",
            None,
        )

    except requests.RequestException as exc:

        return (
            False,
            f"Could not connect to backend: {exc}",
            None,
        )


def ask_question(
    doc_id: str,
    question: str,
) -> tuple[bool, str, list[dict[str, Any]]]:

    payload = {
        "doc_id": doc_id,
        "question": question,
        "top_k": 3,
    }

    try:

        response = requests.post(
            api_url("/questions/ask"),
            json=payload,
            timeout=180,
        )

        if response.status_code == 200:

            data = response.json()

            answer = data.get(
                "answer",
                "",
            )

            sources = data.get(
                "sources",
                [],
            )

            # Always show at most 3.
            sources = sources[:3]

            return (
                True,
                answer,
                sources,
            )

        try:
            error = response.json()
            detail = error.get(
                "detail",
                response.text,
            )
        except Exception:
            detail = response.text

        return (
            False,
            f"Question failed: {detail}",
            [],
        )

    except requests.Timeout:

        return (
            False,
            "The question request timed out.",
            [],
        )

    except requests.RequestException as exc:

        return (
            False,
            f"Could not connect to backend: {exc}",
            [],
        )


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Settings"
    )

    st.session_state.backend_url = st.text_input(
        "Backend URL",
        value=st.session_state.backend_url,
        help="FastAPI backend URL.",
    ).rstrip("/")

    st.divider()

    # ------------------------------------------------------------
    # Backend status
    # ------------------------------------------------------------

    st.markdown(
        "### 🔌 Backend"
    )

    health = check_backend()

    if health:

        st.markdown(
            '<span class="status-online">'
            '● Backend Online'
            '</span>',
            unsafe_allow_html=True,
        )

        st.caption(
            f"Embedding: "
            f"{health.get('embedding_model', 'Unknown')}"
        )

        st.caption(
            f"LLM: "
            f"{health.get('llm_model', 'Unknown')}"
        )

    else:

        st.markdown(
            '<span class="status-offline">'
            '● Backend Offline'
            '</span>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Make sure FastAPI is running."
        )

    st.divider()

    # ------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------

    st.markdown(
        "### 📤 Upload Documents"
    )

    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        help=(
            "Upload one or more PDF/TXT documents. "
            "Each document will be processed and indexed."
        ),
    )

    if uploaded_files:

        if st.button(
            "🚀 Upload & Index",
            use_container_width=True,
            type="primary",
        ):

            progress = st.progress(0)

            total = len(uploaded_files)

            successful = 0

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                with st.status(
                    f"Processing {uploaded_file.name}...",
                    expanded=False,
                ) as status:

                    success, message, data = upload_document(
                        uploaded_file
                    )

                    if success:

                        successful += 1

                        status.update(
                            label=(
                                f"✓ {uploaded_file.name} "
                                "indexed"
                            ),
                            state="complete",
                        )

                    else:

                        status.update(
                            label=(
                                f"✗ {uploaded_file.name} "
                                "failed"
                            ),
                            state="error",
                        )

                        st.error(message)

                progress.progress(
                    (index + 1) / total
                )

            if successful:

                st.success(
                    f"{successful} document(s) indexed."
                )

                st.session_state.documents = (
                    get_documents()
                )

                st.rerun()

    st.divider()

    # ------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------

    st.markdown(
        "### 📚 Indexed Documents"
    )

    if st.button(
        "🔄 Refresh Documents",
        use_container_width=True,
    ):

        st.session_state.documents = (
            get_documents()
        )

        st.rerun()

    documents = st.session_state.documents

    if not documents:

        documents = get_documents()

        st.session_state.documents = documents

    if documents:

        document_options = {}

        for document in documents:

            filename = document.get(
                "filename",
                "Unknown document",
            )

            doc_id = document.get(
                "doc_id"
            )

            document_options[
                f"📄 {filename}"
            ] = doc_id

        selected_label = st.selectbox(
            "Select document",
            options=list(
                document_options.keys()
            ),
        )

        selected_id = document_options[
            selected_label
        ]

        st.session_state.selected_doc_id = (
            selected_id
        )

        selected_document = next(
            (
                doc
                for doc in documents
                if doc.get("doc_id")
                == selected_id
            ),
            None,
        )

        if selected_document:

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Pages",
                    selected_document.get(
                        "num_pages",
                        0,
                    ),
                )

            with col2:

                st.metric(
                    "Chunks",
                    selected_document.get(
                        "num_chunks",
                        0,
                    ),
                )

    else:

        st.info(
            "No documents indexed yet."
        )

    st.divider()

    # ------------------------------------------------------------
    # Chat controls
    # ------------------------------------------------------------

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================================
# Main Header
# ============================================================================

st.markdown(
    '<div class="main-title">'
    '📚 Document Intelligence RAG'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your uploaded documents '
    'using semantic search and Retrieval-Augmented Generation.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================================
# Selected Document Check
# ============================================================================

selected_doc_id = (
    st.session_state.selected_doc_id
)

documents = st.session_state.documents

selected_document = None

if selected_doc_id:

    selected_document = next(
        (
            doc
            for doc in documents
            if doc.get("doc_id")
            == selected_doc_id
        ),
        None,
    )


# ============================================================================
# Document Information
# ============================================================================

if selected_document:

    filename = selected_document.get(
        "filename",
        "Unknown",
    )

    pages = selected_document.get(
        "num_pages",
        0,
    )

    chunks = selected_document.get(
        "num_chunks",
        0,
    )

    st.info(
        f"📄 **{filename}**  •  "
        f"{pages} pages  •  "
        f"{chunks} chunks"
    )

else:

    st.warning(
        "👈 Upload and select a document from the sidebar "
        "before asking a question."
    )


# ============================================================================
# Chat History
# ============================================================================

for message in st.session_state.messages:

    role = message.get(
        "role"
    )

    content = message.get(
        "content",
        "",
    )

    with st.chat_message(role):

        if role == "assistant":

            st.markdown(
                content
            )

            sources = message.get(
                "sources",
                [],
            )

            if sources:

                st.markdown(
                    f"### 📚 Sources ({len(sources)})"
                )

                for index, source in enumerate(
                    sources[:3],
                    start=1,
                ):

                    page_number = source.get(
                        "page_number",
                        0,
                    )

                    relevance = source.get(
                        "relevance_score",
                        0,
                    )

                    snippet = source.get(
                        "snippet",
                        "",
                    )

                    st.markdown(
                        f"**Source {index} — "
                        f"Page {page_number}**"
                    )

                    st.caption(
                        f"Relevance: {float(relevance):.4f}"
                    )

                    if snippet:

                        st.markdown(
                            f'> "{snippet}"'
                        )

                    # ------------------------------------------------
                    # Full page
                    # ------------------------------------------------

                    full_page = source.get(
                        "full_page_text",
                        "",
                    )

                    if full_page:

                        with st.expander(
                            f"📖 Expand full page — "
                            f"Page {page_number}",
                            expanded=False,
                        ):

                            st.markdown(
                                full_page
                            )

                    else:

                        with st.expander(
                            f"📖 Expand source — "
                            f"Page {page_number}",
                            expanded=False,
                        ):

                            st.warning(
                                "Full page text is not available "
                                "for this source. "
                                "Update the backend page-store "
                                "implementation and re-index the document."
                            )

        else:

            st.markdown(
                content
            )


# ============================================================================
# Question Input
# ============================================================================

question = st.chat_input(
    "Ask a question about your document..."
)


# ============================================================================
# Question Processing
# ============================================================================

if question:

    question = question.strip()

    if not question:
        st.stop()

    if not selected_doc_id:

        st.warning(
            "Please select a document first."
        )

        st.stop()

    # ------------------------------------------------------------
    # Add user message
    # ------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # ------------------------------------------------------------
    # Generate answer
    # ------------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching the document and generating an answer..."
        ):

            success, answer, sources = ask_question(
                selected_doc_id,
                question,
            )

        if success:

            st.markdown(
                answer
            )

            # --------------------------------------------------------
            # Sources
            # --------------------------------------------------------

            if sources:

                sources = sources[:3]

                st.markdown(
                    f"### 📚 Sources ({len(sources)})"
                )

                for index, source in enumerate(
                    sources,
                    start=1,
                ):

                    page_number = source.get(
                        "page_number",
                        0,
                    )

                    relevance = source.get(
                        "relevance_score",
                        0,
                    )

                    snippet = source.get(
                        "snippet",
                        "",
                    )

                    st.markdown(
                        f"**Source {index} — "
                        f"Page {page_number}**"
                    )

                    st.caption(
                        f"Relevance: {float(relevance):.4f}"
                    )

                    if snippet:

                        st.markdown(
                            f'> "{snippet}"'
                        )

                    full_page = source.get(
                        "full_page_text",
                        "",
                    )

                    if full_page:

                        with st.expander(
                            f"📖 Expand full page — "
                            f"Page {page_number}",
                            expanded=False,
                        ):

                            st.markdown(
                                full_page
                            )

                    else:

                        with st.expander(
                            f"📖 Expand source — "
                            f"Page {page_number}",
                        ):

                            st.warning(
                                "Full page text is not available."
                            )

            else:

                st.info( 
                    "No sources were returned." 
                    ) 
        else:
            st.error( 
                answer 
            )

    # ------------------------------------------------------------
    # Save assistant response 
    # ------------------------------------------------------------

    st.session_state.messages.append( 
        { "role": "assistant", 
         "content": answer, 
         "sources": sources, 
         } 
    )