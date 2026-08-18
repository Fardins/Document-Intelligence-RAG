/* Docket AI — framework-free frontend for the FastAPI RAG API */

const API_BASE = "https://document-intelligence-rag-1.onrender.com";

const $ = (id) => document.getElementById(id);

const els = {
  sidebar: $("sidebar"),
  mobileMenu: $("mobileMenu"),
  mobileOverlay: $("mobileOverlay"),
  statusDot: $("statusDot"),
  statusText: $("statusText"),
  docCount: $("docCount"),
  docRegistry: $("docRegistry"),
  registryEmpty: $("registryEmpty"),
  dropzone: $("dropzone"),
  fileInput: $("fileInput"),
  emptyState: $("emptyState"),
  chatView: $("chatView"),
  activeDocName: $("activeDocName"),
  activeDocMeta: $("activeDocMeta"),
  activeDocIcon: $("activeDocIcon"),
  changeDocButton: $("changeDocButton"),
  newChatButton: $("newChatButton"),
  qaThread: $("qaThread"),
  askForm: $("askForm"),
  questionInput: $("questionInput"),
  askButton: $("askButton"),
  suggestions: $("suggestions"),
  uploadProgress: $("uploadProgress"),
  progressFill: $("progressFill"),
  progressLabel: $("progressLabel"),
  progressPercent: $("progressPercent"),
  toastContainer: $("toastContainer"),
};

let documents = [];
let activeDocId = null;
let isAsking = false;

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  checkHealth();
});

function bindEvents() {
  els.dropzone.addEventListener("click", (e) => {
    if (!e.target.closest(".browse-btn")) return els.fileInput.click();
    els.fileInput.click();
  });

  els.fileInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
    e.target.value = "";
  });

  ["dragenter", "dragover"].forEach(evt => {
    els.dropzone.addEventListener(evt, e => {
      e.preventDefault();
      els.dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(evt => {
    els.dropzone.addEventListener(evt, e => {
      e.preventDefault();
      els.dropzone.classList.remove("dragover");
    });
  });

  els.dropzone.addEventListener("drop", e => {
    const file = e.dataTransfer.files?.[0];
    if (file) uploadFile(file);
  });

  els.askForm.addEventListener("submit", e => {
    e.preventDefault();
    askQuestion();
  });

  els.questionInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  });

  els.questionInput.addEventListener("input", autoResize);

  els.suggestions.addEventListener("click", e => {
    const button = e.target.closest("button");
    if (!button) return;
    els.questionInput.value = button.dataset.question || "";
    autoResize();
    els.questionInput.focus();
  });

  els.newChatButton.addEventListener("click", () => {
    if (!activeDocId) return;
    startConversation();
    closeMobileSidebar();
  });

  els.changeDocButton.addEventListener("click", () => {
    closeMobileSidebar();
    els.emptyState.hidden = false;
    els.chatView.hidden = true;
    document.querySelector(".workspace").scrollTop = 0;
  });

  els.mobileMenu.addEventListener("click", () => {
    els.sidebar.classList.add("open");
    els.mobileOverlay.classList.add("show");
  });

  els.mobileOverlay.addEventListener("click", closeMobileSidebar);
}

function closeMobileSidebar() {
  els.sidebar.classList.remove("open");
  els.mobileOverlay.classList.remove("show");
}

async function checkHealth() {
  setStatus("checking", "Connecting…");
  try {
    const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!res.ok) throw new Error("Backend unavailable");
    const health = await res.json();
    setStatus("online", "Backend online");
    await refreshDocuments();
    console.info("Docket API:", health);
  } catch (err) {
    setStatus("offline", "Backend unavailable");
    showToast("Cannot connect to the FastAPI backend.", "error");
  }
}

function setStatus(state, text) {
  els.statusDot.className = `status-dot ${state}`;
  els.statusText.textContent = text;
}

async function refreshDocuments() {
  try {
    const res = await fetch(`${API_BASE}/documents`);
    if (!res.ok) throw new Error();
    documents = await res.json();
    renderDocuments();
  } catch {
    documents = [];
    renderDocuments();
  }
}

function renderDocuments() {
  els.docCount.textContent = documents.length;
  els.docRegistry.querySelectorAll(".document-item").forEach(n => n.remove());
  els.registryEmpty.hidden = documents.length > 0;

  documents.forEach(doc => {
    const item = document.createElement("button");
    item.className = "document-item" + (doc.doc_id === activeDocId ? " active" : "");
    item.type = "button";
    item.innerHTML = `
      <span class="doc-mini ${fileType(doc.filename)}">${fileType(doc.filename).toUpperCase()}</span>
      <span class="doc-item-copy">
        <strong>${escapeHtml(doc.filename)}</strong>
        <small>${doc.num_pages} page${doc.num_pages === 1 ? "" : "s"} · ${doc.num_chunks} chunks</small>
      </span>
      <span class="doc-arrow">›</span>
    `;
    item.addEventListener("click", () => selectDocument(doc.doc_id));
    els.docRegistry.appendChild(item);
  });
}

function selectDocument(docId) {
  const doc = documents.find(d => d.doc_id === docId);
  if (!doc) return;

  activeDocId = docId;
  renderDocuments();

  els.emptyState.hidden = true;
  els.chatView.hidden = false;
  els.activeDocName.textContent = doc.filename;
  els.activeDocMeta.textContent =
    `${doc.num_pages} page${doc.num_pages === 1 ? "" : "s"} · ${doc.num_chunks} chunks indexed`;
  els.activeDocIcon.textContent = fileType(doc.filename).toUpperCase();
  els.activeDocIcon.className = `doc-symbol ${fileType(doc.filename)}`;

  startConversation();
  closeMobileSidebar();
  els.questionInput.focus();
}

function startConversation() {
  els.qaThread.innerHTML = `
    <div class="conversation-intro">
      <div class="assistant-avatar">✦</div>
      <div>
        <span class="intro-label">DOCUMENT READY</span>
        <h3>What would you like to know?</h3>
        <p>I’ll search the document, rerank the most relevant passages, and show the evidence behind my answer.</p>
      </div>
    </div>
  `;
  els.suggestions.hidden = false;
  els.questionInput.value = "";
  autoResize();
}

async function uploadFile(file) {
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (![".pdf", ".txt"].includes(ext)) {
    showToast("Only PDF and TXT files are supported.", "error");
    return;
  }

  const maxBytes = 25 * 1024 * 1024;
  if (file.size > maxBytes) {
    showToast("File is larger than the 25 MB limit.", "error");
    return;
  }

  setUploadProgress(true, 8, `Uploading ${file.name}…`);

  const formData = new FormData();
  formData.append("file", file);

  try {
    setUploadProgress(true, 25, "Uploading document…");

    const res = await fetch(`${API_BASE}/documents/upload`, {
      method: "POST",
      body: formData
    });
    console.log("Upload response:", res.status, res.statusText);

    setUploadProgress(true, 62, "Extracting text & building chunks…");

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Upload failed (${res.status})`);
    }

    const data = await res.json();

    setUploadProgress(true, 88, "Generating embeddings & indexing…");
    await refreshDocuments();

    setUploadProgress(true, 100, "Document indexed successfully");
    selectDocument(data.doc_id);

    showToast(`${data.filename} is ready to query.`, "success");

    setTimeout(() => setUploadProgress(false), 900);
  } catch (err) {
    setUploadProgress(true, 100, err.message || "Upload failed");
    els.progressFill.classList.add("error");
    showToast(err.message || "Document upload failed.", "error");
    setTimeout(() => {
      els.progressFill.classList.remove("error");
      setUploadProgress(false);
    }, 2200);
  }
}

function setUploadProgress(show, percent = 0, label = "") {
  els.uploadProgress.hidden = !show;
  if (!show) {
    els.progressFill.style.width = "0%";
    els.progressPercent.textContent = "0%";
    return;
  }
  els.progressFill.style.width = `${percent}%`;
  els.progressPercent.textContent = `${percent}%`;
  els.progressLabel.textContent = label;
}

async function askQuestion() {
  const question = els.questionInput.value.trim();
  if (!question || !activeDocId || isAsking) return;

  isAsking = true;
  els.askButton.disabled = true;
  els.questionInput.disabled = true;
  els.suggestions.hidden = true;

  const item = document.createElement("article");
  item.className = "qa-item";
  item.innerHTML = `
    <div class="user-message">
      <div class="user-avatar">You</div>
      <div class="user-question">${escapeHtml(question).replace(/\n/g, "<br>")}</div>
    </div>
    <div class="assistant-message">
      <div class="assistant-avatar small">✦</div>
      <div class="answer-wrap">
        <div class="answer-block loading">
          <div class="thinking">
            <span></span><span></span><span></span>
            <em>Searching your document…</em>
          </div>
        </div>
      </div>
    </div>
  `;

  els.qaThread.appendChild(item);
  els.questionInput.value = "";
  autoResize();
  scrollToBottom();

  try {
    const res = await fetch(`${API_BASE}/questions/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        doc_id: activeDocId,
        question
      })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    renderAnswer(item, data);
  } catch (err) {
    const answer = item.querySelector(".answer-block");
    answer.classList.remove("loading");
    answer.innerHTML = `
      <div class="error-answer">
        <strong>Something went wrong</strong>
        <p>${escapeHtml(err.message || "Unable to generate an answer.")}</p>
      </div>
    `;
  } finally {
    isAsking = false;
    els.askButton.disabled = false;
    els.questionInput.disabled = false;
    els.questionInput.focus();
  }
}

function renderAnswer(item, data) {
  const answer = item.querySelector(".answer-block");
  answer.classList.remove("loading");

  answer.innerHTML = formatAnswer(data.answer || "No answer returned.");

  if (Array.isArray(data.sources) && data.sources.length) {
    const sourceSection = document.createElement("div");
    sourceSection.className = "sources-section";

    const header = document.createElement("div");
    header.className = "sources-header";
    header.innerHTML = `
      <div>
        <span class="source-kicker">RETRIEVED EVIDENCE</span>
        <strong>${data.sources.length} source${data.sources.length === 1 ? "" : "s"}</strong>
      </div>
      <span class="source-note">Ranked by relevance</span>
    `;
    sourceSection.appendChild(header);

    data.sources.forEach((src, index) => {
      const card = document.createElement("div");
      card.className = "source-card";
      const score = Math.max(0, Math.min(100, Math.round(Number(src.relevance_score || 0) * 100)));
      const page = src.page_number ? `Page ${src.page_number}` : "Document";

      card.innerHTML = `
        <div class="source-number">${String(index + 1).padStart(2, "0")}</div>
        <div class="source-body">
          <div class="source-meta">
            <span class="page-badge">${escapeHtml(page)}</span>
            <span>Chunk ${escapeHtml(String(src.chunk_index ?? "—"))}</span>
            <span class="score"><i style="width:${score}%"></i>${score}% match</span>
          </div>
          <p>${escapeHtml(src.snippet || "")}</p>
        </div>
      `;
      sourceSection.appendChild(card);
    });

    item.querySelector(".answer-wrap").appendChild(sourceSection);
  }

  scrollToBottom();
}

function formatAnswer(text) {
  let safe = escapeHtml(text).replace(/\r\n/g, "\n");

  // Lightweight formatting while keeping backend response plain-text compatible.
  safe = safe
    .replace(/^Answer:\s*$/gim, '<span class="answer-heading">Answer</span>')
    .replace(/^Document Evidence:\s*$/gim, '<span class="evidence-heading">Document Evidence</span>')
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");

  return safe;
}

function autoResize() {
  const el = els.questionInput;
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 150) + "px";
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    els.qaThread.scrollTo({
      top: els.qaThread.scrollHeight,
      behavior: "smooth"
    });
  });
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${type === "success" ? "✓" : type === "error" ? "!" : "i"}</span>
    <span>${escapeHtml(message)}</span>
  `;
  els.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("out");
    setTimeout(() => toast.remove(), 250);
  }, 3200);
}

function fileType(filename) {
  return filename.toLowerCase().endsWith(".txt") ? "txt" : "pdf";
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value ?? "");
  return div.innerHTML;
}
