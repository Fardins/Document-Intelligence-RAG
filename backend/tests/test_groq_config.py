import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import create_temp_upload_path
from app import config


def test_default_groq_model_is_supported():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    env_text = env_path.read_text(encoding="utf-8")
    assert "GROQ_MODEL=groq/compound" in env_text or "GROQ_MODEL=llama-3.1-8b-instant" not in env_text


def test_create_temp_upload_path_uses_tempdir_not_render_storage():
    path = create_temp_upload_path("report.pdf")

    try:
        assert path.suffix == ".pdf"
        assert path.parent != config.UPLOAD_DIR
        assert str(path.parent).startswith(tempfile.gettempdir())
    finally:
        path.unlink(missing_ok=True)
