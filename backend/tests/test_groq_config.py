import os
from pathlib import Path


def test_default_groq_model_is_supported():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    env_text = env_path.read_text(encoding="utf-8")
    assert "GROQ_MODEL=groq/compound" in env_text or "GROQ_MODEL=llama-3.1-8b-instant" not in env_text
