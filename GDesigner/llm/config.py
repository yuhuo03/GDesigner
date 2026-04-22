import json
import os
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv


load_dotenv()

OPENAI_BACKEND = "openai"
VLLM_BACKEND = "vllm"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_VLLM_MODEL = "Qwen3-8B"


def get_llm_backend() -> str:
    backend = os.getenv("LLM_BACKEND", OPENAI_BACKEND).strip().lower()
    if backend not in {OPENAI_BACKEND, VLLM_BACKEND}:
        raise ValueError(f"Unsupported LLM_BACKEND={backend!r}. Use 'openai' or 'vllm'.")
    return backend


def get_openai_base_url() -> str:
    return os.getenv("BASE_URL", "").strip()


def get_openai_api_key() -> str:
    return os.getenv("API_KEY", "").strip()


def get_vllm_base_url() -> str:
    return os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").strip()


def get_vllm_api_key() -> str:
    return os.getenv("VLLM_API_KEY", "EMPTY").strip()


def get_vllm_model() -> str:
    return os.getenv("VLLM_MODEL", DEFAULT_VLLM_MODEL).strip() or DEFAULT_VLLM_MODEL


def get_vllm_extra_body() -> Optional[Dict[str, Any]]:
    raw = os.getenv("VLLM_EXTRA_BODY_JSON", "").strip()
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("VLLM_EXTRA_BODY_JSON must decode to a JSON object.")
    return value


def resolve_runtime_model_name(model_name: Optional[str] = None) -> str:
    if get_llm_backend() == VLLM_BACKEND:
        return get_vllm_model()
    return model_name or DEFAULT_OPENAI_MODEL


def sanitize_model_name(model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("._")
    return safe or "model"
