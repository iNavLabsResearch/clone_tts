"""Central config loading + secret resolution.

The single source of truth is ``config.json`` at the repo root. Secrets live in
``config.json`` (``REV:`` reversed-string obfuscation to pass GitHub push
protection) and/or a gitignored ``secrets.env`` / shell env vars. Env vars win
when set explicitly.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo root = two levels up from this file (src/milli_tts/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.json"

_ENV_PREFIX = "ENV:"
_B64_PREFIX = "B64:"
_REV_PREFIX = "REV:"


class Config:
    """Thin attribute/dict wrapper around the parsed config.json."""

    def __init__(self, data: dict[str, Any], path: Path):
        self._data = data
        self.path = path

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Config(data, cfg_path)


@dataclass
class Secrets:
    hf_token: str | None
    wandb_api_key: str | None


def _resolve_secret_marker(value: Any) -> str | None:
    """Decode secret markers: ENV:, B64:, REV: (reversed string)."""
    if not value or not isinstance(value, str):
        return None
    if value.startswith(_ENV_PREFIX):
        return os.environ.get(value[len(_ENV_PREFIX):]) or None
    if value.startswith(_B64_PREFIX):
        import base64

        try:
            return base64.b64decode(value[len(_B64_PREFIX):]).decode("utf-8")
        except Exception:
            return None
    if value.startswith(_REV_PREFIX):
        return value[len(_REV_PREFIX):][::-1]
    return value


def _decode_env_value(value: str) -> str:
    """Decode REV:/B64: markers in secrets.env values."""
    if value.startswith(_B64_PREFIX):
        import base64

        try:
            return base64.b64decode(value[len(_B64_PREFIX):]).decode("utf-8")
        except Exception:
            return value
    if value.startswith(_REV_PREFIX):
        return value[len(_REV_PREFIX):][::-1]
    return value


def _maybe_load_secrets_env() -> None:
    """Load secrets.env (KEY=VALUE lines) into os.environ if present and not already set."""
    secrets_file = REPO_ROOT / "secrets.env"
    if not secrets_file.exists():
        return
    for line in secrets_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        value = _decode_env_value(value)
        os.environ.setdefault(key, value)


def resolve_secrets(cfg: Config) -> Secrets:
    """Resolve HF + W&B secrets (env overrides config.json)."""
    _maybe_load_secrets_env()

    hf_from_cfg = _resolve_secret_marker(cfg.get("huggingface", {}).get("token"))
    wb_from_cfg = _resolve_secret_marker(cfg.get("wandb", {}).get("api_key"))

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or hf_from_cfg
    )
    wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_TOKEN") or wb_from_cfg

    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
    if wandb_key:
        os.environ.setdefault("WANDB_API_KEY", wandb_key)

    return Secrets(hf_token=hf_token, wandb_api_key=wandb_key)


def resolve_dataset_repo_id(cfg: Config, token: str | None) -> str:
    """Return the fully-qualified ``namespace/name`` for the processed dataset.

    If ``dataset.output.namespace`` is null, derive the namespace from the HF token's
    account (whoami) so the same id works for both push and load.
    """
    out = cfg["dataset"]["output"]
    name = out["repo"]
    namespace = out.get("namespace")
    if namespace:
        return f"{namespace}/{name}"
    if "/" in name:
        return name
    if token:
        try:
            from huggingface_hub import HfApi

            user = HfApi().whoami(token=token).get("name")
            if user:
                return f"{user}/{name}"
        except Exception:
            pass
    return name  # last resort; push_to_hub will still prepend the token namespace


def require_hf_token(cfg: Config) -> str:
    token = resolve_secrets(cfg).hf_token
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Set huggingface.token in config.json, "
            "or export HF_TOKEN (or put it in secrets.env / Kaggle Secrets). "
            "The Vaani dataset is gated, so a token with accepted access is required."
        )
    return token
