"""Central config loading + secret resolution.

The single source of truth is ``config.json`` at the repo root. Secrets are NEVER
stored in config or code; they are read from environment variables (or a gitignored
``secrets.env`` loaded by setup.sh / the launch scripts, or Kaggle Secrets).
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
        os.environ.setdefault(key, value)


def resolve_secrets(cfg: Config) -> Secrets:
    """Resolve HF + W&B secrets from env (after optionally sourcing secrets.env)."""
    _maybe_load_secrets_env()
    sec = cfg.get("secrets", {})
    hf_env = sec.get("hf_token_env", "HF_TOKEN")
    wb_env = sec.get("wandb_api_key_env", "WANDB_API_KEY")
    # Accept a couple of common aliases too.
    hf_token = os.environ.get(hf_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    wandb_key = os.environ.get(wb_env) or os.environ.get("WANDB_TOKEN")
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
        env = cfg.get("secrets", {}).get("hf_token_env", "HF_TOKEN")
        raise RuntimeError(
            f"No Hugging Face token found. Set ${env} (or put it in secrets.env / Kaggle Secrets). "
            f"The Vaani dataset is gated, so a token with accepted access is required."
        )
    return token
