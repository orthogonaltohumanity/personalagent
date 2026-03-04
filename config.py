from pathlib import Path

import yaml
from ollama import generate as ollama_generate

# ── Load config ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / 'config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)


def resolve_path(path_value: str):
    """Resolve a config path relative to repo root when not absolute."""
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p

_molt_client = None

def get_molt_client():
    global _molt_client
    if _molt_client is None:
        from moltbook import Moltbook
        _molt_client = Moltbook()
    return _molt_client


# ── Model helpers ────────────────────────────────────────────────────────────

_models_cfg = cfg['models']

def get_model(role):
    """Returns the model name for a given role."""
    if role not in _models_cfg:
        # Backward compatibility for older configs that still use tool_selector
        if role in ('tool_group_chooser', 'tool_user') and 'tool_selector' in _models_cfg:
            role = 'tool_selector'
        else:
            raise KeyError(f"Model role '{role}' not found in config.models")
    entry = _models_cfg[role]
    if isinstance(entry, dict):
        return entry['model']
    return entry  # backward compat: plain string

def get_model_think(role):
    """Returns whether a model supports think/CoT mode."""
    if role not in _models_cfg:
        if role in ('tool_group_chooser', 'tool_user') and 'tool_selector' in _models_cfg:
            role = 'tool_selector'
        else:
            return False
    entry = _models_cfg[role]
    if isinstance(entry, dict):
        return entry.get('think', False)
    return False

def get_code_model():
    """Returns the code generation model, with fallback."""
    code_model = cfg.get('code_model', 'qwen2.5-coder:7b')
    code_model_fallback = cfg.get('code_model_fallback', 'qwen2.5-coder:7b')
    try:
        ollama_generate(model=code_model, prompt="test", options={'num_predict': 1})
        return code_model
    except Exception:
        print(f"[code_model] {code_model} unavailable, falling back to {code_model_fallback}")
        return code_model_fallback
