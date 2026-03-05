import inspect
import sys
from ollama import chat
from config import cfg


MAX_RETRIES = 2


# ── Tool schema builder ──────────────────────────────────────────────────────

def build_tool_schemas(functions):
    """Build Ollama tool schemas from a list of functions."""
    schemas = []
    for func in functions:
        sig = inspect.signature(func)
        full_doc = inspect.getdoc(func) or func.__name__
        # Build per-parameter descriptions from the full docstring + signature
        param_names = list(sig.parameters.keys())
        props = {}
        required = []
        for pname, param in sig.parameters.items():
            ann = param.annotation
            if ann == int:
                ptype = "integer"
            elif ann == bool:
                ptype = "boolean"
            elif ann == float:
                ptype = "number"
            else:
                ptype = "string"
            # Build a useful description: parameter name, type, and default if any
            desc = pname
            if param.default is not inspect.Parameter.empty:
                desc += f" (default: {param.default!r})"
            props[pname] = {"type": ptype, "description": desc}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        # Include full docstring + parameter names so the model knows the signature
        func_desc = full_doc
        if param_names:
            func_desc += f"\nParameters: {', '.join(param_names)}"
        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func_desc,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                }
            }
        }
        schemas.append(schema)
    return schemas


def _warn(msg, on_chunk=None):
    """Log a warning via on_chunk callback or stderr."""
    if on_chunk:
        on_chunk('content', f"\n[WARNING: {msg}]\n")
    else:
        print(f"\n[WARNING: {msg}]", file=sys.stderr)


# ── Streaming Ollama query ───────────────────────────────────────────────────

def stream_ollama(model, messages, tools=None, think=True, on_chunk=None):
    """
    Stream a response from an Ollama model.
    Returns (thinking, content, tool_calls).

    Retries on Ollama errors (e.g. JSON parse failures from malformed tool
    calls). Tool schemas are preserved across retries so tool-calling behavior
    remains available on every attempt.

    on_chunk: optional callback(chunk_type, text)
        chunk_type is 'thinking_start', 'thinking', 'answer_start', 'content'
    """
    tool_schemas = build_tool_schemas(tools) if tools else None

    for attempt in range(MAX_RETRIES + 1):
        try:
            stream = chat(
                model=model,
                messages=messages,
                tools=tool_schemas,
                think=think,
                stream=True,
                options={'num_ctx': cfg['ollama_context_window'], 'num_predict': -1}
            )
            in_thinking = False
            content = ''
            thinking = ''
            tool_calls = []
            for chunk in stream:
                if chunk.message.thinking:
                    if not in_thinking:
                        in_thinking = True
                        if on_chunk:
                            on_chunk('thinking_start', '')
                        else:
                            print('Thinking:\n', end='', flush=True)
                    if on_chunk:
                        on_chunk('thinking', chunk.message.thinking)
                    else:
                        print(chunk.message.thinking, end='', flush=True)
                    thinking += chunk.message.thinking
                elif chunk.message.content:
                    if in_thinking:
                        in_thinking = False
                        if on_chunk:
                            on_chunk('answer_start', '')
                        else:
                            print('\n\nAnswer:\n', end='', flush=True)
                    if on_chunk:
                        on_chunk('content', chunk.message.content)
                    else:
                        print(chunk.message.content, end='', flush=True)
                    content += chunk.message.content
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)
            if not on_chunk:
                print()
            return thinking, content, tool_calls

        except Exception as e:
            if attempt < MAX_RETRIES:
                _warn(f"Ollama error: {e} — retrying with same tool configuration (attempt {attempt + 1})", on_chunk)
            else:
                _warn(f"Ollama error after {MAX_RETRIES + 1} attempts: {e}", on_chunk)
                return '', '', []


# ── Non-streaming Ollama query ───────────────────────────────────────────────

def query_ollama(model, messages, tools=None, think=False):
    """
    Non-streaming Ollama query. Used for tool_group_chooser and tool_user.
    Returns (thinking, content, tool_calls).

    Retries on failure while preserving tool schemas across attempts.
    """
    tool_schemas = build_tool_schemas(tools) if tools else None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = chat(
                model=model,
                messages=messages,
                tools=tool_schemas,
                think=think,
                stream=False,
                options={'num_ctx': cfg['ollama_context_window'], 'num_predict': -1}
            )
            thinking = getattr(response.message, 'thinking', '') or ''
            content = response.message.content or ''
            tool_calls = response.message.tool_calls or []
            return thinking, content, tool_calls

        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"[WARNING: Ollama error: {e} — retrying with same tool configuration (attempt {attempt + 1})]",
                      file=sys.stderr)
            else:
                print(f"[WARNING: Ollama error after {MAX_RETRIES + 1} attempts: {e}]",
                      file=sys.stderr)
                return '', '', []
