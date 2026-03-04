# Sequential Task-Loop Agent

An autonomous AI agent with a structured planner → executor → verifier loop, persistent memory, tool groups, social media integration, and dynamic tool creation.

---

## Architecture Overview

```
User Task
    ↓
[Planner] ── text-only, no tool calls ── produces numbered subtask list
    ↓                                      (includes memory recall/save as subtasks)
For each subtask:
    [Tool Group Chooser] → picks group
    ↓
    [Tool User] → calls tools in chosen group → records results
    ↓
[Verifier] → COMPLETE or INCOMPLETE
    ↓
  COMPLETE → save memories → ask user for next task
  INCOMPLETE → remaining work fed back to Planner → loop
```

### File Map

```
config.yaml          Configuration: models, limits, paths, tool group constraints
config.py            Loads config.yaml, model helpers (get_model, get_code_model)
state.py             Centralized mutable state: memories, PDF index, iteration counter
providers.py         LLM query layer: Ollama streaming + non-streaming
tools.py             All tool functions + flat tool registry builder
tool_groups.py       Tool group definitions and selection logic
tui.py               Rich TUI components (AgentTUI, layout, state model)
main.py              Entry point: TUI mode (default) or --no-tui for plain output
system_prompt.md     The agent's system prompt / prime directive
memories.json        Persistent key-value memory store
pdf_index.json       Vector index for ingested documents
Playground/          Working directory for agent-generated files
downloads/           Downloaded PDFs and CSVs
```

### Three Models, Three Roles

| Role | Default Model | Think | Purpose |
|------|---------------|-------|---------|
| planner | qwen3.5:9b | yes | Text-only — no tool calls. Outputs numbered subtask list with suggested tool groups. |
| tool_group_chooser | qwen3:1.7b | no | Picks a tool group for each subtask based on planner subtask text and suggested group. |
| tool_user | ministral-3:3b | no | Calls up to 3 tools from the chosen group for each subtask. |
| verifier | qwen3:4b | yes | Reviews original task + all results, responds COMPLETE or INCOMPLETE |

Configured in `config.yaml` under `models:`. Each model entry has a `think` flag for chain-of-thought mode. The planner receives NO tool schemas — it produces text only. This avoids compatibility issues with models that can't reliably handle thinking + tool calling together.

### Data Flow (Detailed)

```
while True:
  user_task = input()
  planning_input = user_task

  while verification_loops <= max:

    1. PLAN PHASE
       - build_planner_messages() assembles:
           system_prompt.md + working dir + short-term goal
           + planner role instructions
           + group summary (group names + descriptions, NO individual tool names)
       - Planner streams response via stream_ollama() with tools=None, think=True
       - No tool calls — planner is text-only
       - Planner can suggest groups; execution is handled by tool group chooser + tool user
       - Text output is parsed into subtask list via parse_subtasks()

    2. EXECUTE PHASE (for each subtask)
       a. build_selector_messages() provides subtask + previous results + group list
       b. Tool group chooser (non-streaming, no thinking) picks a group name
       c. pick_group() extracts group name from chooser output (first line match, then full scan)
       d. If no group found → defaults to web_search
       e. Second query to tool user with chosen group's tool schemas → generates tool calls
       f. execute_tool_calls() runs each call, records results
       g. Results accumulate — later subtasks see earlier results for context

    2b. FAILURE HANDLING
       - If a subtask fails (error result, no tool calls, no tools in group, or timeout):
         execution stops immediately — remaining subtasks are skipped
       - Failure context (what failed + results so far) is fed back to the planner
       - Verification is skipped — goes straight to re-planning
       - Per-subtask timeout: 5 min default, 15 min for web_search/document_processing groups

    3. VERIFY PHASE
       - build_verifier_messages() provides system prompt + original task + all results
       - Verifier streams response with think=True
       - First word checked: exact match on 'COMPLETE' → task done
       - Otherwise: INCOMPLETE → remaining work extracted → fed back as new planning_input
       - After max_verification_loops: asks user to continue, stop, or provide new instructions
```

### Planner Isolation

The planner is strictly text-only — it receives `tools=None` and cannot make tool calls. It sees tool group names and descriptions but **not individual tool names** (via `get_group_summary(include_tools=False)`). This prevents the model from hallucinating tool calls or getting confused between planning and executing.

Memory operations (search_memory, save_memory) and user clarification (check_in) are handled as regular subtasks that the tool user executes.

---

## Tool Groups

Tools are organized into 9 groups. The tool group chooser picks ONE group per subtask, then the tool user calls tools from it. This forces clean task decomposition.

| Group | Description | Tools |
|-------|-------------|-------|
| web_search | Search the web and download files | search_web, search_and_download_files |
| social_media | Moltbook social media platform | 16 tools (posts, file-to-post, comments, votes, profiles, communities) |
| document_processing | PDFs and CSVs | ingest_pdf, ingest_csv, query_documents, list_downloaded_files |
| file_operations | Read/write files | read_file, edit, list_working_files |
| code_generation | AI code models | generate_code, generate_code_edit |
| text_generation | Generate or edit written text | write_text, edit_text, write_text_from_source |
| version_control | Git operations | git_init, git_status, git_add, git_commit, git_log, git_diff, git_diff_staged, git_branch, git_checkout, git_list_branches |
| memory | Persistent memory | search_memory, save_memory, open_memory, edit_memory, delete_memory, list_memory_keys, memory_stats, set_short_term_goal |
| system | Meta-tools | create_tool, list_custom_tools, remove_custom_tool, check_in |

Group definitions are in `tool_groups.py`. Key functions:
- `get_group_summary(include_tools=True)` → formatted string for prompts. Pass `include_tools=False` for the planner to hide individual tool names.
- `get_tools_in_group(name)` → list of function objects
- `get_group_tool_schemas(name)` → Ollama tool schemas for that group

### Adding a New Tool

1. Define the function in `tools.py` with type annotations and a docstring
2. Register it in `build_tool_registry()`
3. Add it to the appropriate group in `TOOL_GROUPS` in `tool_groups.py`

### Adding a New Tool Group

Add an entry to `TOOL_GROUPS` in `tool_groups.py`:

```python
TOOL_GROUPS["my_group"] = {
    "description": "What this group does",
    "tools": ["tool_name_1", "tool_name_2"]
}
```

---

## Configuration

### config.yaml

```yaml
models:
  planner:
    model: "qwen3.5:9b"
    think: true              # chain-of-thought mode (model must support it)
  tool_group_chooser:
    model: "qwen3:1.7b"
    think: false
  tool_user:
    model: "ministral-3:3b"
    think: false
  verifier:
    model: "qwen3:4b"
    think: true

code_model: "qwen2.5-coder:7b"
code_model_fallback: "qwen2.5-coder:7b"
embedding_model: "embeddinggemma"

max_tools_per_task: 3          # max tools from a group per subtask
max_subtasks: 20               # max subtasks per plan
max_verification_loops: 12     # max re-plans before asking user

ollama_context_window: 32768
max_tool_calls_per_step: 10
subtask_timeout_seconds: 300           # 5-min default per-subtask timeout
download_subtask_timeout_seconds: 900  # 15-min timeout for web_search, document_processing, write/edit_text
max_web_search_results: 5
max_download_search_results: 10
download_timeout_seconds: 15

working_directory: "Playground/"
downloads_directory: "downloads"
pdf_index_path: "pdf_index.json"
memories_path: "memories.json"
system_prompt_path: "system_prompt.md"

short_term_goal: "Short Term Goal Not Given. To Be Determined"
```

### config.py

Simple module — no classes:
- `cfg` — the parsed YAML dict
- `models` — shortcut to `cfg['models']`
- `get_model(role)` — returns model name for a role (handles dict or plain string entries)
- `get_model_think(role)` — returns whether a role's model has thinking enabled
- `get_code_model()` — returns code model with fallback
- `get_molt_client()` — lazy-loaded Moltbook client

---

## Providers (providers.py)

Ollama-only. Three functions:

- **`build_tool_schemas(functions)`** — Inspects function signatures to generate Ollama tool schemas. Type annotations → JSON types, docstrings → descriptions, required params detected automatically.
- **`stream_ollama(model, messages, tools, think, on_chunk)`** — Streaming chat. Returns `(thinking, content, tool_calls)`. Retries up to 2 times on Ollama errors — first retry drops tools so the model can still produce text. When `on_chunk` is set, tokens stream to the callback instead of stdout. Chunk types: `thinking_start`, `thinking`, `answer_start`, `content`.
- **`query_ollama(model, messages, tools, think)`** — Non-streaming chat. Used by tool_group_chooser and tool_user for speed. Same retry logic. Returns `(thinking, content, tool_calls)`.

---

## State (state.py)

Centralized mutable state:
- `iteration_count` — simple counter, resets per user task
- `memories` — persistent key-value store (auto-saved when dirty)
- `pdf_index` — vector index for ingested documents
- `working_directory` — where agent files go
- `short_term_goal` — the agent's current goal

No token budgets, no clocks, no model switching.

---

## Memory System

Persistent key-value store in `memories.json`. Each entry:

```json
{
  "key_name": {
    "text": "content (string or list of strings)",
    "created": "2025-01-15T10:30:00",
    "accessed": "2025-01-15T14:20:00",
    "access_count": 5
  }
}
```

- `save_memory(key, text)` — appends to existing key or creates new
- `edit_memory(key, text)` — overwrites (replaces, not appends)
- `search_memory(query, top_k)` — semantic search using embeddings
- `open_memory(key)` — direct access, updates timestamp
- Saved to disk after each completed user task and whenever dirty

The planner can suggest memory-focused subtasks (e.g. "search memory for identity and goals", "save what we learned"). All memory tools are available to the executor via the memory tool group.

---

## Document System

### Ingestion

```
PDF/CSV → extract text → adaptive chunking → embed each chunk → store in pdf_index.json
```

Chunk sizes adapt to document length:

| Pages | Chunk Size | Overlap | Page Step |
|-------|-----------|---------|-----------|
| ≤10   | 300 chars | 50      | 1         |
| ≤50   | 600 chars | 80      | 1         |
| ≤150  | 1000 chars| 100     | 2         |
| >150  | 1500 chars| 150     | 3         |

### Querying

`query_documents(query, top_k)` embeds the query and finds the most similar chunks via cosine similarity.

### Auto-download

`search_and_download_files(query, filetype)` searches DuckDuckGo, downloads matching files, and auto-ingests them.

---

## Text & Code Generation

### Text Generation

- `write_text(filename, prompt)` — generates written text (articles, posts, essays, docs, creative writing) using the planner model with memory context, saves to working directory
- `edit_text(filename, prompt)` — reads an existing text file, sends it with editing instructions to the planner model with memory context, overwrites the file
- `write_text_from_source(filename, source_filename, prompt)` — reads a source file as reference and generates new text in a separate output file. Designed for multi-stage writing pipelines (e.g. outline → draft → polished work). The source file is not modified.
- `post_file_to_social_media(community, title, filename)` — reads a file from the working directory and posts its contents to a Moltbook community

All three writing tools (`write_text`, `edit_text`, `write_text_from_source`) have a 15-minute timeout (configurable via `download_subtask_timeout_seconds`).

### Code Generation

Uses a dedicated code model (separate from conversation models).

- `generate_code(filename, prompt)` — generates code, saves to working directory
- `generate_code_edit(filename, prompt)` — reads existing file, sends with prompt, overwrites

Code is written but **never executed**. The code model is tested on first use; if unavailable, falls back to `code_model_fallback`.

---

## Dynamic Tool Creation

The agent can create tools at runtime via `create_tool(function_name, description, code)`:

1. Agent writes a Python function as a string
2. Code is shown to the user for approval (60s timeout)
3. If approved: `exec(code)` → registered in `available_functions` → persisted to `custom_tools.json`
4. Survives restarts via `load_custom_tools()` at startup
5. Remove with `remove_custom_tool(function_name)` (also requires approval)

---

## TUI (tui.py)

The agent runs in a fullscreen Rich terminal UI by default. The TUI shows all three phases in real-time with auto-scrolling panels.

### Layout

```
┌─ Header ─────────────────────────────────────────────────┐
│ Task-Loop Agent  qwen3.5:9b      PLANNING  Loop 1/20     │
├──────────────────────┬───────────────────────────────────┤
│ Thinking             │ Response                           │
│ [streaming dim text] │ [streaming answer text]            │
│ (auto-scrolls)       │ (auto-scrolls)                     │
├──────────────────────┴──────┬────────────────────────────┤
│ Subtasks                    │ Tool Calls                  │
│ OK 1. Search the web        │ search_web("python")        │
│ >> 2. Download PDFs          │   -> 5 results              │
│    3. Summarize              ├────────────────────────────┤
│ (scrolls to current)        │ Log                          │
│                              │ 14:32:01 Planning started   │
│                              │ 14:32:15 Group: web_search  │
├──────────────────────────────┴───────────────────────────┤
│ Task> type here_                                          │
└──────────────────────────────────────────────────────────┘
```

### Architecture

- **`TUIState`** — dataclass holding all display state (phase, thinking/response text, subtasks, tool calls, log, input state)
- **`build_layout(state, height, width)`** — builds a Rich Layout from the current state, calculates visible lines from terminal dimensions so all panels auto-scroll
- **`AgentTUI`** — controller class with callbacks for streaming chunks, tool logs, and phase transitions
- **`Rich Live`** — fullscreen (`screen=True`), refreshes at 8 fps, stays visible the entire session
- **Input** — read character-by-character in a background thread; typed text appears in the footer panel in real-time (supports backspace, Ctrl+U, Ctrl+W)

### Scrolling & Wrapping

All panels auto-scroll to show the latest content:
- **Thinking/Response** — tail the last N lines based on terminal height, wrap-aware line counting
- **Subtasks** — window around the currently running subtask
- **Tool Calls/Log** — show most recent entries that fit

The subtasks, tool calls, and log panels use `Text(no_wrap=True, overflow="ellipsis")` so Rich crops long lines instead of wrapping. This ensures each entry takes exactly 1 visual row and scroll calculations stay accurate.

### Streaming

`providers.py` accepts an `on_chunk` callback:
- `on_chunk('thinking_start', '')` — thinking begins
- `on_chunk('thinking', text)` — thinking token
- `on_chunk('answer_start', '')` — answer begins
- `on_chunk('content', text)` — answer token

The TUI's `on_stream_chunk` method updates `TUIState` and triggers a refresh.

### Tool Output Capture

`tools.py` uses `_log()` instead of `print()`. The TUI sets a callback via `set_log_callback()` that routes messages to the Log panel. Similarly, `set_input_callback()` routes user input prompts (from `check_in` and tool approval) through the TUI's footer input system. Without these callbacks, both fall back to stdout/stdin.

### User Interaction in TUI

When `check_in` is called (by the tool user during execution), the question appears as a prompt in the TUI footer. The user types their response (visible in real-time), presses Enter, and the response is returned to the calling tool.

### Plain Mode

Run with `--no-tui` to get the original stdout-printing behavior:

```bash
python main.py --no-tui
```

---

## System Prompt (system_prompt.md)

The system prompt is loaded fresh at the start of each user task and injected into both the planner and verifier messages. It defines:

- **Identity** — the agent's personality (curious, resourceful, driven) and that identity persists through memory
- **Loop structure** — PLANNER → TOOL GROUP CHOOSER → TOOL USER → VERIFIER → re-plan
- **Memory as identity** — what to remember (opinions, preferences, decisions, failures, user patterns)

The system prompt is kept intentionally short (~6 lines) and role-neutral — it's shared by both planner and verifier. Role-specific instructions (e.g. "you are the PLANNER", "you are the VERIFIER") are appended by `build_planner_messages()` and `build_verifier_messages()` respectively. This keeps the prompt concise for small models (8-9B) that lose coherence with verbose instructions.

---

## Running

```bash
# Set env vars (only needed for social media features)
export MOLTBOOK_API_KEY="..."

# Pull required models
ollama pull qwen3.5:9b
ollama pull ministral-3:3b
ollama pull qwen3:4b
ollama pull qwen2.5-coder:7b
ollama pull embeddinggemma

# Run (TUI mode, default)
python main.py

# Run (plain stdout mode)
python main.py --no-tui
```

The agent prompts for tasks interactively. Type a task, watch the planner break it down, the executor call tools, and the verifier confirm completion.

---

## Troubleshooting

### Planner returns empty output
The planner receives `tools=None` so this shouldn't happen with tool-related issues. Check that the model is running (`ollama list`) and that the context window isn't exceeded. The retry logic in `providers.py` will attempt recovery on Ollama errors.

### Ollama JSON parse errors
Some models (e.g. qwen3.5 with tool schemas) produce malformed JSON that crashes Ollama. The provider layer retries without tools on first failure, then returns empty results after 2 retries. If persistent, switch to a model with better tool calling support (qwen3 family is reliable).

### "Model not found" errors
Pull the model: `ollama pull <model-name>`

### Ollama connection errors
Ensure Ollama is running: `ollama serve`

### Memory/PDF index corruption
Delete `memories.json` or `pdf_index.json`. The agent starts fresh.

### High memory usage
Large PDF indexes consume RAM. Prune old entries or reduce `ollama_context_window`.
