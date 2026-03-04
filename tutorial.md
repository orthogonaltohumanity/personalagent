# PersonalAgent Tutorial

This tutorial shows how to set up and run the agent in both TUI and legacy terminal modes.

## 1) Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- Model(s) pulled that match your `config.yaml`

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Configure the agent

Edit `config.yaml` and check at least:

- `models.planner`
- `models.tool_selector`
- `models.verifier`
- `system_prompt_path`
- `working_directory`
- timeout values (`subtask_timeout_seconds`, etc.)

## 4) Run the agent

### TUI mode (default)

```bash
python main.py
```

Type your task at `Task>` and press Enter.

### Legacy mode (fallback)

If your terminal has issues with fullscreen TUI behavior:

```bash
python main.py --no-tui
```

## 5) How execution/replanning works

The loop is:

1. **Planner** creates subtasks.
2. **Tool selector/executor** picks a tool group and runs tool calls.
3. **Verifier** checks completeness.
4. If incomplete or failed, the agent replans.

### Failure handling behavior

When a subtask fails, replanning now uses a **sanitized retry prompt**:

- Includes high-level context (failed subtask + likely failing tool)
- Includes successful prior results
- Avoids injecting raw tool error payloads back into planning context

This reduces the chance of repeated first-subtask failures caused by error echoing.

## 6) Useful files

- `main.py` — loop orchestration (planner/executor/verifier)
- `tools.py` — tool implementations
- `tool_groups.py` — grouping/selectability of tools
- `providers.py` — Ollama query helpers
- `state.py` — memory/index persistence
- `system_prompt.md` — system behavior guidance

## 7) Troubleshooting

- **No tool calls generated**: adjust model/tool-selector model or improve prompt specificity.
- **Frequent timeouts**: increase timeout settings in `config.yaml`.
- **Model errors**: verify Ollama is running and required models are pulled.
- **TUI input issues**: run in legacy mode.
