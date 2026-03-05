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
- `models.tool_group_chooser`
- `models.tool_user`
- `models.verifier`
- `system_prompt_path`
- `planner_step_prompt_path`
- `tool_group_chooser_prompt_path`
- `tool_user_prompt_path`
- `verifier_prompt_path`
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

### Conversational chat mode (single model)

```bash
python chat.py
```

This uses `chat_model` from `config.yaml` (or falls back to planner model) and can use tools for web browsing, memory, and document download/read workflows.

## 5) Full pipeline (including failures and retries)

For each user task, the agent runs a planner → executor → verifier loop with retries at multiple layers.

### Step-by-step flow

1. **Plan phase (text-only planner)**
   - The planner receives system instructions, working-directory context, and tool-group summaries.
   - It returns a numbered list of subtasks (no tool calls are allowed in this phase).
   - Subtasks are parsed into an ordered execution list.

2. **Execute phase (per subtask)**
   - **2a. Tool group chooser** selects one group for the current subtask.
   - **2b. Tool user** gets only that group's tool schemas.
   - **2c. Tool calls execute** and results are appended to shared context so later subtasks can use earlier outputs.
   - This repeats until all subtasks succeed, or one subtask fails.

3. **Verify phase**
   - The verifier reviews the original user goal plus collected results.
   - If it returns `COMPLETE`, the run ends successfully.
   - Otherwise, remaining work is fed back into planning for another loop.

### Where retries happen

- **Model-call retries (provider level):**
  - Planner/chooser/tool-user/verifier model calls use built-in retry logic.
  - On Ollama/API errors, calls are retried (with safe fallback behavior where applicable).

- **Subtask-level recovery (execution level):**
  - If a subtask has an error result, no tool call, invalid group/tools, or timeout, execution stops immediately for that loop.
  - The agent does **not** continue to later subtasks in that failed pass.

- **Replanning retries (task level):**
  - After a failed subtask (or verifier `INCOMPLETE`), the agent starts a new planning loop.
  - Replanning uses a **sanitized retry prompt** that includes:
    - high-level failure context (failed subtask + likely failing tool),
    - successful prior outputs,
    - and no raw error payload echoing.
  - This reduces repeated failure loops caused by overfitting to noisy tool errors.

- **Loop limit / user control:**
  - Replan/verify loops are capped by `max_verification_loops`.
  - At the limit, the system asks whether to continue, stop, or provide new instructions.

### Timeout behavior

- Each subtask has a timeout budget.
- Some heavy groups (for example download/document flows) can use longer timeout settings.
- Timeout failures are treated as subtask failures and trigger replanning.

## 6) Useful files

- `main.py` — loop orchestration (planner/executor/verifier)
- `tools.py` — tool implementations
- `tool_groups.py` — grouping/selectability of tools
- `providers.py` — Ollama query helpers
- `state.py` — memory/index persistence
- `system_prompt.md` — system behavior guidance

## 7) How to edit code manually

If you want to make changes yourself (instead of asking the agent to do it), use this quick loop:

1. Open the project in your editor.
2. Make a small change in one file.
3. Run the app to verify behavior.
4. Repeat until complete.

### Common manual edits

- **Change agent behavior text**: edit `system_prompt.md`.
- **Adjust planning/execution instructions**: edit files in `prompts/`.
- **Tune runtime/model settings**: edit `config.yaml`.
- **Add or modify tools**: edit `tools.py` and update grouping in `tool_groups.py`.
- **Adjust main loop behavior**: edit `main.py`.

### Recommended safe workflow

```bash
# 1) Create a branch for your edits
git checkout -b docs-or-code-change

# 2) Run the app before edits (baseline)
python main.py --no-tui

# 3) After each set of edits, run again
python main.py --no-tui

# 4) Review changes
git diff

# 5) Commit when satisfied
git add .
git commit -m "Describe your manual code edit"
```

Tips:
- Prefer small, focused commits.
- If a change affects prompts/config, test with a short real task.
- If a change affects tools, test the exact tool flow you modified.

## 8) Troubleshooting

- **No tool calls generated**: adjust the `tool_user` model or improve prompt specificity.
- **Frequent timeouts**: increase timeout settings in `config.yaml`.
- **Model errors**: verify Ollama is running and required models are pulled.
- **TUI input issues**: run in legacy mode.
