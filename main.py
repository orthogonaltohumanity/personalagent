import json
import re
import sys
import time
import threading

from config import cfg, get_model, get_model_think, resolve_path
from state import state
from tools import build_tool_registry, available_functions, set_log_callback, set_input_callback, set_stream_callback
from tool_groups import get_group_summary, get_tools_in_group, get_group_names, TOOL_GROUPS
from providers import stream_ollama, query_ollama

from tui import AgentTUI, Phase

# ── Build tool registry ──────────────────────────────────────────────────────

build_tool_registry()

# ── Helpers ──────────────────────────────────────────────────────────────────

def run_with_timeout(fn, timeout_seconds):
    """Run fn() in a thread with a timeout. Returns (result, timed_out).
    If timed out, result is None. The thread may continue running in the background."""
    result_box = [None]
    error_box = [None]

    def wrapper():
        try:
            result_box[0] = fn()
        except Exception as e:
            error_box[0] = e

    t = threading.Thread(target=wrapper, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)
    if t.is_alive():
        return None, True
    if error_box[0]:
        raise error_box[0]
    return result_box[0], False


def load_system_prompt():
    system_prompt_path = resolve_path(cfg['system_prompt_path'])
    try:
        with open(system_prompt_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""


def load_step_prompt(path_key: str, default_text: str = ""):
    """Load a configurable step prompt from config path key."""
    path_value = cfg.get(path_key)
    if not path_value:
        return default_text
    try:
        with open(resolve_path(path_value), 'r') as f:
            content = f.read().strip()
            return content if content else default_text
    except FileNotFoundError:
        return default_text


def render_prompt_template(text: str, values: dict):
    """Simple token replacement for prompt files using {{token}} syntax."""
    rendered = text
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


def parse_subtasks(text):
    """Parse planner output into a clean subtask list.

    Accepts JSON (`{"subtasks": [...]}` or list), numbered/bulleted text,
    and strict bracket-tag lines like `[web_search] ...`.
    """
    if not text:
        return []

    normalized = text.strip()

    # Strip optional fenced code wrapper if the model returns markdown.
    fence_match = re.match(r"^```(?:json|text)?\s*(.*?)\s*```$", normalized, re.DOTALL | re.IGNORECASE)
    if fence_match:
        normalized = fence_match.group(1).strip()

    try:
        data = json.loads(normalized)
        if isinstance(data, dict) and 'subtasks' in data:
            return [str(s).strip() for s in data['subtasks'] if str(s).strip()]
        if isinstance(data, list):
            return [str(s).strip() for s in data if str(s).strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    lines = normalized.split('\n')
    subtasks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Preferred strict planner format: [group] Action
        if re.match(r'^\[[a-z_]+\]\s+.+', line, re.IGNORECASE):
            subtasks.append(line)
            continue

        match = re.match(r'^(?:\d+[\.\)]\s*|[-*]\s+)(.+)', line)
        if match:
            subtasks.append(match.group(1).strip())

    # Fallback: keep non-empty text as single subtask.
    return subtasks if subtasks else [normalized]


def pick_group(selector_content):
    """Parse a tool group name from the chooser's response.

    Expected format is strict: first line should be either '[group_name]' or
    'group_name'. This reduces accidental substring matches.
    """
    first_line = selector_content.strip().split('\n')[0].strip().lower()

    bracket_match = re.match(r'^\[([a-z_]+)\]$', first_line)
    if bracket_match:
        candidate = bracket_match.group(1)
        if candidate in get_group_names():
            return candidate

    if first_line in get_group_names():
        return first_line

    return None


def execute_tool_calls(tool_calls, group_name, tui=None):
    """Execute tool calls. Returns list of (fn_name, fn_args, result) tuples."""
    results = []
    max_calls = cfg.get('max_tool_calls_per_step', 10)
    try:
        for tc in tool_calls[:max_calls]:
            if isinstance(tc, dict):
                fn_name = tc['function']['name']
                fn_args = tc['function']['arguments']
            else:
                fn_name = tc.function.name
                fn_args = tc.function.arguments

            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except json.JSONDecodeError:
                    fn_args = {}

            if fn_name in available_functions:
                if tui:
                    tui.state.add_log(f"Calling {fn_name}({fn_args})")
                    tui._refresh()
                else:
                    print(f"  Calling {fn_name}({fn_args})")
                try:
                    result = available_functions[fn_name](**fn_args)
                except Exception as e:
                    result = f"Error: {e}"
                result_str = str(result)
                results.append((fn_name, fn_args, result_str))
                if tui:
                    tui.record_tool_call(fn_name, fn_args, result_str[:300])
                else:
                    print(f"    -> {result_str[:300]}")
            else:
                msg = f"Tool '{fn_name}' not found, skipping"
                if tui:
                    tui.state.add_log(msg)
                else:
                    print(f"  {msg}")
    finally:
        state.save_memories_if_dirty()
    return results


def get_planner_memory_context():
    """Preload planner context with fixed identity/personality memory query."""
    query = "identity personality preferences goals"
    search_memory = available_functions.get('search_memory')
    if not search_memory:
        return ""
    try:
        memory_context = search_memory(query=query, top_k=3)
    except Exception:
        return ""
    if not memory_context or memory_context in ("No memories stored yet.", "No memories found."):
        return ""
    return f"\n\nPreloaded memory recall ({query}):\n{memory_context}"


def detect_subtask_failure(results):
    """Check if tool call results indicate a failure.
    Returns (failed: bool, reason: str)."""
    if not results:
        return True, "No tool calls were executed"
    for fn_name, fn_args, result_str in results:
        if result_str.startswith("Error:"):
            return True, f"{fn_name} failed: {result_str}"
        lower = result_str.lower()
        if lower in ('none', '', 'null'):
            return True, f"{fn_name} returned empty result"
        if "not found" in lower and "error" in lower:
            return True, f"{fn_name}: {result_str[:200]}"
    return False, ""


def build_retry_planning_input(user_task, failure_reason, all_results):
    """Build a re-planning prompt after execution failure.

    Include the direct failure/error details and require strategy diversity on
    retry so the planner does not repeat the same failing path.
    """
    summarized_results = []
    for task_desc, results in all_results:
        if not results:
            continue
        successful_calls = [
            f"{fn}→{res[:100]}"
            for fn, _, res in results
            if not str(res).startswith("Error:")
        ]
        if successful_calls:
            summarized_results.append(f"  {task_desc}: {', '.join(successful_calls)}")

    # Keep only structured, high-level failure context.
    # Expected format: Subtask N '...' failed: ...
    subtask_match = re.search(r"Subtask\s+\d+\s+'([^']+)'", failure_reason)
    failed_subtask = subtask_match.group(1).strip() if subtask_match else "(unknown subtask)"

    tool_match = re.search(r"failed:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+failed", failure_reason)
    failed_tool = tool_match.group(1).strip() if tool_match else "(unknown tool)"

    raw_failure_lines = []
    for task_desc, results in all_results:
        if task_desc != failed_subtask:
            continue
        for fn, args, res in results:
            raw_failure_lines.append(f"  {fn}({args}) -> {res}")

    retry_prompt = (
        f"Original task: {user_task}\n\n"
        f"Previous execution failed.\n"
        f"Failed subtask: {failed_subtask}\n"
        f"Likely failing tool: {failed_tool}\n"
        f"Raw failure message: {failure_reason}\n\n"
        f"Re-plan with strategy diversity. Try a meaningfully different method than the failed attempt, such as:\n"
        f"- choose a different tool group,\n"
        f"- change tool order or intermediate artifacts,\n"
        f"- gather missing evidence before acting,\n"
        f"- split the failed work into smaller safer subtasks.\n"
        f"Do not repeat the same failing tool path unchanged."
    )
    if raw_failure_lines:
        retry_prompt += "\n\nDirect tool outputs from failed subtask (including errors):\n" + "\n".join(raw_failure_lines)
    if summarized_results:
        retry_prompt += "\n\nUseful successful results so far:\n" + "\n".join(summarized_results)
    return retry_prompt


def build_planner_messages(system_prompt, planning_input):
    group_summary = get_group_summary(include_tools=False)
    memory_context = get_planner_memory_context()
    default_planner_prompt = (
        f"You are the PLANNER. You produce ONLY a numbered subtask list — nothing else.\n"
        f"You do NOT call tools, write code, or perform tasks. A separate TOOL GROUP CHOOSER + TOOL USER pair executes your plan.\n\n"
        f"RULES:\n"
        f"- REQUIRED FORMAT: every subtask must start with a bracketed tool group tag followed by an action, e.g., '[web_search] Find ...'.\n"
        f"- Use exact group tags from the list below; do not invent new group names.\n"
        f"- Use memory tools distinctly: search_memory/open_memory to retrieve, save_memory to add new facts, edit_memory to correct existing facts.\n"
        f"- For writing subtasks, specify intent clearly: write_text for net-new writing, write_text_from_source when based on a file, edit_text for revising an existing file.\n"
        f"- Each subtask = one tool group. Be specific about what the executor should do.\n"
        f"- Prefer subtasks that naturally require multiple tool calls when evidence gathering + action are both needed.\n"
        f"- If re-planning after a failure, use a diverse strategy rather than repeating the same approach.\n"
        f"- Max {cfg['max_subtasks']} subtasks. 2-5 per phase — the re-plan loop handles the rest.\n\n"
        f"{group_summary}"
    )
    planner_template = load_step_prompt('planner_step_prompt_path', default_planner_prompt)
    planner_prompt = render_prompt_template(planner_template, {
        'max_subtasks': cfg['max_subtasks'],
        'group_summary': group_summary,
    })
    return [
        {'role': 'system', 'content': (
            f"{system_prompt}\n\n"
            f"WORKING DIRECTORY: {state.working_directory}\n"
            f"Short term goal: {state.short_term_goal}\n\n"
            f"Use this preloaded memory context when relevant:{memory_context}\n\n"
            f"{planner_prompt}"
        )},
        {'role': 'user', 'content': planning_input}
    ]


def build_tool_executor_system_prompt(chosen_group):
    """System prompt for tool execution, configurable via prompt file."""
    default_tool_user_prompt = (
        f"You are the tool user. You MUST respond ONLY with tool calls — no text, no explanations, no commentary. "
        f"Do NOT write content yourself. Use the provided tools to accomplish the subtask. "
        f"Use up to {cfg['max_tools_per_task']} tool calls. Prefer 2+ tool calls when they improve quality (e.g., retrieve then write/save). "
        f"Use memory tools distinctly: search/open for recall, save for new durable knowledge, edit for corrections."
    )
    if chosen_group == 'text_generation':
        default_tool_user_prompt += (
            "\n\nFor text_generation group:"
            "\n- Use write_text for net-new writing when no source file is required."
            "\n- Use write_text_from_source when writing must be based on an existing source/reference file."
            "\n- Use edit_text only to revise an existing output file."
            "\n- Prefer source-grounded writing over free-form writing when either can satisfy the subtask."
            "\n- When practical, chain multiple calls (e.g., read_file -> write_text_from_source -> edit_text) for better quality."
        )

    template = load_step_prompt('tool_user_prompt_path', default_tool_user_prompt)
    return render_prompt_template(template, {
        'max_tools_per_task': cfg['max_tools_per_task'],
        'chosen_group': chosen_group,
    })


def build_selector_messages(subtask, all_results):
    context_parts = [f"Current subtask: {subtask}"]
    recommended_group = None
    subtask_lower = subtask.lower()
    tag_match = re.search(r'\[([a-z_]+)\]', subtask_lower)
    if tag_match:
        tagged = tag_match.group(1)
        if tagged in get_group_names():
            recommended_group = tagged
    if recommended_group:
        context_parts.append(f"Planner-recommended group: {recommended_group}")
    if all_results:
        context_parts.append("\nPrevious subtask results:")
        for prev_task, prev_results in all_results:
            context_parts.append(f"  Task: {prev_task}")
            for fn, args, res in prev_results:
                context_parts.append(f"    {fn}: {res[:200]}")

    group_list = "\n".join(
        f"  {g}: {TOOL_GROUPS[g]['description']}"
        for g in get_group_names()
    )
    default_chooser_prompt = (
        f"You are a tool group chooser. Given a subtask, pick the best tool group based on the planner's subtask description and any recommended group.\n\n"
        f"Available groups:\n{group_list}\n\n"
        f"Selection rules:\n"
        f"- Choose text_generation for writing/editing tasks.\n"
        f"- Distinguish writing tools: write_text (net-new), write_text_from_source (source-based), edit_text (revise existing).\n"
        f"- Distinguish memory tools: search/open for retrieval, save for new durable information, edit for corrections.\n"
        f"- Prefer multiple tool calls when helpful (e.g., find/read context, then write, then save key results to memory).\n\n"
        f"Reply with ONLY one line in this exact format: [group_name]. Do not call tools and do not add any other text."
    )
    chooser_template = load_step_prompt('tool_group_chooser_prompt_path', default_chooser_prompt)
    chooser_prompt = render_prompt_template(chooser_template, {
        'group_list': group_list,
        'max_tools_per_task': cfg['max_tools_per_task'],
    })
    messages = [
        {'role': 'system', 'content': chooser_prompt},
        {'role': 'user', 'content': "\n".join(context_parts)}
    ]
    return messages, context_parts


def build_verifier_messages(system_prompt, user_task, all_results):
    results_summary = []
    for task_desc, results in all_results:
        results_summary.append(f"Subtask: {task_desc}")
        if results:
            for fn, args, res in results:
                results_summary.append(f"  {fn}({args}) -> {res[:300]}")
        else:
            results_summary.append("  (no tool calls executed)")
    default_verifier_prompt = (
        "You are the VERIFIER. Review whether the original task has been completed.\n"
        "If the task is COMPLETE, respond with exactly 'COMPLETE' on the first line, followed by a summary.\n"
        "If the task is INCOMPLETE, respond with exactly 'INCOMPLETE' on the first line, followed by a description of what still needs to be done."
    )
    verifier_template = load_step_prompt('verifier_prompt_path', default_verifier_prompt)
    verifier_prompt = render_prompt_template(verifier_template, {})
    return [
        {'role': 'system', 'content': (
            f"{system_prompt}\n\n"
            f"{verifier_prompt}"
        )},
        {'role': 'user', 'content': (
            f"Original task: {user_task}\n\n"
            f"Results:\n" + "\n".join(results_summary)
        )}
    ]


# ── TUI Main Loop ───────────────────────────────────────────────────────────

def main_tui():
    tui = AgentTUI(
        models={r: get_model(r) for r in ('planner', 'tool_group_chooser', 'tool_user', 'verifier')},
        max_loops=cfg['max_verification_loops']
    )

    set_log_callback(tui.on_tool_log)
    set_input_callback(lambda prompt: tui.get_user_input(prompt))
    set_stream_callback(tui.on_stream_chunk)

    # Start TUI immediately — stays fullscreen the entire time
    tui.start()
    tui.set_status("Type a task below and press Enter")
    tui.state.add_log("Agent started")
    tui.state.add_log(f"Planner: {get_model('planner')}")
    tui.state.add_log(f"Tool Group Chooser: {get_model('tool_group_chooser')}")
    tui.state.add_log(f"Tool User: {get_model('tool_user')}")
    tui.state.add_log(f"Verifier: {get_model('verifier')}")

    try:
        while True:
            # Input prompt shows in the TUI footer, stdin read in background thread
            try:
                user_task = tui.get_user_input("Task> ")
            except (EOFError, KeyboardInterrupt):
                tui.stop()
                print("Goodbye!")
                state.save_memories()
                return

            if not user_task:
                continue

            state.reset_iteration()
            system_prompt = load_system_prompt()
            verification_loops = 0
            planning_input = user_task
            tui.reset_for_task()
            tui.set_status(f"Task: {user_task[:60]}")
            tui.state.add_log(f"New task: {user_task[:80]}")

            while verification_loops <= cfg['max_verification_loops']:
                # ── PLAN PHASE ────────────────────────────────────────
                tui.set_loop(verification_loops + 1)
                tui.set_phase(Phase.PLANNING, get_model('planner'))
                tui.state.add_log(f"Planning (loop {verification_loops + 1})")

                planner_messages = build_planner_messages(system_prompt, planning_input)
                _, planner_content, _ = stream_ollama(
                    get_model('planner'), planner_messages,
                    tools=None,
                    think=get_model_think('planner'), on_chunk=tui.on_stream_chunk
                )

                subtasks = parse_subtasks(planner_content)
                subtasks = subtasks[:cfg['max_subtasks']]
                tui.set_subtasks(subtasks)
                tui.state.add_log(f"Plan: {len(subtasks)} subtask(s)")

                # ── EXECUTE PHASE ─────────────────────────────────────
                tui.set_phase(Phase.EXECUTING, get_model('tool_group_chooser'))

                all_results = []
                execution_failed = False
                failure_reason = ""
                default_timeout = cfg.get('subtask_timeout_seconds', 300)
                download_timeout = cfg.get('download_subtask_timeout_seconds', 900)
                slow_groups = {'web_search', 'document_processing', 'text_generation'}

                for i, subtask in enumerate(subtasks):
                    state.increment_iteration()
                    tui.start_subtask(i)
                    tui.set_status(f"Subtask {i+1}/{len(subtasks)}: {subtask[:50]}")
                    tui.state.add_log(f"Subtask {i+1}/{len(subtasks)}: {subtask[:60]}")

                    # ── Group selection (fast, no timeout needed) ──
                    selector_messages, context_parts = build_selector_messages(subtask, all_results)
                    _, selector_content, _ = query_ollama(
                        get_model('tool_group_chooser'), selector_messages, think=False
                    )

                    chosen_group = pick_group(selector_content)
                    if not chosen_group:
                        tui.state.add_log(f"Could not determine group, defaulting to web_search")
                        chosen_group = 'web_search'

                    tui.set_subtask_group(i, chosen_group)
                    tui.state.add_log(f"Group: {chosen_group}")
                    tui.state.add_log(f"Tool user model: {get_model('tool_user')}")

                    group_tools = get_tools_in_group(chosen_group)
                    if not group_tools:
                        tui.state.add_log(f"No tools in group '{chosen_group}'")
                        all_results.append((subtask, []))
                        tui.finish_subtask(i, success=False)
                        execution_failed = True
                        failure_reason = f"Subtask {i+1} '{subtask}' failed: no tools in group '{chosen_group}'"
                        break

                    # ── Tool execution (timeout-wrapped) ──
                    tui.set_phase(Phase.EXECUTING, get_model('tool_user'))
                    timeout = download_timeout if chosen_group in slow_groups else default_timeout

                    def run_tool_calls(_subtask=subtask, _context_parts=context_parts, _group_tools=group_tools, _chosen_group=chosen_group):
                        tool_messages = [
                            {'role': 'system', 'content': build_tool_executor_system_prompt(_chosen_group)},
                            {'role': 'user', 'content': "\n".join(_context_parts)}
                        ]
                        _, _, tool_calls = query_ollama(
                            get_model('tool_user'), tool_messages,
                            tools=_group_tools, think=False
                        )
                        if not tool_calls:
                            return 'no_calls', []
                        results = execute_tool_calls(tool_calls, _chosen_group, tui=tui)
                        return 'ok', results

                    outcome, timed_out = run_with_timeout(run_tool_calls, timeout)

                    if timed_out:
                        tui.state.add_log(f"Subtask timed out after {timeout}s")
                        all_results.append((subtask, []))
                        tui.finish_subtask(i, success=False)
                        execution_failed = True
                        failure_reason = f"Subtask {i+1} '{subtask}' timed out after {timeout}s"
                        break

                    status, results = outcome

                    if status == 'no_calls':
                        tui.state.add_log("No tool calls generated")
                        all_results.append((subtask, []))
                        tui.finish_subtask(i, success=False)
                        execution_failed = True
                        failure_reason = f"Subtask {i+1} '{subtask}' failed: no tool calls generated"
                        break
                    else:
                        failed, reason = detect_subtask_failure(results)
                        all_results.append((subtask, results))
                        if failed:
                            tui.finish_subtask(i, success=False)
                            tui.state.add_log(f"Subtask failed: {reason}")
                            execution_failed = True
                            failure_reason = f"Subtask {i+1} '{subtask}' failed: {reason}"
                            break
                        tui.finish_subtask(i, success=True)

                # If execution failed, skip verification and re-plan immediately
                if execution_failed:
                    verification_loops += 1
                    if verification_loops > cfg['max_verification_loops']:
                        tui.state.add_log("Max loops reached after execution failure")
                        tui.set_status("Max loops reached — asking user")
                        tui.set_phase(Phase.IDLE)
                        try:
                            user_input = tui.get_user_input("Execution failed. Continue? (y/n or new instructions): ")
                        except (EOFError, KeyboardInterrupt):
                            break
                        if user_input.lower() in ('n', 'no', ''):
                            break
                        planning_input = (
                            user_input
                            if user_input.lower() not in ('y', 'yes')
                            else build_retry_planning_input(user_task, failure_reason, all_results)
                        )
                        verification_loops = 0
                    else:
                        tui.state.add_log(f"Execution failed — re-planning (loop {verification_loops + 1})")
                        planning_input = build_retry_planning_input(user_task, failure_reason, all_results)
                    continue

                # ── VERIFY PHASE ──────────────────────────────────────
                tui.set_phase(Phase.VERIFYING, get_model('verifier'))
                tui.set_status("Verifying task completion...")
                tui.state.add_log("Verifying task completion")

                verifier_messages = build_verifier_messages(system_prompt, user_task, all_results)
                _, verifier_content, _ = stream_ollama(
                    get_model('verifier'), verifier_messages,
                    think=get_model_think('verifier'), on_chunk=tui.on_stream_chunk
                )

                first_word = verifier_content.strip().split()[0].upper() if verifier_content.strip() else ''
                if first_word == 'COMPLETE':
                    tui.state.add_log("TASK COMPLETE")
                    tui.set_status("Task complete!")
                    tui.set_phase(Phase.IDLE)
                    state.save_memories_if_dirty()
                    break
                else:
                    verification_loops += 1
                    if verification_loops > cfg['max_verification_loops']:
                        tui.state.add_log(f"Max verification loops reached")
                        tui.set_status("Max loops reached — asking user")
                        tui.set_phase(Phase.IDLE)
                        try:
                            user_input = tui.get_user_input("Incomplete. Continue? (y/n or new instructions): ")
                        except (EOFError, KeyboardInterrupt):
                            break
                        if user_input.lower() in ('n', 'no', ''):
                            break
                        planning_input = user_input if user_input.lower() not in ('y', 'yes') else verifier_content
                        verification_loops = 0
                    else:
                        remaining_work = verifier_content.split('\n', 1)
                        planning_input = (
                            f"Original task: {user_task}\n\n"
                            f"Previous attempt was incomplete. What still needs to be done:\n"
                            f"{remaining_work[1] if len(remaining_work) > 1 else remaining_work[0]}"
                        )
                        tui.state.add_log(f"INCOMPLETE — re-planning (loop {verification_loops + 1})")

            state.save_memories()

    finally:
        tui.stop()
        tui.console.print("[bold]Goodbye![/]")


# ── Legacy (no-TUI) Main Loop ───────────────────────────────────────────────

def main_legacy():
    print("=" * 60)
    print("  Sequential Task-Loop Agent")
    print("  Planner -> Executor -> Verifier")
    print("=" * 60)
    print(f"  Planner:       {get_model('planner')}")
    print(f"  Tool Group Chooser: {get_model('tool_group_chooser')}")
    print(f"  Tool User:          {get_model('tool_user')}")
    print(f"  Verifier:      {get_model('verifier')}")
    print("=" * 60)

    while True:
        print()
        try:
            user_task = input("Task> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            state.save_memories()
            break

        if not user_task:
            continue

        state.reset_iteration()
        system_prompt = load_system_prompt()
        verification_loops = 0
        planning_input = user_task

        while verification_loops <= cfg['max_verification_loops']:
            # PLAN
            print(f"\n{'─'*40}")
            print(f"PLANNING (loop {verification_loops + 1})")
            print(f"{'─'*40}")

            planner_messages = build_planner_messages(system_prompt, planning_input)
            print(f"\n[Planner] ({get_model('planner')})")
            _, planner_content, _ = stream_ollama(
                get_model('planner'), planner_messages,
                tools=None, think=get_model_think('planner')
            )

            subtasks = parse_subtasks(planner_content)
            subtasks = subtasks[:cfg['max_subtasks']]
            print(f"\n[Plan] {len(subtasks)} subtask(s):")
            for i, st in enumerate(subtasks, 1):
                print(f"  {i}. {st}")

            # EXECUTE
            print(f"\n{'─'*40}")
            print("EXECUTING")
            print(f"{'─'*40}")

            all_results = []
            execution_failed = False
            failure_reason = ""
            default_timeout = cfg.get('subtask_timeout_seconds', 300)
            download_timeout = cfg.get('download_subtask_timeout_seconds', 900)
            slow_groups = {'web_search', 'document_processing', 'text_generation'}

            for i, subtask in enumerate(subtasks):
                state.increment_iteration()
                print(f"\n[Subtask {i+1}/{len(subtasks)}] {subtask}")

                # ── Group selection (fast, no timeout needed) ──
                selector_messages, context_parts = build_selector_messages(subtask, all_results)
                print(f"  [Tool Group Chooser] ({get_model('tool_group_chooser')})")
                _, selector_content, _ = query_ollama(
                    get_model('tool_group_chooser'), selector_messages, think=False
                )

                chosen_group = pick_group(selector_content)
                if not chosen_group:
                    print(f"  Could not determine group, defaulting to web_search")
                    chosen_group = 'web_search'
                print(f"  Selected group: {chosen_group}")

                group_tools = get_tools_in_group(chosen_group)
                if not group_tools:
                    print(f"  No tools in group '{chosen_group}'")
                    all_results.append((subtask, []))
                    execution_failed = True
                    failure_reason = f"Subtask {i+1} '{subtask}' failed: no tools in group '{chosen_group}'"
                    break

                # ── Tool execution (timeout-wrapped) ──
                timeout = download_timeout if chosen_group in slow_groups else default_timeout
                print(f"  [Tool User] ({get_model('tool_user')})")

                def run_tool_calls(_subtask=subtask, _context_parts=context_parts, _group_tools=group_tools, _chosen_group=chosen_group):
                    tool_messages = [
                        {'role': 'system', 'content': build_tool_executor_system_prompt(_chosen_group)},
                        {'role': 'user', 'content': "\n".join(_context_parts)}
                    ]
                    _, _, tool_calls = query_ollama(
                        get_model('tool_user'), tool_messages,
                        tools=_group_tools, think=False
                    )
                    if not tool_calls:
                        return 'no_calls', []
                    results = execute_tool_calls(tool_calls, _chosen_group)
                    return 'ok', results

                outcome, timed_out = run_with_timeout(run_tool_calls, timeout)

                if timed_out:
                    print(f"  [TIMED OUT] after {timeout}s")
                    all_results.append((subtask, []))
                    execution_failed = True
                    failure_reason = f"Subtask {i+1} '{subtask}' timed out after {timeout}s"
                    break

                status, results = outcome

                if status == 'no_calls':
                    print(f"  No tool calls generated")
                    all_results.append((subtask, []))
                    execution_failed = True
                    failure_reason = f"Subtask {i+1} '{subtask}' failed: no tool calls generated"
                    break
                else:
                    failed, reason = detect_subtask_failure(results)
                    all_results.append((subtask, results))
                    if failed:
                        print(f"  [FAILED] {reason}")
                        execution_failed = True
                        failure_reason = f"Subtask {i+1} '{subtask}' failed: {reason}"
                        break

            # If execution failed, skip verification and re-plan immediately
            if execution_failed:
                verification_loops += 1
                if verification_loops > cfg['max_verification_loops']:
                    print(f"\n[Max loops ({cfg['max_verification_loops']}) reached after execution failure]")
                    try:
                        user_input = input("Execution failed. Continue? (y/n or new instructions): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if user_input.lower() in ('n', 'no', ''):
                        break
                    planning_input = (
                        user_input
                        if user_input.lower() not in ('y', 'yes')
                        else build_retry_planning_input(user_task, failure_reason, all_results)
                    )
                    verification_loops = 0
                else:
                    print(f"\n[Execution failed — re-planning (loop {verification_loops + 1})]")
                    planning_input = build_retry_planning_input(user_task, failure_reason, all_results)
                continue

            # VERIFY
            print(f"\n{'─'*40}")
            print("VERIFYING")
            print(f"{'─'*40}")

            verifier_messages = build_verifier_messages(system_prompt, user_task, all_results)
            print(f"\n[Verifier] ({get_model('verifier')})")
            _, verifier_content, _ = stream_ollama(
                get_model('verifier'), verifier_messages, think=get_model_think('verifier')
            )

            first_word = verifier_content.strip().split()[0].upper() if verifier_content.strip() else ''
            if first_word == 'COMPLETE':
                print(f"\n[TASK COMPLETE]")
                state.save_memories_if_dirty()
                break
            else:
                verification_loops += 1
                if verification_loops > cfg['max_verification_loops']:
                    print(f"\n[Max verification loops ({cfg['max_verification_loops']}) reached]")
                    try:
                        user_input = input("Task incomplete. Continue? (y/n or new instructions): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if user_input.lower() in ('n', 'no', ''):
                        break
                    planning_input = user_input if user_input.lower() not in ('y', 'yes') else verifier_content
                    verification_loops = 0
                else:
                    remaining_work = verifier_content.split('\n', 1)
                    planning_input = (
                        f"Original task: {user_task}\n\n"
                        f"Previous attempt was incomplete. What still needs to be done:\n"
                        f"{remaining_work[1] if len(remaining_work) > 1 else remaining_work[0]}"
                    )
                    print(f"\n[INCOMPLETE — re-planning (loop {verification_loops + 1})]")

        state.save_memories()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if '--no-tui' in sys.argv:
        main_legacy()
    else:
        main_tui()
