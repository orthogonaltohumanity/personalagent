import json
import re
import sys
import time
import threading

from config import cfg, get_model, get_model_think
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
    with open(cfg['system_prompt_path'], 'r') as f:
        return f.read()


def parse_subtasks(text):
    """Parse a numbered list or JSON subtask list from planner output."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and 'subtasks' in data:
            return data['subtasks']
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    lines = text.strip().split('\n')
    subtasks = []
    for line in lines:
        line = line.strip()
        match = re.match(r'^(?:\d+[\.\)]\s*|[-*]\s+)(.+)', line)
        if match:
            subtasks.append(match.group(1).strip())
    return subtasks if subtasks else [text.strip()]


def pick_group(selector_content):
    """Parse a tool group name from the selector's response."""
    first_line = selector_content.strip().split('\n')[0].strip().lower()
    for gname in get_group_names():
        if gname in first_line:
            return gname
    for gname in get_group_names():
        if gname in selector_content.lower():
            return gname
    return None


def execute_tool_calls(tool_calls, group_name, tui=None):
    """Execute tool calls. Returns list of (fn_name, fn_args, result) tuples."""
    results = []
    max_calls = cfg.get('max_tool_calls_per_step', 10)
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
    return results


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

    Important: do not feed raw `Error:` strings back into the planner.
    Those error prefixes tend to get echoed into the next plan and can cause
    the first subtask of subsequent loops to repeat the same failure.
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

    clean_failure_reason = re.sub(r"\bError:\s*", "", failure_reason)
    clean_failure_reason = clean_failure_reason.strip()

    retry_prompt = (
        f"Original task: {user_task}\n\n"
        f"Execution failed during subtask execution:\n{clean_failure_reason}\n\n"
        f"Do not repeat the same failed approach unchanged. Re-plan using an alternative tool group or strategy."
    )
    if summarized_results:
        retry_prompt += "\n\nUseful successful results so far:\n" + "\n".join(summarized_results)
    return retry_prompt


def build_planner_messages(system_prompt, planning_input):
    group_summary = get_group_summary(include_tools=False)
    return [
        {'role': 'system', 'content': (
            f"{system_prompt}\n\n"
            f"WORKING DIRECTORY: {state.working_directory}\n"
            f"Short term goal: {state.short_term_goal}\n\n"
            f"You are the PLANNER. You produce ONLY a numbered subtask list — nothing else.\n"
            f"You do NOT call tools, write code, or perform tasks. A separate TOOL SELECTOR executes your plan.\n\n"
            f"RULES:\n"
            f"- First subtask should always be a memory recall (search_memory for identity, goals, context)\n"
            f"- Last subtask should save important info to memory when appropriate\n"
            f"- Each subtask = one tool group. Be specific about what the executor should do.\n"
            f"- Max {cfg['max_subtasks']} subtasks. 2-5 per phase — the re-plan loop handles the rest.\n\n"
            f"{group_summary}"
        )},
        {'role': 'user', 'content': planning_input}
    ]


def build_selector_messages(subtask, all_results):
    context_parts = [f"Current subtask: {subtask}"]
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
    messages = [
        {'role': 'system', 'content': (
            f"You are a tool selector. Given a subtask, first pick the best tool group, "
            f"then call up to {cfg['max_tools_per_task']} tools from that group to accomplish the subtask.\n\n"
            f"Available groups:\n{group_list}\n\n"
            f"Reply with the group name on the first line, then call the appropriate tools."
        )},
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
    return [
        {'role': 'system', 'content': (
            f"{system_prompt}\n\n"
            f"You are the VERIFIER. Review whether the original task has been completed.\n"
            f"If the task is COMPLETE, respond with exactly 'COMPLETE' on the first line, "
            f"followed by a summary.\n"
            f"If the task is INCOMPLETE, respond with exactly 'INCOMPLETE' on the first line, "
            f"followed by a description of what still needs to be done."
        )},
        {'role': 'user', 'content': (
            f"Original task: {user_task}\n\n"
            f"Results:\n" + "\n".join(results_summary)
        )}
    ]


# ── TUI Main Loop ───────────────────────────────────────────────────────────

def main_tui():
    tui = AgentTUI(
        models={r: get_model(r) for r in ('planner', 'tool_selector', 'verifier')},
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
    tui.state.add_log(f"Tool Selector: {get_model('tool_selector')}")
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
                tui.set_phase(Phase.EXECUTING, get_model('tool_selector'))

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
                        get_model('tool_selector'), selector_messages, think=False
                    )

                    chosen_group = pick_group(selector_content)
                    if not chosen_group:
                        tui.state.add_log(f"Could not determine group, defaulting to web_search")
                        chosen_group = 'web_search'

                    tui.set_subtask_group(i, chosen_group)
                    tui.state.add_log(f"Group: {chosen_group}")

                    group_tools = get_tools_in_group(chosen_group)
                    if not group_tools:
                        tui.state.add_log(f"No tools in group '{chosen_group}'")
                        all_results.append((subtask, []))
                        tui.finish_subtask(i, success=False)
                        execution_failed = True
                        failure_reason = f"Subtask {i+1} '{subtask}' failed: no tools in group '{chosen_group}'"
                        break

                    # ── Tool execution (timeout-wrapped) ──
                    timeout = download_timeout if chosen_group in slow_groups else default_timeout

                    def run_tool_calls(_subtask=subtask, _context_parts=context_parts, _group_tools=group_tools, _chosen_group=chosen_group):
                        tool_messages = [
                            {'role': 'system', 'content': (
                                f"You are a tool executor. You MUST respond ONLY with tool calls — no text, no explanations, no commentary. "
                                f"Do NOT write content yourself. Use the provided tools to accomplish the subtask. "
                                f"Use up to {cfg['max_tools_per_task']} tool calls."
                            )},
                            {'role': 'user', 'content': "\n".join(_context_parts)}
                        ]
                        _, _, tool_calls = query_ollama(
                            get_model('tool_selector'), tool_messages,
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
    print(f"  Tool Selector: {get_model('tool_selector')}")
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
                print(f"  [Tool Selector] ({get_model('tool_selector')})")
                _, selector_content, _ = query_ollama(
                    get_model('tool_selector'), selector_messages, think=False
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

                def run_tool_calls(_subtask=subtask, _context_parts=context_parts, _group_tools=group_tools, _chosen_group=chosen_group):
                    tool_messages = [
                        {'role': 'system', 'content': (
                            f"You are a tool executor. Call the appropriate tools to accomplish this subtask. "
                            f"Use up to {cfg['max_tools_per_task']} tool calls."
                        )},
                        {'role': 'user', 'content': "\n".join(_context_parts)}
                    ]
                    _, _, tool_calls = query_ollama(
                        get_model('tool_selector'), tool_messages,
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
