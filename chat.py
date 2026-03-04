import json

from config import cfg, resolve_path
from state import state
from providers import query_ollama
from tools import build_tool_registry, available_functions
from tool_groups import get_tools_in_group


def _load_system_prompt():
    try:
        with open(resolve_path(cfg['system_prompt_path']), 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _get_chat_model():
    """Resolve chat model from config with sane fallback."""
    if cfg.get('chat_model'):
        return cfg['chat_model']
    models = cfg.get('models', {})
    if isinstance(models.get('chat'), dict):
        return models['chat'].get('model')
    if isinstance(models.get('chat'), str):
        return models['chat']
    if isinstance(models.get('planner'), dict):
        return models['planner'].get('model')
    return models.get('planner', 'qwen3.5:9b')


def _allowed_tools():
    """Tools for conversational assistant with memory + research/document access."""
    selected = []
    for group in ('web_search', 'memory', 'document_processing', 'file_operations'):
        selected.extend(get_tools_in_group(group))
    return selected


def _execute_tool_calls(tool_calls):
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

        if fn_name not in available_functions:
            results.append((fn_name, fn_args, f"Error: tool '{fn_name}' not found"))
            continue

        try:
            result = available_functions[fn_name](**fn_args)
        except Exception as e:
            result = f"Error: {e}"
        results.append((fn_name, fn_args, str(result)))
    return results


def _chat_turn(model, system_prompt, history, user_text, tools):
    """Single conversational turn with optional tool use."""
    tool_context = []
    max_rounds = cfg.get('max_tools_per_task', 3)

    for _ in range(max_rounds):
        context_bits = [f"User message: {user_text}"]
        if tool_context:
            context_bits.append("\nTool results so far:")
            context_bits.extend(tool_context)

        messages = [
            {'role': 'system', 'content': (
                f"{system_prompt}\n\n"
                f"WORKING DIRECTORY: {state.working_directory}\n"
                f"Short term goal: {state.short_term_goal}\n\n"
                f"You are a conversational assistant. Be helpful and concise. "
                f"You may use tools when needed for factual lookup, downloading/reading documents, "
                f"and memory management. Distinguish memory tools clearly: search/open for recall, "
                f"save for new durable information, edit for corrections. "
                f"Prefer multiple tool calls when useful (e.g., search/read -> answer -> save key memory)."
            )},
            *history,
            {'role': 'user', 'content': "\n".join(context_bits)}
        ]

        _, content, tool_calls = query_ollama(model, messages, tools=tools, think=False)

        if not tool_calls:
            return content.strip() or "(No response)"

        results = _execute_tool_calls(tool_calls)
        tool_context = [f"- {fn}({args}) -> {res[:500]}" for fn, args, res in results]

    return "I ran tools but couldn't complete a clean response yet. Please refine your request."


def main():
    build_tool_registry()
    model = _get_chat_model()
    system_prompt = _load_system_prompt()
    tools = _allowed_tools()
    history = []

    print("=" * 60)
    print("  Conversational Agent (single-model)")
    print("=" * 60)
    print(f"  Model: {model}")
    print("  Type 'exit' to quit")
    print("=" * 60)

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            state.save_memories_if_dirty()
            break

        if not user_text:
            continue
        if user_text.lower() in ('exit', 'quit'):
            print("Goodbye!")
            state.save_memories_if_dirty()
            break

        answer = _chat_turn(model, system_prompt, history, user_text, tools)
        print(f"Assistant> {answer}\n")

        history.append({'role': 'user', 'content': user_text})
        history.append({'role': 'assistant', 'content': answer})
        if len(history) > 20:
            history = history[-20:]


if __name__ == '__main__':
    main()
