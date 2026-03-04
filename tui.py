import os
import sys
import termios
import threading
import tty
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table


# ── Data model ───────────────────────────────────────────────────────────────

class Phase(Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"


@dataclass
class SubtaskStatus:
    description: str
    status: str = "pending"       # pending, running, done, failed
    tool_group: str = ""


@dataclass
class ToolCallRecord:
    name: str
    args: str
    result: str = ""


@dataclass
class TUIState:
    phase: Phase = Phase.IDLE
    loop_number: int = 0
    max_loops: int = 3
    thinking_text: str = ""
    response_text: str = ""
    subtasks: list = field(default_factory=list)
    current_subtask: int = -1
    tool_calls: list = field(default_factory=list)
    log_lines: list = field(default_factory=list)
    active_model: str = ""
    status_line: str = "Waiting for task..."
    input_text: str = ""          # current user input being typed
    waiting_for_input: bool = False

    def add_log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_lines.append((ts, msg))
        if len(self.log_lines) > 100:
            self.log_lines = self.log_lines[-100:]

    def reset_stream(self):
        self.thinking_text = ""
        self.response_text = ""

    def set_subtasks(self, descriptions):
        self.subtasks = [SubtaskStatus(d) for d in descriptions]
        self.current_subtask = -1

    def start_subtask(self, index):
        self.current_subtask = index
        if 0 <= index < len(self.subtasks):
            self.subtasks[index].status = "running"

    def finish_subtask(self, index, success=True):
        if 0 <= index < len(self.subtasks):
            self.subtasks[index].status = "done" if success else "failed"


# ── Layout builder ───────────────────────────────────────────────────────────

def _wrapped_line_count(line, width):
    """Count how many visual lines a single line occupies when wrapped."""
    if width <= 0:
        return 1
    length = len(line)
    if length == 0:
        return 1
    return max(1, (length + width - 1) // width)


def _tail_lines(text, max_lines, width=0):
    """Return the last lines of text that fit in max_lines visual rows.

    If width > 0, accounts for line wrapping so long lines that wrap to
    multiple visual rows are counted correctly.
    """
    if not text:
        return text
    lines = text.split('\n')
    if width <= 0:
        # Simple mode: just count raw lines
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        return '\n'.join(lines)
    # Wrap-aware mode: walk backwards, accumulating visual rows
    kept = []
    visual_rows = 0
    for line in reversed(lines):
        rows = _wrapped_line_count(line, width)
        if visual_rows + rows > max_lines and kept:
            break
        kept.append(line)
        visual_rows += rows
    kept.reverse()
    return '\n'.join(kept)


def build_layout(state: TUIState, height: int = 24, width: int = 80) -> Layout:
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    layout["body"].split_column(
        Layout(name="stream", ratio=3),
        Layout(name="bottom", ratio=2),
    )

    layout["stream"].split_row(
        Layout(name="thinking", ratio=1),
        Layout(name="response", ratio=1),
    )

    layout["bottom"].split_row(
        Layout(name="subtasks", ratio=3),
        Layout(name="tools_and_log", ratio=2),
    )

    layout["tools_and_log"].split_column(
        Layout(name="tools", ratio=1),
        Layout(name="log", ratio=1),
    )

    # Calculate available lines for each panel region
    # Each Panel has 2 lines of border (top + bottom)
    body_h = height - 6                       # header(3) + footer(3)
    stream_h = max(2, body_h * 3 // 5 - 2)   # 3/5 ratio, minus panel borders
    bottom_h = max(2, body_h * 2 // 5 - 2)   # 2/5 ratio, minus panel borders
    half_bottom_h = max(1, bottom_h // 2 - 2) # tools & log split

    # Calculate inner widths for wrap-aware scrolling
    # Each panel border takes 2 cols; padding takes 2 cols (1 each side)
    stream_panel_w = max(10, width // 2 - 4)   # thinking / response each get half
    subtasks_w = max(10, width * 3 // 5 - 4)   # subtasks panel inner width
    tools_log_w = max(10, width * 2 // 5 - 4)  # tools & log inner width

    # ── Header
    phase_style = {
        Phase.IDLE: "dim",
        Phase.PLANNING: "cyan bold",
        Phase.EXECUTING: "yellow bold",
        Phase.VERIFYING: "green bold",
    }[state.phase]

    header_table = Table.grid(expand=True)
    header_table.add_column(ratio=1)
    header_table.add_column(ratio=1, justify="right")
    header_table.add_row(
        f"[bold blue]Task-Loop Agent[/]  [dim]{state.active_model}[/]",
        f"[{phase_style}]{state.phase.value}[/]  [dim]Loop {state.loop_number}/{state.max_loops}[/]"
    )
    layout["header"].update(Panel(header_table, style="blue"))

    # ── Thinking panel (auto-scroll to bottom, wrap-aware)
    thinking_display = _tail_lines(state.thinking_text, stream_h, stream_panel_w)
    layout["thinking"].update(Panel(
        Text(thinking_display) if state.thinking_text else Text("...", style="dim"),
        title="[dim]Thinking[/]",
        border_style="dim",
    ))

    # ── Response panel (auto-scroll to bottom, wrap-aware)
    response_display = _tail_lines(state.response_text, stream_h, stream_panel_w)
    layout["response"].update(Panel(
        Text(response_display) if state.response_text else Text("...", style="dim"),
        title="[bold]Response[/]",
        border_style="green",
    ))

    # ── Subtasks panel (scroll to keep current subtask visible)
    subtask_text = Text(no_wrap=True, overflow="ellipsis")
    visible_subtasks = state.subtasks
    max_visible = max(1, bottom_h)
    if len(state.subtasks) > max_visible:
        # Show window centered on current subtask, biased towards bottom
        anchor = max(state.current_subtask, 0)
        start = max(0, anchor - max_visible // 2)
        end = start + max_visible
        if end > len(state.subtasks):
            end = len(state.subtasks)
            start = max(0, end - max_visible)
        visible_subtasks = state.subtasks[start:end]
        offset = start
    else:
        offset = 0
    # Truncate descriptions to fit panel width (leave room for prefix + index)
    max_desc = max(20, subtasks_w - 12)
    for i, st in enumerate(visible_subtasks):
        real_i = i + offset
        prefix_map = {
            "done":    ("[green]OK[/green] ", "green"),
            "running": ("[yellow]>> [/yellow]", "yellow bold"),
            "failed":  ("[red]XX[/red] ", "red"),
            "pending": ("[dim]   [/dim]", "dim"),
        }
        prefix, style = prefix_map.get(st.status, ("   ", "dim"))
        group_tag = f" [dim]({st.tool_group})[/dim]" if st.tool_group else ""
        desc = st.description[:max_desc] + ("..." if len(st.description) > max_desc else "")
        line = Text.from_markup(f" {prefix}{real_i+1}. {desc}{group_tag}\n")
        line.no_wrap = True
        line.overflow = "ellipsis"
        subtask_text.append_text(line)
    if not state.subtasks:
        subtask_text = Text("...", style="dim")
    layout["subtasks"].update(Panel(subtask_text, title="Subtasks"))

    # ── Tool calls panel (show most recent that fit, ~2 lines per entry)
    tool_text = Text(no_wrap=True, overflow="ellipsis")
    max_tool_entries = max(1, half_bottom_h // 2)
    recent_tools = state.tool_calls[-max_tool_entries:]
    max_args = max(10, tools_log_w - 8)
    max_result = max(10, tools_log_w - 8)
    for tc in recent_tools:
        args_short = tc.args[:max_args] + "..." if len(tc.args) > max_args else tc.args
        result_short = tc.result[:max_result] + "..." if len(tc.result) > max_result else tc.result
        tool_text.append(f" {tc.name}", style="cyan")
        tool_text.append(f"({args_short})\n", style="dim")
        if tc.result:
            tool_text.append(f"  -> {result_short}\n", style="dim")
    if not state.tool_calls:
        tool_text = Text("...", style="dim")
    layout["tools"].update(Panel(tool_text, title="Tool Calls", border_style="cyan"))

    # ── Log panel (show most recent that fit, truncate to panel width)
    log_text = Text(no_wrap=True, overflow="ellipsis")
    max_log_msg = max(10, tools_log_w - 12)  # leave room for timestamp
    for ts, msg in state.log_lines[-max(1, half_bottom_h):]:
        log_text.append(f" {ts} ", style="dim cyan")
        msg_short = msg[:max_log_msg] + "..." if len(msg) > max_log_msg else msg
        log_text.append(f"{msg_short}\n")
    if not state.log_lines:
        log_text = Text("...", style="dim")
    layout["log"].update(Panel(log_text, title="Log", border_style="dim"))

    # ── Footer
    if state.waiting_for_input:
        footer_content = Text.from_markup(
            f" [bold cyan]Task>[/] [bold]{state.input_text}[/][blink]_[/blink]"
        )
    else:
        footer_content = Text(f" {state.status_line}", style="bold")
    layout["footer"].update(Panel(footer_content, style="blue"))

    return layout


# ── TUI controller ───────────────────────────────────────────────────────────

class AgentTUI:
    def __init__(self, models: dict, max_loops: int):
        self.console = Console()
        self.state = TUIState(max_loops=max_loops)
        self.models = models
        self.live = None
        self._input_result = None
        self._input_event = threading.Event()

    def _refresh(self):
        if self.live:
            self.live.update(build_layout(self.state, self.console.height, self.console.width))

    def start(self):
        """Start the Live display (full screen)."""
        if self.live:
            return
        self.live = Live(
            build_layout(self.state, self.console.height, self.console.width),
            console=self.console,
            refresh_per_second=8,
            screen=True,
        )
        self.live.start()

    def stop(self):
        """Stop the Live display."""
        if self.live:
            self.live.stop()
            self.live = None

    # ── Streaming callback ────────────────────────────────────────────────

    def on_stream_chunk(self, chunk_type, text):
        if chunk_type == 'thinking_start':
            self.state.thinking_text = ""
        elif chunk_type == 'thinking':
            self.state.thinking_text += text
        elif chunk_type == 'answer_start':
            self.state.response_text = ""
        elif chunk_type == 'content':
            self.state.response_text += text
        self._refresh()

    # ── Tool log callback ─────────────────────────────────────────────────

    def on_tool_log(self, message):
        self.state.add_log(message)
        self._refresh()

    # ── Phase management ──────────────────────────────────────────────────

    def set_phase(self, phase: Phase, model: str = ""):
        self.state.phase = phase
        self.state.active_model = model
        self.state.reset_stream()
        self._refresh()

    def set_loop(self, n):
        self.state.loop_number = n
        self._refresh()

    def set_status(self, text):
        self.state.status_line = text
        self._refresh()

    # ── Subtask management ────────────────────────────────────────────────

    def set_subtasks(self, descriptions):
        self.state.set_subtasks(descriptions)
        self._refresh()

    def start_subtask(self, index):
        self.state.start_subtask(index)
        self._refresh()

    def finish_subtask(self, index, success=True):
        self.state.finish_subtask(index, success)
        self._refresh()

    def set_subtask_group(self, index, group):
        if 0 <= index < len(self.state.subtasks):
            self.state.subtasks[index].tool_group = group
        self._refresh()

    # ── Tool call recording ───────────────────────────────────────────────

    def record_tool_call(self, name, args, result):
        self.state.tool_calls.append(ToolCallRecord(name, str(args), str(result)))
        self._refresh()

    # ── User input (char-by-char in background thread, TUI stays visible) ──

    def _read_input_chars(self):
        """Read input character by character, updating the TUI footer live."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            buf = []
            while True:
                ch = os.read(fd, 1)
                if not ch:  # EOF
                    self._input_result = None
                    break
                ch = ch.decode('utf-8', errors='replace')
                if ch in ('\r', '\n'):  # Enter
                    self._input_result = ''.join(buf)
                    break
                elif ch == '\x03':  # Ctrl+C
                    self._input_result = None
                    break
                elif ch == '\x04':  # Ctrl+D
                    self._input_result = None
                    break
                elif ch in ('\x7f', '\x08'):  # Backspace
                    if buf:
                        buf.pop()
                        self.state.input_text = ''.join(buf)
                        self._refresh()
                elif ch == '\x15':  # Ctrl+U (clear line)
                    buf.clear()
                    self.state.input_text = ''
                    self._refresh()
                elif ch == '\x17':  # Ctrl+W (delete word)
                    while buf and buf[-1] == ' ':
                        buf.pop()
                    while buf and buf[-1] != ' ':
                        buf.pop()
                    self.state.input_text = ''.join(buf)
                    self._refresh()
                elif ch >= ' ':  # Printable
                    buf.append(ch)
                    self.state.input_text = ''.join(buf)
                    self._refresh()
        except Exception:
            self._input_result = None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        self._input_event.set()

    def get_user_input(self, prompt_text="Task> "):
        """Show input prompt in TUI footer, read input char-by-char."""
        self.state.waiting_for_input = True
        self.state.input_text = ""
        self._input_result = None
        self._input_event.clear()
        self._refresh()

        reader = threading.Thread(target=self._read_input_chars, daemon=True)
        reader.start()

        self._input_event.wait()

        self.state.waiting_for_input = False
        self._refresh()

        if self._input_result is None:
            raise EOFError()
        return self._input_result.strip()

    # ── Reset for new task ────────────────────────────────────────────────

    def reset_for_task(self):
        self.state.thinking_text = ""
        self.state.response_text = ""
        self.state.subtasks = []
        self.state.tool_calls = []
        self.state.current_subtask = -1
        self.state.loop_number = 0
        self.state.phase = Phase.IDLE
