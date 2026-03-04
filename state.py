import json
import os
from datetime import datetime
from config import cfg, resolve_path


class State:
    """Centralized mutable state for the agent."""

    def __init__(self):
        self.short_term_goal = cfg['short_term_goal']
        self.working_directory = str(resolve_path(cfg['working_directory']))
        os.makedirs(self.working_directory, exist_ok=True)
        self.iteration_count = 0  # resets per user task
        self.pdf_index = self._load_pdf_index()
        self.memories = self._load_memories()
        self._memories_dirty = False
        self._migrate_memories()

    # ── Iteration tracking ────────────────────────────────────────────────

    def reset_iteration(self):
        self.iteration_count = 0

    def increment_iteration(self):
        self.iteration_count += 1

    # ── Memories ──────────────────────────────────────────────────────────

    def _load_memories(self):
        try:
            with open(resolve_path(cfg['memories_path']), 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _migrate_memories(self):
        now = datetime.now().isoformat()
        for k, v in self.memories.items():
            if not (isinstance(v, dict) and 'text' in v):
                self.memories[k] = {'text': v, 'created': now, 'accessed': now, 'access_count': 0}
                self._memories_dirty = True

    def mark_memories_dirty(self):
        self._memories_dirty = True

    def save_memories_if_dirty(self):
        if self._memories_dirty:
            self.save_memories()

    def save_memories(self):
        """Force-save memories."""
        memories_path = resolve_path(cfg['memories_path'])
        memories_path.parent.mkdir(parents=True, exist_ok=True)
        with open(memories_path, 'w') as f:
            json.dump(self.memories, f, indent=4)
        self._memories_dirty = False

    # ── PDF index ─────────────────────────────────────────────────────────

    def _load_pdf_index(self):
        pdf_index_path = resolve_path(cfg['pdf_index_path'])
        if os.path.exists(pdf_index_path):
            with open(pdf_index_path, 'r') as f:
                return json.load(f)
        return []

    def save_pdf_index(self):
        pdf_index_path = resolve_path(cfg['pdf_index_path'])
        pdf_index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pdf_index_path, 'w') as f:
            json.dump(self.pdf_index, f)


state = State()
