import json
import os
from datetime import datetime
from config import cfg


class State:
    """Centralized mutable state for the agent."""

    def __init__(self):
        self.short_term_goal = cfg['short_term_goal']
        self.working_directory = cfg['working_directory']
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
            with open(cfg['memories_path'], 'r') as f:
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
        with open(cfg['memories_path'], 'w') as f:
            json.dump(self.memories, f, indent=4)
        self._memories_dirty = False

    # ── PDF index ─────────────────────────────────────────────────────────

    def _load_pdf_index(self):
        if os.path.exists(cfg['pdf_index_path']):
            with open(cfg['pdf_index_path'], 'r') as f:
                return json.load(f)
        return []

    def save_pdf_index(self):
        with open(cfg['pdf_index_path'], 'w') as f:
            json.dump(self.pdf_index, f)


state = State()
