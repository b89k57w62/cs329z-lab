from __future__ import annotations

import json
import os
from pathlib import Path

import jsonschema

from cs329z_hw1.pipelines import bm25
from cs329z_hw1.tokens import count_tokens

STORE_FILE = "memory.json"
STORE_SCHEMA = {
    "type": "object",
    "required": ["facts"],
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
}
RECALL_K = 10
EMPTY_MESSAGE = "Nothing is stored yet."
ORDER_NOTE = "Oldest first; a later fact replaces an earlier one on the same subject."


class MemoryStore:
    """Facts in one JSON file under memory_dir. With no directory, facts last for the session."""

    def __init__(self, directory: Path | None) -> None:
        self._path = Path(directory) / STORE_FILE if directory is not None else None
        self._session_facts: tuple[str, ...] = ()

    def facts(self) -> tuple[str, ...]:
        if self._path is None:
            return self._session_facts
        if not self._path.exists():
            return ()
        stored = json.loads(self._path.read_text(encoding="utf-8"))
        jsonschema.validate(stored, STORE_SCHEMA)
        return tuple(stored["facts"])

    def remember(self, fact: str) -> str:
        known = self.facts()
        if fact in known:
            return f"Already stored: {fact}"
        self._write((*known, fact))
        return f"Stored: {fact}"

    def recall(self, query: str) -> str:
        facts = self.facts()
        if not facts:
            return EMPTY_MESSAGE
        hits = bm25.search(bm25.build_index(list(facts)), query, RECALL_K)
        if hits:
            oldest_first = sorted(fact_id for fact_id, _ in hits)
            return f"{ORDER_NOTE}\n" + "\n".join(f"- {facts[fact_id]}" for fact_id in oldest_first)
        listed = "\n".join(f"- {fact}" for fact in facts[-RECALL_K:])
        return f"No stored fact matched that query. Most recent facts. {ORDER_NOTE}\n{listed}"

    def prompt_block(self, max_tokens: int) -> str:
        facts = self.facts()
        newest_first = [
            fact
            for position, fact in enumerate(reversed(facts))
            if count_tokens("\n".join(tuple(reversed(facts))[: position + 1])) <= max_tokens
        ]
        return "\n".join(f"- {fact}" for fact in reversed(newest_first))

    def _write(self, facts: tuple[str, ...]) -> None:
        if self._path is None:
            self._session_facts = facts
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"facts": list(facts)}, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self._path)
