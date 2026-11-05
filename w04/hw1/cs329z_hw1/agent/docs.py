from __future__ import annotations

import functools
import re
import threading
from dataclasses import dataclass

from cs329z_hw1 import data
from cs329z_hw1.pipelines import bm25
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import CHARS_PER_TOKEN, count_tokens
from cs329z_hw1.types import DocWindow, SearchResult

MIN_CHUNK_CHARS = 200
MAX_RESULTS_PER_DOC = 2
SNIPPET_MAX_TOKENS = 150
WINDOW_MAX_TOKENS = 1500
HEADING_PREFIX = "#"

_LOCK = threading.Lock()


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    heading: str
    start_line: int
    text: str


@dataclass(frozen=True)
class DocIndex:
    lines_by_doc: dict[str, tuple[str, ...]]
    chunks: tuple[Chunk, ...]
    bm25_index: bm25.BM25Index


def build_doc_index(docs: list[dict]) -> DocIndex:
    chunks = tuple(chunk for doc in docs for chunk in _chunks_of(doc))
    return DocIndex(
        lines_by_doc={doc["doc_id"]: tuple(doc["text"].splitlines()) for doc in docs},
        chunks=chunks,
        bm25_index=bm25.build_index([f"{c.title}\n{c.heading}\n{c.text}" for c in chunks]),
    )


def shared_doc_index() -> DocIndex:
    with _LOCK:
        return _load_shared_doc_index()


@functools.cache
def _load_shared_doc_index() -> DocIndex:
    return build_doc_index(data.load_docs())


def _chunks_of(doc: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading = ""
    pending: list[str] = []
    start = 0
    for number, line in enumerate(doc["text"].splitlines(), start=1):
        if not line.strip():
            continue
        if line.startswith(HEADING_PREFIX):
            heading = line.lstrip(HEADING_PREFIX).strip()
        if not pending:
            start = number
        pending = [*pending, line]
        if sum(map(len, pending)) >= MIN_CHUNK_CHARS:
            chunks = [*chunks, Chunk(doc["doc_id"], doc["title"], heading, start, "\n".join(pending))]
            pending = []
    if pending:
        chunks = [*chunks, Chunk(doc["doc_id"], doc["title"], heading, start, "\n".join(pending))]
    return chunks


def search_docs(index: DocIndex, query: str, k: int = 5, snippet_tokens: int = SNIPPET_MAX_TOKENS) -> list[SearchResult]:
    hits = bm25.search(index.bm25_index, query, len(index.chunks))
    ranked = [index.chunks[chunk_id] for chunk_id, _ in hits]
    limited = [
        chunk
        for position, chunk in enumerate(ranked)
        if sum(other.doc_id == chunk.doc_id for other in ranked[:position]) < MAX_RESULTS_PER_DOC
    ]
    return [_to_result(chunk, query, snippet_tokens) for chunk in limited[:k]]


def _to_result(chunk: Chunk, query: str, snippet_tokens: int) -> dict:
    prefix = f"{chunk.title} / {chunk.heading}: " if chunk.heading else f"{chunk.title}: "
    room = snippet_tokens * CHARS_PER_TOKEN - len(prefix)
    return {
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "snippet": prefix + best_excerpt(chunk.text, query, room),
        "start_line": chunk.start_line,
    }


def best_excerpt(text: str, query: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    terms = list(dict.fromkeys(tokenize(query)))
    starts = sorted(match.start() for term in terms for match in re.finditer(re.escape(term), text, re.IGNORECASE))
    if not starts:
        return text[:max_chars]
    best = max(starts, key=lambda start: (sum(start <= other < start + max_chars for other in starts), -start))
    begin = min(best, len(text) - max_chars)
    return text[begin : begin + max_chars]


def read_doc(index: DocIndex, doc_id: str, start_line: int = 1, max_tokens: int = WINDOW_MAX_TOKENS) -> DocWindow:
    if doc_id not in index.lines_by_doc:
        raise LookupError(f"Unknown doc_id {doc_id!r}. Use the doc_id exactly as search_docs gives it.")
    lines = index.lines_by_doc[doc_id]
    if not 1 <= start_line <= len(lines):
        raise ValueError(f"start_line {start_line} is outside {doc_id}, which has lines 1 to {len(lines)}.")
    end_line = _window_end(lines, start_line, max_tokens)
    return {
        "doc_id": doc_id,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": len(lines),
        "text": "\n".join(lines[start_line - 1 : end_line]),
    }


def _window_end(lines: tuple[str, ...], start_line: int, max_tokens: int) -> int:
    fitting = (
        end
        for end in range(start_line + 1, len(lines) + 1)
        if count_tokens("\n".join(lines[start_line - 1 : end])) > max_tokens
    )
    return next(fitting, len(lines) + 1) - 1
