from __future__ import annotations

import functools
import hashlib
import threading
from pathlib import Path
from typing import Any

from cs329z_hw1 import data, sandbox
from cs329z_hw1.agent.compaction import clip_middle
from cs329z_hw1.agent.docs import best_excerpt, read_doc, search_docs, shared_doc_index
from cs329z_hw1.agent.memory import MemoryStore
from cs329z_hw1.cardinal import cardinal_spec
from cs329z_hw1.pipelines import bm25
from cs329z_hw1.pipelines.daily_digest import write_daily_digest
from cs329z_hw1.pipelines.email_qa import answer_question
from cs329z_hw1.tokens import CHARS_PER_TOKEN
from cs329z_hw1.types import AgentConfig, Email, LMCallable, ToolSpec

MAX_SEARCH_RESULTS = 20
EMAIL_EXCERPT_CHARS = 300
EMAIL_MAX_TOKENS = 1500
DOC_SNIPPET_TOKENS = 90
DOC_WINDOW_TOKENS = 500
SHOWN_OUTPUT_TOKENS = 2000
OUTPUT_BUDGET_DIVISOR = 5
STDOUT_SHARE_NUMERATOR, STDOUT_SHARE_DENOMINATOR = 3, 4
OUTPUT_DIR = "outputs"
HASH_CHARS = 10

_ARCHIVE_LOCK = threading.Lock()


class TerminalFailure(Exception):
    pass


def shared_email_index() -> bm25.BM25Index:
    with _ARCHIVE_LOCK:
        return _load_email_index()


@functools.cache
def _load_email_index() -> bm25.BM25Index:
    return bm25.build_index([data.email_text(email) for email in data.load_emails()])


def _search_archive(query: str, k: int) -> list[Email]:
    archive = data.load_emails()
    return [archive[doc_id] for doc_id, _ in bm25.search(shared_email_index(), query, k)]


def _search_emails(query: str, k: int = 5) -> str:
    hits = _search_archive(query, min(k, MAX_SEARCH_RESULTS))
    if not hits:
        return "No emails matched. Try other keywords, such as names."
    return "\n\n".join(
        f"{email['id']} | {email['date']} | from {email['from']} | {email['subject']}\n"
        f"{best_excerpt(email['body'], query, EMAIL_EXCERPT_CHARS)}"
        for email in hits
    )


def _read_email(email_id: str) -> str:
    match = next((email for email in data.load_emails() if email["id"] == email_id), None)
    if match is None:
        raise LookupError(f"No email has the id {email_id!r}.")
    return (
        f"Id: {match['id']}\nThread: {match['thread_id']}\nFrom: {match['from']}\nTo: {', '.join(match['to'])}\n"
        f"Date: {match['date']}\nSubject: {match['subject']}\n\n{clip_middle(match['body'], EMAIL_MAX_TOKENS)}"
    )


def _format_answer(result: dict) -> str:
    if result["answer"] is None:
        return "No answer was found in the archive. Try search_emails with other keywords."
    return f"Answer: {result['answer']}\nSupporting emails: {', '.join(result['support']) or 'none cited'}"


def _search_docs_tool(query: str, k: int = 5) -> str:
    hits = search_docs(shared_doc_index(), query, k, DOC_SNIPPET_TOKENS)
    if not hits:
        return "No document matched. Try other keywords."
    return "\n\n".join(f"{hit['doc_id']} (start_line {hit['start_line']})\n{hit['snippet']}" for hit in hits)


def _read_doc_tool(doc_id: str, start_line: int = 1) -> str:
    window = read_doc(shared_doc_index(), doc_id, start_line, DOC_WINDOW_TOKENS)
    more = window["end_line"] < window["total_lines"]
    footer = f"\n[more: read_doc with start_line {window['end_line'] + 1}]" if more else "\n[end of document]"
    return f"{doc_id}, lines {window['start_line']} to {window['end_line']} of {window['total_lines']}\n{window['text']}{footer}"


def _workspace_path(workspace: Path | None) -> Path:
    return sandbox.make_workspace(workspace) if workspace is not None else sandbox.default_workspace()


def _save_output(stdout: str, workspace: Path | None) -> str:
    name = f"{hashlib.sha256(stdout.encode('utf-8', 'replace')).hexdigest()[:HASH_CHARS]}.txt"
    directory = _workspace_path(workspace) / OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(stdout, encoding="utf-8", errors="replace")
    return f"{OUTPUT_DIR}/{name}"


def _describe_run(result: Any, config: AgentConfig) -> str:
    limit = min(SHOWN_OUTPUT_TOKENS, config.context_budget // OUTPUT_BUDGET_DIVISOR)
    stdout_limit = limit * STDOUT_SHARE_NUMERATOR // STDOUT_SHARE_DENOMINATOR
    too_long = len(result.stdout) > stdout_limit * CHARS_PER_TOKEN
    saved = _save_output(result.stdout, config.workspace) if too_long else None
    note = f"Read the full output with commands such as: sed -n '100,120p' {saved}" if saved else ""
    header = f"exit code: {result.exit_code}\n" + (
        f"stdout is {len(result.stdout.splitlines())} lines and was cut; the full stdout is saved in {saved}\n" if saved else ""
    )
    return (
        f"{header}stdout:\n{clip_middle(result.stdout, stdout_limit, note)}\n"
        f"stderr:\n{clip_middle(result.stderr, limit - stdout_limit)}"
    )


def _terminal_tool(config: AgentConfig):
    def run_terminal(cmd: str) -> str:
        result = sandbox.run_terminal(cmd, workspace=config.workspace, timeout=config.terminal_timeout)
        if result.rejected:
            raise TerminalFailure(f"The sandbox rejected the command: {result.reason}")
        shown = _describe_run(result, config)
        if result.timed_out:
            raise TerminalFailure(f"The command timed out after {config.terminal_timeout} seconds.\n{shown}")
        if result.exit_code != 0:
            raise TerminalFailure(shown)
        return shown

    return run_terminal


def _ask_user_tool(config: AgentConfig):
    def ask_user(question: str) -> str:
        if config.user is None:
            raise RuntimeError("No user is available to answer; decide from what you have, or say what is missing.")
        return config.user.ask(question)

    return ask_user


def ask_user_spec(config: AgentConfig) -> ToolSpec:
    return cardinal_spec(
        "ask_user",
        "Ask the user one short question when the request cannot be answered without a detail only they know. Returns their answer.",
        _ask_user_tool(config),
    )


def memory_specs(store: MemoryStore) -> list[ToolSpec]:
    def remember(fact: str) -> str:
        return store.remember(fact)

    def recall(query: str) -> str:
        return store.recall(query)

    return [
        cardinal_spec(
            "remember",
            "Store one lasting fact about the user for future conversations, as a complete sentence that makes sense alone.",
            remember,
        ),
        cardinal_spec("recall", "Search the facts stored in earlier conversations.", recall),
    ]


def cardinal_specs(config: AgentConfig, aux_lm: LMCallable, store: MemoryStore) -> list[ToolSpec]:
    def email_qa(question: str) -> str:
        return _format_answer(answer_question(question, _search_archive, aux_lm))

    def daily_digest(date: str) -> str:
        emails = data.emails_on(date)
        if not emails:
            return f"The archive has no emails dated {date}. It covers 2001-06-01 to 2001-08-31."
        return write_daily_digest(emails, aux_lm)

    return [
        cardinal_spec(
            "search_emails",
            "Keyword search over the email archive. Returns ids, headers and a short excerpt; use read_email for the full text.",
            _search_emails,
        ),
        cardinal_spec("read_email", "Read one email by its id, such as em-01234.", _read_email),
        cardinal_spec(
            "email_qa",
            "Answer a factual question from the email archive, following references across emails. Returns the answer and the ids of the emails that support it.",
            email_qa,
        ),
        cardinal_spec("daily_digest", "Write the morning summary of all emails of one day (YYYY-MM-DD).", daily_digest),
        cardinal_spec(
            "search_docs",
            "Keyword search over the company documents. Returns doc_ids, line numbers and short excerpts; use read_doc to read around a line.",
            _search_docs_tool,
        ),
        cardinal_spec(
            "read_doc",
            "Read a short window of one document starting at start_line. The result says how to read the next window.",
            _read_doc_tool,
        ),
        cardinal_spec(
            "run_terminal",
            "Run one shell command in a sandbox whose working directory holds emails.jsonl and docs/. Allowed programs include python3, grep, sort, uniq, wc, sed, awk, head, tail, cut and cat. Use it to compute counts and listings. Multi-line scripts go in python3 -c '...'; here-documents and $(...) are rejected. There is no network.",
            _terminal_tool(config),
        ),
        *memory_specs(store),
        ask_user_spec(config),
    ]
