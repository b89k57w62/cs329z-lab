"""Pytest plugin: send live-test model calls to the local `claude -p` CLI.

Opt in per run; scores are not comparable to the grading model's thresholds:

    uv run python -m pytest -p dev_tools.claude_cli_lm tests/test_priority.py -m live -s
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from collections.abc import Sequence

import jsonschema

from cs329z_hw1 import llm

MODEL = "haiku"
JUDGE_ROLE = "judge"
JUDGE_MODEL = "sonnet"
TIMEOUT_SECONDS = 75
MAX_ATTEMPTS = 3
MAX_CONCURRENT_CALLS = 16
SYSTEM_ROLE = "system"
CONVERSATION_HEADER = "The conversation so far, oldest first. Write only the next assistant reply.\n\n"
TRUNCATED_STOP_REASON = "max_tokens"
NESTED_SESSION_ENV_VAR = "CLAUDECODE"
ISOLATED_SETTINGS = json.dumps({"disableAllHooks": True})
_CALL_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_CALLS)

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["is_error", "result", "usage"],
    "properties": {
        "is_error": {"type": "boolean"},
        "result": {"type": "string"},
        "stop_reason": {"type": ["string", "null"]},
        "usage": {
            "type": "object",
            "required": ["output_tokens"],
            "properties": {"output_tokens": {"type": "integer"}},
        },
    },
}


class ClaudeCliLM:
    """Stands in for llm.LM. No cache, ledger or budget: every call uses Claude quota."""

    def __init__(
        self, role_or_model: str | None = None, *, max_tokens: int = 4096, salt: str = "", tag: str = ""
    ) -> None:
        self.model = JUDGE_MODEL if role_or_model == JUDGE_ROLE else MODEL
        self.tag = tag
        self.calls = 0

    def __call__(self, messages: Sequence[llm.Message]) -> str:
        if messages[0]["role"] != SYSTEM_ROLE or SYSTEM_ROLE in {message["role"] for message in messages[1:]}:
            raise ValueError("ClaudeCliLM needs exactly one system message, in first position")
        self.calls += 1
        with _CALL_SLOTS:
            return _reply_text(_call_with_retries(self.model, messages[0]["content"], _render(messages[1:])), self.model)

    def usage(self) -> dict:
        return {"model": self.model, "calls": self.calls, "cached_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "seconds": 0.0}


def _render(messages: Sequence[llm.Message]) -> str:
    if len(messages) == 1:
        return messages[0]["content"]
    return CONVERSATION_HEADER + "\n\n".join(f"[{message['role']}]\n{message['content']}" for message in messages)


def _call_with_retries(model: str, system: str, user: str) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _parse_output(_run_cli(model, system, user))
        except (subprocess.TimeoutExpired, RuntimeError):
            if attempt == MAX_ATTEMPTS:
                raise
    raise AssertionError("unreachable")


def _run_cli(model: str, system: str, user: str) -> subprocess.CompletedProcess[str]:
    command = [
        "claude",
        "-p",
        "--setting-sources", "",
        "--settings", ISOLATED_SETTINGS,
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--tools", "",
        "--no-session-persistence",
        "--model", model,
        "--output-format", "json",
        "--system-prompt", system,
    ]
    env = {key: value for key, value in os.environ.items() if key != NESTED_SESSION_ENV_VAR}
    with tempfile.TemporaryDirectory() as empty_cwd:
        return subprocess.run(
            command,
            input=user,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=empty_cwd,
            env=env,
        )


def _parse_output(process: subprocess.CompletedProcess[str]) -> dict:
    try:
        output = json.loads(process.stdout)
        jsonschema.validate(output, OUTPUT_SCHEMA)
    except (json.JSONDecodeError, jsonschema.ValidationError) as error:
        raise RuntimeError(f"claude -p exited {process.returncode} with unexpected output") from error
    if output["is_error"]:
        raise RuntimeError(f"claude -p failed: {output['result']}")
    return output


def _reply_text(output: dict, model: str) -> str:
    if output.get("stop_reason") == TRUNCATED_STOP_REASON:
        tokens = output["usage"]["output_tokens"]
        raise llm.OutputTruncated(output["result"], model=model, max_tokens=tokens, output_tokens=tokens)
    return output["result"]


def pytest_terminal_summary(terminalreporter) -> None:
    terminalreporter.write_line(f"LM: claude -p ({MODEL}; judge {JUDGE_MODEL}), NOT the grading models; thresholds are for reference only")


llm.LM = ClaudeCliLM
