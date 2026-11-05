from __future__ import annotations

import json
import re

from cs329z_hw1.types import ParsedResponse, ToolCall

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"
TRAILING_CLOSE = re.compile(rf"\s*{re.escape(CLOSE_TAG)}")
MANGLED_CALL = re.compile(r"""</?(?:invoke|parameter\b|function_call|tool_use|tool_code)|"name"\s*:\s*"[^"]*"\s*,\s*"args"\s*:""")
JSON_DECODER = json.JSONDecoder()

PROTOCOL_TEXT = f"""# Calling tools

To call a tool, write one block in exactly this form, with one JSON object inside:

{OPEN_TAG}
{{"name": "tool_name", "args": {{"parameter": "value"}}}}
{CLOSE_TAG}

Rules:
- Use only the tools listed below and only the parameters they list. Give every required parameter, and leave out optional ones you do not need. Use {{}} for args when a tool takes none.
- Match the JSON types: numbers and booleans have no quotes, and a string keeps its exact value.
- No other tag or syntax works: only a block in the form above runs a tool.
- Write at most one block in a reply. You will get the result in the next message.
- A reply with no block is your final answer to the user. If you need a tool, write the block now instead of saying you will.

# Tools
"""


def render_tools(schemas: list[dict]) -> str:
    return PROTOCOL_TEXT + "\n".join(_render_tool(schema) for schema in schemas)


def _render_tool(schema: dict) -> str:
    parameters = schema["parameters"]
    properties = json.dumps(parameters.get("properties", {}), ensure_ascii=False)
    required = ", ".join(parameters.get("required", [])) or "none"
    return f"\n## {schema['name']}\n{schema['description']}\nParameters: {properties}\nRequired: {required}\n"


def format_tool_call(text: str, call: ToolCall) -> str:
    body = json.dumps({"name": call.name, "args": call.args}, ensure_ascii=False)
    block = f"{OPEN_TAG}\n{body}\n{CLOSE_TAG}"
    return f"{text.strip()}\n{block}" if text.strip() else block


def parse_response(text: str) -> ParsedResponse:
    start = text.find(OPEN_TAG)
    if start < 0:
        if MANGLED_CALL.search(text):
            return _failure(text, f'That looks like a tool call in the wrong format, so nothing ran. Write it exactly as {OPEN_TAG} then {{"name": "tool_name", "args": {{...}}}} then {CLOSE_TAG}, and nothing else.')
        return ParsedResponse(text=text.strip())
    before = text[:start]
    try:
        payload, end = _decode_object(text, start + len(OPEN_TAG))
    except (ValueError, RecursionError):
        return _failure(before, f"The {OPEN_TAG} block does not hold one complete JSON object; it may be cut off. Write the whole block again.")
    closing = TRAILING_CLOSE.match(text, end)
    after = text[closing.end() if closing else end :]
    if OPEN_TAG in after:
        return _failure(before, f"The reply has more than one {OPEN_TAG} block. Write one call per reply.")
    call = _to_call(payload)
    if call is None:
        return _failure(before, 'A tool call must be a JSON object with a string "name" and an object "args".')
    return ParsedResponse(text=f"{before}{after}".strip(), tool_call=call)


def _decode_object(text: str, position: int) -> tuple[object, int]:
    start = position + len(text[position:]) - len(text[position:].lstrip())
    return JSON_DECODER.raw_decode(text, start)


def _to_call(payload: object) -> ToolCall | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        return None
    args = payload.get("args", {})
    return ToolCall(payload["name"], args) if isinstance(args, dict) else None


def _failure(text: str, error: str) -> ParsedResponse:
    return ParsedResponse(text=text.strip(), error=error)
