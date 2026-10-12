import json, subprocess
from pathlib import Path

LLM_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
MAX_ATTEMPTS = MAX_RETRIES + 1
SAMPLE_COUNT = 20
SAMPLES_PATH = Path(__file__).parent / "data" / "samples.jsonl"
SCHEMA = {
    "type": "object",
    "required": ["title", "date", "action_items", "priority"],
    "properties": {
        "title": {"type": "string"},
        "date": {"type": "string"},
        "action_items": {"type": "array", "items": {"type": "string"}},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}
ROOT_PATH = "$"
TYPE_CHECKS = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
}
JSON_TYPE_NAMES = {
    dict: "object", list: "array", str: "string", bool: "boolean",
    int: "integer", float: "number", type(None): "null",
}
JSON_DECODER = json.JSONDecoder()
SYSTEM_PROMPT_TEMPLATE = """You extract structured data from meeting notes.
Respond with a single JSON object that conforms to this JSON schema:
{schema}"""

class JSONParseError(ValueError):
    pass

class ExtractionError(Exception):
    def __init__(self, attempts: int, errors: list[str]):
        super().__init__(f"Extraction failed after {attempts} attempts: {'; '.join(errors)}")
        self.attempts = attempts
        self.errors = errors

def call_llm(system: str, transcript: str) -> str:
    result = subprocess.run(
        ["claude", "-p", transcript,
         "--system-prompt", system,
         "--model", "haiku",
         "--tools", "",
         "--setting-sources", "",
         "--no-session-persistence",
         "--output-format", "json"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=LLM_TIMEOUT_SECONDS, check=True,
    )
    return json.loads(result.stdout)["result"]

def load_samples() -> list[dict]:
    lines = SAMPLES_PATH.read_text(encoding="utf-8").splitlines()
    samples = [json.loads(line) for line in lines if line.strip()]
    if len(samples) != SAMPLE_COUNT:
        raise ValueError(f"Expected {SAMPLE_COUNT} samples, got {len(samples)}")
    return samples

def _json_type_name(value) -> str:
    return JSON_TYPE_NAMES.get(type(value), type(value).__name__)

def _validate_object(obj: dict, schema: dict, path: str) -> list[str]:
    missing = [
        f"{path}: missing required field '{key}'"
        for key in schema.get("required", []) if key not in obj
    ]
    nested = [
        error
        for key, subschema in schema.get("properties", {}).items() if key in obj
        for error in validate(obj[key], subschema, f"{path}.{key}")
    ]
    return missing + nested

def _validate_array(obj: list, schema: dict, path: str) -> list[str]:
    item_schema = schema.get("items")
    if item_schema is None:
        return []
    return [
        error
        for index, item in enumerate(obj)
        for error in validate(item, item_schema, f"{path}[{index}]")
    ]

def validate(obj, schema: dict, path: str = ROOT_PATH) -> list[str]:
    expected = schema.get("type")
    if expected is not None:
        if expected not in TYPE_CHECKS:
            raise ValueError(f"Unsupported schema type: {expected}")
        if not TYPE_CHECKS[expected](obj):
            return [f"{path}: expected {expected}, got {_json_type_name(obj)}"]
    if "enum" in schema and obj not in schema["enum"]:
        return [f"{path}: {obj!r} is not one of {schema['enum']}"]
    if expected == "object":
        return _validate_object(obj, schema, path)
    if expected == "array":
        return _validate_array(obj, schema, path)
    return []

def parse_json(text: str) -> dict:
    first_error = None
    for start in (index for index, char in enumerate(text) if char == "{"):
        try:
            obj, _ = JSON_DECODER.raw_decode(text, start)
            return obj
        except json.JSONDecodeError as error:
            first_error = first_error or error
    detail = (
        f": {first_error.msg} (line {first_error.lineno}, column {first_error.colno})"
        if first_error else ""
    )
    raise JSONParseError(f"No valid JSON object found in the response{detail}")

def _check(reply: str, schema: dict) -> tuple[dict | None, list[str]]:
    try:
        obj = parse_json(reply)
    except JSONParseError as error:
        return None, [str(error)]
    return obj, validate(obj, schema)

def _feedback(reply: str, errors: list[str]) -> str:
    bullet_list = "\n".join(f"- {error}" for error in errors)
    return (
        f"\n\nYour previous response:\n{reply}\n\n"
        f"It failed validation:\n{bullet_list}\n\n"
        "Respond again with only the corrected JSON object."
    )

def extract(text: str, schema: dict, feedback: bool = True) -> tuple[dict, int]:
    system = SYSTEM_PROMPT_TEMPLATE.format(schema=json.dumps(schema, ensure_ascii=False, indent=2))
    transcript = f"Meeting notes:\n{text}"
    errors = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply = call_llm(system, transcript)
        result, errors = _check(reply, schema)
        if not errors:
            return result, attempt
        if feedback:
            transcript += _feedback(reply, errors)
    raise ExtractionError(MAX_ATTEMPTS, errors)
