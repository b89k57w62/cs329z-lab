import json, subprocess

LLM_TIMEOUT_SECONDS = 60
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

class JSONParseError(ValueError):
    pass

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
