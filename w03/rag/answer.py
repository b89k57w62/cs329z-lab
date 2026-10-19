import json
from dataclasses import dataclass
from .chunker import Chunk
from .config import MAX_ATTEMPTS
from .llm import JSONParseError, call_llm, parse_json, validate
from .retrieve import retrieve

INSUFFICIENT = "資料不足"
NO_LLM_ATTEMPTS = 0
SCHEMA = {
    "type": "object",
    "required": ["answer", "citations"],
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
}
SYSTEM_PROMPT_TEMPLATE = """You answer questions using only the notes provided by the user.
Rules:
- Use only information stated in the notes; never use outside knowledge.
- Cite every claim with the chunk id of its source in square brackets, e.g. [path/to/note.md#3].
- List every chunk id you cited in "citations".
- If the notes do not contain enough information to answer, set "answer" to exactly "{insufficient}" and "citations" to [].
- Write the answer in Traditional Chinese.
Respond with a single JSON object that conforms to this JSON schema:
{schema}"""

@dataclass(frozen=True)
class Answer:
    answer: str
    citations: tuple[str, ...]
    attempts: int

    @property
    def is_refusal(self) -> bool:
        return self.answer.strip() == INSUFFICIENT

class AnswerError(Exception):
    def __init__(self, attempts: int, errors: list[str]):
        super().__init__(f"Answer failed after {attempts} attempts: {'; '.join(errors)}")
        self.attempts = attempts
        self.errors = errors

def _format_context(results: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{chunk.id}] {chunk.heading_path}\n{chunk.text}" for chunk, _ in results)

def _citation_errors(obj: dict, allowed_ids: frozenset[str]) -> list[str]:
    unknown = [
        f"$.citations: '{chunk_id}' is not one of the provided chunk ids"
        for chunk_id in obj["citations"] if chunk_id not in allowed_ids
    ]
    uncited = (
        [f"$.citations: an answer other than '{INSUFFICIENT}' must cite at least one chunk id"]
        if obj["answer"].strip() != INSUFFICIENT and not obj["citations"] else []
    )
    return unknown + uncited

def _check(reply: str, allowed_ids: frozenset[str]) -> tuple[dict | None, list[str]]:
    try:
        obj = parse_json(reply)
    except JSONParseError as error:
        return None, [str(error)]
    schema_errors = validate(obj, SCHEMA)
    return obj, schema_errors or _citation_errors(obj, allowed_ids)

def _feedback(reply: str, errors: list[str]) -> str:
    bullet_list = "\n".join(f"- {error}" for error in errors)
    return (
        f"\n\nYour previous response:\n{reply}\n\n"
        f"It failed validation:\n{bullet_list}\n\n"
        "Respond again with only the corrected JSON object."
    )

def answer(query: str) -> Answer:
    results = retrieve(query)
    if not results:
        return Answer(INSUFFICIENT, (), NO_LLM_ATTEMPTS)
    allowed_ids = frozenset(chunk.id for chunk, _ in results)
    system = SYSTEM_PROMPT_TEMPLATE.format(
        insufficient=INSUFFICIENT, schema=json.dumps(SCHEMA, ensure_ascii=False, indent=2)
    )
    transcript = f"Notes:\n{_format_context(results)}\n\nQuestion: {query}"
    errors = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply = call_llm(system, transcript)
        obj, errors = _check(reply, allowed_ids)
        if not errors:
            return Answer(obj["answer"], tuple(obj["citations"]), attempt)
        transcript += _feedback(reply, errors)
    raise AnswerError(MAX_ATTEMPTS, errors)
