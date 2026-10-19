import json
from dataclasses import dataclass
from pathlib import Path
from .answer import answer
from .chunker import Chunk, load_chunks
from .config import (
    CHUNK_MAX_CHARS, EVAL_MAX_QUESTIONS, EVAL_MIN_QUESTIONS,
    EVAL_NO_ANSWER_QUESTIONS, EVAL_SET_PATH, TOP_K,
)
from .llm import validate
from .retrieve import retrieve

NOT_FOUND_RANK = 0
FIRST_RANK = 1
EVAL_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["question", "gold_chunk_paths"],
        "properties": {
            "question": {"type": "string"},
            "gold_chunk_paths": {"type": "array", "items": {"type": "string"}},
        },
    },
}

@dataclass(frozen=True)
class EvalCase:
    question: str
    gold_chunk_paths: frozenset[str]

    @property
    def answerable(self) -> bool:
        return bool(self.gold_chunk_paths)

@dataclass(frozen=True)
class RetrievalResult:
    case: EvalCase
    rank: int
    gated: bool

@dataclass(frozen=True)
class RefusalResult:
    case: EvalCase
    refused: bool

@dataclass(frozen=True)
class EvalReport:
    chunk_max_chars: int
    top_k: int
    retrievals: tuple[RetrievalResult, ...]
    refusals: tuple[RefusalResult, ...]

    @property
    def recall_at_k(self) -> float:
        return sum(result.rank != NOT_FOUND_RANK for result in self.retrievals) / len(self.retrievals)

    @property
    def mrr(self) -> float:
        return sum(
            1 / result.rank for result in self.retrievals if result.rank != NOT_FOUND_RANK
        ) / len(self.retrievals)

    @property
    def gated_count(self) -> int:
        return sum(result.gated for result in self.retrievals)

    @property
    def refusal_rate(self) -> float:
        return sum(result.refused for result in self.refusals) / len(self.refusals)

def _check_cases(cases: tuple[EvalCase, ...], path: Path) -> None:
    if not EVAL_MIN_QUESTIONS <= len(cases) <= EVAL_MAX_QUESTIONS:
        raise ValueError(
            f"Eval set {path} must have {EVAL_MIN_QUESTIONS}-{EVAL_MAX_QUESTIONS} questions, got {len(cases)}"
        )
    no_answer = sum(not case.answerable for case in cases)
    if no_answer != EVAL_NO_ANSWER_QUESTIONS:
        raise ValueError(
            f"Eval set {path} must have {EVAL_NO_ANSWER_QUESTIONS} no-answer questions, got {no_answer}"
        )
    known_paths = {chunk.path for chunk in load_chunks()}
    unknown = sorted(gold for case in cases for gold in case.gold_chunk_paths - known_paths)
    if unknown:
        raise ValueError(f"Eval set {path} has gold paths missing from the corpus: {unknown}")

def load_eval_set(path: Path = EVAL_SET_PATH) -> tuple[EvalCase, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Eval set not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate(payload, EVAL_SCHEMA)
    if errors:
        raise ValueError(f"Invalid eval set {path}: {'; '.join(errors)}")
    cases = tuple(EvalCase(item["question"], frozenset(item["gold_chunk_paths"])) for item in payload)
    _check_cases(cases, path)
    return cases

def _first_relevant_rank(results: list[tuple[Chunk, float]], gold_paths: frozenset[str]) -> int:
    return next(
        (rank for rank, (chunk, _) in enumerate(results, start=FIRST_RANK) if chunk.path in gold_paths),
        NOT_FOUND_RANK,
    )

def _evaluate_retrieval(case: EvalCase) -> RetrievalResult:
    results = retrieve(case.question)
    return RetrievalResult(case, _first_relevant_rank(results, case.gold_chunk_paths), not results)

def _evaluate_refusal(case: EvalCase) -> RefusalResult:
    return RefusalResult(case, answer(case.question).is_refusal)

def run_eval(cases: tuple[EvalCase, ...]) -> EvalReport:
    return EvalReport(
        chunk_max_chars=CHUNK_MAX_CHARS,
        top_k=TOP_K,
        retrievals=tuple(_evaluate_retrieval(case) for case in cases if case.answerable),
        refusals=tuple(_evaluate_refusal(case) for case in cases if not case.answerable),
    )
