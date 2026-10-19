from pathlib import Path

CORPUS_DIR = Path.home() / "dev" / "SecondBrain"
PRIVATE_DATA_DIR = Path(__file__).resolve().parent.parent / "corpus"
EXCLUDED_FILES = frozenset({"decisions/TEMPLATE.md"})
CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP_CHARS = 100
INDEX_PATH = PRIVATE_DATA_DIR / f"bm25_index_{CHUNK_MAX_CHARS}_{CHUNK_OVERLAP_CHARS}.json"
BM25_K1 = 1.5
BM25_B = 0.75
TOP_K = 5
MIN_SCORE_TO_ANSWER = 5.0
MAX_RETRIES = 3
MAX_ATTEMPTS = MAX_RETRIES + 1
EVAL_SET_PATH = PRIVATE_DATA_DIR / "eval_set.json"
EVAL_MIN_QUESTIONS = 15
EVAL_MAX_QUESTIONS = 20
EVAL_NO_ANSWER_QUESTIONS = 3

if not 0 <= CHUNK_OVERLAP_CHARS < CHUNK_MAX_CHARS:
    raise ValueError("CHUNK_OVERLAP_CHARS must be in [0, CHUNK_MAX_CHARS)")
if BM25_K1 < 0 or not 0 <= BM25_B <= 1:
    raise ValueError("BM25_K1 must be >= 0 and BM25_B must be in [0, 1]")
if TOP_K < 1 or MIN_SCORE_TO_ANSWER < 0:
    raise ValueError("TOP_K must be >= 1 and MIN_SCORE_TO_ANSWER must be >= 0")
if MAX_RETRIES < 0:
    raise ValueError("MAX_RETRIES must be >= 0")
if not 0 < EVAL_NO_ANSWER_QUESTIONS < EVAL_MIN_QUESTIONS <= EVAL_MAX_QUESTIONS:
    raise ValueError("Eval question counts must satisfy 0 < no-answer < min <= max")
