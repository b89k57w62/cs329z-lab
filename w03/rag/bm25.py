import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from functools import cached_property
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from .chunker import Chunk
from .config import BM25_K1, BM25_B

TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[㐀-䶿一-鿿]+")
CJK_NGRAM_SIZE = 2
INDEX_KEYS = ("chunks", "doc_lengths", "postings")

@dataclass(frozen=True)
class BM25Index:
    chunks: tuple[Chunk, ...]
    doc_lengths: tuple[int, ...]
    postings: dict[str, dict[int, int]]

    @cached_property
    def avg_doc_length(self) -> float:
        return sum(self.doc_lengths) / len(self.doc_lengths)

def _run_tokens(run: str) -> list[str]:
    if run.isascii() or len(run) < CJK_NGRAM_SIZE:
        return [run]
    return [run[start:start + CJK_NGRAM_SIZE] for start in range(len(run) - CJK_NGRAM_SIZE + 1)]

def tokenize(text: str) -> list[str]:
    return [token for run in TOKEN_PATTERN.findall(text.lower()) for token in _run_tokens(run)]

def build_index(chunks: list[Chunk]) -> BM25Index:
    if not chunks:
        raise ValueError("Cannot build an index from zero chunks")
    term_counts = [Counter(tokenize(chunk.index_text)) for chunk in chunks]
    triples = sorted(
        (term, doc, freq) for doc, counts in enumerate(term_counts) for term, freq in counts.items()
    )
    postings = {
        term: {doc: freq for _, doc, freq in group}
        for term, group in groupby(triples, key=itemgetter(0))
    }
    return BM25Index(
        chunks=tuple(chunks),
        doc_lengths=tuple(sum(counts.values()) for counts in term_counts),
        postings=postings,
    )

def _idf(index: BM25Index, term: str) -> float:
    doc_freq = len(index.postings[term])
    total = len(index.chunks)
    return math.log((total - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

def _term_score(index: BM25Index, term: str, freq: int, doc: int) -> float:
    length_norm = 1 - BM25_B + BM25_B * index.doc_lengths[doc] / index.avg_doc_length
    return _idf(index, term) * freq * (BM25_K1 + 1) / (freq + BM25_K1 * length_norm)

def score(index: BM25Index, query: str) -> dict[int, float]:
    terms = sorted(set(tokenize(query)) & index.postings.keys())
    contributions = sorted(
        ((doc, _term_score(index, term, freq, doc))
         for term in terms for doc, freq in index.postings[term].items()),
        key=itemgetter(0),
    )
    return {
        doc: sum(term_score for _, term_score in group)
        for doc, group in groupby(contributions, key=itemgetter(0))
    }

def save_index(index: BM25Index, path: Path) -> None:
    payload = {
        "chunks": [asdict(chunk) for chunk in index.chunks],
        "doc_lengths": list(index.doc_lengths),
        "postings": index.postings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

def load_index(path: Path) -> BM25Index:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in INDEX_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Index file {path} is missing keys: {missing}")
    if len(payload["chunks"]) != len(payload["doc_lengths"]):
        raise ValueError(f"Index file {path} has mismatched chunks and doc_lengths")
    return BM25Index(
        chunks=tuple(Chunk(**chunk) for chunk in payload["chunks"]),
        doc_lengths=tuple(payload["doc_lengths"]),
        postings={
            term: {int(doc): freq for doc, freq in docs.items()}
            for term, docs in payload["postings"].items()
        },
    )
