from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from cs329z_hw1.tokenizer import tokenize

K1 = 1.5
B = 0.75
IDF_SMOOTHING = 0.5


@dataclass(frozen=True)
class BM25Index:
    postings: dict[str, tuple[tuple[int, int], ...]]
    length_norms: tuple[float, ...]


def build_index(docs: list[str]) -> BM25Index:
    postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths = []
    for doc_id, doc in enumerate(docs):
        tokens = tokenize(doc)
        lengths.append(len(tokens))
        for term, frequency in Counter(tokens).items():
            postings[term].append((doc_id, frequency))
    return BM25Index(
        postings={term: tuple(entries) for term, entries in postings.items()},
        length_norms=_length_norms(lengths),
    )


def search(index: BM25Index, query: str, k: int) -> list[tuple[int, float]]:
    scores: defaultdict[int, float] = defaultdict(float)
    document_count = len(index.length_norms)
    for term in dict.fromkeys(tokenize(query)):
        entries = index.postings.get(term, ())
        idf = _idf(document_count, len(entries))
        for doc_id, frequency in entries:
            scores[doc_id] += idf * frequency * (K1 + 1) / (frequency + index.length_norms[doc_id])
    return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))


def _length_norms(lengths: list[int]) -> tuple[float, ...]:
    total = sum(lengths)
    if total == 0:
        return ()
    average = total / len(lengths)
    return tuple(K1 * (1 - B + B * length / average) for length in lengths)


def _idf(document_count: int, document_frequency: int) -> float:
    ratio = (document_count - document_frequency + IDF_SMOOTHING) / (document_frequency + IDF_SMOOTHING)
    return math.log(ratio + 1)
