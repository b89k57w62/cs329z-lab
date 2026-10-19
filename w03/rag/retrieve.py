from functools import cache
from .bm25 import BM25Index, build_index, load_index, save_index, score
from .chunker import Chunk, load_chunks
from .config import INDEX_PATH, MIN_SCORE_TO_ANSWER, TOP_K

@cache
def _index() -> BM25Index:
    if INDEX_PATH.exists():
        return load_index(INDEX_PATH)
    index = build_index(load_chunks())
    save_index(index, INDEX_PATH)
    return index

def retrieve(query: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
    if not query.strip():
        raise ValueError("Query must not be blank")
    if k < 1:
        raise ValueError("k must be >= 1")
    index = _index()
    ranked = sorted(score(index, query).items(), key=lambda item: (-item[1], item[0]))[:k]
    if not ranked or ranked[0][1] < MIN_SCORE_TO_ANSWER:
        return []
    return [(index.chunks[doc], doc_score) for doc, doc_score in ranked]
