from dataclasses import dataclass

import numpy as np

from .embeddings import SemanticEncoder
from .index import HybridIndex


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    lexical_score: float
    semantic_score: float
    function_id: str
    language: str
    file_path: str
    symbol_name: str
    qualified_name: str
    signature: str
    docstring: str
    start_line: int
    end_line: int


class HybridSearcher:
    def __init__(
        self,
        index: HybridIndex,
        encoder: SemanticEncoder,
        alpha: float = 0.5,
        fusion: str = "zero_anchored",
        rrf_k: int = 60,
    ) -> None:
        if fusion == "zero_anchored" and not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if fusion not in {"zero_anchored", "rrf"}:
            raise ValueError("fusion must be 'zero_anchored' or 'rrf'")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.index = index
        self.encoder = encoder
        self.alpha = alpha
        self.fusion = fusion
        self.rrf_k = rrf_k
        self._function_ids = np.array([item.id for item in self.index.functions])
        self._semantic_matrix = self._prepare_semantic_matrix()

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.index.functions:
            return []

        lexical_scores = self._lexical_scores(query)
        semantic_scores = self._semantic_scores(query)
        final_scores = self._fuse(lexical_scores, semantic_scores)

        if self.fusion == "rrf":
            # Took me a while to figure out the key is the 2nd entry.
            order = np.lexsort((self._function_ids, -final_scores))[:top_k]
        else:
            order = np.argsort(-final_scores)[:top_k]

        results = []
        for rank, idx in enumerate(order, start=1):
            item = self.index.functions[int(idx)]
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(final_scores[idx]),
                    lexical_score=float(lexical_scores[idx]),
                    semantic_score=float(semantic_scores[idx]),
                    function_id=item.id,
                    language=item.language,
                    file_path=item.file_path,
                    symbol_name=item.symbol_name,
                    qualified_name=item.qualified_name,
                    signature=item.signature,
                    docstring=item.docstring,
                    start_line=item.start_line,
                    end_line=item.end_line,
                )
            )
        return results

    def _lexical_scores(self, query: str) -> np.ndarray:
        query_vec = self.index.vectorizer.transform([query])
        scores = query_vec @ self.index.tfidf_matrix.T  # type: ignore
        return scores.toarray().ravel()

    def _semantic_scores(self, query: str) -> np.ndarray:
        query_vec = self.encoder.encode([query])[0]
        query_vec = self._normalize_vector(query_vec)
        return self._semantic_matrix @ query_vec

    def _fuse(self, lexical: np.ndarray, semantic: np.ndarray) -> np.ndarray:
        if self.fusion == "rrf":
            return self._reciprocal_rank_fusion(lexical, semantic)

        lexical = self._normalize_scores(lexical)
        semantic = self._normalize_scores(semantic)
        return (1.0 - self.alpha) * lexical + self.alpha * semantic

    def _reciprocal_rank_fusion(
        self, lexical: np.ndarray, semantic: np.ndarray
    ) -> np.ndarray:
        lexical_contribution = np.zeros(len(lexical), dtype=np.float64)
        matches = lexical > 0
        lexical_contribution[matches] = 1.0 / (
            self.rrf_k + self._average_ranks(lexical[matches])
        )
        semantic_contribution = 1.0 / (self.rrf_k + self._average_ranks(semantic))
        return lexical_contribution + semantic_contribution

    @staticmethod
    def _average_ranks(scores: np.ndarray) -> np.ndarray:
        ordered = np.sort(scores)
        left = np.searchsorted(ordered, scores, side="left")
        right = np.searchsorted(ordered, scores, side="right")
        return len(scores) - (left + right - 1) / 2.0

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        high = float(scores.max())
        if high == 0.0:
            return np.zeros_like(scores, dtype=np.float64)
        return scores / high

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return vector
        return vector / norm

    def _prepare_semantic_matrix(self) -> np.ndarray:
        # If we indexed with the same encoder, we can reuse the matrix.
        if self.index.encoder_name == self.encoder.name:
            return self.index.semantic_matrix
        # Re-encode otherwise, mostly used for benchmarks.
        texts = [item.semantic_text() for item in self.index.functions]
        matrix = self.encoder.encode(texts)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)
