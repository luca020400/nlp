import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .embeddings import SemanticEncoder
from .ir import FunctionIR


@dataclass
class HybridIndex:
    functions: list[FunctionIR]
    vectorizer: TfidfVectorizer
    tfidf_matrix: sparse.spmatrix
    semantic_matrix: np.ndarray
    encoder_name: str

    @classmethod
    def build(
        cls, functions: list[FunctionIR], encoder: SemanticEncoder
    ) -> "HybridIndex":
        if not functions:
            raise ValueError("No functions were found")
        vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"\b\w+\b")
        lexical_texts = [item.lexical_text() for item in functions]
        tfidf_matrix = vectorizer.fit_transform(lexical_texts)
        semantic_texts = [item.semantic_text() for item in functions]
        semantic_matrix = encoder.encode(semantic_texts)
        semantic_matrix = _normalize_rows(semantic_matrix)
        return cls(functions, vectorizer, tfidf_matrix, semantic_matrix, encoder.name)

    # Dumps all the index to disk, metadata is plain json
    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "functions.json").write_text(
            json.dumps([item.to_dict() for item in self.functions], indent=2),
            encoding="utf-8",
        )
        with (path / "vectorizer.pkl").open("wb") as handle:
            pickle.dump(self.vectorizer, handle)
        sparse.save_npz(path / "tfidf.npz", self.tfidf_matrix)
        np.save(path / "semantic.npy", self.semantic_matrix)
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "encoder_name": self.encoder_name,
                    "function_count": len(self.functions),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "HybridIndex":
        path = Path(directory)
        functions = [
            FunctionIR.from_dict(item)
            for item in json.loads(
                (path / "functions.json").read_text(encoding="utf-8")
            )
        ]
        with (path / "vectorizer.pkl").open("rb") as handle:
            vectorizer = pickle.load(handle)
        tfidf_matrix = sparse.load_npz(path / "tfidf.npz")
        semantic_matrix = np.load(path / "semantic.npy")
        semantic_matrix = _normalize_rows(semantic_matrix)
        encoder_name = json.loads(
            (path / "metadata.json").read_text(encoding="utf-8")
        ).get(
            "encoder_name",
            "unknown",
        )
        return cls(functions, vectorizer, tfidf_matrix, semantic_matrix, encoder_name)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
