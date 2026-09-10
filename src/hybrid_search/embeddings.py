from typing import Protocol

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .unixcoder import UniXcoder


class SemanticEncoder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class ModelSemanticEncoder:
    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = batch.pop("special_tokens_mask")
        with torch.no_grad():
            output = self._model(**batch)

        token_mask = batch["attention_mask"] * (1 - special_tokens_mask)
        token_mask = token_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
        token_count = token_mask.sum(dim=1).clamp_min(1.0)

        pooled = (output.last_hidden_state * token_mask).sum(dim=1) / token_count
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return pooled.numpy().astype(np.float32)


class UniXcoderSemanticEncoder:
    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self._model = UniXcoder(model_name)
        self._model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        source_ids = self._model.tokenize(
            texts,
            mode="<encoder-only>",
            max_length=512,
            padding=True,
        )
        source_ids = torch.tensor(source_ids, dtype=torch.long)
        with torch.no_grad():
            _, sentence_embeddings = self._model(source_ids)
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings,
            p=2,
            dim=1,
        )
        return sentence_embeddings.numpy().astype(np.float32)


def make_encoder(name: str) -> SemanticEncoder:
    key = name.lower()
    if key in {"codebert", "microsoft/codebert-base"}:
        return ModelSemanticEncoder("microsoft/codebert-base")
    if key in {"unixcoder", "unixcoder-base", "microsoft/unixcoder-base"}:
        return UniXcoderSemanticEncoder("microsoft/unixcoder-base")
    raise ValueError(f"Unknown encoder: {name}")
