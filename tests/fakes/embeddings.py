"""Deterministic embeddings — keeps unit tests off the 90 MB sentence-transformer."""

import hashlib
from typing import List

from langchain_core.embeddings import Embeddings


class DeterministicFakeEmbeddings(Embeddings):
    """Hash-derived vectors: same text always yields the same vector."""

    def __init__(self, size: int = 64):
        self.size = size

    def _vector(self, text: str) -> List[float]:
        values: List[float] = []
        counter = 0
        while len(values) < self.size:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            values.extend(byte / 255.0 for byte in digest)
            counter += 1
        return values[: self.size]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)
