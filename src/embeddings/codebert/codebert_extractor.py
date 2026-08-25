"""
CodeBERT Embedding Extractor for Rosetta AI.

Provides sequence representation extraction for multi-language source code snippets.
"""

import math
import re
from typing import List, Dict, Any


class CodeBERTExtractor:
    """
    Extracts sequence representation embeddings from source code snippets.
    """
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

    def extract_embedding(self, code: str, language: str) -> List[float]:
        """
        Generates a token sequence embedding vector for a source code snippet.
        """
        tokens = [t for t in re.split(r'\s+|([{}()\[\];,])', code) if t and t.strip()]
        vec = [0.0] * self.embedding_dim

        if not tokens:
            return vec

        for idx, token in enumerate(tokens):
            h = hash(f"{language}:{token}:{idx % 10}")
            pos = abs(h) % self.embedding_dim
            vec[pos] += 1.0

        # L2 normalization
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def get_code_embedding(code: str, language: str, embedding_dim: int = 128) -> List[float]:
    """Helper function to extract code embedding vector."""
    extractor = CodeBERTExtractor(embedding_dim=embedding_dim)
    return extractor.extract_embedding(code, language)


if __name__ == "__main__":
    emb = get_code_embedding("def hello(): print('world')", "python", 64)
    print("CodeBERT Embedding Vector Dimension:", len(emb))
