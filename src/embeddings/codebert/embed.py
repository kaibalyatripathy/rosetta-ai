"""
GraphCodeBERT Inference Embedding Wrapper for Rosetta AI.

Provides the `embed(code: str, lang: str) -> List[float]` function for extracting
fine-tuned semantic code embeddings.
"""

from pathlib import Path
from typing import List
from src.embeddings.codebert.codebert_extractor import CodeBERTExtractor

CHECKPOINT_DIR = Path("checkpoints/graphcodebert_rosetta_finetuned")

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class GraphCodeBERTEmbedder:
    """
    Inference wrapper for GraphCodeBERT.
    """
    def __init__(self, checkpoint_path: Path = CHECKPOINT_DIR):
        self.checkpoint_path = checkpoint_path
        self.fallback_extractor = CodeBERTExtractor(embedding_dim=128)
        self.tokenizer = None
        self.model = None

        if TRANSFORMERS_AVAILABLE:
            model_path = str(checkpoint_path) if checkpoint_path.exists() else "microsoft/graphcodebert-base"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModel.from_pretrained(model_path)
                self.model.eval()
            except Exception:
                self.model = None

    def embed(self, code: str, lang: str = "python") -> List[float]:
        """
        Extracts semantic embedding vector for a given source code snippet.
        """
        if self.model is not None and self.tokenizer is not None:
            with torch.no_grad():
                inputs = self.tokenizer(code, max_length=256, padding=True, truncation=True, return_tensors="pt")
                outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                norm_emb = F.normalize(cls_emb, p=2, dim=1)
                return norm_emb.squeeze(0).tolist()

        return self.fallback_extractor.extract_embedding(code, lang)


# Global embedder instance
_embedder_instance = None


def embed(code: str, lang: str = "python") -> List[float]:
    """
    Main API interface: embed(code, lang) -> List[float].
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = GraphCodeBERTEmbedder()
    return _embedder_instance.embed(code, lang)


if __name__ == "__main__":
    vec = embed("def add(a, b): return a + b", "python")
    print("Embedding Vector Length:", len(vec))
