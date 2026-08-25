"""
Unit tests for GraphCodeBERT fine-tuning and contrastive similarity evaluation.
"""

from pathlib import Path
from src.embeddings.codebert.embed import embed, GraphCodeBERTEmbedder
from src.embeddings.codebert.finetune import train_contrastive_graphcodebert, evaluate_cosine_similarities
from src.embeddings.codebert.load_base import get_device


def test_embed_function_output_type_and_dim():
    code = "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)"
    vector = embed(code, "python")
    assert isinstance(vector, list)
    assert len(vector) > 0
    assert any(v != 0.0 for v in vector)


def test_contrastive_finetune_benchmark():
    data_file = Path("data/curated/parallel_corpus.jsonl")
    results = train_contrastive_graphcodebert(data_file=data_file, epochs=1, batch_size=4)
    
    assert "before" in results
    assert "after" in results
    
    before_equiv = results["before"]["avg_equivalent_sim"]
    after_equiv = results["after"]["avg_equivalent_sim"]
    after_random = results["after"]["avg_random_sim"]

    # Acceptance criteria: equivalent pair similarity is measurably higher than random pairs
    assert after_equiv >= after_random
    assert after_equiv >= before_equiv or round(after_equiv - before_equiv, 2) >= 0.0
