"""
Unit tests for Semantic IR Fusion Layer and 3-Way Similarity Evaluation.
"""

from pathlib import Path
from src.semantic_ir.fusion import fuse_embeddings, RepresentationFusionMLP, run_3way_similarity_benchmark


def test_fuse_embeddings_output_shape():
    cb_vec = [0.1] * 128
    gnn_vec = [0.2] * 128
    fused_vec = fuse_embeddings(cb_vec, gnn_vec, fused_dim=128)

    assert isinstance(fused_vec, list)
    assert len(fused_vec) == 128
    assert any(v != 0.0 for v in fused_vec)


def test_3way_similarity_benchmark():
    results = run_3way_similarity_benchmark(epochs=2)

    assert "codebert_only" in results
    assert "gnn_only" in results
    assert "fused_representation" in results

    # Verify GNN AST-Only achieves strong separation gap
    assert results["gnn_only"]["delta"] > 0.50
    assert results["gnn_only"]["avg_equivalent_sim"] > 0.85
