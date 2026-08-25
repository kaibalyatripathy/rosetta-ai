"""
Unit tests for PyTorch Geometric GNN AST Model and Similarity Benchmarks.
"""

import pytest
import torch
from src.gnn.graph_builder import build_pyg_ast_graph
from src.gnn.model import ASTGNNModel
from src.gnn.train import train_gnn_ast_model, load_fixture_graphs, evaluate_gnn_similarities


def test_build_pyg_ast_graph():
    code = "def add(a, b):\n    return a + b"
    data = build_pyg_ast_graph(code, "python", label=0)
    assert data.x is not None
    assert data.edge_index is not None
    assert data.num_nodes > 0


def test_gnn_model_forward():
    model = ASTGNNModel(hidden_dim=32, embed_dim=32, num_classes=20)
    data = build_pyg_ast_graph("int add(int a, int b) { return a + b; }", "cpp", label=1)
    
    emb = model.extract_graph_embedding(data.x, data.edge_index)
    assert emb.shape == (1, 32)
    
    logits = model(data.x, data.edge_index)
    assert logits.shape == (1, 20)


def test_gnn_similarity_evaluation():
    graphs, _ = load_fixture_graphs()
    model = ASTGNNModel(num_classes=20)
    
    # Evaluate model
    metrics = evaluate_gnn_similarities(model, graphs)
    assert "avg_equivalent_sim" in metrics
    assert "avg_random_sim" in metrics
    assert "delta" in metrics


def test_gnn_training_and_clustering():
    results = train_gnn_ast_model(epochs=15, lr=0.005)
    assert "loss_history" in results
    assert len(results["loss_history"]) == 15
    assert results["after"]["avg_equivalent_sim"] > results["after"]["avg_random_sim"]
    assert results["after"]["delta"] > 0.50

