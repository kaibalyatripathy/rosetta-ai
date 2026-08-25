"""
Unit tests for GNN Graph Builder and Model Architecture.
"""

import pytest
from src.gnn.graph_builder import build_ast_graph, ASTGraph
from src.gnn.model import encode_ast_graph, ASTGraphGNN, TORCH_AVAILABLE


def test_python_ast_graph_builder():
    code = "def add(a, b):\n    return a + b"
    graph: ASTGraph = build_ast_graph(code, "python")
    assert graph.language == "python"
    assert graph.num_nodes > 0
    assert len(graph.edges) > 0
    edge_index = graph.get_edge_index()
    assert len(edge_index) == 2
    assert len(edge_index[0]) == len(graph.edges)


def test_java_cpp_js_graph_builder():
    for lang in ["java", "cpp", "javascript"]:
        code = "int square(int x) { return x * x; }"
        graph: ASTGraph = build_ast_graph(code, lang)
        assert graph.language == lang
        assert graph.num_nodes > 0
        assert len(graph.edges) > 0


def test_encode_ast_graph():
    code = "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)"
    graph = build_ast_graph(code, "python")
    emb = encode_ast_graph(graph, hidden_dim=64)
    assert len(emb) == 64
    assert any(x != 0.0 for x in emb)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_pytorch_gnn_model_forward():
    import torch
    model = ASTGraphGNN(hidden_dim=32, output_dim=32)
    nodes = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = model(nodes, edges)
    assert out.shape == (32,)
