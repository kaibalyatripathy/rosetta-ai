"""
GNN Model Architecture for Rosetta AI AST Graph Representation.

Implements ASTGraphGNN: Graph Convolutional / Attention layers for AST graphs.
Transforms AST nodes and structural/data-flow edges into a dense graph embedding vector.
"""

import math
from typing import Dict, List, Any, Union
from src.gnn.graph_builder import ASTGraph

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class ASTGraphGNN(nn.Module):
        """
        PyTorch GNN Module for AST Graph Structural Encoding.
        """
        def __init__(self, vocab_size: int = 1000, hidden_dim: int = 128, output_dim: int = 128):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            
            # Simple Graph Convolution Weights
            self.w_self = nn.Linear(hidden_dim, hidden_dim)
            self.w_neigh = nn.Linear(hidden_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, output_dim)

        def forward(self, node_indices: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """
            Forward pass for AST Graph GNN.
            node_indices: (N,) tensor of node type/vocab IDs
            edge_index: (2, E) tensor of graph edges
            Returns: (output_dim,) graph representation vector
            """
            h = self.embedding(node_indices)  # (N, hidden_dim)
            
            # Message Passing
            h_self = self.w_self(h)
            
            # Aggregate neighbor messages
            if edge_index.numel() > 0:
                src_nodes, dst_nodes = edge_index[0], edge_index[1]
                msg = h[src_nodes]  # (E, hidden_dim)
                
                # Scatter add messages to dst nodes
                h_neigh = torch.zeros_like(h)
                h_neigh.index_add_(0, dst_nodes, msg)
                h_neigh = self.w_neigh(h_neigh)
                h_next = F.relu(h_self + h_neigh)
            else:
                h_next = F.relu(h_self)

            # Global Mean Pooling over all AST nodes in the graph
            graph_emb = torch.mean(h_next, dim=0)  # (hidden_dim,)
            out = self.fc_out(graph_emb)
            return out
else:
    class ASTGraphGNN:
        """Fallback GNN class representation when PyTorch is unavailable."""
        def __init__(self, vocab_size: int = 1000, hidden_dim: int = 128, output_dim: int = 128):
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim



class DummyASTEncoder:
    """Fallback AST Graph Encoder using hashing & pooling when PyTorch is not available."""
    
    def __init__(self, output_dim: int = 128):
        self.output_dim = output_dim

    def encode(self, graph: ASTGraph) -> List[float]:
        vec = [0.0] * self.output_dim
        if not graph.nodes:
            return vec

        for idx, node in enumerate(graph.nodes):
            val_str = f"{node.label}_{node.node_type}_{node.value}"
            h = hash(val_str)
            dim_idx = abs(h) % self.output_dim
            vec[dim_idx] += 1.0

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def encode_ast_graph(graph: ASTGraph, hidden_dim: int = 128) -> Union[List[float], Any]:
    """Encodes an ASTGraph into a dense representation vector."""
    if TORCH_AVAILABLE:
        model = ASTGraphGNN(hidden_dim=hidden_dim, output_dim=hidden_dim)
        model.eval()
        
        # Simple node vocabulary indexing
        vocab_map: Dict[str, int] = {}
        node_ids = []
        for n in graph.nodes:
            key = f"{n.node_type}:{n.label}"
            if key not in vocab_map:
                vocab_map[key] = (hash(key) % 900) + 10
            node_ids.append(vocab_map[key])

        node_tensor = torch.tensor(node_ids, dtype=torch.long)
        edge_index_list = graph.get_edge_index()
        edge_tensor = torch.tensor(edge_index_list, dtype=torch.long)

        with torch.no_grad():
            emb = model(node_tensor, edge_tensor)
            return emb.tolist()
    else:
        encoder = DummyASTEncoder(output_dim=hidden_dim)
        return encoder.encode(graph)


if __name__ == "__main__":
    from src.gnn.graph_builder import build_ast_graph
    g = build_ast_graph("def foo(x):\n    return x * 2", "python")
    emb = encode_ast_graph(g, hidden_dim=64)
    print("Graph Embedding Vector Dimension:", len(emb))
