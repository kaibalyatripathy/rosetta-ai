"""
PyTorch Geometric Graph Neural Network (GNN) Model for AST Structural Representation.

Implements ASTGNNModel:
- Embedding layer mapping node types to continuous vectors
- 2-Layer GCN Graph Convolutional Network
- Global Mean Pooling over all AST graph nodes
- Linear projection head for structural embedding & auxiliary classification
"""

from typing import List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

TORCH_AVAILABLE = True


class ASTGNNModel(nn.Module):
    """
    2-Layer GCN Graph Neural Network for AST Structural Encoding.
    """
    def __init__(self, vocab_size: int = 150, node_dim: int = 64, hidden_dim: int = 128, embed_dim: int = 128, num_classes: int = 20):
        super().__init__()
        self.node_embedding = nn.Embedding(vocab_size, node_dim)
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)
        self.fc_proj = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def extract_graph_embedding(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Computes pooled graph-level embedding for AST graph input.
        x: (num_nodes,) integer node type IDs
        edge_index: (2, num_edges) edge indices
        batch: (num_nodes,) graph batch index assignment
        Returns: (batch_size, embed_dim) normalized embedding tensor
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.node_embedding(x)  # (num_nodes, node_dim)
        h = F.relu(self.conv1(h, edge_index))
        h = F.dropout(h, p=0.1, training=self.training)
        h = F.relu(self.conv2(h, edge_index))

        # Global Mean Pooling over AST nodes
        graph_emb = global_mean_pool(h, batch)  # (batch_size, embed_dim)
        out_emb = F.normalize(self.fc_proj(graph_emb), p=2, dim=1)
        return out_emb

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass returning classification logits for algorithm family.
        """
        emb = self.extract_graph_embedding(x, edge_index, batch)
        logits = self.classifier(emb)
        return logits


class ASTGraphGNN(ASTGNNModel):
    """Alias for ASTGNNModel."""
    pass


def encode_ast_graph(graph: Any, hidden_dim: int = 128) -> List[float]:
    """
    Encodes an ASTGraph or PyG Data object into a dense representation vector.
    """
    from src.gnn.graph_builder import build_pyg_ast_graph
    
    if hasattr(graph, 'x') and hasattr(graph, 'edge_index'):
        data = graph
    else:
        # Fallback build PyG graph
        data = build_pyg_ast_graph(getattr(graph, 'language', 'python'), getattr(graph, 'language', 'python'))

    model = ASTGNNModel(hidden_dim=hidden_dim, embed_dim=hidden_dim)
    model.eval()
    with torch.no_grad():
        emb = model.extract_graph_embedding(data.x, data.edge_index)
        return emb.squeeze(0).tolist()


if __name__ == "__main__":
    from src.gnn.graph_builder import build_pyg_ast_graph
    data = build_pyg_ast_graph("def add(a, b): return a + b", "python")
    model = ASTGNNModel()
    emb = model.extract_graph_embedding(data.x, data.edge_index)
    print("AST GNN Output Embedding Shape:", emb.shape)
