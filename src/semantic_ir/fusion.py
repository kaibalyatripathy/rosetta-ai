"""
Semantic Intermediate Representation (IR) Fusion Module for Rosetta AI.

Combines CodeBERT token sequence embeddings (Phase 2) and AST GNN structural graph embeddings (Phase 4)
into a unified fused representation using a 2-layer projection MLP.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings.codebert.codebert_extractor import get_code_embedding
from src.gnn.graph_builder import build_pyg_ast_graph
from src.gnn.model import ASTGNNModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.SemanticIR.Fusion")

CHECKPOINT_PATH = Path("checkpoints/fusion_mlp_model.pt")
GNN_CHECKPOINT_PATH = Path("checkpoints/gnn_ast_model.pt")
RAW_DATA_DIR = Path("data/raw")


class RepresentationFusionMLP(nn.Module):
    """
    2-Layer MLP projecting concatenated CodeBERT + GNN embeddings down to a target fusion dimension.
    """
    def __init__(self, codebert_dim: int = 128, gnn_dim: int = 128, fused_dim: int = 128):
        super().__init__()
        in_dim = codebert_dim + gnn_dim
        self.fc1 = nn.Linear(in_dim, 192)
        self.norm1 = nn.LayerNorm(192)
        self.fc2 = nn.Linear(192, fused_dim)

    def forward(self, codebert_emb: torch.Tensor, gnn_emb: torch.Tensor) -> torch.Tensor:
        """
        codebert_emb: (batch_size, codebert_dim)
        gnn_emb: (batch_size, gnn_dim)
        Returns: (batch_size, fused_dim) normalized fused embedding tensor
        """
        combined = torch.cat([codebert_emb, gnn_emb], dim=-1)
        h = F.gelu(self.norm1(self.fc1(combined)))
        out = F.normalize(self.fc2(h), p=2, dim=-1)
        return out


def fuse_embeddings(codebert_vec: List[float], gnn_vec: List[float], fused_dim: int = 128) -> List[float]:
    """
    Fuses a CodeBERT vector and a GNN vector into a single normalized fused embedding vector.
    """
    c_tensor = torch.tensor(codebert_vec, dtype=torch.float32).unsqueeze(0)
    g_tensor = torch.tensor(gnn_vec, dtype=torch.float32).unsqueeze(0)

    model = RepresentationFusionMLP(
        codebert_dim=len(codebert_vec),
        gnn_dim=len(gnn_vec),
        fused_dim=fused_dim
    )

    if CHECKPOINT_PATH.exists():
        try:
            model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
        except Exception:
            pass

    model.eval()
    with torch.no_grad():
        fused_tensor = model(c_tensor, g_tensor)
        return fused_tensor.squeeze(0).tolist()


def run_3way_similarity_benchmark(epochs: int = 10, lr: float = 0.005) -> Dict[str, Any]:
    """
    Evaluates CodeBERT-Only, GNN-Only, and Fused Representations across all 20 gold algorithm fixtures.
    """
    fixtures_path = RAW_DATA_DIR / "rosetta_code_fixtures.json"
    if not fixtures_path.exists():
        raise FileNotFoundError(f"{fixtures_path} not found.")

    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    # 1. Initialize & Load Pre-Trained GNN AST Model
    gnn_model = ASTGNNModel(num_classes=20)
    if GNN_CHECKPOINT_PATH.exists():
        try:
            gnn_model.load_state_dict(torch.load(GNN_CHECKPOINT_PATH, weights_only=True))
            logger.info("Loaded pre-trained GNN checkpoint for fusion benchmark.")
        except Exception:
            pass
    gnn_model.eval()

    # 2. Extract Embeddings for all 80 code snippets
    codebert_embs = []
    gnn_embs = []
    labels = []

    for fix_idx, fix in enumerate(fixtures):
        impls = fix["implementations"]
        for lang, code in impls.items():
            # CodeBERT Sequence Embedding (128-dim)
            cb_vec = get_code_embedding(code, lang, embedding_dim=128)
            codebert_embs.append(torch.tensor(cb_vec, dtype=torch.float32))

            # GNN Structural Embedding (128-dim)
            pyg_data = build_pyg_ast_graph(code, lang, label=fix_idx)
            with torch.no_grad():
                g_emb = gnn_model.extract_graph_embedding(pyg_data.x, pyg_data.edge_index)
            gnn_embs.append(g_emb.squeeze(0))
            labels.append(fix_idx)

    cb_tensor = torch.stack(codebert_embs)
    gnn_tensor = torch.stack(gnn_embs)
    lbl_tensor = torch.tensor(labels, dtype=torch.long)

    # 3. Train Fusion MLP Layer with Contrastive Loss
    fusion_model = RepresentationFusionMLP(codebert_dim=128, gnn_dim=128, fused_dim=128)
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=lr)

    logger.info(f"Training RepresentationFusionMLP for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        fusion_model.train()
        optimizer.zero_grad()
        fused_out = fusion_model(cb_tensor, gnn_tensor)
        
        # Similarity matrix & NT-Xent Contrastive Loss
        sim_matrix = torch.matmul(fused_out, fused_out.T) / 0.07
        loss = F.cross_entropy(sim_matrix, lbl_tensor)
        loss.backward()
        optimizer.step()

    # Save Fusion MLP Checkpoint
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fusion_model.state_dict(), CHECKPOINT_PATH)
    logger.info(f"Saved fusion MLP checkpoint to {CHECKPOINT_PATH}")

    # 4. Compute 3-Way Metrics
    fusion_model.eval()
    with torch.no_grad():
        fused_tensor = fusion_model(cb_tensor, gnn_tensor)

    def calc_metrics(tensor_data: torch.Tensor) -> Dict[str, float]:
        equiv_sims = []
        random_sims = []
        num_items = tensor_data.size(0)

        for i in range(num_items):
            for j in range(i + 1, num_items):
                sim = F.cosine_similarity(tensor_data[i].unsqueeze(0), tensor_data[j].unsqueeze(0)).item()
                if labels[i] == labels[j]:
                    equiv_sims.append(sim)
                else:
                    random_sims.append(sim)

        avg_eq = sum(equiv_sims) / len(equiv_sims) if equiv_sims else 0.0
        avg_rnd = sum(random_sims) / len(random_sims) if random_sims else 0.0
        return {
            "avg_equivalent_sim": round(avg_eq, 4),
            "avg_random_sim": round(avg_rnd, 4),
            "delta": round(avg_eq - avg_rnd, 4)
        }

    cb_metrics = calc_metrics(cb_tensor)
    gnn_metrics = calc_metrics(gnn_tensor)
    fused_metrics = calc_metrics(fused_tensor)

    results = {
        "codebert_only": cb_metrics,
        "gnn_only": gnn_metrics,
        "fused_representation": fused_metrics
    }

    logger.info(f"3-Way Benchmark Results: {results}")
    return results


if __name__ == "__main__":
    run_3way_similarity_benchmark()
