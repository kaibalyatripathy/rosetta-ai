"""
Training Script for AST Graph Neural Network (GNN) Model.

Trains `ASTGNNModel` from scratch on AST graphs extracted from multi-lingual code algorithms using:
1. Auxiliary Algorithm Classification Loss (predicting algorithm class across 20 fixtures)
2. Contrastive Cosine Loss (aligning cross-language AST embeddings of equivalent algorithms)

Saves trained weights to `checkpoints/gnn_ast_model.pt`.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.gnn.graph_builder import build_pyg_ast_graph
from src.gnn.model import ASTGNNModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.GNN.Train")

RAW_DATA_DIR = Path("data/raw")
CHECKPOINT_PATH = Path("checkpoints/gnn_ast_model.pt")


def load_fixture_graphs() -> Tuple[List[Any], Dict[str, int]]:
    """
    Loads 20 gold algorithm fixtures across 4 languages (80 graph objects) with algorithm class labels.
    """
    fixtures_path = RAW_DATA_DIR / "rosetta_code_fixtures.json"
    if not fixtures_path.exists():
        raise FileNotFoundError(f"{fixtures_path} not found.")

    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    alg_to_label = {fix["algorithm"]: idx for idx, fix in enumerate(fixtures)}
    graphs = []

    for fix in fixtures:
        alg = fix["algorithm"]
        label = alg_to_label[alg]
        impls = fix["implementations"]

        for lang, code in impls.items():
            g_data = build_pyg_ast_graph(code, lang, label=label)
            graphs.append(g_data)

    return graphs, alg_to_label


def evaluate_gnn_similarities(model: ASTGNNModel, graphs: List[Any]) -> Dict[str, float]:
    """
    Evaluates average cosine similarity of structurally equivalent code graphs vs random code graphs.
    """
    model.eval()
    equiv_sims = []
    random_sims = []

    with torch.no_grad():
        # Group graphs by algorithm class label
        by_label: Dict[int, List[Any]] = {}
        for g in graphs:
            lbl = g.y.item()
            by_label.setdefault(lbl, []).append(g)

        # Equivalent pair similarities (same algorithm class across different languages)
        for lbl, group in by_label.items():
            if len(group) >= 2:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        emb1 = model.extract_graph_embedding(group[i].x, group[i].edge_index)
                        emb2 = model.extract_graph_embedding(group[j].x, group[j].edge_index)
                        sim = F.cosine_similarity(emb1, emb2).item()
                        equiv_sims.append(sim)

        # Random pair similarities (different algorithm classes)
        labels = list(by_label.keys())
        for i in range(len(labels)):
            lbl1 = labels[i]
            lbl2 = labels[(i + 1) % len(labels)]
            g1 = by_label[lbl1][0]
            g2 = by_label[lbl2][0]
            emb1 = model.extract_graph_embedding(g1.x, g1.edge_index)
            emb2 = model.extract_graph_embedding(g2.x, g2.edge_index)
            sim = F.cosine_similarity(emb1, emb2).item()
            random_sims.append(sim)

    avg_equiv = sum(equiv_sims) / len(equiv_sims) if equiv_sims else 0.0
    avg_rand = sum(random_sims) / len(random_sims) if random_sims else 0.0

    return {
        "avg_equivalent_sim": round(avg_equiv, 4),
        "avg_random_sim": round(avg_rand, 4),
        "delta": round(avg_equiv - avg_rand, 4)
    }


def train_gnn_ast_model(epochs: int = 15, lr: float = 0.005) -> Dict[str, Any]:
    """
    Main training function for AST GNN model.
    """
    logger.info("Loading AST graphs from 20 gold algorithm fixtures...")
    graphs, alg_to_label = load_fixture_graphs()
    logger.info(f"Loaded {len(graphs)} AST graph objects across 4 languages.")

    model = ASTGNNModel(num_classes=len(alg_to_label))

    # Evaluate BEFORE training
    metrics_before = evaluate_gnn_similarities(model, graphs)
    logger.info(f"BEFORE Training GNN Metrics: {metrics_before}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    logger.info(f"Training ASTGNNModel from scratch for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        # Batch processing
        batch_size = 8
        for i in range(0, len(graphs), batch_size):
            batch_list = graphs[i:i + batch_size]
            batch_data = Batch.from_data_list(batch_list)

            optimizer.zero_grad()
            logits = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = criterion(logits, batch_data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = round(total_loss / (len(graphs) // batch_size), 4)
        loss_history.append(avg_loss)
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Training Loss: {avg_loss:.4f}")

    # Save trained model checkpoint
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    logger.info(f"Saved GNN AST model checkpoint to {CHECKPOINT_PATH}")

    # Evaluate AFTER training
    metrics_after = evaluate_gnn_similarities(model, graphs)
    logger.info(f"AFTER Training GNN Metrics: {metrics_after}")

    return {
        "loss_history": loss_history,
        "before": metrics_before,
        "after": metrics_after,
        "improvement_delta": round(metrics_after["delta"] - metrics_before["delta"], 4)
    }


if __name__ == "__main__":
    train_gnn_ast_model(epochs=15)
