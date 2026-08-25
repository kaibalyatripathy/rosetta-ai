"""
Inference Generator Wrapper for Conditioned Seq2Seq Transformer Model.

Provides `generate(source_code, source_lang, target_lang)` function for automated code translation.
"""

from typing import Optional
import torch

from src.embeddings.codebert.codebert_extractor import get_code_embedding
from src.gnn.graph_builder import build_pyg_ast_graph
from src.gnn.model import ASTGNNModel
from src.semantic_ir.fusion import fuse_embeddings
from src.seq2seq.model import ConditionedSeq2SeqModel, DEFAULT_MODEL_NAME
from src.seq2seq.train import CHECKPOINT_PATH, GNN_CHECKPOINT_PATH


def generate(
    source_code: str,
    source_lang: str,
    target_lang: str,
    model_name: str = "t5-small",
    checkpoint_path: Optional[str] = None
) -> str:
    """
    Translates source code snippet to target language using fine-tuned ConditionedSeq2SeqModel.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Pre-compute Fused Semantic Vector (Phase 5)
    cb_vec = get_code_embedding(source_code, source_lang, embedding_dim=128)
    
    gnn_model = ASTGNNModel(num_classes=20)
    if GNN_CHECKPOINT_PATH.exists():
        try:
            gnn_model.load_state_dict(torch.load(GNN_CHECKPOINT_PATH, weights_only=True))
        except Exception:
            pass
    gnn_model.eval()

    pyg_data = build_pyg_ast_graph(source_code, source_lang)
    with torch.no_grad():
        g_emb = gnn_model.extract_graph_embedding(pyg_data.x, pyg_data.edge_index)
    gnn_vec = g_emb.squeeze(0).tolist()

    fused_vec = fuse_embeddings(cb_vec, gnn_vec, fused_dim=128)

    # 2. Load Conditioned Seq2Seq Model
    model = ConditionedSeq2SeqModel(model_name=model_name)
    ckpt = checkpoint_path or (CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None)
    if ckpt:
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        except Exception:
            pass

    model.to(device)
    target_code = model.generate_code(
        source_code=source_code,
        source_lang=source_lang,
        target_lang=target_lang,
        fused_vec=fused_vec,
        device=device
    )
    return target_code


if __name__ == "__main__":
    py_snippet = "def add(a, b):\n    return a + b"
    js_code = generate(py_snippet, "python", "javascript")
    print("Generated Translation (Python -> JavaScript):")
    print(js_code)
