"""
Fine-tuning Script for Conditioned Seq2Seq Code Translation Transformer.

Splits parallel corpus into Train (80%), Val (10%), and Test (10%),
trains `ConditionedSeq2SeqModel` using cross-entropy loss with teacher forcing,
and reports training and validation loss curves.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.embeddings.codebert.codebert_extractor import get_code_embedding
from src.gnn.graph_builder import build_pyg_ast_graph
from src.gnn.model import ASTGNNModel
from src.semantic_ir.fusion import fuse_embeddings
from src.seq2seq.model import ConditionedSeq2SeqModel, DEFAULT_MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.Seq2Seq.Train")

PARALLEL_CORPUS_PATH = Path("data/curated/parallel_corpus_large.jsonl")
SPLITS_PATH = Path("data/curated/dataset_splits.json")
CHECKPOINT_PATH = Path("checkpoints/seq2seq_codet5_finetuned.pt")
GNN_CHECKPOINT_PATH = Path("checkpoints/gnn_ast_model.pt")


class RosettaParallelDataset(Dataset):
    """
    PyTorch Dataset for Code Translation with Fused Semantic Vectors.
    """
    def __init__(self, data_list: List[Dict[str, Any]], tokenizer: Any, max_len: int = 256):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        src_code = item["source_code"]
        src_lang = item["source_lang"]
        tgt_lang = item["target_lang"]
        tgt_code = item["target_code"]
        fused_vec = item.get("fused_vec", [0.0] * 128)

        prompt = f"translate {src_lang} to {tgt_lang}: {src_code}"

        src_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        tgt_enc = self.tokenizer(
            tgt_code,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        labels = tgt_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": src_enc["input_ids"].squeeze(0),
            "attention_mask": src_enc["attention_mask"].squeeze(0),
            "labels": labels,
            "fused_embed": torch.tensor(fused_vec, dtype=torch.float32)
        }


def prepare_dataset_splits(seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads parallel_corpus.jsonl and performs an 80/10/10 train/val/test split.
    """
    if not PARALLEL_CORPUS_PATH.exists():
        raise FileNotFoundError(f"{PARALLEL_CORPUS_PATH} not found.")

    pairs = []
    with open(PARALLEL_CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    # Pre-compute fused vectors for dataset
    logger.info(f"Loaded {len(pairs)} parallel pairs. Pre-computing fused IR vectors...")
    
    gnn_model = ASTGNNModel(num_classes=20)
    if GNN_CHECKPOINT_PATH.exists():
        try:
            gnn_model.load_state_dict(torch.load(GNN_CHECKPOINT_PATH, weights_only=True))
        except Exception:
            pass
    gnn_model.eval()

    processed_pairs = []
    for item in pairs:
        src_code = item["source_code"]
        src_lang = item["source_lang"]
        
        # CodeBERT & GNN Vectors
        cb_vec = get_code_embedding(src_code, src_lang, embedding_dim=128)
        pyg_data = build_pyg_ast_graph(src_code, src_lang)
        with torch.no_grad():
            g_emb = gnn_model.extract_graph_embedding(pyg_data.x, pyg_data.edge_index)
        gnn_vec = g_emb.squeeze(0).tolist()

        fused_vec = fuse_embeddings(cb_vec, gnn_vec, fused_dim=128)
        item_copy = dict(item)
        item_copy["fused_vec"] = fused_vec
        processed_pairs.append(item_copy)

    # Shuffle with fixed seed
    rng = random.Random(seed)
    rng.shuffle(processed_pairs)

    n_total = len(processed_pairs)
    n_train = int(0.80 * n_total)
    n_val = int(0.10 * n_total)

    train_data = processed_pairs[:n_train]
    val_data = processed_pairs[n_train:n_train + n_val]
    test_data = processed_pairs[n_train + n_val:]

    logger.info(f"Dataset Split -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Save split indices/records to disk
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "test_samples": test_data
        }, f, indent=2)

    return train_data, val_data, test_data


def train_seq2seq_model(
    model_name: str = "t5-small",
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 5e-4,
    sanity_check: bool = False
) -> Dict[str, Any]:
    """
    Fine-tunes ConditionedSeq2SeqModel and outputs training/val loss history.
    """
    train_data, val_data, test_data = prepare_dataset_splits()

    if sanity_check:
        logger.info("Running SANITY CHECK fine-tuning on a 32-sample subset...")
        train_data = train_data[:32]
        val_data = val_data[:8]
        epochs = 2

    model = ConditionedSeq2SeqModel(model_name=model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_dataset = RosettaParallelDataset(train_data, model.tokenizer)
    val_dataset = RosettaParallelDataset(val_data, model.tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    logger.info(f"Starting Seq2Seq Fine-Tuning on {device} ({epochs} epochs)...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            fused_embed = batch["fused_embed"].to(device)

            outputs = model(input_ids, attention_mask, fused_embed, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
            if (batch_idx + 1) % 200 == 0:
                logger.info(f"Epoch {epoch:02d} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = round(total_train_loss / len(train_loader), 4)
        train_losses.append(avg_train_loss)

        # Validation Pass
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                fused_embed = batch["fused_embed"].to(device)

                outputs = model(input_ids, attention_mask, fused_embed, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = round(total_val_loss / len(val_loader), 4)
        val_losses.append(avg_val_loss)

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save Checkpoint
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    logger.info(f"Saved Seq2Seq fine-tuned model checkpoint to {CHECKPOINT_PATH}")

    return {
        "device": device,
        "epochs": epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_size": len(test_data)
    }


if __name__ == "__main__":
    train_seq2seq_model(sanity_check=False)
