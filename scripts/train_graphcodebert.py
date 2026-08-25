"""
Standalone Executable Script for GraphCodeBERT Fine-Tuning.

Usage:
    python scripts/train_graphcodebert.py --epochs 3 --batch-size 8
"""

import argparse
from pathlib import Path
from src.embeddings.codebert.finetune import train_contrastive_graphcodebert


def main():
    parser = argparse.ArgumentParser(description="Rosetta AI GraphCodeBERT Fine-Tuning")
    parser.add_argument("--data-file", type=str, default="data/curated/parallel_corpus.jsonl", help="Path to parallel corpus JSONL")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    print("Starting GraphCodeBERT Fine-Tuning Pipeline...")
    results = train_contrastive_graphcodebert(
        data_file=Path(args.data_file),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    print("\n" + "=" * 60)
    print("GRAPHCODEBERT FINE-TUNING BENCHMARK RESULTS")
    print("=" * 60)
    if "before" in results:
        print("BEFORE Fine-Tuning Equivalent Pair Cosine Sim:", results["before"].get("avg_equivalent_sim"))
        print("BEFORE Fine-Tuning Random Pair Cosine Sim:    ", results["before"].get("avg_random_sim"))
    if "after" in results:
        print("AFTER Fine-Tuning Equivalent Pair Cosine Sim: ", results["after"].get("avg_equivalent_sim"))
        print("AFTER Fine-Tuning Random Pair Cosine Sim:     ", results["after"].get("avg_random_sim"))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
