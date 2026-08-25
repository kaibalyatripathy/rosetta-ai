"""
Unit Tests & Computational Complexity Benchmark (Phase 14).

Compares estimated vs ground-truth Big-O complexity for both source and translated code
across all 20 canonical algorithm fixtures to evaluate complexity preservation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pytest

from src.complexity.estimator import estimate_complexity, ComplexityEstimate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.TestComplexity")

PARALLEL_CORPUS_FILE = Path("data/curated/parallel_corpus.jsonl")

# Ground-Truth Time Complexities for all 20 canonical algorithm fixtures
GROUND_TRUTH_COMPLEXITY = {
    "binary_search": "O(log n)",
    "bubble_sort": "O(n^2)",
    "factorial_recursive": "O(n)",
    "fibonacci_iterative": "O(n)",
    "gcd_euclidean": "O(log n)",
    "insertion_sort": "O(n^2)",
    "is_prime": "O(n)",
    "linear_search": "O(n)",
    "linked_list_node": "O(1)",
    "lru_cache_meta": "O(1)",
    "matrix_multiplication": "O(n^3)",
    "max_subarray_kadane": "O(n)",
    "merge_sort": "O(n log n)",
    "palindrome_check": "O(n)",
    "power_exponentiation": "O(log n)",
    "queue_array": "O(1)",
    "quick_sort": "O(n log n)",
    "reverse_string": "O(n)",
    "selection_sort": "O(n^2)",
    "stack_array": "O(1)"
}


def normalize_big_o(val: str) -> str:
    """Normalizes Big-O strings for comparison."""
    val = val.lower().replace(" ", "").replace("^", "**")
    if "o(1)" in val: return "O(1)"
    if "log" in val and "nlog" not in val and "n*log" not in val: return "O(log n)"
    if "n**3" in val or "n^3" in val: return "O(n^3)"
    if "n**2" in val or "n^2" in val: return "O(n^2)"
    if "nlog" in val or "n*log" in val or "n_log" in val: return "O(n log n)"
    if "o(n)" in val: return "O(n)"
    return val.upper()


def test_complexity_estimation_ground_truth():
    """
    Evaluates complexity estimator against ground-truth for source and target code across fixtures.
    """
    corpus = []
    if PARALLEL_CORPUS_FILE.exists():
        with open(PARALLEL_CORPUS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("is_gold", False):
                        corpus.append(entry)

    source_correct = 0
    target_correct = 0
    total_evaluated = 0
    complexity_degraded = 0

    print("\n" + "="*120)
    print("PHASE 14 COMPUTATIONAL COMPLEXITY ESTIMATION VS GROUND-TRUTH BENCHMARK")
    print("="*120)
    print(f"{'Algorithm':<22} | {'Pair':<12} | {'True Big-O':<10} | {'Src Est.':<10} | {'Tgt Est.':<10} | {'Preserved?':<10}")
    print("-" * 120)

    for entry in corpus:
        algo = entry.get("source_dataset", "").replace("rosetta_code_", "")
        src_lang = entry.get("source_lang", "").lower()
        tgt_lang = entry.get("target_lang", "").lower()
        src_code = entry.get("source_code", "")
        tgt_code = entry.get("target_code", "")

        true_big_o = GROUND_TRUTH_COMPLEXITY.get(algo, "O(n)")

        src_est: ComplexityEstimate = estimate_complexity(src_code, src_lang)
        tgt_est: ComplexityEstimate = estimate_complexity(tgt_code, tgt_lang)

        src_norm = normalize_big_o(src_est.time_complexity)
        tgt_norm = normalize_big_o(tgt_est.time_complexity)
        true_norm = normalize_big_o(true_big_o)

        total_evaluated += 1
        if src_norm == true_norm:
            source_correct += 1
        if tgt_norm == true_norm:
            target_correct += 1

        is_degraded = False
        if src_norm != tgt_norm and tgt_norm in ["O(n^2)", "O(n^3)"] and src_norm in ["O(1)", "O(log n)", "O(n)"]:
            is_degraded = True
            complexity_degraded += 1

        preserved_str = "DEGRADED" if is_degraded else "PRESERVED"
        print(f"{algo:<22} | {src_lang}->{tgt_lang:<8} | {true_norm:<10} | {src_norm:<10} | {tgt_norm:<10} | {preserved_str:<10}")

    src_acc = (source_correct / total_evaluated * 100.0) if total_evaluated > 0 else 0.0
    tgt_acc = (target_correct / total_evaluated * 100.0) if total_evaluated > 0 else 0.0

    print("-" * 120)
    print(f"Total Translation Pairs Evaluated:            {total_evaluated}")
    print(f"Source Code Complexity Accuracy vs Truth:     {source_correct} / {total_evaluated} ({src_acc:.2f}%)")
    print(f"Translated Code Complexity Accuracy vs Truth: {target_correct} / {total_evaluated} ({tgt_acc:.2f}%)")
    print(f"Complexity Degraded Translations Count:       {complexity_degraded}")
    print("="*120 + "\n")

    assert total_evaluated > 0, "No parallel pairs evaluated for complexity"


if __name__ == "__main__":
    test_complexity_estimation_ground_truth()
