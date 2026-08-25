"""
Script to execute Phase 15 Composite Semantic Preservation Scoring Benchmark.
"""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath("."))

from src.scoring.preservation_score import calculate_preservation_score, PreservationScoreReport

PARALLEL_CORPUS_FILE = Path("data/curated/parallel_corpus.jsonl")
SCORES_SUMMARY_FILE = Path("data/curated/preservation_scores_summary.json")


def main():
    print("\n" + "="*80)
    print("EXECUTING PHASE 15 COMPOSITE SEMANTIC PRESERVATION SCORING BENCHMARK")
    print("="*80 + "\n")

    corpus = []
    if PARALLEL_CORPUS_FILE.exists():
        with open(PARALLEL_CORPUS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("is_gold", False):
                        corpus.append(entry)

    scores_list = []
    grade_counts = {"EXCELLENT": 0, "GOOD": 0, "MODERATE / MARGINAL": 0, "POOR / FAILING": 0}

    for entry in corpus:
        algo = entry.get("source_dataset", "").replace("rosetta_code_", "")
        src_lang = entry.get("source_lang", "").lower()
        tgt_lang = entry.get("target_lang", "").lower()
        src_code = entry.get("source_code", "")
        tgt_code = entry.get("target_code", "")

        report: PreservationScoreReport = calculate_preservation_score(
            source_code=src_code,
            source_lang=src_lang,
            target_code=tgt_code,
            target_lang=tgt_lang,
            algorithm_name=algo,
            round_trip_passed=False
        )

        grade_counts[report.quality_grade] = grade_counts.get(report.quality_grade, 0) + 1
        scores_list.append({
            "algorithm": algo,
            "pair": f"{src_lang}->{tgt_lang}",
            "composite_score": report.composite_score,
            "grade": report.quality_grade,
            "score_equiv": report.score_equiv,
            "score_risk": report.score_risk,
            "score_complexity": report.score_complexity,
            "score_round_trip": report.score_round_trip
        })

    avg_score = (sum(s["composite_score"] for s in scores_list) / len(scores_list)) if scores_list else 0.0

    summary = {
        "total_pairs_evaluated": len(scores_list),
        "average_preservation_score": avg_score,
        "grade_counts": grade_counts,
        "scores_list": scores_list
    }

    with open(SCORES_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Total Translation Pairs Evaluated:  {len(scores_list)}")
    print(f"Average Composite Preservation Score: {avg_score:.2f} / 100")
    print("-" * 80)
    print("QUALITY GRADE DISTRIBUTION:")
    for grade, count in grade_counts.items():
        print(f"  - {grade:<22}: {count}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
