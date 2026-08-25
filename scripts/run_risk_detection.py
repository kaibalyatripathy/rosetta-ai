"""
Script to execute Phase 13 Deterministic Semantic Risk Detector across all translation pairs.
"""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath("."))

from src.risk_detection.risk_rules import detect_semantic_risks, RiskAnalysisReport

PARALLEL_CORPUS_FILE = Path("data/curated/parallel_corpus.jsonl")
RISK_SUMMARY_FILE = Path("data/curated/semantic_risk_summary.json")

LANGUAGES = ["python", "java", "cpp", "javascript"]
LANGUAGE_PAIRS = [(src, tgt) for src in LANGUAGES for tgt in LANGUAGES if src != tgt]


def main():
    print("\n" + "="*80)
    print("EXECUTING PHASE 13 DETERMINISTIC SEMANTIC RISK DETECTION BENCHMARK")
    print("="*80 + "\n")

    corpus = []
    if PARALLEL_CORPUS_FILE.exists():
        with open(PARALLEL_CORPUS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("is_gold", False):
                        corpus.append(entry)

    flagged_pairs = []
    category_counts = {
        "INTEGER_OVERFLOW": 0,
        "INDEX_BOUNDARY": 0,
        "TYPE_COERCION": 0,
        "FLOAT_PRECISION": 0,
        "MEMORY_MANAGEMENT": 0
    }

    for entry in corpus:
        algo = entry.get("source_dataset", "").replace("rosetta_code_", "")
        src_lang = entry.get("source_lang", "").lower()
        tgt_lang = entry.get("target_lang", "").lower()
        src_code = entry.get("source_code", "")
        tgt_code = entry.get("target_code", "")

        report: RiskAnalysisReport = detect_semantic_risks(src_code, src_lang, tgt_code, tgt_lang)
        if report.flagged_risks:
            flagged_pairs.append({
                "algorithm": algo,
                "pair": f"{src_lang}->{tgt_lang}",
                "risk_score": report.risk_score,
                "has_high_risk": report.has_high_risk,
                "risks": [
                    {
                        "category": r.category,
                        "severity": r.severity,
                        "description": r.description,
                        "matched_pattern": r.matched_pattern
                    }
                    for r in report.flagged_risks
                ]
            })
            for r in report.flagged_risks:
                category_counts[r.category] = category_counts.get(r.category, 0) + 1

    summary = {
        "total_gold_pairs_scanned": len(corpus),
        "total_risk_flagged_pairs": len(flagged_pairs),
        "category_counts": category_counts,
        "flagged_pairs": flagged_pairs
    }

    with open(RISK_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Total Parallel Code Pairs Scanned: {len(corpus)}")
    print(f"Total Risk-Flagged Translation Pairs: {len(flagged_pairs)}")
    print("-" * 80)
    print("RISK CATEGORY DISTRIBUTION:")
    for cat, cnt in category_counts.items():
        print(f"  - {cat:<22}: {cnt}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
