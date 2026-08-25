"""
Round-Trip Integration Benchmark & Semantic Drift Test (Phase 12).

Executes A -> B -> A round-trip translation pipeline across all 20 canonical fixtures
using Python -> Java -> Python and Python -> C++ -> Python directions.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pytest

from src.constrained_decoding.grammar_decoder import ConstrainedGrammarDecoder
from src.round_trip.round_trip import round_trip_check, RoundTripReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.RoundTripTest")

PARALLEL_CORPUS_FILE = Path("data/curated/parallel_corpus.jsonl")
SUMMARY_OUTPUT_FILE = Path("data/curated/round_trip_summary.json")

INTERMEDIATE_LANGS = ["java", "cpp"]


def load_gold_python_corpus() -> Dict[str, str]:
    """Loads gold Python source code for each canonical algorithm fixture."""
    corpus_map = {}
    if not PARALLEL_CORPUS_FILE.exists():
        return corpus_map

    with open(PARALLEL_CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("is_gold", False) and entry.get("source_lang", "").lower() == "python":
                algo = entry.get("source_dataset", "").replace("rosetta_code_", "")
                code = entry.get("source_code", "")
                if algo not in corpus_map:
                    corpus_map[algo] = code
    return corpus_map


def _eval_single_round_trip(
    algo: str,
    source_code: str,
    source_lang: str,
    intermediate_lang: str,
    decoder: ConstrainedGrammarDecoder
) -> Tuple[str, str, Dict[str, Any]]:
    report: RoundTripReport = round_trip_check(
        source_code=source_code,
        source_lang=source_lang,
        intermediate_lang=intermediate_lang,
        algorithm_name=algo,
        decoder=decoder
    )
    
    pair_str = f"{source_lang}->{intermediate_lang}->{source_lang}"
    return algo, pair_str, {
        "passed": report.passed,
        "passed_inputs": report.passed_inputs,
        "total_inputs": report.total_inputs,
        "pass_rate": report.pass_rate,
        "semantic_drift_detected": report.semantic_drift_detected,
        "drift_details": report.drift_details,
        "intermediate_code_sample": report.intermediate_code[:100],
        "final_code_sample": report.final_code[:100]
    }


def run_round_trip_benchmark(max_workers: int = 4) -> Dict[str, Any]:
    corpus = load_gold_python_corpus()
    decoder = ConstrainedGrammarDecoder()

    tasks = []
    for algo, code in sorted(corpus.items()):
        for mid_lang in INTERMEDIATE_LANGS:
            tasks.append((algo, code, "python", mid_lang))

    total_evaluations = len(tasks)
    passed_evaluations = 0
    drift_count = 0
    results_matrix: Dict[str, Dict[str, Any]] = {}

    print("\n" + "="*80)
    print(f"STARTING PHASE 12 ROUND-TRIP BENCHMARK ({total_evaluations} PAIRS across {max_workers} WORKERS)")
    print("="*80 + "\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_eval_single_round_trip, algo, code, src, mid, decoder): (algo, mid)
            for algo, code, src, mid in tasks
        }

        for future in as_completed(futures):
            algo, pair_str, res = future.result()
            if algo not in results_matrix:
                results_matrix[algo] = {}
            results_matrix[algo][pair_str] = res

            if res["passed"]:
                passed_evaluations += 1
            if res["semantic_drift_detected"]:
                drift_count += 1

    overall_pass_rate = (passed_evaluations / total_evaluations * 100.0) if total_evaluations > 0 else 0.0

    summary = {
        "total_evaluations": total_evaluations,
        "passed_evaluations": passed_evaluations,
        "overall_pass_rate": overall_pass_rate,
        "drift_count": drift_count,
        "results_matrix": results_matrix
    }

    with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_round_trip_report(summary)
    return summary


def print_round_trip_report(summary: Dict[str, Any]):
    print("\n" + "="*80)
    print("PHASE 12 ROUND-TRIP VERIFICATION REPORT (A -> B -> A)")
    print("="*80)
    print(f"Total Round-Trip Paths Evaluated: {summary['total_evaluations']}")
    print(f"Passed Round-Trip Paths:          {summary['passed_evaluations']} / {summary['total_evaluations']} ({summary['overall_pass_rate']:.2f}%)")
    print(f"Semantic Drift Cases Detected:    {summary['drift_count']}")
    print("-" * 80)
    print("DETAILED FIXTURE BREAKDOWN:")
    for algo, paths in sorted(summary["results_matrix"].items()):
        row = f"  - {algo:<25}: "
        parts = []
        for path_name, data in sorted(paths.items()):
            st = "PASS" if data["passed"] else "FAIL"
            parts.append(f"{path_name} => {st}")
        print(row + " | ".join(parts))
    print("="*80 + "\n")


def test_round_trip_pipeline():
    summary = run_round_trip_benchmark(max_workers=4)
    assert summary["total_evaluations"] > 0, "No round trip paths evaluated"


if __name__ == "__main__":
    run_round_trip_benchmark(max_workers=4)
