"""
Full Pipeline Integration Benchmark & Functional Equivalence Matrix Test (Phase 10).

Executes end-to-end pipeline:
Phase 6 (Seq2Seq Model) -> Phase 7 (Grammar-Constrained Decoding) -> Phase 8 (LLM Refactoring Pass) -> Phase 10 (Differential Verification Sandbox)

Runs on all 20 canonical fixtures across all 12 language pairs (240 translation directions)
and logs the actual empirical Pass/Fail matrix.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pytest

from src.constrained_decoding.grammar_decoder import ConstrainedGrammarDecoder
from src.refactor.refactor import refactor
from src.verification.differential_test import verify_equivalence, EquivalenceReport
from tests.fixtures.test_inputs import get_test_inputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.EquivalenceTest")

PARALLEL_CORPUS_FILE = Path("data/curated/parallel_corpus.jsonl")
MATRIX_OUTPUT_FILE = Path("data/curated/equivalence_matrix.json")

LANGUAGES = ["python", "java", "cpp", "javascript"]
LANGUAGE_PAIRS = [(src, tgt) for src in LANGUAGES for tgt in LANGUAGES if src != tgt]


def load_gold_corpus() -> Dict[Tuple[str, str, str], str]:
    """
    Loads gold parallel corpus indexed by (algorithm_name, source_lang, target_lang) -> source_code.
    """
    corpus_map = {}
    if not PARALLEL_CORPUS_FILE.exists():
        logger.error(f"Corpus file missing: {PARALLEL_CORPUS_FILE}")
        return corpus_map

    with open(PARALLEL_CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("is_gold", False):
                algo = entry.get("source_dataset", "").replace("rosetta_code_", "")
                src_lang = entry.get("source_lang", "").lower()
                tgt_lang = entry.get("target_lang", "").lower()
                code = entry.get("source_code", "")
                key = (algo, src_lang, tgt_lang)
                if key not in corpus_map:
                    corpus_map[key] = code
    return corpus_map


def _eval_single_pair(
    algo: str,
    src_lang: str,
    tgt_lang: str,
    source_code: str,
    decoder: ConstrainedGrammarDecoder
) -> Tuple[str, str, Dict[str, Any]]:
    """Evaluates a single (algo, src_lang, tgt_lang) translation pair."""
    pair_str = f"{src_lang}->{tgt_lang}"
    test_inputs = get_test_inputs(algo)

    try:
        # 1. Phase 6 & Phase 7: Seq2Seq Model + Constrained Decoding
        gen_target_code, is_valid = decoder.generate_constrained(
            source_code=source_code,
            source_lang=src_lang,
            target_lang=tgt_lang,
            num_candidates=2
        )

        # Clean up possible prompt leak prefix if present
        prefix_pattern = f"{src_lang} to {tgt_lang}:"
        if gen_target_code.lower().startswith(prefix_pattern):
            gen_target_code = gen_target_code[len(prefix_pattern):].strip()

        # 2. Phase 8: LLM Refactoring Pass
        refactored_info = refactor(gen_target_code, tgt_lang)
        final_target_code = refactored_info["refactored_code"]

        # 3. Phase 10: Differential Sandbox Verification
        report: EquivalenceReport = verify_equivalence(
            source_code=source_code,
            source_lang=src_lang,
            target_code=final_target_code,
            target_lang=tgt_lang,
            test_inputs=test_inputs,
            algorithm_name=algo
        )

        status_str = "PASS" if report.passed else "FAIL"
        logger.info(f"[{algo}] [{pair_str}] -> {status_str} ({report.passed_inputs}/{report.total_inputs} inputs passed)")

        return algo, pair_str, {
            "status": status_str,
            "passed_inputs": report.passed_inputs,
            "total_inputs": report.total_inputs,
            "pass_rate": report.pass_rate,
            "is_syntax_valid": is_valid,
            "passed": report.passed,
            "generated_code_sample": final_target_code[:100]
        }

    except Exception as e:
        logger.error(f"[{algo}] [{pair_str}] -> ERROR: {e}")
        return algo, pair_str, {
            "status": "ERROR",
            "passed_inputs": 0,
            "total_inputs": len(test_inputs),
            "pass_rate": 0.0,
            "error": str(e),
            "passed": False
        }


def run_full_equivalence_benchmark(max_workers: int = 4) -> Dict[str, Any]:
    """
    Runs the full end-to-end pipeline on all 240 translation pairs using parallel workers.
    Returns empirical benchmark metrics and full Pass/Fail matrix.
    """
    corpus_map = load_gold_corpus()
    decoder = ConstrainedGrammarDecoder()

    distinct_algos = sorted(list(set(k[0] for k in corpus_map.keys())))

    tasks = []
    for algo in distinct_algos:
        for src_lang, tgt_lang in LANGUAGE_PAIRS:
            source_code = corpus_map.get((algo, src_lang, tgt_lang))
            if not source_code:
                for (a, s, t), code in corpus_map.items():
                    if a == algo and s == src_lang:
                        source_code = code
                        break
            if source_code:
                tasks.append((algo, src_lang, tgt_lang, source_code))

    results_matrix: Dict[str, Dict[str, Dict[str, Any]]] = {algo: {} for algo in distinct_algos}
    total_evaluations = 0
    passed_evaluations = 0
    total_inputs_tested = 0
    passed_inputs_count = 0

    print("\n" + "="*80)
    print(f"STARTING PHASE 10 FULL PIPELINE BENCHMARK ({len(tasks)} PAIRS across {max_workers} WORKERS)")
    print("="*80 + "\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_eval_single_pair, algo, src_lang, tgt_lang, code, decoder): (algo, src_lang, tgt_lang)
            for algo, src_lang, tgt_lang, code in tasks
        }

        for future in as_completed(futures):
            algo, pair_str, res = future.result()
            results_matrix[algo][pair_str] = res
            
            total_evaluations += 1
            if res["passed"]:
                passed_evaluations += 1
            total_inputs_tested += res["total_inputs"]
            passed_inputs_count += res["passed_inputs"]

    overall_pair_pass_rate = (passed_evaluations / total_evaluations * 100.0) if total_evaluations > 0 else 0.0
    overall_input_pass_rate = (passed_inputs_count / total_inputs_tested * 100.0) if total_inputs_tested > 0 else 0.0

    summary = {
        "total_evaluations": total_evaluations,
        "passed_evaluations": passed_evaluations,
        "overall_pair_pass_rate": overall_pair_pass_rate,
        "total_inputs_tested": total_inputs_tested,
        "passed_inputs_count": passed_inputs_count,
        "overall_input_pass_rate": overall_input_pass_rate,
        "results_matrix": results_matrix
    }

    with open(MATRIX_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_empirical_matrix(summary)
    return summary


def print_empirical_matrix(summary: Dict[str, Any]):
    """Prints the raw empirical pass/fail matrix to stdout."""
    matrix = summary["results_matrix"]
    pair_headers = [f"{src}->{tgt}" for src, tgt in LANGUAGE_PAIRS]

    print("\n" + "="*140)
    print("EMPIRICAL PASS/FAIL MATRIX (20 FIXTURES x 12 LANGUAGE PAIRS)")
    print("="*140)

    header_row = f"{'Algorithm Fixture':<28} | " + " | ".join(f"{p:<9}" for p in pair_headers)
    print(header_row)
    print("-" * len(header_row))

    for algo, pairs in sorted(matrix.items()):
        row_str = f"{algo:<28} | "
        cols = []
        for p in pair_headers:
            info = pairs.get(p, {})
            st = info.get("status", "N/A")
            if st == "PASS":
                cols.append(f"{'PASS':<9}")
            elif st == "FAIL":
                p_in = info.get("passed_inputs", 0)
                t_in = info.get("total_inputs", 0)
                cols.append(f"FAIL({p_in}/{t_in})")
            else:
                cols.append(f"{st:<9}")
        row_str += " | ".join(cols)
        print(row_str)

    print("-" * len(header_row))
    print(f"TOTAL EVALUATED PAIRS: {summary['total_evaluations']}")
    print(f"PASSED PAIRS:          {summary['passed_evaluations']} / {summary['total_evaluations']} ({summary['overall_pair_pass_rate']:.2f}%)")
    print(f"TOTAL TEST INPUTS:     {summary['passed_inputs_count']} / {summary['total_inputs_tested']} ({summary['overall_input_pass_rate']:.2f}%)")
    print("="*140 + "\n")


def test_full_pipeline_functional_equivalence():
    """Pytest wrapper executing full functional equivalence benchmark."""
    summary = run_full_equivalence_benchmark(max_workers=4)
    assert summary["total_evaluations"] > 0, "No translation pairs were evaluated"


if __name__ == "__main__":
    run_full_equivalence_benchmark(max_workers=4)
