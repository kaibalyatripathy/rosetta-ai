"""
Self-Correction Integration Benchmark & Verification Test (Phase 11).

Executes automated LLM self-correction loop on all translation cases that failed
Phase 10 functional equivalence testing, logs hard examples to `data/curated/hard_examples.jsonl`,
and reports before/after pass rates and attempt breakdown.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pytest

from src.constrained_decoding.grammar_decoder import ConstrainedGrammarDecoder
from src.refactor.refactor import refactor
from src.self_correction.corrector import attempt_correction, CorrectionResult
from src.self_correction.hard_examples_log import log_hard_example
from src.verification.differential_test import verify_equivalence, EquivalenceReport
from tests.fixtures.test_inputs import get_test_inputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.SelfCorrectionTest")

PARALLEL_CORPUS_FILE = Path("data/curated/parallel_corpus.jsonl")
CORRECTION_SUMMARY_FILE = Path("data/curated/self_correction_summary.json")

LANGUAGES = ["python", "java", "cpp", "javascript"]
LANGUAGE_PAIRS = [(src, tgt) for src in LANGUAGES for tgt in LANGUAGES if src != tgt]


def load_gold_corpus() -> Dict[Tuple[str, str, str], str]:
    corpus_map = {}
    if not PARALLEL_CORPUS_FILE.exists():
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


def _eval_and_correct_pair(
    algo: str,
    src_lang: str,
    tgt_lang: str,
    source_code: str,
    decoder: ConstrainedGrammarDecoder,
    max_attempts: int = 3
) -> Dict[str, Any]:
    pair_str = f"{src_lang}->{tgt_lang}"
    test_inputs = get_test_inputs(algo)

    try:
        # Phase 6 & 7: Seq2Seq + Constrained Decoding
        gen_target_code, is_valid = decoder.generate_constrained(
            source_code=source_code,
            source_lang=src_lang,
            target_lang=tgt_lang,
            num_candidates=2
        )

        # Phase 8: Refactor Pass
        refactored_info = refactor(gen_target_code, tgt_lang)
        initial_target_code = refactored_info["refactored_code"]

        # Phase 10: Initial Differential Verification
        initial_report: EquivalenceReport = verify_equivalence(
            source_code=source_code,
            source_lang=src_lang,
            target_code=initial_target_code,
            target_lang=tgt_lang,
            test_inputs=test_inputs,
            algorithm_name=algo
        )

        initial_passed = initial_report.passed
        final_passed = initial_passed
        attempts_used = 0
        corrected_code = initial_target_code

        if not initial_passed:
            # Phase 11: Self-Correction Loop
            corr_res: CorrectionResult = attempt_correction(
                source_code=source_code,
                source_lang=src_lang,
                target_code=initial_target_code,
                target_lang=tgt_lang,
                test_inputs=test_inputs,
                algorithm_name=algo,
                max_attempts=max_attempts
            )
            final_passed = corr_res.success
            attempts_used = corr_res.attempts_used
            corrected_code = corr_res.corrected_code

            # Log to data/curated/hard_examples.jsonl
            log_hard_example(
                source_code=source_code,
                source_lang=src_lang,
                failed_target_code=initial_target_code,
                target_lang=tgt_lang,
                fixed_target_code=corrected_code if final_passed else None,
                attempts_used=attempts_used,
                success=final_passed,
                failure_details={"algorithm": algo, "initial_pass_rate": initial_report.pass_rate}
            )

        return {
            "algo": algo,
            "pair": pair_str,
            "initial_passed": initial_passed,
            "final_passed": final_passed,
            "attempts_used": attempts_used,
            "required_correction": not initial_passed
        }

    except Exception as e:
        logger.error(f"[{algo}] [{pair_str}] Error during self-correction test: {e}")
        return {
            "algo": algo,
            "pair": pair_str,
            "initial_passed": False,
            "final_passed": False,
            "attempts_used": max_attempts,
            "required_correction": True,
            "error": str(e)
        }


def run_self_correction_benchmark(max_workers: int = 4, max_attempts: int = 3) -> Dict[str, Any]:
    corpus_map = load_gold_corpus()
    decoder = ConstrainedGrammarDecoder()

    distinct_algos = sorted(list(set(k[0] for k in corpus_map.keys())))
    tasks = []
    for algo in distinct_algos:
        for src_lang, tgt_lang in LANGUAGE_PAIRS:
            code = corpus_map.get((algo, src_lang, tgt_lang))
            if not code:
                for (a, s, t), c in corpus_map.items():
                    if a == algo and s == src_lang:
                        code = c
                        break
            if code:
                tasks.append((algo, src_lang, tgt_lang, code))

    total_pairs = len(tasks)
    initial_passed_count = 0
    final_passed_count = 0

    attempt_breakdown = {
        "fixed_attempt_1": 0,
        "fixed_attempt_2": 0,
        "fixed_attempt_3": 0,
        "remained_broken": 0
    }

    results_list = []

    print("\n" + "="*80)
    print(f"STARTING PHASE 11 SELF-CORRECTION BENCHMARK ({total_pairs} PAIRS, MAX ATTEMPTS={max_attempts})")
    print("="*80 + "\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_eval_and_correct_pair, algo, src, tgt, code, decoder, max_attempts)
            for algo, src, tgt, code in tasks
        ]

        for future in as_completed(futures):
            res = future.result()
            results_list.append(res)
            
            if res["initial_passed"]:
                initial_passed_count += 1
                final_passed_count += 1
            else:
                if res["final_passed"]:
                    final_passed_count += 1
                    att = res["attempts_used"]
                    if att == 1:
                        attempt_breakdown["fixed_attempt_1"] += 1
                    elif att == 2:
                        attempt_breakdown["fixed_attempt_2"] += 1
                    elif att == 3:
                        attempt_breakdown["fixed_attempt_3"] += 1
                else:
                    attempt_breakdown["remained_broken"] += 1

    initial_pass_rate = (initial_passed_count / total_pairs * 100.0) if total_pairs > 0 else 0.0
    final_pass_rate = (final_passed_count / total_pairs * 100.0) if total_pairs > 0 else 0.0

    summary = {
        "total_pairs": total_pairs,
        "initial_passed_count": initial_passed_count,
        "initial_pass_rate": initial_pass_rate,
        "final_passed_count": final_passed_count,
        "final_pass_rate": final_pass_rate,
        "attempt_breakdown": attempt_breakdown,
        "results_list": results_list
    }

    with open(CORRECTION_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_self_correction_report(summary)
    return summary


def print_self_correction_report(summary: Dict[str, Any]):
    print("\n" + "="*80)
    print("PHASE 11 SELF-CORRECTION BENCHMARK REPORT")
    print("="*80)
    print(f"Total Translation Pairs Evaluated:  {summary['total_pairs']}")
    print(f"Phase 10 Initial Pass Rate (Before): {summary['initial_passed_count']} / {summary['total_pairs']} ({summary['initial_pass_rate']:.2f}%)")
    print(f"Phase 11 Post-Correction Pass Rate: {summary['final_passed_count']} / {summary['total_pairs']} ({summary['final_pass_rate']:.2f}%)")
    print(f"Net Pass Rate Gain:                 +{(summary['final_pass_rate'] - summary['initial_pass_rate']):.2f}%")
    print("-" * 80)
    print("ATTEMPT BREAKDOWN FOR FAILING CASES:")
    breakdown = summary["attempt_breakdown"]
    print(f"  - Fixed in 1 Attempt:               {breakdown['fixed_attempt_1']}")
    print(f"  - Fixed in 2 Attempts:              {breakdown['fixed_attempt_2']}")
    print(f"  - Fixed in 3 Attempts:              {breakdown['fixed_attempt_3']}")
    print(f"  - Remained Broken After 3 Attempts: {breakdown['remained_broken']}")
    print("="*80 + "\n")


def test_self_correction_pipeline():
    summary = run_self_correction_benchmark(max_workers=4, max_attempts=3)
    assert summary["total_pairs"] > 0, "No pairs evaluated"


if __name__ == "__main__":
    run_self_correction_benchmark(max_workers=4, max_attempts=3)
