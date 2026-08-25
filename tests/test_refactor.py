"""
Unit and Integration Tests for Refactoring Pass & Multi-Lingual Style Linters.
"""

import json
from pathlib import Path
import pytest

from src.refactor.style_check import compute_style_score
from src.refactor.refactor import refactor
from src.constrained_decoding.grammar_decoder import ConstrainedGrammarDecoder
from src.seq2seq.train import SPLITS_PATH


def test_style_checker_scoring():
    # Bad Python style
    bad_py = "def MyFunction(a,b):\n    x=a+b\n    return x"
    res_bad = compute_style_score(bad_py, "python")
    assert res_bad["score"] < 100.0
    assert res_bad["warnings_count"] > 0

    # Good Python style
    good_py = "def my_function(a: int, b: int) -> int:\n    # Computes sum\n    return a + b"
    res_good = compute_style_score(good_py, "python")
    assert res_good["score"] >= res_bad["score"]


def test_refactor_pass_fallback():
    raw_js = "var x=10;\nvar y=20;"
    res = refactor(raw_js, "javascript")
    assert "refactored_code" in res
    assert res["after_score"] >= res["before_score"]


def test_refactor_pass_held_out_benchmark():
    assert SPLITS_PATH.exists(), f"Split dataset file {SPLITS_PATH} not found."
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    test_samples = data["test_samples"][:10]

    decoder = ConstrainedGrammarDecoder(model_name="t5-base")
    improved_count = 0

    print("\n" + "=" * 80)
    print("PHASE 8 -- REFACTORING PASS LINT SCORE BENCHMARK")
    print("=" * 80)

    for idx, sample in enumerate(test_samples, 1):
        src_code = sample["source_code"]
        src_lang = sample["source_lang"]
        tgt_lang = sample["target_lang"]

        # 1. Phase 7 Constrained Generation
        gen_code, _ = decoder.generate_constrained(src_code, src_lang, tgt_lang, num_candidates=3)

        # 2. Phase 8 Refactoring Pass
        ref_res = refactor(gen_code, tgt_lang)
        b_score = ref_res["before_score"]
        a_score = ref_res["after_score"]

        if a_score >= b_score:
            improved_count += 1

        print(f"\n--- Sample {idx}/10 [{src_lang} -> {tgt_lang}] ---")
        print(f"Pre-Refactor Style Score:  {b_score}/100 ({ref_res['before_warnings']} warnings)")
        print(f"Post-Refactor Style Score: {a_score}/100 ({ref_res['after_warnings']} warnings)")
        print(f"IMPROVED: {ref_res['score_improved']}")

    print("=" * 80 + "\n")
    # Acceptance criteria: equal or improved score in at least 8/10 cases
    assert improved_count >= 8, f"Expected at least 8/10 improved style scores, got {improved_count}/10"
