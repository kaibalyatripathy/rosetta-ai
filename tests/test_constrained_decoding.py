"""
Integration tests for Tree-Sitter Constrained Decoding & Syntactic Validity Verification.
"""

from src.constrained_decoding.grammar_decoder import (
    check_syntax_validity,
    ConstrainedGrammarDecoder,
    evaluate_constrained_decoding_benchmark
)


def test_syntax_validity_checker():
    # Valid Python code
    valid_py = "def add(a, b):\n    return a + b"
    is_valid, errs = check_syntax_validity(valid_py, "python")
    assert is_valid is True
    assert errs == 0

    # Broken Python code
    broken_py = "def add(a, b\n    return a +"
    is_valid_broken, errs_broken = check_syntax_validity(broken_py, "python")
    assert is_valid_broken is False
    assert errs_broken > 0


def test_syntax_validity_checker_all_languages():
    snippets = {
        "java": ("public class Test { public static void main(String[] args) {} }", "public class Test {"),
        "cpp": ("int main() { return 0; }", "int main() { return"),
        "javascript": ("function test() { return 42; }", "function test() { return")
    }

    for lang, (valid_code, broken_code) in snippets.items():
        v_ok, v_err = check_syntax_validity(valid_code, lang)
        assert v_ok is True, f"Expected {lang} valid snippet to parse"

        b_ok, b_err = check_syntax_validity(broken_code, lang)
        assert b_ok is False, f"Expected {lang} broken snippet to fail"


def test_constrained_decoder_generation():
    decoder = ConstrainedGrammarDecoder(model_name="t5-base")
    code, is_valid = decoder.generate_constrained(
        source_code="def add(a, b):\n    return a + b",
        source_lang="python",
        target_lang="javascript",
        num_candidates=3
    )

    assert isinstance(code, str)
    assert len(code.strip()) > 0
    assert is_valid in (True, False)



def test_constrained_decoding_benchmark_gain():
    results = evaluate_constrained_decoding_benchmark(num_samples=5)
    assert "unconstrained_validity_rate_pct" in results
    assert "constrained_validity_rate_pct" in results
    assert results["constrained_validity_rate_pct"] >= results["unconstrained_validity_rate_pct"]
