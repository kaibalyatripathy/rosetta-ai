"""
Unit tests for Tree-Sitter AST Analysis Layer.
"""

import json
import pytest
from pathlib import Path
from src.ast_analysis.extract import extract_structure


@pytest.fixture(scope="module")
def gold_fixtures():
    fixtures_file = Path("data/raw/rosetta_code_fixtures.json")
    assert fixtures_file.exists(), f"Gold fixtures file {fixtures_file} not found."
    with open(fixtures_file, "r", encoding="utf-8") as f:
        return json.load(f)


def test_extract_structure_all_20_fixtures(gold_fixtures):
    """Runs extract_structure on all 20 algorithm fixtures in Python, Java, C++, JavaScript."""
    assert len(gold_fixtures) == 20

    for fix in gold_fixtures:
        alg = fix["algorithm"]
        impls = fix["implementations"]

        for lang in ["python", "java", "cpp", "javascript"]:
            code = impls[lang]
            res = extract_structure(code, lang)

            assert isinstance(res["function_names"], list)
            assert isinstance(res["parameter_count"], int)
            assert isinstance(res["loops"], list)
            assert isinstance(res["conditionals"], list)
            assert isinstance(res["has_recursion"], bool)

            # Specific assertion checks against actual fixture properties
            if alg in {"bubble_sort", "selection_sort", "insertion_sort"}:
                assert "for" in res["loops"] or "while" in res["loops"], f"Expected loops in {alg} ({lang})"
            if alg in {"binary_search", "linear_search"}:
                assert "while" in res["loops"] or "for" in res["loops"], f"Expected loop in {alg} ({lang})"
            if "recursive" in alg:
                assert res["has_recursion"] is True, f"Expected recursion in {alg} ({lang})"


def test_bubble_sort_side_by_side_comparison(gold_fixtures):
    """
    Extracts and prints side-by-side structural comparison for the Bubble Sort fixture in all 4 languages.
    """
    bs_fixture = next(f for f in gold_fixtures if f["algorithm"] == "bubble_sort")
    impls = bs_fixture["implementations"]

    print("\n" + "=" * 80)
    print("BUBBLE SORT -- AST STRUCTURAL FACT EXTRACTION SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    for lang in ["python", "java", "cpp", "javascript"]:
        res = extract_structure(impls[lang], lang)
        print(f"\n--- Language: {lang.upper()} ---")
        print(f"Functions Extracted:  {res['function_names']}")
        print(f"Parameter Count:      {res['parameter_count']} ({res['parameter_names']})")
        print(f"Loops Present:        {res['loops']}")
        print(f"Conditionals Present: {res['conditionals']}")
        print(f"Recursion Detected:   {res['has_recursion']}")

    print("=" * 80 + "\n")
