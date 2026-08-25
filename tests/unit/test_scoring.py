"""
Unit Tests & Example Report Generator for Semantic Preservation Scoring (Phase 15).

Generates 3 full example reports (High-Scoring, Middling-Scoring, Low-Scoring) to verify
that composite scores correlate sensibly with translation quality across evaluation phases.
"""

import logging
from pathlib import Path
import pytest

from src.scoring.preservation_score import calculate_preservation_score, PreservationScoreReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.TestScoring")

OUTPUT_DIR = Path("docs/example_reports")


def test_generate_three_example_reports():
    """Generates High, Middling, and Low scoring example reports for user review."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    py_bs_src = """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

    # -------------------------------------------------------------------------
    # 1. HIGH-SCORING EXAMPLE (Binary Search, Python -> JS, 100% Pass, No Risk, Round-Trip Stable)
    # -------------------------------------------------------------------------
    js_bs_tgt = """function binarySearch(arr, target) {
    let left = 0, right = arr.length - 1;
    while (left <= right) {
        let mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
"""

    high_rep: PreservationScoreReport = calculate_preservation_score(
        source_code=py_bs_src,
        source_lang="python",
        target_code=js_bs_tgt,
        target_lang="javascript",
        algorithm_name="binary_search",
        round_trip_passed=True
    )

    high_file = OUTPUT_DIR / "high_scoring_report.md"
    with open(high_file, "w", encoding="utf-8") as f:
        f.write(high_rep.markdown_report)

    # -------------------------------------------------------------------------
    # 2. MIDDLING-SCORING EXAMPLE (Binary Search, Python -> JS, Passes Sandbox, Round-Trip Failed & Risk Flagged)
    # -------------------------------------------------------------------------
    js_bs_loose = """function binarySearch(arr, target) {
    let left = 0, right = arr.length - 1;
    while (left <= right) {
        let mid = (left + right) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
"""

    mid_rep: PreservationScoreReport = calculate_preservation_score(
        source_code=py_bs_src,
        source_lang="python",
        target_code=js_bs_loose,
        target_lang="javascript",
        algorithm_name="binary_search",
        round_trip_passed=False
    )

    mid_file = OUTPUT_DIR / "middling_scoring_report.md"
    with open(mid_file, "w", encoding="utf-8") as f:
        f.write(mid_rep.markdown_report)

    # -------------------------------------------------------------------------
    # 3. LOW-SCORING EXAMPLE (Failing Translation with Prompt Leak & Syntax Fail)
    # -------------------------------------------------------------------------
    py_fail_src = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""

    fail_tgt = """python to javascript: def bubble_sort(arr): n = len(arr) for i in range(n): if arr[j] > "5" + 3: arr[j] = arr[j+1];"""

    low_rep: PreservationScoreReport = calculate_preservation_score(
        source_code=py_fail_src,
        source_lang="python",
        target_code=fail_tgt,
        target_lang="javascript",
        algorithm_name="bubble_sort",
        round_trip_passed=False
    )

    low_file = OUTPUT_DIR / "low_scoring_report.md"
    with open(low_file, "w", encoding="utf-8") as f:
        f.write(low_rep.markdown_report)

    print("\n" + "="*80)
    print("PHASE 15 THREE EXAMPLE REPORTS GENERATED FOR USER REVIEW")
    print("="*80)
    print(f"1. High-Scoring Report:     {high_rep.composite_score:.1f} / 100 ({high_rep.quality_grade}) -> {high_file}")
    print(f"2. Middling-Scoring Report: {mid_rep.composite_score:.1f} / 100 ({mid_rep.quality_grade}) -> {mid_file}")
    print(f"3. Low-Scoring Report:      {low_rep.composite_score:.1f} / 100 ({low_rep.quality_grade}) -> {low_file}")
    print("="*80 + "\n")

    assert high_rep.composite_score > mid_rep.composite_score > low_rep.composite_score, f"Score ordering broken: {high_rep.composite_score} vs {mid_rep.composite_score} vs {low_rep.composite_score}"


if __name__ == "__main__":
    test_generate_three_example_reports()
