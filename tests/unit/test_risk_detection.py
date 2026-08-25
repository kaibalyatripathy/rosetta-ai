"""
Unit Tests & Targeted Adversarial Risk Verification (Phase 13).

Evaluates the rule-based semantic risk detector across translation pairs and runs
targeted adversarial inputs through the Phase 9 Docker sandbox to prove flagged risks.
"""

import logging
import pytest

from src.risk_detection.risk_rules import detect_semantic_risks, RiskAnalysisReport
from src.sandbox.runner import run_in_sandbox

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.TestRiskDetection")


def test_rule_based_risk_detector_unit():
    """Unit test for risk detector rules."""
    # 1. Test Integer Overflow rule
    py_code = "def power(base, exp):\n    return base ** exp"
    cpp_code = "int power(int base, int exp) {\n    int res = 1;\n    for(int i=0; i<exp; i++) res *= base;\n    return res;\n}"
    report: RiskAnalysisReport = detect_semantic_risks(py_code, "python", cpp_code, "cpp")
    assert report.has_high_risk
    assert any(r.category == "INTEGER_OVERFLOW" for r in report.flagged_risks)

    # 2. Test Float Division rule
    py_div = "def mid(l, r):\n    return (l + r) // 2"
    js_div = "function mid(l, r) {\n    return (l + r) / 2;\n}"
    report_div = detect_semantic_risks(py_div, "python", js_div, "javascript")
    assert any(r.category == "FLOAT_PRECISION" for r in report_div.flagged_risks)

    # 3. Test Negative Indexing rule
    py_idx = "def last(arr):\n    return arr[-1]"
    cpp_idx = "int last(std::vector<int> arr) {\n    return arr[-1];\n}"
    report_idx = detect_semantic_risks(py_idx, "python", cpp_idx, "cpp")
    assert any(r.category == "INDEX_BOUNDARY" for r in report_idx.flagged_risks)


def test_adversarial_risk_verification_in_sandbox():
    """
    Executes 3 targeted adversarial test inputs specifically designed to expose flagged risks
    and verifies through the Phase 9 Docker sandbox that the risks are genuine.
    """
    print("\n" + "="*80)
    print("RUNNING PHASE 13 TARGETED ADVERSARIAL RISK VERIFICATION IN SANDBOX")
    print("="*80)

    # =========================================================================
    # ADVERSARIAL CASE 1: INTEGER OVERFLOW (Python -> C++)
    # =========================================================================
    py_power_src = """import sys
data = sys.stdin.read().split()
base, exp = int(data[0]), int(data[1])
def power(b, e):
    return b ** e
print(power(base, exp))
"""

    cpp_power_tgt = """#include <iostream>

int power(int base, int exp) {
    int res = 1;
    for(int i = 0; i < exp; ++i) {
        res *= base;
    }
    return res;
}

int main() {
    int b, e;
    if (std::cin >> b >> e) {
        std::cout << power(b, e) << std::endl;
    }
    return 0;
}
"""

    adv_stdin_1 = "2 62"
    py_res = run_in_sandbox(py_power_src, "python", stdin_input=adv_stdin_1)
    cpp_res = run_in_sandbox(cpp_power_tgt, "cpp", stdin_input=adv_stdin_1)

    print("\n[ADVERSARIAL PROOF 1 - INTEGER OVERFLOW]")
    print(f"Risk Flagged:     INTEGER_OVERFLOW (Python arbitrary precision vs C++ 32-bit signed int)")
    print(f"Input Stdin:      {adv_stdin_1}")
    print(f"Python Output:    {py_res.stdout.strip()} (Expected exact 2^62)")
    print(f"C++ Output:       {cpp_res.stdout.strip()} (Exposed 32-bit Integer Overflow Wraparound to 0!)")
    assert py_res.stdout.strip() != cpp_res.stdout.strip(), "Adversarial overflow failed to demonstrate risk"

    # =========================================================================
    # ADVERSARIAL CASE 2: FLOAT PRECISION / INTEGER DIVISION (Python -> JS)
    # =========================================================================
    py_div_src = """import sys
data = sys.stdin.read().split()
l, r = int(data[0]), int(data[1])
print((l + r) // 2)
"""

    js_div_tgt = """const fs = require('fs');
const input = fs.readFileSync(0, 'utf-8').trim().split(/\\s+/);
const l = parseInt(input[0]);
const r = parseInt(input[1]);
console.log((l + r) / 2);
"""

    adv_stdin_2 = "1 4"
    py_div_res = run_in_sandbox(py_div_src, "python", stdin_input=adv_stdin_2)
    js_div_res = run_in_sandbox(js_div_tgt, "javascript", stdin_input=adv_stdin_2)

    print("\n[ADVERSARIAL PROOF 2 - FLOAT PRECISION / INTEGER DIVISION]")
    print(f"Risk Flagged:     FLOAT_PRECISION (Python floor '//' vs JS float '/')")
    print(f"Input Stdin:      {adv_stdin_2}")
    print(f"Python Output:    {py_div_res.stdout.strip()} (Integer index 2)")
    print(f"JS Output:        {js_div_res.stdout.strip()} (Exposed Float Division 2.5!)")
    assert py_div_res.stdout.strip() != js_div_res.stdout.strip(), "Adversarial float division failed to demonstrate risk"

    # =========================================================================
    # ADVERSARIAL CASE 3: NEGATIVE INDEX BOUNDARY (Python -> C++)
    # =========================================================================
    py_idx_src = """print([10, 20, 30][-1])"""
    cpp_idx_tgt = """#include <iostream>
#include <vector>

int main() {
    std::vector<int> arr = {10, 20, 30};
    int* ptr = arr.data();
    std::cout << ptr[-1] << std::endl;
    return 0;
}
"""

    py_idx_res = run_in_sandbox(py_idx_src, "python")
    cpp_idx_res = run_in_sandbox(cpp_idx_tgt, "cpp")

    print("\n[ADVERSARIAL PROOF 3 - NEGATIVE INDEX BOUNDARY]")
    print(f"Risk Flagged:     INDEX_BOUNDARY (Python arr[-1] vs C++ pointer/vector raw [-1])")
    print(f"Python Output:    {py_idx_res.stdout.strip()} (Expected last element 30)")
    print(f"C++ Output:       {cpp_idx_res.stdout.strip()} (Exposed Out-of-Bounds Memory Garbage / Undefined Behavior!)")
    assert py_idx_res.stdout.strip() != cpp_idx_res.stdout.strip(), "Adversarial index boundary failed to demonstrate risk"

    print("="*80 + "\n")


if __name__ == "__main__":
    test_rule_based_risk_detector_unit()
    test_adversarial_risk_verification_in_sandbox()
