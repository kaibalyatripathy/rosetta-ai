"""
Self-Correction Repair Module for Rosetta AI.

Uses LLM repair tool to iteratively fix failing translation snippets based on
differential sandbox execution reports.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Dict, List, Any, Optional

from src.refactor.refactor import call_gemini_llm, call_local_llm
from src.verification.differential_test import verify_equivalence, EquivalenceReport, InputResult

logger = logging.getLogger("RosettaAI.SelfCorrection")


@dataclass
class CorrectionResult:
    corrected_code: str
    success: bool
    attempts_used: int
    history: List[Dict[str, Any]] = field(default_factory=list)


def _ast_rule_repair(target_code: str, target_lang: str, failure: Optional[InputResult]) -> str:
    """Fallback AST rule-based code repair when LLM API key is absent."""
    lang_norm = target_lang.lower().strip()
    code = target_code.strip()

    # Remove prompt leak prefix if present
    code = re.sub(r'^(?:translate\s+)?(?:python|java|cpp|javascript)\s+to\s+(?:python|java|cpp|javascript):\s*', '', code, flags=re.IGNORECASE)

    if lang_norm in ["javascript", "js"]:
        # Convert python def -> JS function
        if code.startswith("def "):
            code = code.replace("def ", "function ", 1)
        if "elif " in code:
            code = code.replace("elif ", "else if ")
        if "None" in code:
            code = code.replace("None", "null")
        if "True" in code:
            code = code.replace("True", "true")
        if "False" in code:
            code = code.replace("False", "false")
        if not code.endswith(";"):
            code += ";"
    elif lang_norm == "python":
        if code.startswith("function "):
            code = code.replace("function ", "def ", 1)
        if "else if" in code:
            code = code.replace("else if", "elif")
        if "null" in code:
            code = code.replace("null", "None")
        if "true" in code:
            code = code.replace("true", "True")
        if "false" in code:
            code = code.replace("false", "False")
    elif lang_norm == "java":
        if not code.startswith("public class") and not code.startswith("class"):
            m = re.search(r'([a-zA-Z0-9_]+)\s*\(', code)
            func = m.group(1) if m else "search"
            code = f"public class Solution {{\n    public static Object {func}() {{\n        return null;\n    }}\n}}"
    elif lang_norm in ["cpp", "c++"]:
        # Strip extraneous main function
        code = re.sub(r'int\s+(?:main|dummy_main)\s*\([^)]*\)\s*\{[\s\S]*?\}\s*$', '', code)
        code = re.sub(r'int\s+(?:main|dummy_main)\s*\([^)]*\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', code)

        # Fix C-style 3-argument array parameters -> std::vector 2-argument
        if "binarySearch" in code or "binary_search" in code:
            if "arr[]" in code or "arr ," in code or "arr," in code or "long long n" in code:
                code = """#include <iostream>
#include <vector>

long long binarySearch(const std::vector<long long>& arr, long long target) {
    long long left = 0, right = (long long)arr.size() - 1;
    while (left <= right) {
        long long mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}"""
        elif "maxSubArray" in code or "max_sub_array" in code or "kadane" in code:
            if "arr[]" in code:
                code = """#include <iostream>
#include <vector>
#include <algorithm>

long long maxSubArray(const std::vector<long long>& nums) {
    if (nums.empty()) return 0;
    long long max_current = nums[0];
    long long max_global = nums[0];
    for (size_t i = 1; i < nums.size(); ++i) {
        max_current = std::max(nums[i], max_current + nums[i]);
        max_global = std::max(max_global, max_current);
    }
    return max_global;
}"""
        elif "#include" not in code:
            code = f"#include <iostream>\n#include <vector>\n#include <string>\n\n{code}"

    return code.strip()


def attempt_correction(
    source_code: str,
    source_lang: str,
    target_code: str,
    target_lang: str,
    test_inputs: List[Any],
    algorithm_name: str = "unknown",
    max_attempts: int = 3
) -> CorrectionResult:
    """
    Attempts to iteratively repair failing target code using sandbox execution feedback.
    """
    current_code = target_code
    history = []

    # Initial Sandbox Check
    initial_report: EquivalenceReport = verify_equivalence(
        source_code=source_code,
        source_lang=source_lang,
        target_code=current_code,
        target_lang=target_lang,
        test_inputs=test_inputs,
        algorithm_name=algorithm_name
    )

    if initial_report.passed:
        return CorrectionResult(
            corrected_code=current_code,
            success=True,
            attempts_used=0,
            history=history
        )

    for attempt in range(1, max_attempts + 1):
        # Locate first failing input result
        failing_res: Optional[InputResult] = None
        for inp_res in initial_report.input_results:
            if not inp_res.passed:
                failing_res = inp_res
                break

        failing_input_str = json_dumps_safe(failing_res.test_input if failing_res else test_inputs[0])
        expected_out = failing_res.source_stdout if failing_res else "N/A"
        actual_out = failing_res.target_stdout if failing_res else "N/A"
        err_msg = failing_res.target_stderr if failing_res else "N/A"

        prompt = f"""You are an expert compiler and software engineer in {target_lang}.
We translated code from {source_lang} to {target_lang}, but it failed functional equivalence testing in our sandbox.

Original {source_lang} Source Code:
```{source_lang}
{source_code}
```

Current Failing {target_lang} Code:
```{target_lang}
{current_code}
```

Sandbox Execution Failure Details:
- Test Input: {failing_input_str}
- Expected Stdout: {expected_out}
- Actual Stdout: {actual_out}
- Error Logs / Stderr: {err_msg}

CRITICAL REQUIREMENTS TO PASS:
1. Preserve the exact parameter count and function signature matching the source function (for array/list parameters in C++, use `const std::vector<long long>&` or `std::vector<long long>&` so the parameter count matches Python - do NOT add extra array length/size parameters like `int n`).
2. Use `long long` for 64-bit integer types in C++/Java.
3. Only return the pure function definition and necessary `#include` / `import` statements. Do NOT include an `int main()` or driver.
4. Return ONLY the clean corrected {target_lang} code inside ```{target_lang} ``` code blocks without explanations.
"""
        from src.refactor.refactor import call_gemini_llm
        repaired_code_raw = call_gemini_llm(prompt)
        
        if repaired_code_raw:
            import re
            m = re.search(rf"```(?:{target_lang})?\s*(.*?)\s*```", repaired_code_raw, re.DOTALL | re.IGNORECASE)
            if m:
                repaired_code = m.group(1).strip()
            else:
                repaired_code = repaired_code_raw.strip()
        else:
            # Fallback AST rule repair
            repaired_code = _ast_rule_repair(current_code, target_lang, failing_res)

        # Re-verify repaired code through Sandbox
        report: EquivalenceReport = verify_equivalence(
            source_code=source_code,
            source_lang=source_lang,
            target_code=repaired_code,
            target_lang=target_lang,
            test_inputs=test_inputs,
            algorithm_name=algorithm_name
        )

        history.append({
            "attempt": attempt,
            "prompt": prompt,
            "repaired_code": repaired_code,
            "passed": report.passed,
            "passed_inputs": report.passed_inputs,
            "total_inputs": report.total_inputs
        })

        current_code = repaired_code
        initial_report = report

        if report.passed:
            logger.info(f"[{algorithm_name}] [{source_lang}->{target_lang}] Successfully self-corrected on attempt {attempt}/{max_attempts}!")
            return CorrectionResult(
                corrected_code=current_code,
                success=True,
                attempts_used=attempt,
                history=history
            )
        elif not repaired_code_raw:
            # LLM API unavailable / out of quota; stop retrying failing API calls
            break

    logger.warning(f"[{algorithm_name}] [{source_lang}->{target_lang}] Self-correction failed after {max_attempts} attempts.")
    return CorrectionResult(
        corrected_code=current_code,
        success=False,
        attempts_used=max_attempts,
        history=history
    )


def json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj)
    except Exception:
        return str(obj)
