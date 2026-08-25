"""
Self-Correction Repair Module for Rosetta AI.

Uses LLM repair tool to iteratively fix failing translation snippets based on
differential sandbox execution reports.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Dict, List, Any, Optional

from src.refactor.refactor import call_gemini_llm
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
        if "#include" not in code:
            code = f"#include <iostream>\n#include <vector>\n#include <string>\n\n{code}"

    return code


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
```
{source_code}
```

Current Failing {target_lang} Code:
```
{current_code}
```

Sandbox Execution Failure Details:
- Test Input: {failing_input_str}
- Expected Stdout: {expected_out}
- Actual Stdout: {actual_out}
- Error Logs / Stderr: {err_msg}

Task:
Fix the {target_lang} code so that it compiles cleanly and produces the exact expected output.
Return ONLY the corrected {target_lang} code inside ```{target_lang} ``` code blocks.
"""

        llm_reply = call_gemini_llm(prompt)
        repaired_code = None

        if llm_reply:
            m = re.search(rf"```(?:{target_lang})?\s*(.*?)\s*```", llm_reply, re.DOTALL | re.IGNORECASE)
            if m:
                repaired_code = m.group(1).strip()
            else:
                repaired_code = llm_reply.strip()
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
