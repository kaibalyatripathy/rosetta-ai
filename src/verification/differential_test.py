"""
Differential Testing Verification Engine for Rosetta AI.

Executes source and target code through Phase 9 Docker sandbox on matching test inputs,
compares normalized outputs, and generates EquivalenceReport.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Dict, List, Any, Optional

from src.sandbox.runner import run_in_sandbox, ExecutionResult


@dataclass
class InputResult:
    test_input: Any
    passed: bool
    source_stdout: str
    target_stdout: str
    source_exit_code: int
    target_exit_code: int
    source_stderr: str = ""
    target_stderr: str = ""
    source_timed_out: bool = False
    target_timed_out: bool = False


@dataclass
class EquivalenceReport:
    algorithm_name: str
    source_lang: str
    target_lang: str
    passed: bool
    pass_rate: float
    total_inputs: int
    passed_inputs: int
    input_results: List[InputResult] = field(default_factory=list)


def normalize_output(raw_output: str) -> str:
    """Normalizes stdout output for robust comparison across language format differences."""
    out = raw_output.strip()
    if not out:
        return ""
    # Strip all whitespaces and newlines for ultra-lenient equivalence checking
    out = re.sub(r'\s+', '', out)
    out_lower = out.lower()
    # Normalize booleans
    if out_lower in ["true", "1", "1.0"]:
        return "true"
    if out_lower in ["false", "0", "0.0"]:
        return "false"
    # Normalize array representations e.g. [1, 2, 3] vs 1 2 3
    cleaned = out_lower.replace("[", "").replace("]", "").replace(",", "").strip()
    return cleaned


def _extract_func_name(code: str, lang: str) -> Optional[str]:
    """Extracts main function name from code snippet."""
    lang_lower = lang.lower().strip()
    if lang_lower in ["python", "py"]:
        m = re.search(r'def\s+([a-zA-Z0-9_]+)\s*\(', code)
        return m.group(1) if m else None
    elif lang_lower in ["javascript", "js", "node"]:
        m = re.search(r'function\s+([a-zA-Z0-9_]+)\s*\(', code)
        return m.group(1) if m else None
    elif lang_lower in ["java"]:
        for m in re.finditer(r'public\s+static\s+(?:[a-zA-Z0-9_<>\s\[\]]+)\s+([a-zA-Z0-9_]+)\s*\(', code):
            if m.group(1) not in ["main"]:
                return m.group(1)
        return None
    elif lang_lower in ["cpp", "c++"]:
        for m in re.finditer(r'(?:[a-zA-Z0-9_<>]+\s+)+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*\{', code):
            if m.group(1) not in ["main", "if", "while", "for", "switch"]:
                return m.group(1)
        return None
    return None


def to_java_val(val: Any) -> str:
    if isinstance(val, list):
        if not val: return "new int[]{}"
        if isinstance(val[0], int):
            return "new int[]{" + ", ".join(map(str, val)) + "}"
        elif isinstance(val[0], str):
            return "new String[]{" + ", ".join(f'"{v}"' for v in val) + "}"
    elif isinstance(val, str):
        return f'"{val}"'
    elif isinstance(val, bool):
        return "true" if val else "false"
    elif val is None:
        return "null"
    return str(val)


def to_cpp_val(val: Any) -> str:
    if isinstance(val, list):
        if not val: return "{}"
        if isinstance(val[0], int):
            return "{" + ", ".join(map(str, val)) + "}"
        elif isinstance(val[0], str):
            return "{" + ", ".join(f'"{v}"' for v in val) + "}"
    elif isinstance(val, str):
        return f'"{val}"'
    elif isinstance(val, bool):
        return "true" if val else "false"
    elif val is None:
        return "NULL"
    return str(val)


def attach_test_driver(code: str, lang: str, test_input: Any) -> str:
    """
    Attaches a test driver call to code if it does not already contain a main execution driver.
    """
    lang_lower = lang.lower().strip()
    
    # Do not bail out immediately for Java/C++ if they have a main method, as we need to replace it.
    if lang_lower not in ["java", "cpp", "c++"]:
        if "main(" in code or "print(" in code or "console.log" in code or "System.out.print" in code:
            return code

    func_name = _extract_func_name(code, lang)
    if not func_name:
        return code

    # Format input arguments
    if isinstance(test_input, dict):
        args_repr_py = ", ".join(f"{v!r}" for v in test_input.values())
        args_repr_js = ", ".join(json.dumps(v) for v in test_input.values())
        args_repr_java = ", ".join(to_java_val(v) for v in test_input.values())
        args_repr_cpp = ", ".join(to_cpp_val(v) for v in test_input.values())
    else:
        args_repr_py = repr(test_input)
        args_repr_js = json.dumps(test_input)
        args_repr_java = to_java_val(test_input)
        args_repr_cpp = to_cpp_val(test_input)

    if lang_lower in ["python", "py"]:
        driver = f"\n\nif __name__ == '__main__':\n    res = {func_name}({args_repr_py})\n    if res is not None:\n        print(res)\n"
        return code + driver
    elif lang_lower in ["javascript", "js"]:
        driver = f"\n\nconst res = {func_name}({args_repr_js});\nif (res !== undefined) console.log(res);\n"
        return code + driver
    elif lang_lower in ["java"]:
        code = re.sub(r'public\s+static\s+void\s+main\s*\(', 'public static void dummy_main(', code)
        driver = f"\n    public static void main(String[] args) {{\n        System.out.println({func_name}({args_repr_java}));\n    }}\n"
        last_brace = code.rfind('}')
        if last_brace != -1:
            code = code[:last_brace] + driver + code[last_brace:]
        return code
    elif lang_lower in ["cpp", "c++"]:
        code = re.sub(r'int\s+main\s*\(', 'int dummy_main(', code)
        driver = f"\n\n#include <iostream>\nint main() {{\n    std::cout << {func_name}({args_repr_cpp}) << std::endl;\n    return 0;\n}}\n"
        return code + driver
    
    return code


def verify_equivalence(
    source_code: str,
    source_lang: str,
    target_code: str,
    target_lang: str,
    test_inputs: List[Any],
    algorithm_name: str = "unknown_algorithm",
    timeout_sec: float = 10.0
) -> EquivalenceReport:
    """
    Runs source code and target code through Phase 9 Docker sandbox on test inputs,
    compares normalized outputs, and returns EquivalenceReport.
    """
    input_results: List[InputResult] = []
    passed_count = 0

    for inp in test_inputs:
        src_run_code = attach_test_driver(source_code, source_lang, inp)
        tgt_run_code = attach_test_driver(target_code, target_lang, inp)

        src_res: ExecutionResult = run_in_sandbox(src_run_code, source_lang, timeout_sec=timeout_sec)
        tgt_res: ExecutionResult = run_in_sandbox(tgt_run_code, target_lang, timeout_sec=timeout_sec)

        src_norm = normalize_output(src_res.stdout)
        tgt_norm = normalize_output(tgt_res.stdout)

        # Output matching rule
        both_zero_exit = (src_res.exit_code == 0) and (tgt_res.exit_code == 0)
        outputs_match = (src_norm == tgt_norm) and bool(src_norm)
        no_compile_errors = (not src_res.compile_error) and (not tgt_res.compile_error)
        no_timeouts = (not src_res.timed_out) and (not tgt_res.timed_out)

        passed = both_zero_exit and outputs_match and no_compile_errors and no_timeouts

        if passed:
            passed_count += 1

        input_results.append(InputResult(
            test_input=inp,
            passed=passed,
            source_stdout=src_res.stdout.strip(),
            target_stdout=tgt_res.stdout.strip(),
            source_exit_code=src_res.exit_code,
            target_exit_code=tgt_res.exit_code,
            source_stderr=src_res.stderr.strip() or src_res.compile_stderr.strip(),
            target_stderr=tgt_res.stderr.strip() or tgt_res.compile_stderr.strip(),
            source_timed_out=src_res.timed_out,
            target_timed_out=tgt_res.timed_out
        ))

    total = len(test_inputs)
    pass_rate = (passed_count / total * 100.0) if total > 0 else 0.0
    overall_passed = (passed_count == total) and (total > 0)

    return EquivalenceReport(
        algorithm_name=algorithm_name,
        source_lang=source_lang.lower().strip(),
        target_lang=target_lang.lower().strip(),
        passed=overall_passed,
        pass_rate=pass_rate,
        total_inputs=total,
        passed_inputs=passed_count,
        input_results=input_results
    )
