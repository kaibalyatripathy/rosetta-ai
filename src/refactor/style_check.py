"""
Multi-Lingual Style & Linting Checker for Rosetta AI.

Evaluates style compliance and counts lint warnings for Python, Java, C++, and JavaScript code snippets.
Returns a 0-100 style score and warning count per snippet.
"""

import re
from typing import Dict, Any, List


def compute_style_score(code: str, lang: str) -> Dict[str, Any]:
    """
    Computes a 0-100 style score and counts lint warnings for a given code snippet.
    """
    if not code or not code.strip():
        return {"score": 0.0, "warnings_count": 10, "warnings": ["Empty code snippet."]}

    warnings: List[str] = []
    lines = code.splitlines()

    norm_lang = lang.lower().strip()

    # 1. Line Length Check (>100 chars)
    long_lines = [idx + 1 for idx, l in enumerate(lines) if len(l.rstrip()) > 100]
    if long_lines:
        warnings.append(f"Lines exceeding 100 chars: {long_lines}")

    # 2. Trailing Whitespace Check
    trailing_ws = [idx + 1 for idx, l in enumerate(lines) if l != l.rstrip()]
    if trailing_ws:
        warnings.append(f"Trailing whitespace on lines: {trailing_ws[:3]}")

    # 3. Language-Specific Style Rules
    if norm_lang == "python":
        # Snake_case function/variable naming
        if re.search(r"def\s+[a-z]+[A-Z]", code):
            warnings.append("Python function name uses camelCase instead of snake_case.")
        # PEP8 spacing around operators
        if re.search(r"[a-zA-Z0-9]=[a-zA-Z0-9]", code):
            warnings.append("Missing spaces around assignment operator '='.")
        # Missing docstring/comment
        if '"""' not in code and "#" not in code:
            warnings.append("Missing inline comments or docstring.")

    elif norm_lang in {"java", "javascript"}:
        # camelCase function naming
        if norm_lang == "java" and re.search(r"\b(public|private|protected|static)\s+[a-zA-Z0-9_<>]+(?:\s+([a-z]+_[a-z0-9_]+))\s*\(", code):
            warnings.append("Java method uses snake_case instead of camelCase.")
        # Var usage in JS
        if norm_lang == "javascript" and re.search(r"\bvar\s+", code):
            warnings.append("JavaScript uses legacy 'var' keyword instead of 'const'/'let'.")
        # Missing semicolons in JS/Java
        missing_semi = [idx + 1 for idx, l in enumerate(lines) if l.strip() and not l.strip().endswith((';', '{', '}', ':', '//')) and not l.strip().startswith(('for', 'if', 'while', '//', '/*'))]
        if missing_semi and norm_lang == "java":
            warnings.append(f"Missing statement terminating semicolon on lines: {missing_semi[:3]}")

    elif norm_lang == "cpp":
        # C++ using namespace std in global header
        if "using namespace std;" in code:
            warnings.append("Global 'using namespace std;' directive present.")
        # C++ raw pointer usage without const/references
        if re.search(r"\bint\s*\*\s*", code) and "const" not in code:
            warnings.append("Raw un-const pointer usage detected.")

    # 4. Calculate Score (Base 100 - 10 points per warning type)
    score = max(0.0, 100.0 - (len(warnings) * 12.5))

    return {
        "language": norm_lang,
        "score": round(score, 1),
        "warnings_count": len(warnings),
        "warnings": warnings
    }


if __name__ == "__main__":
    py_code = "def MyFunction(a,b):\n    x=a+b\n    return x"
    print("Python Style Check:", compute_style_score(py_code, "python"))
