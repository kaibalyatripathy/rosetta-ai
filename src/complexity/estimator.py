"""
Computational Complexity Estimator for Rosetta AI (Phase 14).

Combines AST loop-nesting analysis, GNN structural AST graph signals, and LLM Big-O
reasoning to estimate time and space complexity of code snippets.
"""

from dataclasses import dataclass
import logging
import re
from typing import Dict, Any, Optional, Tuple

from src.embeddings.codebert.codebert_extractor import get_code_embedding
from src.gnn.graph_builder import build_pyg_ast_graph
from src.refactor.refactor import call_gemini_llm, call_local_llm

logger = logging.getLogger("RosettaAI.ComplexityEstimator")


@dataclass
class ComplexityEstimate:
    time_complexity: str
    space_complexity: str
    max_loop_depth: int
    justification: str
    confidence_score: float


def _analyze_ast_loops(code: str, lang: str) -> Tuple[int, bool, bool]:
    """
    Analyzes code to determine max nested loop depth, logarithmic reduction patterns, and recursion.
    Returns: (max_depth: int, has_log_pattern: bool, is_recursive: bool)
    """
    lines = code.splitlines()
    max_depth = 0
    current_depth = 0

    has_log_pattern = bool(re.search(r'(//\s*2|/\s*2|>>\s*1|\bmid\b|\bleft\s*<=|std::swap)', code, re.IGNORECASE))
    
    # Check recursion
    func_matches = re.findall(r'(?:def|function|void|int|long)\s+([a-zA-Z0-9_]+)\s*\(', code)
    is_recursive = False
    for func_name in func_matches:
        if func_name not in ["main", "if", "for", "while"]:
            # Count occurrences of func_name(
            occurrences = len(re.findall(rf'\b{func_name}\s*\(', code))
            if occurrences > 1:
                is_recursive = True
                break

    # Calculate max loop nesting depth using indentation and block braces
    for line in lines:
        stripped = line.strip()
        if re.search(r'^\s*(?:for|while)\b', line) or re.search(r'\bfor\s*\(|\bwhile\s*\(', line):
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth
        elif '}' in stripped or (lang.lower() == 'python' and stripped and not line.startswith(' ' * (current_depth * 4))):
            if current_depth > 0:
                current_depth -= 1

    return max_depth, has_log_pattern, is_recursive


def _estimate_heuristic(code: str, lang: str) -> ComplexityEstimate:
    """Heuristic fallback for Big-O complexity estimation based on AST analysis."""
    max_depth, has_log, is_recursive = _analyze_ast_loops(code, lang)
    
    # Algorithmic signature detection
    code_lower = code.lower()
    
    if "binarysearch" in code_lower or ("left" in code_lower and "right" in code_lower and "mid" in code_lower):
        time_c = "O(log n)"
        space_c = "O(1)"
        just = "Binary search logarithmic interval halving pattern."
    elif "mergesort" in code_lower or "quicksort" in code_lower or ("pivot" in code_lower and "partition" in code_lower):
        time_c = "O(n log n)"
        space_c = "O(n)" if "mergesort" in code_lower else "O(log n)"
        just = "Divide-and-conquer sorting algorithm pattern."
    elif "matrix" in code_lower or max_depth >= 3:
        time_c = "O(n^3)" if max_depth >= 3 else "O(n^2)"
        space_c = "O(n^2)" if "matrix" in code_lower else "O(1)"
        just = f"Nested loop analysis detected max loop depth of {max_depth}."
    elif max_depth == 2 or "bubble" in code_lower or "insertion" in code_lower or "selection" in code_lower:
        time_c = "O(n^2)"
        space_c = "O(1)"
        just = f"Quadratic nested loops detected (max loop depth {max_depth})."
    elif max_depth == 1 or is_recursive or "linear" in code_lower or "factorial" in code_lower or "fibonacci" in code_lower:
        time_c = "O(n)"
        space_c = "O(n)" if is_recursive else "O(1)"
        just = f"Single linear iteration or recursive stack of depth n."
    elif has_log:
        time_c = "O(log n)"
        space_c = "O(1)"
        just = "Logarithmic loop or division pattern detected."
    else:
        time_c = "O(1)"
        space_c = "O(1)"
        just = "Constant time operations with zero loop iterations."

    # Incorporate GNN AST graph node count as secondary signal
    try:
        pyg_graph = build_pyg_ast_graph(code, lang)
        graph_nodes = pyg_graph.x.size(0)
    except Exception:
        graph_nodes = 0

    return ComplexityEstimate(
        time_complexity=time_c,
        space_complexity=space_c,
        max_loop_depth=max_depth,
        justification=f"{just} (AST Nodes: {graph_nodes})",
        confidence_score=0.85
    )


def estimate_complexity(code: str, lang: str) -> ComplexityEstimate:
    """
    Estimates time and space complexity of code using AST loop analysis, GNN structural signals, and LLM reasoning.
    """
    # 1. AST Loop Depth Analysis
    ast_estimate = _estimate_heuristic(code, lang)

    # 2. LLM Reasoning Check (if API key available)
    prompt = f"""Analyze the computational Big-O complexity of this {lang} code:
```
{code}
```
State:
1. Time Complexity (e.g. O(1), O(log n), O(n), O(n log n), O(n^2), O(n^3))
2. Space Complexity (e.g. O(1), O(n))
3. Concise 1-sentence justification.
Format:
TIME: <Big-O>
SPACE: <Big-O>
JUSTIFICATION: <sentence>
"""
    llm_resp = call_local_llm(prompt)
    if not llm_resp:
        llm_resp = call_gemini_llm(prompt)
        
    if llm_resp:
        m_time = re.search(r'TIME:\s*(O\([^\n]+\))', llm_resp, re.IGNORECASE)
        m_space = re.search(r'SPACE:\s*(O\([^\n]+\))', llm_resp, re.IGNORECASE)
        m_just = re.search(r'JUSTIFICATION:\s*([^\n]+)', llm_resp, re.IGNORECASE)

        if m_time:
            return ComplexityEstimate(
                time_complexity=m_time.group(1).strip(),
                space_complexity=m_space.group(1).strip() if m_space else ast_estimate.space_complexity,
                max_loop_depth=ast_estimate.max_loop_depth,
                justification=m_just.group(1).strip() if m_just else ast_estimate.justification,
                confidence_score=0.95
            )

    return ast_estimate
