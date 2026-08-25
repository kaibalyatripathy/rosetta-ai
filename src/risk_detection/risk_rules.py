"""
Deterministic Rule-Based Semantic Risk Detector for Rosetta AI (Phase 13).

Evaluates source and target translation pairs against a checklist of known cross-language
semantic divergence patterns without external LLM/model calls.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Any, Optional


@dataclass
class RiskItem:
    category: str  # INTEGER_OVERFLOW, INDEX_BOUNDARY, TYPE_COERCION, FLOAT_PRECISION, MEMORY_MANAGEMENT
    severity: str  # HIGH, MEDIUM, LOW
    description: str
    matched_pattern: str


@dataclass
class RiskAnalysisReport:
    source_lang: str
    target_lang: str
    flagged_risks: List[RiskItem] = field(default_factory=list)
    risk_score: float = 0.0
    has_high_risk: bool = False


def detect_semantic_risks(
    source_code: str,
    source_lang: str,
    target_code: str,
    target_lang: str
) -> RiskAnalysisReport:
    """
    Evaluates translation pair for semantic risk flags using deterministic heuristics.
    """
    src_norm = source_lang.lower().strip()
    tgt_norm = target_lang.lower().strip()

    risks: List[RiskItem] = []

    # 1. Rule 1: INTEGER_OVERFLOW
    # Python int has arbitrary precision; Java/C++ fixed 32-bit/64-bit int can overflow silently.
    if src_norm == "python" and tgt_norm in ["cpp", "c++", "java"]:
        if re.search(r'\*\*|\bpow\b|\*|\bshift\b|<<', source_code):
            if not re.search(r'BigInteger|int64_t|long long|unsigned long', target_code, re.IGNORECASE):
                risks.append(RiskItem(
                    category="INTEGER_OVERFLOW",
                    severity="HIGH",
                    description=f"Arbitrary precision Python integer arithmetic translated to fixed-width {tgt_norm} without BigInteger or int64 protection.",
                    matched_pattern="python arithmetic -> fixed width int"
                ))

    # 2. Rule 2: INDEX_BOUNDARY (Negative Indexing & Slicing)
    # Python supports arr[-1] to access last element. Java/C++/JS evaluate arr[-1] as undefined or out-of-bounds.
    if src_norm == "python" and tgt_norm in ["javascript", "js", "java", "cpp", "c++"]:
        if re.search(r'\[\s*-\s*\d+\s*\]', source_code):
            if re.search(r'\[\s*-\s*\d+\s*\]', target_code):
                risks.append(RiskItem(
                    category="INDEX_BOUNDARY",
                    severity="HIGH",
                    description=f"Python negative array indexing (e.g. arr[-1]) directly copied to {tgt_norm} without bounds translation (length - 1).",
                    matched_pattern="[ -N ]"
                ))

    # 3. Rule 3: TYPE_COERCION
    # JavaScript loose equality '==' vs '===' and implicit string/number addition '+'
    if tgt_norm in ["javascript", "js"]:
        if re.search(r'[^=]==[^=]', target_code):
            risks.append(RiskItem(
                category="TYPE_COERCION",
                severity="MEDIUM",
                description="JavaScript translation uses loose equality '==' instead of strict equality '===', risking unexpected type coercion.",
                matched_pattern="=="
            ))
        if re.search(r'\+\s*["\']|["\']\s*\+', target_code):
            risks.append(RiskItem(
                category="TYPE_COERCION",
                severity="HIGH",
                description="JavaScript implicit string-number addition '+' detected, which can coerce numeric addition into string concatenation.",
                matched_pattern="+ 'str'"
            ))

    # 4. Rule 4: FLOAT_PRECISION & INTEGER DIVISION
    # Python '//' is floor integer division. JS '/' performs floating point division.
    if src_norm == "python" and tgt_norm in ["javascript", "js"]:
        if "//" in source_code:
            if "/" in target_code and not re.search(r'Math\.floor|Math\.trunc|\|0|>>0', target_code):
                risks.append(RiskItem(
                    category="FLOAT_PRECISION",
                    severity="HIGH",
                    description="Python floor integer division '//' translated to JavaScript '/' without Math.floor() or Math.trunc(), producing floating point values.",
                    matched_pattern="// -> /"
                ))

    # 5. Rule 5: MEMORY_MANAGEMENT
    # C++ manual memory management (new/delete, malloc/free, raw pointers) introduced from garbage-collected sources.
    if tgt_norm in ["cpp", "c++"] and src_norm in ["python", "java", "javascript", "js"]:
        if re.search(r'\bnew\b|\bdelete\b|\bmalloc\b|\bfree\b|\bint\s*\*|\bdouble\s*\*|\bchar\s*\*', target_code):
            risks.append(RiskItem(
                category="MEMORY_MANAGEMENT",
                severity="HIGH",
                description="C++ translation introduced manual raw memory management (new/delete/pointers) from a garbage-collected source language, risking memory leaks.",
                matched_pattern="new/delete/pointers"
            ))

    # Compute Overall Risk Score
    severity_weights = {"HIGH": 3.0, "MEDIUM": 1.5, "LOW": 0.5}
    score = sum(severity_weights.get(r.severity, 1.0) for r in risks)
    has_high = any(r.severity == "HIGH" for r in risks)

    return RiskAnalysisReport(
        source_lang=src_norm,
        target_lang=tgt_norm,
        flagged_risks=risks,
        risk_score=score,
        has_high_risk=has_high
    )
