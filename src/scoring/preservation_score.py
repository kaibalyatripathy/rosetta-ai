"""
Explainable Composite Semantic Preservation Scoring Engine for Rosetta AI (Phase 15).

Combines signals from Phases 10, 12, 13, and 14 into an explainable 0-100 weighted score:
- Functional Equivalence (Phase 10): 45% weight (Max 45 pts)
- Semantic Risk Flags (Phase 13): 25% weight (Max 25 pts)
- Complexity Preservation (Phase 14): 15% weight (Max 15 pts)
- Round-Trip Stability (Phase 12): 15% weight (Max 15 pts)
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.complexity.estimator import estimate_complexity, ComplexityEstimate
from src.risk_detection.risk_rules import detect_semantic_risks, RiskAnalysisReport
from src.verification.differential_test import verify_equivalence, EquivalenceReport
from tests.fixtures.test_inputs import get_test_inputs

logger = logging.getLogger("RosettaAI.PreservationScore")
TEMPLATE_FILE = Path("docs/preservation_report_template.md")


@dataclass
class PreservationScoreReport:
    algorithm_name: str
    source_lang: str
    target_lang: str
    source_code: str
    target_code: str
    score_equiv: float
    score_risk: float
    score_complexity: float
    score_round_trip: float
    composite_score: float
    quality_grade: str
    intent_summary: str
    markdown_report: str


def calculate_preservation_score(
    source_code: str,
    source_lang: str,
    target_code: str,
    target_lang: str,
    algorithm_name: str = "unknown",
    round_trip_passed: bool = False
) -> PreservationScoreReport:
    """
    Calculates composite Semantic Preservation Score (0-100) and produces human-readable markdown report.
    """
    src_norm = source_lang.lower().strip()
    tgt_norm = target_lang.lower().strip()
    test_inputs = get_test_inputs(algorithm_name)

    # 1. Phase 10 Signal: Functional Equivalence (45 pts max)
    equiv_report: EquivalenceReport = verify_equivalence(
        source_code=source_code,
        source_lang=src_norm,
        target_code=target_code,
        target_lang=tgt_norm,
        test_inputs=test_inputs,
        algorithm_name=algorithm_name
    )
    score_equiv = (equiv_report.pass_rate / 100.0) * 45.0 if equiv_report.pass_rate > 1.0 else equiv_report.pass_rate * 45.0

    # 2. Phase 13 Signal: Semantic Risk Flags (25 pts max)
    risk_report: RiskAnalysisReport = detect_semantic_risks(source_code, src_norm, target_code, tgt_norm)
    score_risk = 25.0
    for r in risk_report.flagged_risks:
        if r.severity == "HIGH":
            score_risk -= 15.0
        elif r.severity == "MEDIUM":
            score_risk -= 7.0
    score_risk = max(0.0, score_risk)

    # 3. Phase 14 Signal: Complexity Preservation (15 pts max)
    src_comp = estimate_complexity(source_code, src_norm)
    tgt_comp = estimate_complexity(target_code, tgt_norm)
    
    is_degraded = (
        src_comp.time_complexity != tgt_comp.time_complexity and
        tgt_comp.time_complexity in ["O(n^2)", "O(n^3)"] and
        src_comp.time_complexity in ["O(1)", "O(log n)", "O(n)"]
    )
    score_complexity = 0.0 if is_degraded else 15.0

    # 4. Phase 12 Signal: Round-Trip Stability (15 pts max)
    score_round_trip = 15.0 if round_trip_passed else 0.0

    # Composite Score Calculation (0-100)
    composite_score = score_equiv + score_risk + score_complexity + score_round_trip
    composite_score = max(0.0, min(100.0, composite_score))

    # Determine Quality Grade
    if composite_score >= 90.0:
        grade = "EXCELLENT"
        verdict = "Translation cleanly preserves functional semantics, risk boundaries, and computational complexity."
    elif composite_score >= 70.0:
        grade = "GOOD"
        verdict = "Translation is functionally sound with minor syntax or risk warnings."
    elif composite_score >= 45.0:
        grade = "MODERATE / MARGINAL"
        verdict = "Translation passes partial inputs but contains risk flags or structural differences."
    else:
        grade = "POOR / FAILING"
        verdict = "Translation failed functional equivalence sandbox execution or exhibited severe semantic risks."

    intent = f"Algorithmic code translation of {algorithm_name} from {src_norm} to {tgt_norm}."

    # Build Markdown Report
    risk_summary_str = f"{len(risk_report.flagged_risks)} risk flags" if risk_report.flagged_risks else "Zero risks flagged"
    complexity_status = "Degraded" if is_degraded else "Preserved"
    round_trip_status = "Passed" if round_trip_passed else "Failed / Drifted"

    if risk_report.flagged_risks:
        risk_md = "\n".join([f"- **[{r.severity}] {r.category}**: {r.description}" for r in risk_report.flagged_risks])
    else:
        risk_md = "- *No semantic risks or edge-case divergences flagged.*"

    report_text = f"""# Rosetta AI — Semantic Preservation Report

**Translation Pair**: `{src_norm}` $\\rightarrow$ `{tgt_norm}`
**Algorithm Fixture**: `{algorithm_name}`
**Composite Semantic Preservation Score**: `{composite_score:.1f} / 100` (`{grade}`)

---

## 1. Code Comparison

### Original Source Code (`{src_norm}`)
```{src_norm}
{source_code}
```

### Translated & Refactored Code (`{tgt_norm}`)
```{tgt_norm}
{target_code}
```

---

## 2. Intent & Semantic Summary
- **Intent**: {intent}
- **Language Pair Transition**: {src_norm} to {tgt_norm}

---

## 3. Evaluation Signals & Breakdown

| Evaluation Dimension | Phase | Measured Signal / Status | Score Contribution | Max Points |
|:---|:---|:---|:---|:---|
| **Functional Equivalence** | Phase 10 | {equiv_report.passed_inputs}/{equiv_report.total_inputs} inputs passed ({equiv_report.pass_rate:.1f}%) | `{score_equiv:.1f}` | `45.0` |
| **Semantic Risk Detection** | Phase 13 | {risk_summary_str} | `{score_risk:.1f}` | `25.0` |
| **Complexity Preservation** | Phase 14 | Source: `{src_comp.time_complexity}` \\| Target: `{tgt_comp.time_complexity}` ({complexity_status}) | `{score_complexity:.1f}` | `15.0` |
| **Round-Trip Stability** | Phase 12 | Path: `{src_norm}` $\\rightarrow$ `{tgt_norm}` $\\rightarrow$ `{src_norm}` ({round_trip_status}) | `{score_round_trip:.1f}` | `15.0` |

---

## 4. Detailed Risk Flags & Findings
{risk_md}

---

## 5. Final Quality Verdict
**Grade**: `{grade}` ({composite_score:.1f} / 100)
**Summary**: {verdict}
"""

    return PreservationScoreReport(
        algorithm_name=algorithm_name,
        source_lang=src_norm,
        target_lang=tgt_norm,
        source_code=source_code,
        target_code=target_code,
        score_equiv=score_equiv,
        score_risk=score_risk,
        score_complexity=score_complexity,
        score_round_trip=score_round_trip,
        composite_score=composite_score,
        quality_grade=grade,
        intent_summary=intent,
        markdown_report=report_text
    )
