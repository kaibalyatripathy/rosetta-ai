# Rosetta AI — Semantic Preservation Report

**Translation Pair**: `{source_lang}` $\rightarrow$ `{target_lang}`
**Algorithm Fixture**: `{algorithm_name}`
**Composite Semantic Preservation Score**: `{composite_score} / 100` (`{quality_grade}`)

---

## 1. Code Comparison

### Original Source Code (`{source_lang}`)
```{source_lang}
{source_code}
```

### Translated & Refactored Code (`{target_lang}`)
```{target_lang}
{target_code}
```

---

## 2. Intent & Semantic Summary
- **Intent**: {intent_summary}
- **Language Pair Transition**: {source_lang} to {target_lang}

---

## 3. Evaluation Signals & Breakdown

| Evaluation Dimension | Phase | Measured Signal / Status | Score Contribution | Max Points |
|:---|:---|:---|:---|:---|
| **Functional Equivalence** | Phase 10 | {equiv_passed_inputs}/{equiv_total_inputs} inputs passed ({equiv_pass_rate:.1f}%) | `{score_equiv:.1f}` | `45.0` |
| **Semantic Risk Detection** | Phase 13 | {risk_summary_str} | `{score_risk:.1f}` | `25.0` |
| **Complexity Preservation** | Phase 14 | Source: `{src_complexity}` \| Target: `{tgt_complexity}` ({complexity_status}) | `{score_complexity:.1f}` | `15.0` |
| **Round-Trip Stability** | Phase 12 | Path: `{source_lang}` $\rightarrow$ `{target_lang}` $\rightarrow$ `{source_lang}` ({round_trip_status}) | `{score_round_trip:.1f}` | `15.0` |

---

## 4. Detailed Risk Flags & Findings
{risk_details_markdown}

---

## 5. Final Quality Verdict
**Grade**: `{quality_grade}` ({composite_score:.1f} / 100)
**Summary**: {verdict_summary}
