"""
Round-Trip Verification Module for Rosetta AI.

Executes A -> B -> A round-trip translation pipeline and evaluates twice-translated
code against original source code via differential testing in the Phase 9 sandbox.
"""

from dataclasses import dataclass
import logging
import re
from typing import Dict, List, Any, Optional

from src.constrained_decoding.grammar_decoder import ConstrainedGrammarDecoder
from src.refactor.refactor import refactor
from src.verification.differential_test import verify_equivalence, EquivalenceReport
from tests.fixtures.test_inputs import get_test_inputs

logger = logging.getLogger("RosettaAI.RoundTrip")


@dataclass
class RoundTripReport:
    source_code: str
    source_lang: str
    intermediate_lang: str
    intermediate_code: str
    final_code: str
    passed: bool
    passed_inputs: int
    total_inputs: int
    pass_rate: float
    semantic_drift_detected: bool
    drift_details: Optional[str] = None


def round_trip_check(
    source_code: str,
    source_lang: str,
    intermediate_lang: str,
    algorithm_name: str = "unknown",
    decoder: Optional[ConstrainedGrammarDecoder] = None
) -> RoundTripReport:
    """
    Executes A -> B -> A round-trip code translation and tests for behavioral equivalence.
    """
    if decoder is None:
        decoder = ConstrainedGrammarDecoder()

    test_inputs = get_test_inputs(algorithm_name)
    src_norm = source_lang.lower().strip()
    mid_norm = intermediate_lang.lower().strip()

    try:
        # Hop 1: Forward Translation (A -> B)
        gen_b_code, _ = decoder.generate_constrained(
            source_code=source_code,
            source_lang=src_norm,
            target_lang=mid_norm,
            num_candidates=2
        )

        prefix_b = f"{src_norm} to {mid_norm}:"
        if gen_b_code.lower().startswith(prefix_b):
            gen_b_code = gen_b_code[len(prefix_b):].strip()

        b_refactored = refactor(source_code, src_norm, gen_b_code, mid_norm)["refactored_code"]

        # Hop 2: Reverse Translation (B -> A)
        gen_a_code, _ = decoder.generate_constrained(
            source_code=b_refactored,
            source_lang=mid_norm,
            target_lang=src_norm,
            num_candidates=2
        )

        prefix_a = f"{mid_norm} to {src_norm}:"
        if gen_a_code.lower().startswith(prefix_a):
            gen_a_code = gen_a_code[len(prefix_a):].strip()

        a_final = refactor(b_refactored, mid_norm, gen_a_code, src_norm)["refactored_code"]

        # Differential Verification (Original Source A vs Twice-Translated A)
        report: EquivalenceReport = verify_equivalence(
            source_code=source_code,
            source_lang=src_norm,
            target_code=a_final,
            target_lang=src_norm,
            test_inputs=test_inputs,
            algorithm_name=algorithm_name
        )

        semantic_drift = False
        drift_msg = None

        if not report.passed:
            semantic_drift = True
            failing_counts = report.total_inputs - report.passed_inputs
            drift_msg = (
                f"Compounded translation drift across two hops ({src_norm}->{mid_norm}->{src_norm}). "
                f"Failed {failing_counts}/{report.total_inputs} test inputs."
            )
        elif report.passed and (len(a_final) < len(source_code) * 0.3 or len(a_final) > len(source_code) * 3.0):
            semantic_drift = True
            drift_msg = f"Structural drift: twice-translated code length changed significantly ({len(source_code)} -> {len(a_final)} chars)."

        logger.info(f"[{algorithm_name}] Round-Trip ({src_norm}->{mid_norm}->{src_norm}) -> {'PASS' if report.passed else 'FAIL'}")

        return RoundTripReport(
            source_code=source_code,
            source_lang=src_norm,
            intermediate_lang=mid_norm,
            intermediate_code=b_refactored,
            final_code=a_final,
            passed=report.passed,
            passed_inputs=report.passed_inputs,
            total_inputs=report.total_inputs,
            pass_rate=report.pass_rate,
            semantic_drift_detected=semantic_drift,
            drift_details=drift_msg
        )

    except Exception as e:
        logger.error(f"[{algorithm_name}] Round-trip check error: {e}")
        return RoundTripReport(
            source_code=source_code,
            source_lang=src_norm,
            intermediate_lang=mid_norm,
            intermediate_code="",
            final_code="",
            passed=False,
            passed_inputs=0,
            total_inputs=len(test_inputs),
            pass_rate=0.0,
            semantic_drift_detected=True,
            drift_details=f"Pipeline error: {str(e)}"
        )
