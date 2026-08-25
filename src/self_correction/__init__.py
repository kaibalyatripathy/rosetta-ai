"""
Self-Correction Loop Package for Rosetta AI.
"""

from src.self_correction.corrector import attempt_correction, CorrectionResult
from src.self_correction.hard_examples_log import log_hard_example

__all__ = ["attempt_correction", "CorrectionResult", "log_hard_example"]
