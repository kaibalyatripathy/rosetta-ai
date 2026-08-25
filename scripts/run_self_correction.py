"""
Script to execute Phase 11 Self-Correction Loop Benchmark.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

from tests.integration.test_self_correction import run_self_correction_benchmark

if __name__ == "__main__":
    print("Executing Rosetta AI Phase 11 Self-Correction Benchmark...")
    summary = run_self_correction_benchmark(max_workers=4, max_attempts=3)
    print("Self-Correction Benchmark Execution Finished.")
