"""
Script to execute full Phase 10 Functional Equivalence Verification benchmark.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from tests.integration.test_equivalence_full import run_full_equivalence_benchmark

if __name__ == "__main__":
    print("Executing Rosetta AI Phase 10 Functional Equivalence Verification Benchmark...")
    summary = run_full_equivalence_benchmark(max_workers=4)
    print("Benchmark Execution Finished Successfully.")
