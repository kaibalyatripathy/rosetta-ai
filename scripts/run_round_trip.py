"""
Script to execute Phase 12 Round-Trip Verification Benchmark.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

from tests.integration.test_round_trip import run_round_trip_benchmark

if __name__ == "__main__":
    print("Executing Rosetta AI Phase 12 Round-Trip Verification Benchmark...")
    summary = run_round_trip_benchmark(max_workers=4)
    print("Round-Trip Verification Benchmark Finished.")
