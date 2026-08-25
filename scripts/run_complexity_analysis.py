"""
Script to execute Phase 14 Computational Complexity Estimation Benchmark.
"""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath("."))

from tests.unit.test_complexity import test_complexity_estimation_ground_truth

if __name__ == "__main__":
    print("Executing Rosetta AI Phase 14 Computational Complexity Benchmark...")
    test_complexity_estimation_ground_truth()
    print("Complexity Estimation Benchmark Finished.")
