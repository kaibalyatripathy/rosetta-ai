"""
Test inputs loader module for 20 canonical algorithm fixtures.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

INPUTS_FILE = Path(__file__).parent / "test_inputs.json"


def load_all_test_inputs() -> Dict[str, List[Any]]:
    """Loads all test inputs for all 20 canonical fixtures."""
    with open(INPUTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_test_inputs(algorithm_name: str) -> List[Any]:
    """Gets test inputs list for a specific algorithm name (or normalized key)."""
    data = load_all_test_inputs()
    # Handle key normalization (e.g. rosetta_code_binary_search -> binary_search)
    clean_name = algorithm_name.replace("rosetta_code_", "").lower()
    if clean_name in data:
        return data[clean_name]
    if algorithm_name in data:
        return data[algorithm_name]
    return [{"default": True}]
