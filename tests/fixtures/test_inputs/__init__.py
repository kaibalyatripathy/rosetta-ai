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
    clean_name = algorithm_name.replace("rosetta_code_", "").lower().strip()
    if clean_name in data:
        return data[clean_name]
    if algorithm_name in data:
        return data[algorithm_name]
    
    # Substring / alias matching
    for k, v in data.items():
        if clean_name in k or k in clean_name:
            return v
            
    # Aliases
    aliases = {
        "gcd": "gcd_euclidean",
        "prime": "is_prime",
        "primes": "is_prime",
        "kadane": "max_subarray_kadane",
        "max_subarray": "max_subarray_kadane",
        "palindrome": "palindrome_check",
        "power": "power_exponentiation",
        "reverse": "reverse_string"
    }
    if clean_name in aliases and aliases[clean_name] in data:
        return data[aliases[clean_name]]
        
    return [{"default": True}]
