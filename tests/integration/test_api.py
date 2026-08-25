"""
Integration Benchmark Tests for FastAPI Translation API Service (Phase 16).

Hits the FastAPI API with 3 real requests across different language pairs,
asserting real (unmocked) responses come back with all expected fields populated.
"""

import logging
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.TestAPI")

client = TestClient(app)


def test_health_endpoint():
    """Verifies health check endpoint returns 200 OK and valid status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["docker_sandbox_active"] is True
    assert "javascript" in data["supported_languages"]


def test_translate_python_to_javascript():
    """Verifies real end-to-end translation request: Python -> JavaScript."""
    py_code = """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
    payload = {
        "source_code": py_code,
        "source_lang": "python",
        "target_lang": "javascript",
        "algorithm_name": "binary_search"
    }

    response = client.post("/translate", json=payload)
    assert response.status_code == 200, f"API error: {response.text}"
    data = response.json()

    # Assert real responses come back with all expected fields populated
    assert "target_code" in data and len(data["target_code"]) > 0
    assert data["source_lang"] == "python"
    assert data["target_lang"] == "javascript"
    assert data["algorithm_name"] == "binary_search"
    assert "composite_score" in data and isinstance(data["composite_score"], float)
    assert "quality_grade" in data and len(data["quality_grade"]) > 0
    assert "passed_inputs" in data and data["passed_inputs"] >= 0
    assert "total_inputs" in data and data["total_inputs"] > 0
    assert "flagged_risks" in data and isinstance(data["flagged_risks"], list)
    assert "source_complexity" in data and len(data["source_complexity"]) > 0
    assert "target_complexity" in data and len(data["target_complexity"]) > 0
    assert "markdown_report" in data and "Rosetta AI" in data["markdown_report"]

    logger.info(f"Python -> JS Translation Success: Score {data['composite_score']:.1f} ({data['quality_grade']})")


def test_translate_python_to_cpp():
    """Verifies real end-to-end translation request: Python -> C++."""
    py_code = """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    payload = {
        "source_code": py_code,
        "source_lang": "python",
        "target_lang": "cpp",
        "algorithm_name": "factorial_recursive"
    }

    response = client.post("/translate", json=payload)
    assert response.status_code == 200, f"API error: {response.text}"
    data = response.json()

    assert "target_code" in data and len(data["target_code"]) > 0
    assert data["source_lang"] == "python"
    assert data["target_lang"] == "cpp"
    assert data["composite_score"] >= 0.0
    assert data["passed_inputs"] >= 0

    logger.info(f"Python -> C++ Translation Success: Score {data['composite_score']:.1f} ({data['quality_grade']})")


def test_translate_java_to_python():
    """Verifies real end-to-end translation request: Java -> Python."""
    java_code = """public class Solution {
    public static int linearSearch(int[] arr, int target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1;
    }
}
"""
    payload = {
        "source_code": java_code,
        "source_lang": "java",
        "target_lang": "python",
        "algorithm_name": "linear_search"
    }

    response = client.post("/translate", json=payload)
    assert response.status_code == 200, f"API error: {response.text}"
    data = response.json()

    assert "target_code" in data and len(data["target_code"]) > 0
    assert data["source_lang"] == "java"
    assert data["target_lang"] == "python"
    assert data["composite_score"] >= 0.0

    logger.info(f"Java -> Python Translation Success: Score {data['composite_score']:.1f} ({data['quality_grade']})")
