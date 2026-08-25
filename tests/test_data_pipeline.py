"""
Unit tests for Rosetta AI Data Pipeline.
"""

import json
from pathlib import Path
from src.data_pipeline.download_datasets import get_rosetta_code_gold_fixtures
from src.data_pipeline.silver_data_generation import calculate_cost_estimate


def test_gold_fixtures_count():
    fixtures = get_rosetta_code_gold_fixtures()
    assert len(fixtures) == 20
    for fix in fixtures:
        assert "algorithm" in fix
        assert "implementations" in fix
        impls = fix["implementations"]
        assert "python" in impls
        assert "java" in impls
        assert "cpp" in impls
        assert "javascript" in impls


def test_cost_estimate_calculation():
    estimate = calculate_cost_estimate(num_pairs_per_direction=100, model="gpt-4o-mini")
    assert estimate["total_pairs"] == 1200
    assert estimate["total_directions"] == 12
    assert estimate["total_estimated_cost_usd"] > 0


def test_curated_corpus_exists():
    corpus_file = Path("data/curated/parallel_corpus.jsonl")
    assert corpus_file.exists()
    
    with open(corpus_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > 0
        first_record = json.loads(lines[0])
        assert "source_lang" in first_record
        assert "target_lang" in first_record
        assert "source_code" in first_record
        assert "target_code" in first_record
        assert "is_gold" in first_record
        assert "source_dataset" in first_record
