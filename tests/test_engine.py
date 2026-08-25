"""
Unit tests for core Rosetta Translator Engine.
"""

import pytest
from src.rosetta_ai.engine import RosettaTranslator


def test_translator_initialization():
    translator = RosettaTranslator(embedding_dim=64)
    assert translator.embedding_dim == 64


def test_language_validation():
    translator = RosettaTranslator()
    
    # Valid directions
    translator.validate_languages("python", "cpp")
    translator.validate_languages("java", "javascript")

    # Invalid source
    with pytest.raises(ValueError, match="Unsupported source language"):
        translator.validate_languages("ruby", "python")

    # Invalid target
    with pytest.raises(ValueError, match="Unsupported target language"):
        translator.validate_languages("python", "go")

    # Identical languages
    with pytest.raises(ValueError, match="Source and target languages must be different"):
        translator.validate_languages("python", "python")


def test_prepare_translation_context():
    translator = RosettaTranslator(embedding_dim=128)
    code = "public static int add(int a, int b) { return a + b; }"
    context = translator.prepare_translation_context(code, "java", "python")

    assert context["source_lang"] == "java"
    assert context["target_lang"] == "python"
    assert context["ast_embedding_dim"] == 128
    assert context["seq_embedding_dim"] == 128
    assert context["ast_node_count"] > 0
    assert context["status"] == "conditioned_representation_ready"
