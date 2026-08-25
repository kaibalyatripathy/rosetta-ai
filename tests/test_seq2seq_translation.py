"""
Integration tests for Conditioned Seq2Seq Transformer Model & Translation Pipeline.
"""

import json
from pathlib import Path
import pytest

from src.seq2seq.generate import generate
from src.seq2seq.model import ConditionedSeq2SeqModel
from src.seq2seq.train import SPLITS_PATH


@pytest.fixture(scope="module")
def test_dataset():
    assert SPLITS_PATH.exists(), f"Split dataset file {SPLITS_PATH} not found."
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_samples"]


def test_seq2seq_model_instantiation():
    model = ConditionedSeq2SeqModel(model_name="t5-small")
    assert model.tokenizer is not None
    assert model.seq2seq_model is not None


def test_seq2seq_translation_generation_shape(test_dataset):
    sample = test_dataset[0]
    src_code = sample["source_code"]
    src_lang = sample["source_lang"]
    tgt_lang = sample["target_lang"]

    output_code = generate(src_code, src_lang, tgt_lang, model_name="t5-small")
    assert isinstance(output_code, str)
    assert len(output_code.strip()) > 0


def test_held_out_plausibility_benchmark(test_dataset):
    """
    Evaluates syntactical plausibility rate on held-out test set examples.
    """
    plausible_count = 0
    total_eval = min(10, len(test_dataset))

    print("\n" + "=" * 80)
    print("HELD-OUT TEST SET CODE TRANSLATION SAMPLES")
    print("=" * 80)

    for idx in range(total_eval):
        sample = test_dataset[idx]
        src_code = sample["source_code"]
        src_lang = sample["source_lang"]
        tgt_lang = sample["target_lang"]
        gold_tgt = sample["target_code"]

        gen_code = generate(src_code, src_lang, tgt_lang, model_name="t5-small")

        # Syntactical Plausibility Check: non-empty output with structural tokens
        is_plausible = len(gen_code.strip()) > 0 and ("def" in gen_code or "function" in gen_code or "{" in gen_code or ";" in gen_code or "(" in gen_code)
        if is_plausible:
            plausible_count += 1

        print(f"\n--- Sample {idx+1}/{total_eval} [{src_lang} -> {tgt_lang}] ---")
        print(f"SOURCE ({src_lang}):\n{src_code[:120]}...")
        print(f"GENERATED TRANSLATION ({tgt_lang}):\n{gen_code}")
        print(f"PLAUSIBLE: {is_plausible}")

    print("=" * 80 + "\n")
    assert plausible_count > 0
