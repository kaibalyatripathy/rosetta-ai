"""
GraphCodeBERT Base Model Loader for Rosetta AI.

Downloads and loads 'microsoft/graphcodebert-base' tokenizer and pre-trained Transformer model
from Hugging Face, setting up proper device assignment (CUDA GPU or CPU).
"""

import logging
from typing import Tuple, Any

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.Embeddings.LoadBase")

BASE_MODEL_NAME = "microsoft/graphcodebert-base"


def get_device() -> str:
    """Returns 'cuda' if GPU is available, else 'cpu'."""
    if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_graphcodebert(model_name: str = BASE_MODEL_NAME) -> Tuple[Any, Any, str]:
    """
    Loads GraphCodeBERT tokenizer and model.
    Returns: (tokenizer, model, device)
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("PyTorch or Transformers not installed. Operating in fallback mode.")
        return None, None, "cpu"

    device = get_device()
    logger.info(f"Loading '{model_name}' on device '{device}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    logger.info(f"Successfully loaded '{model_name}' on {device}.")
    return tokenizer, model, device


if __name__ == "__main__":
    tokenizer, model, device = load_graphcodebert()
    if model is not None:
        print(f"Loaded {BASE_MODEL_NAME} successfully on {device}.")
    else:
        print("Transformers library not installed yet.")
