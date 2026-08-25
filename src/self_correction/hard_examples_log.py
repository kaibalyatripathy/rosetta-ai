"""
Hard Examples Logging Module for Rosetta AI.

Logs translation cases that required self-correction to `data/curated/hard_examples.jsonl`
for future fine-tuning iterations of the seq2seq model.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

HARD_EXAMPLES_FILE = Path("data/curated/hard_examples.jsonl")
logger = logging.getLogger("RosettaAI.HardExamplesLog")


def log_hard_example(
    source_code: str,
    source_lang: str,
    failed_target_code: str,
    target_lang: str,
    fixed_target_code: Optional[str],
    attempts_used: int,
    success: bool,
    failure_details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Appends a hard example record to `data/curated/hard_examples.jsonl`.
    """
    HARD_EXAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    record = {
        "source_lang": source_lang.lower().strip(),
        "source_code": source_code,
        "target_lang": target_lang.lower().strip(),
        "original_failed_target": failed_target_code,
        "fixed_target_code": fixed_target_code,
        "attempts_used": attempts_used,
        "success": success,
        "failure_details": failure_details or {}
    }
    
    with open(HARD_EXAMPLES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        
    logger.info(f"Logged hard example ({source_lang}->{target_lang}) [success={success}, attempts={attempts_used}] to {HARD_EXAMPLES_FILE}")
