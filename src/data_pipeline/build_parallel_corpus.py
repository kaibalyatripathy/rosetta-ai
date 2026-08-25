"""
Parallel Corpus Builder Module for Rosetta AI.

Constructs standardized (source_lang, source_code, target_lang, target_code) parallel records
combining:
- Gold data: Rosetta Code multi-language algorithm fixtures (20 algorithms x 12 language directions = 240 gold pairs).
- Silver data: Synthetic LLM teacher parallel records.

Saves the combined output to data/curated/parallel_corpus.jsonl.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.DataPipeline.BuildCorpus")

RAW_DATA_DIR = Path("data/raw")
CURATED_DATA_DIR = Path("data/curated")

GOLD_FIXTURES_FILE = RAW_DATA_DIR / "rosetta_code_fixtures.json"
SILVER_PAIRS_FILE = CURATED_DATA_DIR / "silver_pairs.jsonl"
FINAL_CORPUS_FILE = CURATED_DATA_DIR / "parallel_corpus.jsonl"

LANGUAGES = ["python", "java", "cpp", "javascript"]


def build_gold_records_from_fixtures(fixtures_file: Path) -> List[Dict[str, Any]]:
    """
    Expands 4-language algorithm fixtures into directional parallel records across all 12 pairs.
    """
    gold_records = []
    if not fixtures_file.exists():
        logger.warning(f"Gold fixtures file {fixtures_file} not found.")
        return gold_records

    with open(fixtures_file, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    for item in fixtures:
        alg_name = item.get("algorithm", "unknown")
        impls = item.get("implementations", {})

        for src_lang in LANGUAGES:
            for tgt_lang in LANGUAGES:
                if src_lang == tgt_lang:
                    continue
                
                src_code = impls.get(src_lang)
                tgt_code = impls.get(tgt_lang)

                if src_code and tgt_code:
                    gold_records.append({
                        "source_lang": src_lang,
                        "source_code": src_code,
                        "target_lang": tgt_lang,
                        "target_code": tgt_code,
                        "source_dataset": f"rosetta_code_{alg_name}",
                        "is_gold": True
                    })

    return gold_records


def build_parallel_corpus():
    """
    Merges Gold and Silver data sources into the final parallel_corpus.jsonl file.
    """
    CURATED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Gold records
    gold_records = build_gold_records_from_fixtures(GOLD_FIXTURES_FILE)
    logger.info(f"Loaded {len(gold_records)} Gold parallel pairs.")

    # 2. Ingest Silver records
    silver_records = []
    if SILVER_PAIRS_FILE.exists():
        with open(SILVER_PAIRS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    silver_records.append(json.loads(line))
        logger.info(f"Loaded {len(silver_records)} Silver parallel pairs.")
    else:
        logger.info("No silver pairs file found yet. Proceeding with Gold pairs.")

    all_records = gold_records + silver_records

    # 3. Save combined dataset
    with open(FINAL_CORPUS_FILE, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Successfully compiled {len(all_records)} total pairs into {FINAL_CORPUS_FILE}")

    # 4. Generate statistics
    direction_counts: Dict[str, int] = {}
    gold_count = 0
    silver_count = 0

    for r in all_records:
        if r.get("is_gold"):
            gold_count += 1
        else:
            silver_count += 1

        direction = f"{r['source_lang']} -> {r['target_lang']}"
        direction_counts[direction] = direction_counts.get(direction, 0) + 1

    stats = {
        "total_pairs": len(all_records),
        "gold_pairs": gold_count,
        "silver_pairs": silver_count,
        "direction_matrix": direction_counts
    }

    with open(CURATED_DATA_DIR / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    build_parallel_corpus()
