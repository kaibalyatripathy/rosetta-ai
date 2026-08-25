"""
Downloads a medium-sized Code-to-Code translation dataset from HuggingFace
to allow for real local training on the CodeT5p base model.
"""

import json
from pathlib import Path
from datasets import load_dataset

def download_and_format_dataset():
    print("Downloading FULL CodeXGLUE translation dataset (Java <-> C#) from HuggingFace...")
    # Fetch the full train split (~10.3k pairs)
    dataset = load_dataset("google/code_x_glue_cc_code_to_code_trans", split="train")
    
    out_dir = Path("data/curated")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "parallel_corpus_large.jsonl"
    
    # 10.3k * 2 (mirroring) = ~20.6k pairs
    print(f"Formatting and mirroring {len(dataset)} pairs into ~20,600 pairs...")
    
    with open(out_file, "w", encoding="utf-8") as f:
        for item in dataset:
            # 1. Forward Mapping: Java -> C++
            record_fwd = {
                "source_code": item["java"],
                "source_lang": "java",
                "target_code": item["cs"],
                "target_lang": "cpp", # Map C# to C++
            }
            f.write(json.dumps(record_fwd) + "\n")
            
            # 2. Reverse Mapping: C++ -> Java
            record_rev = {
                "source_code": item["cs"],
                "source_lang": "cpp",
                "target_code": item["java"],
                "target_lang": "java",
            }
            f.write(json.dumps(record_rev) + "\n")
            
    print(f"Done! Created massive dataset at {out_file}.")
    print("To train on this, modify PARALLEL_CORPUS_PATH in src/seq2seq/train.py to point here,")
    print("and run `python -m src.seq2seq.train`")

if __name__ == "__main__":
    download_and_format_dataset()
