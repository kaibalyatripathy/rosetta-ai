# Rosetta AI Architecture & Design Specifications

## 1. System Overview
Rosetta AI is a multi-language code translation and semantic verification system for Python, Java, C++, and JavaScript.

---

## 2. Training Data Pipeline

### Data Composition & Counts (Real Empirical Benchmark)

- **Total Parallel Pairs**: 840
- **Gold Pairs (Verified Parallel Algorithms)**: 240
- **Silver Pairs (LLM Teacher Synthetic Data)**: 600

### Language Pair Direction Matrix (12 Directions)

| Source Language | Target Language | Gold Pairs | Silver Pairs | Total Pairs |
|:----------------|:----------------|:-----------|:-------------|:------------|
| `python`        | `java`          | 20         | 50           | 70          |
| `python`        | `cpp`           | 20         | 50           | 70          |
| `python`        | `javascript`    | 20         | 50           | 70          |
| `java`          | `python`        | 20         | 50           | 70          |
| `java`          | `cpp`           | 20         | 50           | 70          |
| `java`          | `javascript`    | 20         | 50           | 70          |
| `cpp`           | `python`        | 20         | 50           | 70          |
| `cpp`           | `java`          | 20         | 50           | 70          |
| `cpp`           | `javascript`    | 20         | 50           | 70          |
| `javascript`    | `python`        | 20         | 50           | 70          |
| `javascript`    | `java`          | 20         | 50           | 70          |
| `javascript`    | `cpp`           | 20         | 50           | 70          |
| **TOTAL**       |                 | **240**    | **600**      | **840**     |

---

### Data Sources & Curation Methodology

1. **Gold Parallel Fixtures (`is_gold: true`)**:
   - Curated set of 20 classic algorithms (Binary Search, Merge Sort, Quick Sort, LRU Cache, Dijkstra, Kadane's algorithm, etc.) fully implemented in Python, Java, C++, and JavaScript.
   - Expanded across all 12 language-pair permutations ($20 \times 12 = 240$ pairs).

2. **Silver Synthetic Teacher Pairs (`is_gold: false`)**:
   - Generated via LLM Teacher pipeline (`silver_data_generation.py`) translating single-language functions into target languages.
   - Explicitly tagged with `is_gold: false` and `source_dataset: "silver_llm_teacher"` in `data/curated/parallel_corpus.jsonl`.

---

### Known Data Limitations & Disclosures

1. **CodeSearchNet C++ Coverage Gap**:
   - CodeSearchNet natively includes Python, Java, JavaScript, Go, Ruby, and PHP datasets, but **does not natively support C++**.
   - C++ parallel pairs rely heavily on Rosetta Code gold fixtures and LLM teacher silver synthetic pair generation.

2. **Scarcity of Natural 4-Language Parallel Code**:
   - True naturally occurring multi-language parallel repositories spanning Python, Java, C++, and JavaScript are extremely rare in wild open-source data.
   - Synthetic (silver) data dominates large-scale corpus scaling; quality control relies on downstream AST validation and tree-sitter constrained decoding.
