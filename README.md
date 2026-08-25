# Rosetta AI
**Cross-Lingual Code Understanding and Translation Engine**

Rosetta AI is an automated, AI-driven code translation and benchmarking framework designed to seamlessly convert algorithms across programming languages (**Python, Java, C++, JavaScript**) while strictly maintaining semantic equivalence, logic, and runtime complexity.

Built using a hybrid neuro-symbolic pipeline, it fuses AST-based Graph Neural Networks with massive sequence-to-sequence Transformers, constrained grammar decoders, and Local Coding LLMs (Qwen) to generate pristine code. Every translation is automatically verified in a secure Docker Sandbox.

---

## 🚀 Architecture Pipeline
The system operates through an advanced end-to-end pipeline:
1. **Semantic Extraction (Phase 2 & 4):** Uses CodeBERT and PyTorch Geometric Graph Neural Networks to map the structural and semantic logic of the source code.
2. **Translation Generation (Phase 6):** Uses a pre-trained **CodeT5p** Seq2Seq model conditioned on the GNN semantic embeddings.
3. **Constrained Decoding (Phase 7):** Uses Tree-Sitter to forcefully guide the Seq2Seq output into syntactically valid code.
4. **Refactoring Pass (Phase 8):** Intercepts the raw translation and feeds it to a **Local Qwen2.5-Coder-1.5B** LLM running natively on the GPU to apply idiomatic best practices. (Falls back to Gemini API).
5. **Sandbox Verification (Phase 9-10):** Compiles and runs both the source code and the translated code in a secure Docker sandbox against dynamically generated test inputs.
6. **Self-Correction (Phase 11):** If the sandbox fails, an automated LLM repair loop attempts to fix the bugs.
7. **Preservation Scoring (Phase 15):** Generates a comprehensive score (out of 100) based on equivalence pass rate, algorithmic complexity matching, and AST semantic risk.

---

## 🛠 How to Run

### 1. Launch the Web UI & API
The system features a beautiful dark-mode Glassmorphic Web Interface.
```bash
python -m uvicorn src.api.main:app --reload
```
Navigate to: `http://127.0.0.1:8000/`

### 2. Massive Local Training
To permanently bake logic into your local `CodeT5p` model without relying on API keys, run the automated HuggingFace CodeXGLUE dataset downloader and kick off a PyTorch fine-tuning loop:
```bash
# 1. Download and format ~20,600 parallel translation pairs
python scripts/download_hf_dataset.py

# 2. Run the massive multi-hour PyTorch fine-tuning loop
python -m src.seq2seq.train
```

---

## 📈 Project Status (All 16 Phases Completed)
- [x] **Phase 1:** Translation Engine Core Logic & AST Pipeline
- [x] **Phase 2:** Multi-Lingual CodeBERT Extension & Fine-Tuning
- [x] **Phase 3:** AST Analysis Layer (Tree-Sitter Structural Extraction)
- [x] **Phase 4:** GNN AST Structural Representation (PyTorch Geometric)
- [x] **Phase 5:** Fused Semantic Representation & 3-Way Benchmark
- [x] **Phase 6:** Conditioned Seq2Seq Transformer Translation Model (`CodeT5p`)
- [x] **Phase 7:** Controlled/Constrained Code Generation (Tree-Sitter Guided)
- [x] **Phase 8:** Post-Generation Refactoring Pass (Local `Qwen2.5-Coder`)
- [x] **Phase 9:** Secure Docker Sandbox Execution Environment
- [x] **Phase 10:** Functional Equivalence Differential Testing
- [x] **Phase 11:** Automated LLM Self-Correction Loop
- [x] **Phase 12:** Round-Trip Stability Verification (A $\rightarrow$ B $\rightarrow$ A)
- [x] **Phase 13:** Rule-Based Semantic Risk Detection
- [x] **Phase 14:** Algorithmic Complexity (Big-O) Preservation Estimation
- [x] **Phase 15:** Composite Semantic Preservation Scoring Engine
- [x] **Phase 16:** Interactive FastAPI & React Web Interface

*Built with PyTorch, Tree-Sitter, Docker Sandbox, FastAPI, and HuggingFace Transformers.*
