"""
Grammar-Constrained Decoding & Syntax Validation Module for Rosetta AI.

Uses Tree-Sitter AST parsing to evaluate syntax validity of generated code candidates
and enforce syntactically valid code generation (Python, Java, C++, JavaScript).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import torch
from src.ast_analysis.parsers import get_parser, LANG_MAP
from src.seq2seq.generate import generate
from src.seq2seq.model import ConditionedSeq2SeqModel
from src.seq2seq.train import CHECKPOINT_PATH, GNN_CHECKPOINT_PATH, SPLITS_PATH
from src.embeddings.codebert.codebert_extractor import get_code_embedding
from src.gnn.graph_builder import build_pyg_ast_graph
from src.gnn.model import ASTGNNModel
from src.semantic_ir.fusion import fuse_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.ConstrainedDecoding")


def check_syntax_validity(code: str, lang: str) -> Tuple[bool, int]:
    """
    Parses code snippet using Tree-Sitter and checks if AST contains ERROR or MISSING nodes.
    Returns: (is_valid: bool, error_count: int)
    """
    if not code or not code.strip():
        return False, 999

    norm_lang = LANG_MAP.get(lang.lower().strip(), lang.lower().strip())
    try:
        parser = get_parser(norm_lang)
    except Exception:
        return False, 999

    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root = tree.root_node

    error_count = 0

    def _traverse(node):
        nonlocal error_count
        if node.type in {"ERROR", "MISSING"}:
            error_count += 1
        for child in node.children:
            _traverse(child)

    _traverse(root)
    is_valid = (not root.has_error) and (error_count == 0)
    return is_valid, error_count


class ConstrainedGrammarDecoder:
    """
    Syntax-Guided Constrained Decoder using Tree-Sitter candidate filtering.
    """
    def __init__(self, model_name: str = "t5-base"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.gnn_model = ASTGNNModel(num_classes=20)
        if GNN_CHECKPOINT_PATH.exists():
            try:
                self.gnn_model.load_state_dict(torch.load(GNN_CHECKPOINT_PATH, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.gnn_model.eval()

        self.seq2seq_model = ConditionedSeq2SeqModel(model_name=model_name)
        if CHECKPOINT_PATH.exists():
            try:
                self.seq2seq_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True))
            except Exception:
                pass
        self.seq2seq_model.to(self.device)
        self.seq2seq_model.eval()

    def generate_constrained(
        self,
        source_code: str,
        source_lang: str,
        target_lang: str,
        num_candidates: int = 5
    ) -> Tuple[str, bool]:
        """
        Generates candidate translations and selects the top-ranking candidate that parses cleanly.
        Returns: (target_code: str, is_valid: bool)
        """
        # 1. Fused Embedding
        cb_vec = get_code_embedding(source_code, source_lang, embedding_dim=128)
        pyg_data = build_pyg_ast_graph(source_code, source_lang)
        with torch.no_grad():
            g_emb = self.gnn_model.extract_graph_embedding(pyg_data.x.to(self.device), pyg_data.edge_index.to(self.device))
        gnn_vec = g_emb.squeeze(0).tolist()
        fused_vec = fuse_embeddings(cb_vec, gnn_vec, fused_dim=128)

        # 2. Sample Top-N Candidates via Beam / Top-k Decoding
        prompt_text = f"translate {source_lang} to {target_lang}: {source_code}"
        tokenizer = self.seq2seq_model.tokenizer
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        fused_tensor = torch.tensor(fused_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            inputs_embeds = self.seq2seq_model.seq2seq_model.encoder.embed_tokens(inputs["input_ids"])
            soft_prompt = self.seq2seq_model.prompt_projection(fused_tensor).unsqueeze(1)
            conditioned_embeds = torch.cat([soft_prompt, inputs_embeds], dim=1)

            prompt_mask = torch.ones((inputs["attention_mask"].size(0), 1), device=self.device)
            conditioned_mask = torch.cat([prompt_mask, inputs["attention_mask"]], dim=1)

            candidate_outputs = self.seq2seq_model.seq2seq_model.generate(
                inputs_embeds=conditioned_embeds,
                attention_mask=conditioned_mask,
                max_length=256,
                num_return_sequences=num_candidates,
                num_beams=max(num_candidates, 4),
                early_stopping=True
            )

        candidates = [tokenizer.decode(c, skip_special_tokens=True) for c in candidate_outputs]

        # 3. Post-Hoc Syntax Repair & Filtering
        repaired_candidates = []
        for cand in candidates:
            cleaned = cand.strip()
            # Post-hoc conversion from Python def -> target language function keyword
            if target_lang in {"javascript", "java", "cpp"} and cleaned.startswith("def "):
                cleaned = cleaned.replace("def ", "function ", 1)
            if target_lang == "javascript" and not cleaned.endswith(";"):
                cleaned += ";"
            repaired_candidates.append(cleaned)

        # 4. Filter & Rank by Syntax Validity
        for cand in repaired_candidates:
            is_valid, err_cnt = check_syntax_validity(cand, target_lang)
            if is_valid:
                return cand, True

        # Fallback: return candidate with minimal Tree-Sitter error count
        scored_candidates = [(cand, check_syntax_validity(cand, target_lang)[1]) for cand in repaired_candidates]
        scored_candidates.sort(key=lambda x: x[1])
        best_cand, min_errs = scored_candidates[0]
        return best_cand, min_errs == 0



def evaluate_constrained_decoding_benchmark(num_samples: int = 15) -> Dict[str, Any]:
    """
    Evaluates Unconstrained vs Constrained Decoding on held-out test set examples.
    """
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"{SPLITS_PATH} not found.")

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    test_samples = data["test_samples"][:num_samples]

    decoder = ConstrainedGrammarDecoder(model_name="t5-base")

    unconstrained_valid_count = 0
    constrained_valid_count = 0

    logger.info(f"Evaluating Constrained Decoding Benchmark on {len(test_samples)} held-out test samples...")

    for idx, sample in enumerate(test_samples, 1):
        src_code = sample["source_code"]
        src_lang = sample["source_lang"]
        tgt_lang = sample["target_lang"]

        # 1. Unconstrained Generation
        unconstrained_code = generate(src_code, src_lang, tgt_lang, model_name="t5-base")
        u_valid, u_errs = check_syntax_validity(unconstrained_code, tgt_lang)
        if u_valid:
            unconstrained_valid_count += 1

        # 2. Constrained Generation
        constrained_code, c_valid = decoder.generate_constrained(src_code, src_lang, tgt_lang, num_candidates=5)
        if c_valid:
            constrained_valid_count += 1

    total = len(test_samples)
    u_rate = round((unconstrained_valid_count / total) * 100, 2)
    c_rate = round((constrained_valid_count / total) * 100, 2)
    gain = round(c_rate - u_rate, 2)

    results = {
        "total_test_samples": total,
        "unconstrained_valid_count": unconstrained_valid_count,
        "unconstrained_validity_rate_pct": u_rate,
        "constrained_valid_count": constrained_valid_count,
        "constrained_validity_rate_pct": c_rate,
        "validity_improvement_gain_pct": gain
    }

    logger.info(f"Constrained Decoding Benchmark Results: {results}")
    return results


if __name__ == "__main__":
    evaluate_constrained_decoding_benchmark(num_samples=10)
