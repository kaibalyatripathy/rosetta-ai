"""
Core Rosetta AI Translator Engine.

Orchestrates multi-language AST graph building, structural GNN encoding,
CodeBERT sequence representation extraction, and translation pipeline conditioning.
"""

import logging
from typing import Dict, Any, List
from src.gnn.graph_builder import build_ast_graph, ASTGraph
from src.gnn.model import encode_ast_graph
from src.embeddings.codebert.codebert_extractor import get_code_embedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.Engine")

SUPPORTED_LANGUAGES = {"python", "java", "cpp", "javascript"}


class RosettaTranslator:
    """
    Main Rosetta AI Translator Engine.
    """
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        logger.info(f"Initialized RosettaTranslator engine with embedding_dim={embedding_dim}")

    def validate_languages(self, source_lang: str, target_lang: str):
        src = source_lang.lower().strip()
        tgt = target_lang.lower().strip()
        if src not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported source language: {source_lang}. Supported: {SUPPORTED_LANGUAGES}")
        if tgt not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported target language: {target_lang}. Supported: {SUPPORTED_LANGUAGES}")
        if src == tgt:
            raise ValueError(f"Source and target languages must be different. Got {src} -> {tgt}")

    def process_code(self, source_code: str, source_lang: str) -> Dict[str, Any]:
        """
        Parses source code into AST graph, GNN graph embedding, and CodeBERT sequence embedding.
        """
        # 1. AST Graph Construction
        ast_graph: ASTGraph = build_ast_graph(source_code, source_lang)
        
        # 2. Structural GNN AST Graph Encoding
        ast_embedding: List[float] = encode_ast_graph(ast_graph, hidden_dim=self.embedding_dim)

        # 3. CodeBERT Token Sequence Representation
        seq_embedding: List[float] = get_code_embedding(source_code, source_lang, embedding_dim=self.embedding_dim)

        return {
            "source_lang": source_lang,
            "ast_graph": ast_graph.to_dict(),
            "ast_embedding": ast_embedding,
            "seq_embedding": seq_embedding,
            "num_nodes": ast_graph.num_nodes,
            "num_edges": len(ast_graph.edges)
        }

    def prepare_translation_context(self, source_code: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Validates translation request and builds multi-modal conditioned representation.
        """
        self.validate_languages(source_lang, target_lang)
        processed = self.process_code(source_code, source_lang)
        
        return {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_code": source_code,
            "ast_embedding_dim": len(processed["ast_embedding"]),
            "seq_embedding_dim": len(processed["seq_embedding"]),
            "ast_node_count": processed["num_nodes"],
            "ast_edge_count": processed["num_edges"],
            "status": "conditioned_representation_ready"
        }


if __name__ == "__main__":
    translator = RosettaTranslator()
    sample_code = "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)"
    context = translator.prepare_translation_context(sample_code, "python", "cpp")
    print("Translation Context Prepared:", context)
