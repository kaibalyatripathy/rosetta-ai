"""
Tree-Sitter Parser Loader for Rosetta AI.

Loads pre-compiled tree-sitter grammars for:
- Python
- Java
- C++
- JavaScript

Note on Dependency Pinning:
Requires `tree-sitter==0.21.3` and `tree-sitter-languages==1.10.2` to prevent C-API mismatches.
"""

import logging
from typing import Dict, Any

try:
    import tree_sitter
    import tree_sitter_languages as tsl
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.ASTAnalysis.Parsers")

LANG_MAP = {
    "python": "python",
    "py": "python",
    "java": "java",
    "cpp": "cpp",
    "c++": "cpp",
    "javascript": "javascript",
    "js": "javascript"
}

_PARSER_CACHE: Dict[str, Any] = {}


def get_parser(language: str) -> Any:
    """
    Returns a tree-sitter Parser object for the specified programming language.
    """
    if not TREE_SITTER_AVAILABLE:
        raise ImportError("tree-sitter or tree-sitter-languages is not installed.")

    norm_lang = LANG_MAP.get(language.lower().strip())
    if not norm_lang:
        raise ValueError(f"Unsupported language for tree-sitter parsing: {language}. Supported: {list(LANG_MAP.keys())}")

    if norm_lang not in _PARSER_CACHE:
        logger.info(f"Initializing Tree-Sitter parser for '{norm_lang}'...")
        parser = tsl.get_parser(norm_lang)
        _PARSER_CACHE[norm_lang] = parser

    return _PARSER_CACHE[norm_lang]


if __name__ == "__main__":
    if TREE_SITTER_AVAILABLE:
        p = get_parser("python")
        tree = p.parse(b"def add(a, b): return a + b")
        print("Tree-Sitter Root Node:", tree.root_node.type)
    else:
        print("Tree-Sitter unavailable.")
