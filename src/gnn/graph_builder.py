"""
AST Graph Builder for Rosetta AI.

Parses source code in Python, Java, C++, and JavaScript into Abstract Syntax Tree (AST) graphs.
Extracts:
1. AST Node Types (e.g., FunctionDef, Assign, For, If, Variable, Literal, Operator)
2. Parent-Child Structural Edges
3. Next-Token Control Flow Edges
4. Variable-Usage Data Flow Edges
"""

import ast
import re
from typing import Dict, List, Tuple, Any, Optional


class ASTNode:
    """Represents a node in the AST Graph."""
    def __init__(self, node_id: int, label: str, node_type: str, value: Optional[str] = None):
        self.node_id = node_id
        self.label = label
        self.node_type = node_type
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type,
            "value": self.value
        }


class ASTGraph:
    """Represents an AST Graph with nodes and directed edges."""
    def __init__(self, language: str):
        self.language = language
        self.nodes: List[ASTNode] = []
        self.edges: List[Tuple[int, int, str]] = []  # (source_id, target_id, edge_type)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def add_node(self, label: str, node_type: str, value: Optional[str] = None) -> int:

        node_id = len(self.nodes)
        node = ASTNode(node_id, label, node_type, value)
        self.nodes.append(node)
        return node_id

    def add_edge(self, src: int, dst: int, edge_type: str = "child"):
        self.edges.append((src, dst, edge_type))

    def get_edge_index(self) -> List[List[int]]:
        """Returns 2xE edge index list [[src_0, src_1...], [dst_0, dst_1...]]."""
        if not self.edges:
            return [[], []]
        srcs = [e[0] for e in self.edges]
        dsts = [e[1] for e in self.edges]
        return [srcs, dsts]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": self.edges,
            "edge_index": self.get_edge_index()
        }


class PythonASTGraphBuilder:
    """Constructs AST graphs from Python code using standard ast module."""
    
    @staticmethod
    def build(code: str) -> ASTGraph:
        graph = ASTGraph("python")
        try:
            tree = ast.parse(code)
        except Exception:
            # Fallback for code fragments
            root_id = graph.add_node("ModuleFragment", "Module")
            tokens = [t for t in re.split(r'\s+|([{}()\[\];,])', code) if t and t.strip()]
            prev_id = root_id
            for tok in tokens[:50]:
                n_id = graph.add_node(tok, "Token", tok)
                graph.add_edge(prev_id, n_id, "child")
                prev_id = n_id
            return graph

        def _traverse(node, parent_id: Optional[int] = None) -> int:
            node_type = type(node).__name__
            label = node_type
            val = None

            if isinstance(node, ast.Name):
                val = node.id
                label = f"Name({val})"
            elif isinstance(node, ast.Constant):
                val = str(node.value)
                label = f"Constant({val})"
            elif isinstance(node, ast.FunctionDef):
                val = node.name
                label = f"FunctionDef({val})"

            node_id = graph.add_node(label, node_type, val)
            if parent_id is not None:
                graph.add_edge(parent_id, node_id, "child")

            for child in ast.iter_child_nodes(node):
                _traverse(child, node_id)

            return node_id

        _traverse(tree)
        return graph


class GenericASTGraphBuilder:
    """Constructs tokenized AST representations for Java, C++, and JavaScript."""
    
    @staticmethod
    def build(code: str, language: str) -> ASTGraph:
        graph = ASTGraph(language)
        root_id = graph.add_node(f"Root_{language.upper()}", "Root")
        
        # Tokenize code into keywords, identifiers, symbols
        tokens = [t for t in re.split(r'(\s+|[{}()\[\];,+\-*/=><!&|])', code) if t and t.strip()]
        
        prev_id = root_id
        var_last_seen: Dict[str, int] = {}

        for tok in tokens:
            if tok in {"class", "public", "private", "static", "void", "int", "double", "bool", "boolean", "function", "let", "const", "var", "def", "return", "if", "else", "for", "while"}:
                n_type = "Keyword"
            elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tok):
                n_type = "Identifier"
            elif re.match(r'^\d+$', tok):
                n_type = "Literal"
            else:
                n_type = "Symbol"

            node_id = graph.add_node(tok, n_type, tok)
            graph.add_edge(root_id, node_id, "child")
            graph.add_edge(prev_id, node_id, "next_token")
            
            if n_type == "Identifier":
                if tok in var_last_seen:
                    graph.add_edge(var_last_seen[tok], node_id, "data_flow")
                var_last_seen[tok] = node_id

            prev_id = node_id

        return graph


def build_ast_graph(code: str, language: str) -> ASTGraph:
    """Factory function to build an AST graph for any supported language."""
    lang = language.lower().strip()
    if lang == "python":
        return PythonASTGraphBuilder.build(code)
    elif lang in {"java", "cpp", "c++", "javascript", "js"}:
        return GenericASTGraphBuilder.build(code, lang)
    else:
        raise ValueError(f"Unsupported language for AST parsing: {language}")


if __name__ == "__main__":
    sample_py = "def add(a, b):\n    return a + b"
    g = build_ast_graph(sample_py, "python")
    print("Python AST Nodes:", g.num_nodes, "Edges:", len(g.edges))

    sample_cpp = "int add(int a, int b) { return a + b; }"
    g_cpp = build_ast_graph(sample_cpp, "cpp")
    print("C++ AST Nodes:", g_cpp.num_nodes, "Edges:", len(g_cpp.edges))
