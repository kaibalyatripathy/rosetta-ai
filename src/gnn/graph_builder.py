"""
PyTorch Geometric (PyG) & AST Graph Builder for Tree-Sitter AST Graphs.

Converts Tree-Sitter AST structures (from Phase 3) into PyTorch Geometric `Data` objects and ASTGraph structures:
- Node features (x): AST node types mapped to integer vocabulary indices
- Edge index (edge_index): 2xE directed edge tensor (parent-child, next-token, data-flow)
"""

from typing import Dict, List, Tuple, Any, Optional
from src.ast_analysis.parsers import get_parser, LANG_MAP

try:
    import torch
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


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


def build_ast_graph(code: str, language: str) -> ASTGraph:
    """
    Constructs an ASTGraph instance for a given code snippet using Tree-Sitter.
    """
    norm_lang = LANG_MAP.get(language.lower().strip(), language.lower().strip())
    parser = get_parser(norm_lang)
    graph = ASTGraph(norm_lang)

    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root = tree.root_node

    def _traverse(node, parent_id: Optional[int] = None) -> int:
        text_val = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
        n_id = graph.add_node(node.type, node.type, text_val)
        if parent_id is not None:
            graph.add_edge(parent_id, n_id, "child")
        for child in node.children:
            _traverse(child, n_id)
        return n_id

    _traverse(root)
    return graph


# AST Node Type Vocabulary Mapping
NODE_TYPE_VOCAB: Dict[str, int] = {
    "module": 1,
    "program": 2,
    "function_definition": 3,
    "function_declaration": 4,
    "method_declaration": 5,
    "method_definition": 6,
    "parameters": 7,
    "formal_parameters": 8,
    "parameter_list": 9,
    "identifier": 10,
    "type_identifier": 11,
    "for_statement": 12,
    "while_statement": 13,
    "do_statement": 14,
    "if_statement": 15,
    "else_clause": 16,
    "binary_operator": 17,
    "binary_expression": 18,
    "return_statement": 19,
    "call": 20,
    "call_expression": 21,
    "method_invocation": 22,
    "assignment": 23,
    "expression_statement": 24,
    "primitive_type": 25,
    "number_literal": 26,
    "string_literal": 27,
    "comment": 28,
    "UNKNOWN": 99
}


def build_pyg_ast_graph(code: str, lang: str, label: Optional[int] = None) -> Any:
    """
    Parses code snippet via Tree-Sitter and constructs a PyTorch Geometric Data object.
    """
    if not PYG_AVAILABLE:
        raise ImportError("PyTorch or PyTorch Geometric is not installed.")

    norm_lang = LANG_MAP.get(lang.lower().strip(), lang.lower().strip())
    parser = get_parser(norm_lang)

    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root = tree.root_node

    node_types: List[int] = []
    edges_src: List[int] = []
    edges_dst: List[int] = []

    def _traverse(node) -> int:
        current_id = len(node_types)
        ntype = node.type
        type_id = NODE_TYPE_VOCAB.get(ntype, NODE_TYPE_VOCAB["UNKNOWN"])
        node_types.append(type_id)

        for child in node.children:
            child_id = _traverse(child)
            # Parent-Child edge
            edges_src.append(current_id)
            edges_dst.append(child_id)

        return current_id

    _traverse(root)

    x = torch.tensor(node_types, dtype=torch.long)
    if edges_src:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y_tensor = torch.tensor([label], dtype=torch.long) if label is not None else None

    return Data(x=x, edge_index=edge_index, y=y_tensor, num_nodes=len(node_types))


if __name__ == "__main__":
    if PYG_AVAILABLE:
        data = build_pyg_ast_graph("def add(a, b): return a + b", "python", label=0)
        print("PyG AST Graph Data Object:", data)
