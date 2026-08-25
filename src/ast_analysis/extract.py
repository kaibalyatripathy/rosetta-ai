"""
AST Structural Feature Extractor for Rosetta AI.

Extracts key AST structural features from Python, Java, C++, and JavaScript code snippets using Tree-Sitter:
- Function/Method names
- Parameter count and parameter names
- Loop constructs present (for, while, do-while, etc.)
- Conditional constructs present (if, else, switch, ternary)
- Recursion detection (bool)
"""

from typing import Dict, List, Any, Set
from src.ast_analysis.parsers import get_parser, LANG_MAP


def extract_structure(code: str, lang: str) -> Dict[str, Any]:
    """
    Parses code snippet using Tree-Sitter and returns extracted structural facts.
    """
    norm_lang = LANG_MAP.get(lang.lower().strip(), lang.lower().strip())
    parser = get_parser(norm_lang)
    
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root = tree.root_node

    function_names: List[str] = []
    parameter_names: List[str] = []
    loops: Set[str] = set()
    conditionals: Set[str] = set()
    call_identifiers: Set[str] = set()

    def get_node_text(node) -> str:
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()

    def find_all_identifiers(node) -> List[str]:
        results = []
        if node.type == "identifier":
            results.append(get_node_text(node))
        for child in node.children:
            results.extend(find_all_identifiers(child))
        return results

    def _traverse(node):
        node_type = node.type

        # 1. Function / Method Detection
        if node_type in {"function_definition", "method_declaration", "function_declaration", "method_definition"}:
            fn_name = None
            for child in node.children:
                if child.type == "identifier":
                    fn_name = get_node_text(child)
                    break
                elif child.type == "function_declarator":
                    for sub in child.children:
                        if sub.type == "identifier":
                            fn_name = get_node_text(sub)
                            break
            if fn_name and fn_name not in function_names:
                function_names.append(fn_name)

        # 2. Parameter Extraction
        if node_type in {"parameters", "formal_parameters", "parameter_list"}:
            for child in node.children:
                if child.type in {"identifier", "formal_parameter", "parameter_declaration"}:
                    # Find last identifier in parameter node (variable name comes after type name)
                    ids = find_all_identifiers(child)
                    if ids:
                        # Exclude type names like vector, int, double, String, std
                        filtered_ids = [i for i in ids if i not in {"int", "double", "bool", "boolean", "void", "String", "vector", "std", "const", "long", "float", "char", "auto"}]
                        if filtered_ids:
                            pname = filtered_ids[-1]
                            if pname not in parameter_names and pname not in {"self", "this", "(", ")", ","}:
                                parameter_names.append(pname)

        # 3. Loop Constructs
        if node_type in {"for_statement", "enhanced_for_statement", "for_in_statement", "for_of_statement", "range_for_statement", "for_range_loop", "for_in_clause"}:
            loops.add("for")
        elif node_type in {"while_statement", "do_statement"}:
            loops.add("while")



        # 4. Conditional Constructs
        if node_type == "if_statement":
            conditionals.add("if")
            if any(c.type == "else" for c in node.children):
                conditionals.add("else")
        elif node_type in {"ternary_expression", "conditional_expression"}:
            conditionals.add("ternary")
        elif node_type in {"switch_statement", "switch_expression"}:
            conditionals.add("switch")

        # 5. Call Expressions & Method Invocations (for Recursion Detection)
        if node_type in {"call", "call_expression", "method_invocation"}:
            ids = find_all_identifiers(node)
            for i in ids:
                call_identifiers.add(i)

        for child in node.children:
            _traverse(child)

    _traverse(root)

    # Detect recursion: if any extracted function name is called within the body
    has_recursion = any(fn in call_identifiers for fn in function_names)

    return {
        "language": norm_lang,
        "function_names": function_names,
        "parameter_count": len(parameter_names),
        "parameter_names": parameter_names,
        "loops": sorted(list(loops)),
        "conditionals": sorted(list(conditionals)),
        "has_recursion": has_recursion
    }


if __name__ == "__main__":
    py_code = "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    res = extract_structure(py_code, "python")
    print("Extracted Python Structure:", res)
