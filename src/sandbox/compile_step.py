"""
Compile Step module for preparing compilation and execution specs
for Python, Java, C++, and JavaScript.
"""

import re
from typing import Dict, List, Tuple, Optional


def extract_java_main_class(code: str) -> str:
    """Extract public class name or fallback to first class or default Solution."""
    public_match = re.search(r'public\s+class\s+([A-Za-z_][A-Za-z0-9_]*)', code)
    if public_match:
        return public_match.group(1)
    
    class_match = re.search(r'class\s+([A-Za-z_][A-Za-z0-9_]*)', code)
    if class_match:
        return class_match.group(1)
        
    return "Solution"


def prepare_sandbox_execution(code: str, lang: str) -> Dict:
    """
    Prepares the file structure and execution commands based on language.
    
    Returns dict with:
        - 'files': Dict[str, str] mapping filename to code
        - 'is_compiled': bool
        - 'compile_cmd': Optional[str] command string for bash execution
        - 'run_cmd': str command string for bash execution
    """
    lang_lower = lang.lower().strip()
    
    if lang_lower in ["python", "py", "python3"]:
        return {
            "files": {"script.py": code},
            "is_compiled": False,
            "compile_cmd": None,
            "run_cmd": "python3 script.py"
        }
    
    elif lang_lower in ["javascript", "js", "node"]:
        return {
            "files": {"script.js": code},
            "is_compiled": False,
            "compile_cmd": None,
            "run_cmd": "node script.js"
        }
        
    elif lang_lower in ["cpp", "c++", "cxx"]:
        return {
            "files": {"main.cpp": code},
            "is_compiled": True,
            "compile_cmd": "g++ -O2 -std=c++17 main.cpp -o main",
            "run_cmd": "./main"
        }
        
    elif lang_lower in ["java"]:
        class_name = extract_java_main_class(code)
        filename = f"{class_name}.java"
        return {
            "files": {filename: code},
            "is_compiled": True,
            "compile_cmd": f"javac {filename}",
            "run_cmd": f"java -Xmx128m {class_name}"
        }
        
    else:
        raise ValueError(f"Unsupported language: {lang}")
