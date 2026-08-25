"""
Post-Generation LLM Refactoring Pass for Rosetta AI.

Uses a Local Coding LLM (Qwen2.5-Coder-1.5B) to transform Phase 7 syntactically valid code
into idiomatic target-language code following modern best practices. 
Falls back to Google Gemini API if the local model is unavailable or fails.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from src.refactor.style_check import compute_style_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.RefactorPass")

# Load environment variables
env_file = Path(".env")
if env_file.exists():
    load_dotenv(env_file)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Global variables for local LLM
_local_llm_pipeline = None

def get_local_llm_pipeline():
    global _local_llm_pipeline
    if _local_llm_pipeline is None:
        try:
            import torch
            from transformers import pipeline
            logger.info("Loading Local Refactoring LLM (Qwen2.5-Coder-1.5B)...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _local_llm_pipeline = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                device=-1 if device == "cpu" else None
            )
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            _local_llm_pipeline = False # Mark as failed to prevent retries
    return _local_llm_pipeline

def call_local_llm(prompt: str) -> Optional[str]:
    pipe = get_local_llm_pipeline()
    if not pipe:
        return None
        
    try:
        messages = [
            {"role": "system", "content": "You are a helpful expert software engineer."},
            {"role": "user", "content": prompt}
        ]
        
        if hasattr(pipe.tokenizer, "apply_chat_template"):
            prompt_input = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_input = f"<|im_start|>system\nYou are a helpful expert software engineer.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
        outputs = pipe(
            prompt_input,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            return_full_text=False
        )
        return outputs[0]["generated_text"].strip()
    except Exception as e:
        logger.error(f"Local LLM generation failed: {e}")
        return None


def call_gemini_llm(prompt: str) -> Optional[str]:
    """
    Calls Google Gemini API (gemini-3.6-flash) with fallback.
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not configured in .env; applying AST rule-based refactoring fallback.")
        return None

    # Try google.genai or google.generativeai
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        return response.text
    except Exception as e1:
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=GEMINI_API_KEY)
            model = genai_legacy.GenerativeModel("gemini-3.6-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e2:
            logger.warning(f"Gemini API call failed ({e1}; {e2}). Applying rule-based refactoring fallback.")
            return None


def refactor(target_code: str, target_lang: str) -> Dict[str, Any]:
    """
    Refactors target code into target-language idiomatic best practices using local LLM or Gemini LLM.
    Returns: Dict containing original code, refactored code, before/after style scores.
    """
    before_metrics = compute_style_score(target_code, target_lang)

    prompt = f"""You are an expert software engineer in {target_lang}.
Refactor the following {target_lang} code snippet to follow modern, clean, idiomatic best practices:
- Apply target language naming conventions (e.g. snake_case for Python, camelCase for JS/Java/C++).
- Use modern language features (e.g. JS const/let and arrow functions, Java Streams/var, C++ range-for, Python list comprehensions).
- Add clear PEP8/idiomatic formatting and proper operator spacing.
- STRICT RULE: Do NOT change the runtime functional behavior or logic of the code.
- Return ONLY the clean refactored code inside ```{target_lang} ``` code blocks.

Input Code ({target_lang}):
{target_code}
"""

    llm_output = call_local_llm(prompt)
    if not llm_output:
        logger.info("Local LLM failed or unavailable. Falling back to Gemini API...")
        llm_output = call_gemini_llm(prompt)
        
    refactored_code = target_code

    if llm_output:
        # Extract code inside markdown code block
        match = re.search(rf"```(?:{target_lang})?\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
        if match:
            refactored_code = match.group(1).strip()
        else:
            refactored_code = llm_output.strip()
    else:
        # Fallback AST Rule-Based Stylistic Enhancements
        refactored_code = _rule_based_refactor(target_code, target_lang)

    after_metrics = compute_style_score(refactored_code, target_lang)

    return {
        "language": target_lang.lower().strip(),
        "original_code": target_code,
        "refactored_code": refactored_code,
        "before_score": before_metrics["score"],
        "before_warnings": before_metrics["warnings_count"],
        "after_score": after_metrics["score"],
        "after_warnings": after_metrics["warnings_count"],
        "score_improved": after_metrics["score"] >= before_metrics["score"]
    }


def _rule_based_refactor(code: str, lang: str) -> str:
    """Fallback rule-based stylistic refactoring when LLM APIs are unavailable."""
    norm_lang = lang.lower().strip()
    lines = code.splitlines()
    cleaned_lines = []

    for line in lines:
        l = line.rstrip()
        if norm_lang == "python":
            # Add spaces around operators
            l = re.sub(r"([a-zA-Z0-9_])=([a-zA-Z0-9_])", r"\1 = \2", l)
        elif norm_lang == "javascript":
            # Replace legacy var with let/const
            l = re.sub(r"\bvar\s+", "let ", l)
            if l.strip() and not l.strip().endswith((';', '{', '}', ':', '//')):
                l += ";"
        cleaned_lines.append(l)

    return "\n".join(cleaned_lines)


if __name__ == "__main__":
    py_code = "def MyFunction(a,b):\n    var_x=a+b\n    return var_x"
    res = refactor(py_code, "python")
    print("Refactoring Pass Results:", res)
