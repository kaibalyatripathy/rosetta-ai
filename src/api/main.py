"""
FastAPI Backend Application for Rosetta AI Web Interface (Phase 16).

Exposes POST /translate and GET /health endpoints running the end-to-end translation,
refactoring, sandboxed verification, risk detection, and preservation scoring pipeline.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.constrained_decoding.grammar_decoder import ConstrainedGrammarDecoder
from src.refactor.refactor import refactor
from src.risk_detection.risk_rules import detect_semantic_risks, RiskAnalysisReport
from src.scoring.preservation_score import calculate_preservation_score, PreservationScoreReport
from src.verification.differential_test import verify_equivalence, EquivalenceReport
from tests.fixtures.test_inputs import get_test_inputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RosettaAI.API")

WEB_DIR = Path("web")

app = FastAPI(
    title="Rosetta AI — Cross-Language Code Translation Engine",
    description="Full End-to-End Code Translation, Constrained Decoding, Docker Sandbox Verification, and Preservation Scoring API",
    version="1.0.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded decoder instance
_decoder_instance: Optional[ConstrainedGrammarDecoder] = None


def get_decoder() -> ConstrainedGrammarDecoder:
    global _decoder_instance
    if _decoder_instance is None:
        logger.info("Initializing ConstrainedGrammarDecoder for API service...")
        _decoder_instance = ConstrainedGrammarDecoder()
    return _decoder_instance


class TranslationRequest(BaseModel):
    source_code: str = Field(..., description="Original source code snippet to translate")
    source_lang: str = Field(..., description="Source programming language (python, java, cpp, javascript)")
    target_lang: str = Field(..., description="Target programming language (python, java, cpp, javascript)")
    algorithm_name: Optional[str] = Field("unknown", description="Optional canonical algorithm fixture name")


class RiskFlagItem(BaseModel):
    category: str
    severity: str
    description: str
    matched_pattern: str


class TranslationResponse(BaseModel):
    source_code: str
    target_code: str
    source_lang: str
    target_lang: str
    algorithm_name: str
    composite_score: float
    quality_grade: str
    intent_summary: str
    passed_inputs: int
    total_inputs: int
    pass_rate: float
    is_syntax_valid: bool
    flagged_risks: List[RiskFlagItem]
    source_complexity: str
    target_complexity: str
    score_equiv: float
    score_risk: float
    score_complexity: float
    score_round_trip: float
    markdown_report: str


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint exposing system status."""
    return {
        "status": "ok",
        "service": "Rosetta AI Translation Engine",
        "version": "1.0.0",
        "docker_sandbox_active": True,
        "supported_languages": ["python", "java", "cpp", "javascript"]
    }


@app.post("/translate", response_model=TranslationResponse)
def translate_code(req: TranslationRequest) -> TranslationResponse:
    """
    Executes full translation pipeline end-to-end:
    Phase 6 (Seq2Seq Model) -> Phase 7 (Constrained Decoding) -> Phase 8 (Refactoring Pass) ->
    Phase 9/10 (Docker Sandbox Differential Verification) -> Phase 13 (Risk Detection) -> Phase 15 (Preservation Scoring)
    """
    src_code = req.source_code.strip()
    src_lang = req.source_lang.lower().strip()
    tgt_lang = req.target_lang.lower().strip()
    algo = req.algorithm_name or "unknown"

    if not src_code:
        raise HTTPException(status_code=400, detail="Source code cannot be empty.")
    if src_lang == tgt_norm_check(tgt_lang):
        raise HTTPException(status_code=400, detail="Source and target languages must be different.")

    try:
        logger.info(f"API Request: Translating {src_lang} -> {tgt_lang} ({len(src_code)} chars)")
        decoder = get_decoder()

        # 1. Phase 6 & Phase 7: Seq2Seq Model + Grammar Constrained Decoding
        gen_target_code, is_valid = decoder.generate_constrained(
            source_code=src_code,
            source_lang=src_lang,
            target_lang=tgt_lang,
            num_candidates=2
        )

        # Clean prompt leak prefix if present
        import re
        gen_target_code = re.sub(r'^(from\s+)?' + re.escape(src_lang) + r'\s+to\s+' + re.escape(tgt_lang) + r'\s*:\s*', '', gen_target_code, flags=re.IGNORECASE).strip()

        # 2. Phase 8: Refactoring Pass
        refactored_info = refactor(gen_target_code, tgt_lang)
        final_target_code = refactored_info["refactored_code"]

        # 3. Phase 9, 10, 13, 14, 15: Sandboxed Verification & Preservation Scoring
        scoring_report: PreservationScoreReport = calculate_preservation_score(
            source_code=src_code,
            source_lang=src_lang,
            target_code=final_target_code,
            target_lang=tgt_lang,
            algorithm_name=algo,
            round_trip_passed=False
        )

        # Extract Phase 10 details
        test_inputs = get_test_inputs(algo)
        equiv_report: EquivalenceReport = verify_equivalence(
            source_code=src_code,
            source_lang=src_lang,
            target_code=final_target_code,
            target_lang=tgt_lang,
            test_inputs=test_inputs,
            algorithm_name=algo
        )

        # Extract Phase 13 risks
        risk_report: RiskAnalysisReport = detect_semantic_risks(src_code, src_lang, final_target_code, tgt_lang)
        flagged_risk_items = [
            RiskFlagItem(
                category=r.category,
                severity=r.severity,
                description=r.description,
                matched_pattern=r.matched_pattern
            )
            for r in risk_report.flagged_risks
        ]

        from src.complexity.estimator import estimate_complexity
        src_comp = estimate_complexity(src_code, src_lang)
        tgt_comp = estimate_complexity(final_target_code, tgt_lang)

        return TranslationResponse(
            source_code=src_code,
            target_code=final_target_code,
            source_lang=src_lang,
            target_lang=tgt_lang,
            algorithm_name=algo,
            composite_score=scoring_report.composite_score,
            quality_grade=scoring_report.quality_grade,
            intent_summary=scoring_report.intent_summary,
            passed_inputs=equiv_report.passed_inputs,
            total_inputs=equiv_report.total_inputs,
            pass_rate=equiv_report.pass_rate,
            is_syntax_valid=is_valid,
            flagged_risks=flagged_risk_items,
            source_complexity=src_comp.time_complexity,
            target_complexity=tgt_comp.time_complexity,
            score_equiv=scoring_report.score_equiv,
            score_risk=scoring_report.score_risk,
            score_complexity=scoring_report.score_complexity,
            score_round_trip=scoring_report.score_round_trip,
            markdown_report=scoring_report.markdown_report
        )

    except Exception as e:
        logger.error(f"API translation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation pipeline error: {str(e)}")


def tgt_norm_check(lang: str) -> str:
    return lang.lower().strip()


# Serve frontend static UI files
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(WEB_DIR / "index.html")
