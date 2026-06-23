"""
llm_explainer.py - Grounded LLM explanation layer for the ENAMED audit dashboard.

It consumes ONLY pre-computed audit numbers (course code, score, national median,
predicted tier, and the ranked local feature contributions) and asks a Gemini
model to phrase them as concise, actionable guidance for a course coordinator.

Design principles
-----------------
- Grounded: the model receives only numbers already produced by the ML pipeline.
  The system prompt forbids inventing any value, percentage, name, or fact that
  is not present in the input.
- Privacy by design: only masked, course-level aggregates are sent (no
  student-level data), consistent with the dashboard's LGPD stance.
- Fail-safe: if the SDK or API key is missing, or the call fails, `explain`
  returns None and the dashboard falls back to its rule-based diagnostic.

Setup
-----
    pip install google-genai
    # then provide a key via Streamlit secrets (.streamlit/secrets.toml):
    #   GEMINI_API_KEY = "your-key"
    # or via environment variable GEMINI_API_KEY
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

try:
    import streamlit as st
except Exception:  # running outside Streamlit (e.g. unit tests)
    st = None

# Newest GA Flash model (released 2026-05-19). For free-tier prototyping you can
# switch to a Flash-Lite string (e.g. "gemini-3.1-flash-lite").
DEFAULT_MODEL = "gemini-3.5-flash"

SYSTEM_INSTRUCTION: Dict[str, str] = {
    "English": (
        "You are an assistant that explains PRE-COMPUTED audit results from a "
        "machine-learning model to a medical-course coordinator. You will receive a "
        "JSON object whose numbers were already calculated by the system. Rules: "
        "(1) Use ONLY the numbers and labels present in the input. NEVER invent "
        "values, percentages, institution names, or facts that are not there. "
        "(2) If a value is absent, do not mention it. (3) Be concise and concrete. "
        "(4) Write in English. (5) Structure your answer as: one short paragraph "
        "summarising the situation; then 2-3 bullet action items derived strictly "
        "from the items whose sentiment is 'friction'; then a one-line caveat that "
        "these are model-derived suggestions to be validated by the coordinator. "
        "Do not echo the JSON back."
    ),
    "Português": (
        "Você é um assistente que explica resultados de auditoria JÁ CALCULADOS por "
        "um modelo de aprendizado de máquina para um coordenador de curso de "
        "medicina. Você receberá um objeto JSON cujos números já foram calculados "
        "pelo sistema. Regras: (1) Use APENAS os números e rótulos presentes no "
        "input. NUNCA invente valores, percentuais, nomes de instituições ou fatos "
        "que não estejam ali. (2) Se um valor não existir, não o mencione. (3) Seja "
        "conciso e concreto. (4) Escreva em português. (5) Estruture a resposta "
        "como: um parágrafo curto resumindo a situação; depois 2-3 itens de ação "
        "derivados estritamente dos itens cujo sentimento é 'friction'; e por fim "
        "uma linha de ressalva de que são sugestões derivadas do modelo, a serem "
        "validadas pelo coordenador. Não repita o JSON na saída."
    ),
}


@dataclass
class AuditContext:
    """All the pre-computed numbers the explanation may use - nothing else."""
    course_code: int
    avg_score: float
    national_median: float
    predicted_tier: str
    contributions: List[Dict[str, Any]]   # {feature,label,value_pct,impact,sentiment}
    language: str = "English"
    confidence_pct: Optional[float] = None  # plug predict_proba here if available

    def to_payload(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("language", None)
        d["delta_vs_median"] = round(self.avg_score - self.national_median, 2)
        return d


def _get_api_key() -> Optional[str]:
    if st is not None:
        try:
            key = st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
            if key:
                return key
        except Exception:
            pass
    return os.environ.get("GEMINI_API_KEY")


def _maybe_cache(func):
    """Use Streamlit's cache when available so identical inputs reuse the call."""
    if st is not None and hasattr(st, "cache_data"):
        return st.cache_data(show_spinner=False)(func)
    return func


@_maybe_cache
def _call_gemini(payload_json: str, language: str, model: str) -> Optional[str]:
    api_key = _get_api_key()
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None
    try:
        client = genai.Client(api_key=api_key)
        sys_inst = SYSTEM_INSTRUCTION.get(language, SYSTEM_INSTRUCTION["English"])
        prompt = (
            "Here are the pre-computed audit numbers as JSON. "
            "Explain them following the rules.\n\n" + payload_json
        )
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=0.2,
                max_output_tokens=600,
            ),
        )
        text = (getattr(resp, "text", "") or "").strip()
        return text or None
    except Exception:
        return None


def explain(ctx: AuditContext, model: str = DEFAULT_MODEL) -> Optional[str]:
    """Return a grounded explanation string, or None to signal rule-based fallback."""
    payload_json = json.dumps(ctx.to_payload(), ensure_ascii=False, sort_keys=True)
    return _call_gemini(payload_json, ctx.language, model)
