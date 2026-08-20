from __future__ import annotations

import json
import os
from typing import Literal, Optional, List, Dict

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import streamlit as st

from .core import Record

MODEL = os.getenv("LEXICORE_LLM_MODEL", "gemini-3.6-flash")

class Citation(BaseModel):
    evidence_id: str
    claim: str

class Answer(BaseModel):
    answer: str
    reasoning: str = ""
    limitations: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

class Weakness(BaseModel):
    weakest_points: list[str] = Field(default_factory=list)
    defense_strategy: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


@st.cache_resource
def get_available_keys() -> list[str]:
    keys = []
    # Collect GOOGLE_API_KEY_1 through GOOGLE_API_KEY_N
    i = 1
    while True:
        k = st.secrets.get(f"GOOGLE_API_KEY_{i}") or os.getenv(f"GOOGLE_API_KEY_{i}")
        if not k:
            break
        keys.append(k)
        i += 1
        
    # Fallback to standard GOOGLE_API_KEY if no numbered keys are found
    if not keys:
        single_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if single_key:
            keys.append(single_key)
            
    if not keys:
        raise RuntimeError("No Google API keys found in Streamlit secrets or environment variables.")
    return keys


def get_next_client() -> genai.Client:
    keys = get_available_keys()
    if "key_index" not in st.session_state:
        st.session_state.key_index = 0
    
    current_idx = st.session_state.key_index % len(keys)
    key = keys[current_idx]
    return genai.Client(api_key=key)


def rotate_key_on_error():
    keys = get_available_keys()
    if len(keys) > 1:
        st.session_state.key_index = (st.session_state.key_index + 1) % len(keys)
        next_num = (st.session_state.key_index % len(keys)) + 1
        st.warning(f"⚠️ API quota limit hit on current key. Automatically switching to API Key #{next_num}...")


def client() -> genai.Client:
    return get_next_client()


def context(records: list[Record], max_chars: int = 30000):
    parts = []
    used = []
    total = 0
    for i, r in enumerate(records, 1):
        block = r.evidence_block(i)
        if total + len(block) > max_chars:
            break
        parts.append(block)
        used.append(r)
        total += len(block)
    return "\n\n".join(parts), used


def instructions(stance: str) -> str:
    base = f"""You are LexiCore, an evidence-grounded theological research, apologetics, and cross-examination assistant. 
Current Research Mode / Stance: {stance}.

CORE INSTRUCTIONS:
- Use the supplied evidence segments to answer the question, counter-questions, or follow-ups thoroughly and structurally.
- RELEVANCY RULE: Strictly filter and match the evidence to the tradition of the question. If the user asks about Christian doctrine, Jesus's divinity, or Christian practices, draw *only* from Christian sources (Scripture, Creeds, Patristic) and completely ignore Islamic or Jewish sources unless explicitly asked for a comparative analysis.
- If the user asks about Islam, draw appropriately on the Quran, hadith, or Islamic history provided.
- Never invent quotations, verses, hadith numbers, or citations. Cite claims using the provided evidence IDs.
"""

    if "Didactic" in stance:
        return base + """
MODE GUIDELINE (Didactic/Explanatory):
- Provide a rich, clear, and vivid explanation unpacking the theological, historical, or textual concepts.
- Guide the user smoothly through how the evidence illuminates the topic, breaking down scriptural references thoroughly.
"""
    elif "Scholarly" in stance:
        return base + """
MODE GUIDELINE (Scholarly / Debate):
- Engage in a rigorous, deep academic debate or cross-examination style, directly handling counter-questions, pushback, and cross-examination.
- Critically evaluate the texts, weigh varying perspectives or tensions across traditions, and present a robust, high-level analysis.
"""
    elif "Skeptical" in stance:
        return base + """
MODE GUIDELINE (Skeptical / Contrarian):
- Adopt a critical, probing, and skeptical lens toward traditional harmonizations or claims.
- Sharply highlight textual discrepancies, historical ambiguities, or potential counter-arguments present in or missing from the sources.
"""
    return base


def _generate(prompt: str, schema, stance: str, temperature: float = 0.2):
    keys = get_available_keys()
    attempts = len(keys)
    
    for attempt in range(attempts):
        try:
            ai_client = get_next_client()
            r = ai_client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=instructions(stance),
                    temperature=temperature,
                    max_output_tokens=4000,  # Increased token ceiling to prevent cuts
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            if getattr(r, "parsed", None) is not None:
                return r.parsed
                
            text = getattr(r, "text", "") or ""
            
            # Strip markdown wrappers if present
            clean_text = text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            return schema.model_validate(json.loads(clean_text))
            
        except Exception as e:
            # If quota or rate limit error occurs, rotate keys and try the next one
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                rotate_key_on_error()
                if attempt == attempts - 1:
                    raise e
            else:
                raise e


def answer(query: str, records: list[Record], stance: str = "Scholarly (Debate)", temperature: float = 0.2, history: Optional[List[Dict[str, str]]] = None):
    ctx, used = context(records)
    
    # Format prior conversation history to support counter-questions and conversational context
    history_text = ""
    if history:
        history_text = "CONVERSATION HISTORY / PREVIOUS TURNS:\n"
        for turn in history:
            history_text += f"User: {turn['query']}\nLexiCore: {turn['answer']}\n\n"

    prompt = f"""{history_text}CURRENT QUERY / COUNTER-QUESTION:\n{query}\n\nEVIDENCE:\n{ctx}\n\nWrite a comprehensive answer matching the requested stance, thoroughly addressing any counter-questions or pushback using the evidence and citing support with evidence IDs."""

    result = _generate(prompt, Answer, stance, temperature)
    valid = {r.id for r in used}
    result.citations = [c for c in result.citations if c.evidence_id in valid]
    return result, used


def assess(argument: str, records: list[Record], stance: str = "Scholarly (Debate)", temperature: float = 0.1):
    ctx, used = context(records)
    result = _generate(
        f"ARGUMENT:\n{argument}\n\nEVIDENCE:\n{ctx}\n\nAct as an adversarial academic reviewer. Identify real logical or evidential weaknesses and propose defensible repairs based on the evidence.",
        Weakness,
        stance,
        temperature,
    )
    return result, used