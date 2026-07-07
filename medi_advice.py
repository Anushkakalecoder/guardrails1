from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    GuardrailFunctionOutput,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
    set_tracing_disabled,
    InputGuardrailTripwireTriggered,
)

load_dotenv()
set_tracing_disabled(True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

GUARDRAIL_MODEL = "gpt-5"


logger = logging.getLogger("guardrails")
logging.basicConfig(level=logging.INFO)

MEDICAL_ADVICE_REFUSAL = (
    "I'm not able to provide medical diagnoses, treatment plans, or medication "
    "guidance. For anything related to your symptoms, a diagnosis, or "
    "medication, please consult a licensed doctor, pharmacist, or other "
    "qualified healthcare professional."
)


# ==========================================================
# Load system prompt(s) from YAML instead of hardcoding them
# ==========================================================

PROMPTS_PATH = os.getenv("CRA_PROMPTS_PATH", "medi_prompts.yaml")


def _load_prompts(path: str = PROMPTS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_PROMPTS = _load_prompts()


def _get_prompt(section: str, key: str) -> str:
    """Fetch prompts[section][key] from cra_prompts.yaml, e.g.
    _get_prompt('guardrails', 'medical_advice_input_classifier')."""
    prompt = (_PROMPTS.get(section) or {}).get(key)
    if not prompt:
        raise ValueError(
            f"Prompt '{key}' not found under '{section}:' in {PROMPTS_PATH}"
        )
    return prompt


MEDICAL_INPUT_CLASSIFIER_PROMPT = _get_prompt("guardrails", "medical_advice_input_classifier")
CRA_ASSISTANT_PROMPT = _get_prompt("agents", "cra_assistant")

MEDICAL_CONFIDENCE_THRESHOLD = (
    (_PROMPTS.get("thresholds") or {}).get("medical_confidence_threshold", 0.60)
)


def _compile_pattern_list(section: str, key: str) -> re.Pattern:
    """Load a list of regex strings from config, validate each one compiles
    individually (so one bad pattern gives a clear error, not a silent
    combined-regex failure), and return a single compiled OR'd pattern."""
    patterns = (_PROMPTS.get(section) or {}).get(key)
    if not patterns:
        raise ValueError(f"No patterns found under '{section}.{key}' in {PROMPTS_PATH}")

    for p in patterns:
        try:
            re.compile(p)
        except re.error as exc:
            raise ValueError(f"Invalid regex in {PROMPTS_PATH} [{section}.{key}]: {p!r} ({exc})")

    return re.compile("|".join(patterns), re.IGNORECASE)


_MEDICAL_FAST_RE = _compile_pattern_list("patterns", "medical_fast_regex")


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        for field_name in ("response", "output", "text", "answer"):
            if hasattr(value, field_name):
                return str(getattr(value, field_name))
        return value.model_dump_json()
    return str(value)


class MedicalAdviceCheckOutput(BaseModel):
    is_medical_advice: bool
    risk_level: str  # "none" | "low" | "high"
    reasoning: str


_medical_input_classifier = Agent(
    name="Medical Advice Guardrail (input)",
    instructions=MEDICAL_INPUT_CLASSIFIER_PROMPT,
    model=GUARDRAIL_MODEL,
)


# ==========================================================
# ✅ STEP 2 — GUARDRAIL FUNCTION
# Only the input guardrail remains — no output/hallucination
# guardrails, since we're only enforcing this on the way in.
# ==========================================================

@input_guardrail(name="Medical Advice Guardrail (input)")
async def medical_advice_input_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:

    text = _extract_text(input)
    logger.info("=" * 60)
    logger.info("Running Medical Advice Guardrail (input)")
    logger.info(text)

    if _MEDICAL_FAST_RE.search(text):
        logger.warning("Blocked by Regex")
        return GuardrailFunctionOutput(
            output_info={"method": "regex", "matched": True, "risk_level": "high"},
            tripwire_triggered=True,
        )

    logger.info("Regex Passed")

    result = await Runner.run(_medical_input_classifier, text, context=ctx.context)
    response = result.final_output
    logger.info(f"Raw Output: {response}")

    verdict = _parse_medical_verdict(response)
    blocked = _medical_blocked(verdict)

    if blocked:
        logger.warning("Blocked by LLM")

    return GuardrailFunctionOutput(output_info=verdict, tripwire_triggered=blocked)


# ==========================================================
# Verdict helpers
# ==========================================================

def _parse_medical_verdict(response: str) -> dict:
    try:
        verdict = json.loads(response)
    except Exception:
        verdict = {
            "is_medical_advice": False,
            "risk_level": "none",
            "reasoning": "Unable to parse classifier output.",
        }
    return verdict


def _medical_blocked(verdict: dict) -> bool:
    return (
        verdict.get("is_medical_advice", False)
        and verdict.get("risk_level") in ("low", "high")
    )


# ==========================================================
# ✅ STEP 3 — MAIN AGENT
# This is the actual CRA assistant the user talks to.
# It's defined at MODULE level (not inside main()) so other
# scripts can `from jb2 import cra_agent` and run it directly.
# It goes here — AFTER the guardrail function is defined
# because it references it in input_guardrails.
# ==========================================================

cra_agent = Agent(
    name="CRA Assistant",
    instructions=CRA_ASSISTANT_PROMPT,
    model=GUARDRAIL_MODEL,
    input_guardrails=[
        medical_advice_input_guardrail,   # ← blocks personal advice requests
    ],
)

