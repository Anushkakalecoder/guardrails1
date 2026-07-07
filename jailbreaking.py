from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import yaml
from pydantic import BaseModel, Field

from openai import AsyncOpenAI
from dotenv import load_dotenv
import json


load_dotenv()

from agents import (
    Agent,
    Runner,
    GuardrailFunctionOutput,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
    set_tracing_disabled,
)
set_tracing_disabled(True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

GUARDRAIL_MODEL = "gpt-5"
logger = logging.getLogger("guardrails")
logging.basicConfig(level=logging.INFO)


# ==========================================================
# Load system prompt(s), regex patterns, and thresholds from
# YAML instead of hardcoding them
# ==========================================================

PROMPTS_PATH = os.getenv("GUARDRAIL_PROMPTS_PATH", "guardrail_prompts.yaml")


def _load_config(path: str = PROMPTS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CONFIG = _load_config()


def _load_prompt(key: str) -> str:
    """Load a named prompt from guardrail_prompts.yaml.

    Expected file shape:
        guardrails:
          jailbreak_classifier: |-
            <instructions text>
    """
    prompt = (_CONFIG.get("guardrails") or {}).get(key)
    if not prompt:
        raise ValueError(
            f"Prompt '{key}' not found under 'guardrails:' in {PROMPTS_PATH}"
        )
    return prompt


def _compile_pattern_list(section: str, key: str) -> re.Pattern:
    """Load a list of regex strings from config, validate each one compiles
    individually (so one bad pattern gives a clear error, not a silent
    combined-regex failure), and return a single compiled OR'd pattern."""
    patterns = (_CONFIG.get(section) or {}).get(key)
    if not patterns:
        raise ValueError(f"No patterns found under '{section}.{key}' in {PROMPTS_PATH}")

    for p in patterns:
        try:
            re.compile(p)
        except re.error as exc:
            raise ValueError(f"Invalid regex in {PROMPTS_PATH} [{section}.{key}]: {p!r} ({exc})")

    return re.compile("|".join(patterns), re.IGNORECASE)


JAILBREAK_CLASSIFIER_PROMPT = _load_prompt("jailbreak_classifier")

JAILBREAK_CONFIDENCE_THRESHOLD = (
    (_CONFIG.get("thresholds") or {}).get("jailbreak_confidence_threshold", 0.60)
)


@dataclass
class GuardrailContext:

    source_documents: list[str] = field(
        default_factory=list
    )


# ==========================================================
# Helper Function
# ==========================================================

def _extract_text(value: Any) -> str:
    """
    Convert different input formats into plain text.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, BaseModel):
        for field_name in (
            "response",
            "output",
            "text",
            "answer",
        ):
            if hasattr(value, field_name):
                return str(getattr(value, field_name))

        return value.model_dump_json()

    return str(value)


# ==========================================================
# Manual JSON Parsing (handles markdown fences, stray text,
# single quotes, trailing commas, etc. before giving up)
# ==========================================================

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([\}\]])")


def _parse_json_output(raw: str) -> dict:
    """
    Manually parse the classifier's JSON output, tolerating common
    formatting issues instead of failing on the first json.loads attempt.
    Only accepts a parsed dict that actually has is_jailbreak/confidence/
    reasoning; otherwise raises so the caller falls back to a safe default.
    """

    if not raw or not raw.strip():
        raise ValueError("Empty classifier output")

    candidates: list[str] = [raw.strip()]

    fence_match = _JSON_FENCE_RE.search(raw)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    obj_match = _JSON_OBJECT_RE.search(raw)
    if obj_match:
        block = obj_match.group(0).strip()
        candidates.append(block)
        candidates.append(_TRAILING_COMMA_RE.sub(r"\1", block))
        candidates.append(
            _TRAILING_COMMA_RE.sub(r"\1", block).replace("'", '"')
        )

    required_keys = {"is_jailbreak", "confidence", "reasoning"}
    last_error: Exception | None = None

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

        if not isinstance(parsed, dict):
            last_error = ValueError(
                f"Parsed JSON was not an object: {type(parsed).__name__}"
            )
            continue

        missing = required_keys - parsed.keys()
        if missing:
            last_error = ValueError(f"Parsed JSON missing keys: {missing}")
            continue

        return parsed

    raise ValueError(f"Could not parse classifier output as JSON: {last_error}")


# ==========================================================
# Fast Regex Jailbreak Detection (patterns loaded from YAML)
# ==========================================================

_JAILBREAK_RE = _compile_pattern_list("patterns", "jailbreak_regex")


#------------------------------------------------------------
#another class to hold the output of the jailbreak check
#----------------------------------------------------------------
class JailbreakCheckOutput(BaseModel):

    is_jailbreak: bool

    confidence: float = Field(
        ge=0,
        le=1,
    )

    reasoning: str


# =====================================================
# LLM Classifier
# =====================================================

_jailbreak_classifier = Agent(

    name="Jailbreak Guardrail",

    instructions=JAILBREAK_CLASSIFIER_PROMPT,

    model=GUARDRAIL_MODEL,

)
@input_guardrail(name="Jailbreak Guardrail")
async def jailbreak_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:

    text = _extract_text(input)

    logger.info("=" * 60)
    logger.info("Running Jailbreak Guardrail")
    logger.info(text)

    # ----------------------------
    # Regex Check
    # ----------------------------

    if _JAILBREAK_RE.search(text):
        logger.warning("Blocked by Regex")

        return GuardrailFunctionOutput(
            output_info={
                "method": "regex",
                "matched": True,
            },
            tripwire_triggered=True,
        )

    logger.info("Regex Passed")

    # ----------------------------
    # LLM Check
    # ----------------------------

    result = await Runner.run(
        _jailbreak_classifier,
        text,
        context=ctx.context,
    )

    response = result.final_output

    logger.info(f"Raw Output: {response}")

    try:
        verdict = _parse_json_output(response)

    except Exception as exc:
        logger.warning(f"Manual JSON parsing failed: {exc}")
        verdict = {
            "is_jailbreak": False,
            "confidence": 0.0,
            "reasoning": "Unable to parse classifier output.",
        }

    blocked = (
        verdict.get("is_jailbreak", False)
        and verdict.get("confidence", 0.0) >= JAILBREAK_CONFIDENCE_THRESHOLD
    )

    if blocked:
        logger.warning("Blocked by LLM")

    return GuardrailFunctionOutput(
    output_info={
        "method": "LLM",
        "is_jailbreak": verdict.get("is_jailbreak", False),
        "confidence": verdict.get("confidence", 0.0),
        "reasoning": verdict.get("reasoning", ""),
    },
    tripwire_triggered=blocked,
)