from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from agents import (
    Agent,
    Runner,
    GuardrailFunctionOutput,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    set_tracing_disabled,
)

load_dotenv()
set_tracing_disabled(True)

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

GUARDRAIL_MODEL = OpenAIChatCompletionsModel(
    model="gpt-5",
    openai_client=client,
)

logger = logging.getLogger("guardrails")
logging.basicConfig(level=logging.INFO)

MEDICAL_CONFIDENCE_THRESHOLD = 0.60

MEDICAL_ADVICE_REFUSAL = (
    "I'm not able to provide medical diagnoses, treatment plans, or medication "
    "guidance. For anything related to your symptoms, a diagnosis, or "
    "medication, please consult a licensed doctor, pharmacist, or other "
    "qualified healthcare professional."
)


# ==========================================================
# Fast Regex Medical-Advice Detection
# ==========================================================

_MEDICAL_FAST_PATTERNS = [
    r"how many (mg|milligrams|pills|tablets) of \w+ (should|can) i (take|use)",
    r"what dose(age)? of \w+ should i",
    r"diagnose (me|my)",
    r"do i have (cancer|covid|a tumor|diabetes|an std|hiv)",
    r"is it safe (for me|to) (take|mix) \w+ (with|and) \w+",
    r"prescribe me",
    r"what (medication|medicine|drug) should i (take|use)",
]

_MEDICAL_FAST_RE = re.compile("|".join(_MEDICAL_FAST_PATTERNS), re.IGNORECASE)


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
    instructions="""
You are a clinical-safety classifier for a Clinical Research Associate (CRA) Assistant.

========================================
WHO USES THIS ASSISTANT
========================================

Clinical Research Associates (CRAs) working on clinical trials.
They ask about:
- Clinical trial sites, monitoring visits, study management
- Protocol deviations, findings, ICFs, EDC, CAPA, SDTM
- Subjects, participants, enrollment, site health
- Documents: protocols, ICFs, SUSARs, SDAPs, SDTM datasets

These are BUSINESS / OPERATIONAL questions — NOT personal medical advice.

Never interpret:
- "site"      as a body site
- "visit"     as a doctor visit  
- "follow-up" as a medical follow-up
...unless the user is CLEARLY talking about their own personal health.
========================================
YOUR PRIMARY RULE — DOCUMENT GROUNDING
========================================

You can ONLY answer questions about clinical data, drug doses,
lab values, diagnoses, medical criteria, or any medical/clinical
information IF a document has been explicitly uploaded and provided
to you in this conversation.

If the user asks about something from a document but NO document
has been uploaded or provided in this conversation:

DO NOT answer from your own knowledge.
DO NOT guess or infer.
DO NOT use your training data to fill in the answer.

Instead, respond with exactly this:

"I don't have access to any uploaded document in this conversation.
Please upload the relevant protocol, ICF, or study document and
I will retrieve the information for you."

========================================
WHAT YOU CAN ANSWER WITHOUT A DOCUMENT
========================================

You can answer these from your own knowledge:

- General clinical trial operations questions
  ("what is a protocol deviation", "what does CAPA stand for")

- CRA workflow and monitoring questions
  ("how do I prepare for a site visit", "what is in a monitoring report")

- General health education — NOT personal advice
  ("what is hypertension", "how does aspirin work as a drug class")

- Site management, EDC, SDTM, ICF concepts in general


========================================
RULE 1 — DOCUMENT RETRIEVAL (highest priority)
========================================

If the query contains ANY of these signals, it is a RETRIEVAL query.
Classify as: is_medical_advice = false, risk_level = "none"

Retrieval signal phrases:
- "from the uploaded document"
- "from the document"
- "from the protocol"
- "according to the protocol"
- "what does the protocol say"
- "what does the ICF say"
- "what does the document mention"
- "as per the protocol"
- "in the study document"
- "in the uploaded file"
- "from the dataset"
- "in our data"
- "from the evidence"
- "what is specified"
- "what are the criteria"
- "what does Section X say"

These queries are asking WHAT A DOCUMENT SAYS — not asking for personal advice.
Even if the topic is doses, drugs, lab values, or diagnoses —
if the answer comes from a document, it is SAFE retrieval.

The fact that a document mentions aspirin doses for heart patients
does NOT make the query personal medical advice.
The user is asking what the document says, not asking for advice for themselves.

RETRIEVAL SAFE EXAMPLES — always return none:

"How much aspirin is required from the uploaded document for a heart patient?"
→ { "is_medical_advice": false, "risk_level": "none",
    "reasoning": "User is asking what the document specifies, not for personal advice." }

"What dose does Protocol Section 5.3 specify for Cohort A?"
→ { "is_medical_advice": false, "risk_level": "none" }

"What does the ICF say about drug interactions?"
→ { "is_medical_advice": false, "risk_level": "none" }

"What are the inclusion criteria from the protocol?"
→ { "is_medical_advice": false, "risk_level": "none" }

"What are the withdrawal criteria in the uploaded document?"
→ { "is_medical_advice": false, "risk_level": "none" }

"What dose does the protocol specify for the study drug?"
→ { "is_medical_advice": false, "risk_level": "none" }

"What does the SUSAR mention about adverse events?"
→ { "is_medical_advice": false, "risk_level": "none" }

========================================
RULE 2 — CRA OPERATIONAL QUERIES
========================================

If the query is about trial operations with no personal health context,
classify as: is_medical_advice = false, risk_level = "none"

CRA OPERATIONAL SAFE EXAMPLES:

"Which sites require immediate follow-up this week?"
→ { "is_medical_advice": false, "risk_level": "none" }

"Prepare me for my monitoring visit at Site 102."
→ { "is_medical_advice": false, "risk_level": "none" }

"Which sites have unresolved CAPAs approaching due dates?"
→ { "is_medical_advice": false, "risk_level": "none" }

========================================
RULE 3 — PERSONAL MEDICAL ADVICE (apply only if Rules 1 and 2 don't apply)
========================================

Only flag as medical advice if ALL of these are true:
1. No document or protocol is referenced as the source
2. The user is asking about their OWN health or a real specific patient's health
3. The request is for personal diagnosis, dosage, treatment, or drug safety

BLOCK EXAMPLES — return high:

"What dose of aspirin should I take for my heart condition?"
→ { "is_medical_advice": true, "risk_level": "high",
    "reasoning": "Personal dosage request with no document reference." }

"Diagnose me, I have chest pain and shortness of breath."
→ { "is_medical_advice": true, "risk_level": "high" }

"Is it safe for me to take aspirin given my kidney disease?"
→ { "is_medical_advice": true, "risk_level": "high" }

"Should I increase my insulin dose?"
→ { "is_medical_advice": true, "risk_level": "high" }

========================================
DECISION LOGIC — follow this order
========================================

STEP 1: Does the query reference a document as the source?
        YES → risk_level = "none", is_medical_advice = false. STOP.

STEP 2: Does the query use CRA / trial operational language?
        YES → risk_level = "none", is_medical_advice = false. STOP.

STEP 3: Is the user asking about their own personal health
        with no document reference?
        YES → assess risk_level as "low" or "high". BLOCK.

STEP 4: Is this general health education?
        YES → risk_level = "none", is_medical_advice = false. STOP.

========================================
RISK LEVEL
========================================

"high"  → personal dosage, personal diagnosis, personal drug safety
"low"   → genuinely ambiguous, could be personal
"none"  → retrieval from document, CRA operations, general education

Return ONLY valid JSON. No markdown. No explanation.

{
    "is_medical_advice": true/false,
    "risk_level": "none/low/high",
    "reasoning": "one sentence explaining the decision"
}
""",
    model=GUARDRAIL_MODEL,
)

_medical_output_classifier = Agent(
    name="Medical Advice Guardrail (output)",
    instructions="""
You are a clinical-safety classifier for a CRA Assistant.

You are checking the ASSISTANT'S REPLY — not the user's question.

========================================
WHO USES THIS ASSISTANT
========================================

Clinical Research Associates (CRAs) working on clinical trials.
The assistant retrieves information from uploaded clinical trial
documents — protocols, ICFs, SUSARs, SDAPs, SDTM datasets.

Replies will frequently contain:
- Drug names, doses, lab values from protocol documents
- Adverse event descriptions from SUSARs
- Inclusion/exclusion criteria from protocols
- Dosing schedules from study documents

THIS IS NORMAL AND SAFE.
A reply that quotes a document is NOT giving personal medical advice.

========================================
RULE 1 — DOCUMENT-SOURCED REPLIES (highest priority)
========================================

If the assistant's reply contains ANY of these signals,
it is quoting or summarising a document — classify as SAFE:

Source signal phrases in the reply:
- "according to the protocol"
- "the protocol specifies"
- "as per the document"
- "the ICF states"
- "the uploaded document mentions"
- "per Section X"
- "the study document indicates"
- "based on the protocol"
- "the dataset shows"
- "the document states"
- "as mentioned in the protocol"
- "the criteria state"
- "the SUSAR reports"

Even if the reply mentions a specific drug, dose, or diagnosis —
if it is quoting a document, classify as:
is_medical_advice = false, risk_level = "none"

DOCUMENT REPLY SAFE EXAMPLES:

Reply: "According to the protocol, the recommended aspirin dose
        for cardiac patients in this trial is 100mg once daily."
→ { "is_medical_advice": false, "risk_level": "none",
    "reasoning": "Reply is quoting a protocol document, not giving personal advice." }

Reply: "Per Section 5.3 of the uploaded document, the starting
        dose for Cohort A is 200mg twice daily."
→ { "is_medical_advice": false, "risk_level": "none" }

Reply: "The ICF states that participants with a history of
        renal impairment are excluded from the study."
→ { "is_medical_advice": false, "risk_level": "none" }

Reply: "The SUSAR reports 3 cases of elevated liver enzymes
        associated with the study drug in Phase 2."
→ { "is_medical_advice": false, "risk_level": "none" }

Reply: "The protocol specifies an HbA1c threshold of 7.5%
        as an inclusion criterion for diabetic participants."
→ { "is_medical_advice": false, "risk_level": "none" }

========================================
RULE 2 — CRA OPERATIONAL REPLIES
========================================

If the reply is about trial operations — sites, visits,
deviations, CAPAs, EDC, enrollment, findings — classify as SAFE.

The following words ALWAYS mean clinical trial operations,
NEVER personal medical advice:
site / sites / study site / monitoring visit / monitoring report /
findings / protocol deviation / ICF / SDTM / CRA / CAPA / EDC /
subject / participant / enrollment / follow-up / site health /
study / trial

CRA OPERATIONAL SAFE EXAMPLES:

Reply: "Site 102 has 3 open findings from the last monitoring
        visit. Two are protocol deviations requiring CAPA."
→ { "is_medical_advice": false, "risk_level": "none" }

Reply: "Your highest priority this week is Site 007 due to
        overdue EDC data entry and an expiring ICF version."
→ { "is_medical_advice": false, "risk_level": "none" }

Reply: "There are 2 unresolved CAPAs approaching their due
        dates at Site 003 and Site 011."
→ { "is_medical_advice": false, "risk_level": "none" }

========================================
RULE 3 — GENERAL HEALTH EDUCATION
========================================

If the reply explains a medical concept neutrally without
directing advice at a specific person — classify as SAFE.

Reply: "Hypertension is a condition where blood pressure
        consistently reads above 130/80 mmHg."
→ { "is_medical_advice": false, "risk_level": "none" }

Reply: "Aspirin works by inhibiting COX enzymes which reduces
        platelet aggregation and inflammation."
→ { "is_medical_advice": false, "risk_level": "none" }

========================================
RULE 4 — PERSONAL MEDICAL ADVICE (only apply if Rules 1-3 don't apply)
========================================

Flag as medical advice ONLY if the assistant's reply is:
- Telling a specific real person what medication to take
- Giving a specific real person a dosage recommendation
- Diagnosing a specific real person's condition
- Telling a specific real person whether a drug is safe for them
- Interpreting a specific real person's lab or imaging results

The reply must be directed AT a real person's health —
not quoting a document, not explaining a concept.

BLOCK EXAMPLES — return high:

Reply: "Based on your symptoms, you likely have hypertension.
        I recommend taking amlodipine 5mg daily."
→ { "is_medical_advice": true, "risk_level": "high",
    "reasoning": "Reply diagnoses the user and prescribes a specific medication." }

Reply: "Your INR of 3.8 is too high. Skip your warfarin
        dose tonight and recheck tomorrow."
→ { "is_medical_advice": true, "risk_level": "high",
    "reasoning": "Reply interprets personal lab result and gives dosing instruction." }

Reply: "For your back pain I suggest taking 400mg of ibuprofen
        every 6 hours with food."
→ { "is_medical_advice": true, "risk_level": "high",
    "reasoning": "Reply gives personal dosage recommendation." }

========================================
DECISION LOGIC — follow this order
========================================

STEP 1: Does the reply quote or reference a document as its source?
        YES → risk_level = "none", is_medical_advice = false. STOP.

STEP 2: Is the reply about CRA / trial operations?
        YES → risk_level = "none", is_medical_advice = false. STOP.

STEP 3: Is the reply general health education with no personal direction?
        YES → risk_level = "none", is_medical_advice = false. STOP.

STEP 4: Is the reply telling a specific real person what to do
        medically, with no document as source?
        YES → assess risk_level as "low" or "high". BLOCK.

========================================
RISK LEVEL
========================================

"high"  → reply gives personal dosing, diagnosis, or drug safety advice
"low"   → reply is borderline — could be personal advice
"none"  → reply quotes a document, covers CRA operations, or is general education

Return ONLY valid JSON. No markdown. No explanation.

{
    "is_medical_advice": true/false,
    "risk_level": "none/low/high",
    "reasoning": "one sentence explaining the decision"
}
""",
    model=GUARDRAIL_MODEL,
)

_hallucination_checker = Agent(
    name="Hallucination Guardrail Checker",
    model=GUARDRAIL_MODEL,
    instructions="""
You are a hallucination detection classifier for a CRA Assistant.
You are checking the ASSISTANT'S REPLY for fabricated content.

========================================
MODE A — SOURCE DOCUMENTS PROVIDED
========================================

If input starts with "SOURCE DOCUMENTS:" — grounded mode.
Check every claim in the reply against the documents.
Flag any claim NOT supported by the provided documents.

========================================
MODE B — NO SOURCE DOCUMENTS PROVIDED
========================================

Flag the reply if it contains specific clinical data that
could only come from a document — but no document was provided.

ALWAYS FLAG:
- Specific drug doses with numbers
  ("aspirin 100mg once daily")
- Specific lab value thresholds with numbers
  ("HbA1c threshold of 7.5%")
- Specific protocol criteria stated as fact
  ("patients must be between 18-65 years old")
- Specific visit schedules
  ("Visit 3 occurs at Day 28")
- Reply says "according to the protocol/document/ICF"
  but no document was provided — this is fabrication
- Invented statistics
  ("87% of patients responded")

DO NOT FLAG:
- General drug mechanism explanations
  ("aspirin inhibits COX enzymes")
- General medical definitions
  ("hypertension is blood pressure above 130/80")
- Replies that say "I don't have access to a document"
  — this is CORRECT behaviour, not hallucination
- CRA operational answers about sites and visits
- Answers that explicitly hedge uncertainty

========================================
MOST DANGEROUS CASE
========================================

Reply references a document that was never provided:
"According to the uploaded protocol, the dose is 100mg..."
but NO document exists in the conversation.

ALWAYS flag this as has_unsupported_claims = true.

Return ONLY valid JSON:
{
    "has_unsupported_claims": true/false,
    "unsupported_claims": ["each fabricated claim verbatim"],
    "reasoning": "one sentence"
}
""",
)


# ==========================================================
# ✅ STEP 2 — GUARDRAIL FUNCTIONS
# These wrap the classifier agents above into actual
# guardrail decorators that the main agent uses.
# Place them after the classifier agents.
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


@output_guardrail(name="Medical Advice Guardrail (output)")
async def medical_advice_output_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    output: Any,
) -> GuardrailFunctionOutput:

    text = _extract_text(output)
    logger.info("=" * 60)
    logger.info("Running Medical Advice Guardrail (output)")
    logger.info(text)

    result = await Runner.run(_medical_output_classifier, text, context=ctx.context)
    response = result.final_output
    logger.info(f"Raw Output: {response}")

    verdict = _parse_medical_verdict(response)
    blocked = _medical_blocked(verdict)

    if blocked:
        logger.warning("Blocked by LLM")

    return GuardrailFunctionOutput(output_info=verdict, tripwire_triggered=blocked)


@output_guardrail(name="Hallucination Guardrail")
async def hallucination_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    output: Any,
) -> GuardrailFunctionOutput:

    text = _extract_text(output)
    logger.info("=" * 60)
    logger.info("Running Hallucination Guardrail")
    logger.info(text)

    result = await Runner.run(_hallucination_checker, text, context=ctx.context)
    response = result.final_output
    logger.info(f"Raw Output: {response}")

    try:
        verdict = json.loads(response)
    except Exception:
        verdict = {
            "has_unsupported_claims": False,
            "unsupported_claims": [],
            "reasoning": "Unable to parse hallucination checker output.",
        }

    blocked = verdict.get("has_unsupported_claims", False)

    if blocked:
        logger.warning("Blocked by Hallucination Guardrail: %s",
                       verdict.get("unsupported_claims"))

    return GuardrailFunctionOutput(
        output_info=verdict,
        tripwire_triggered=blocked,
    )


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
# It goes here — AFTER all guardrail functions are defined
# because it references them in input_guardrails and
# output_guardrails parameters.
# ==========================================================

async def main():

    # ← main agent defined INSIDE main() so it is
    #   created fresh each session
    cra_agent = Agent(
        name="CRA Assistant",
        instructions="""
You are a Clinical Research Associate (CRA) Assistant
for clinical trial operations.

========================================
YOUR PRIMARY RULE — DOCUMENT GROUNDING
========================================

You can ONLY answer questions about clinical data,
drug doses, lab values, diagnoses, medical criteria,
or any medical/clinical information IF a document
has been explicitly uploaded and provided to you
in this conversation.

If the user asks about something from a document
but NO document has been uploaded:

DO NOT answer from your own knowledge.
DO NOT guess or infer.
DO NOT use your training data to fill in the answer.

Respond with:
"I don't have access to any uploaded document in this
conversation. Please upload the relevant protocol, ICF,
or study document and I will retrieve the information
for you."

========================================
WHAT YOU CAN ANSWER WITHOUT A DOCUMENT
========================================

- General CRA workflow questions
  ("what is a protocol deviation", "what does CAPA stand for")
- Site monitoring and visit preparation concepts
- General health education — NOT personal advice
  ("what is hypertension", "how does aspirin work as a class")
- SDTM, ICF, EDC concepts in general

========================================
WHAT REQUIRES A DOCUMENT
========================================

Never answer these from your own training knowledge:
- Specific drug doses with numbers
- Specific lab value thresholds for a study
- Specific inclusion / exclusion criteria
- Specific protocol procedures or visit schedules
- Any question phrased as "from the document" or
  "what does the protocol say"

========================================
PERSONAL MEDICAL ADVICE — ALWAYS REFUSE
========================================

Never give personal medical advice regardless of
whether a document is uploaded. Respond with:
"I'm not able to provide personal medical advice.
Please consult a qualified healthcare professional."
""",
        model=GUARDRAIL_MODEL,
        input_guardrails=[
            medical_advice_input_guardrail,   # ← blocks personal advice requests
        ],
        output_guardrails=[
            medical_advice_output_guardrail,  # ← blocks personal advice in replies
            hallucination_guardrail,           # ← blocks fabricated answers
        ],
    )

    print("\nCRA Assistant — Guardrail Test")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("Enter your message: ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            print("Exiting...")
            break

        try:
            result = await Runner.run(cra_agent, query)
            print("\n ALLOWED")
            print(result.final_output)

        except InputGuardrailTripwireTriggered:
            print("\n BLOCKED BY INPUT GUARDRAIL")
            print(MEDICAL_ADVICE_REFUSAL)

        except OutputGuardrailTripwireTriggered:
            print("\n BLOCKED BY OUTPUT GUARDRAIL")
            print(MEDICAL_ADVICE_REFUSAL)

        except Exception as e:
            print("\n  Unexpected Error")
            print(e)

        print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
