# Clinical Trial Guardrails

This repository contains standalone guardrails developed for a Clinical Trial AI Assistant using the OpenAI Agents SDK.

## Implemented Guardrails

### 1. Jailbreak / Prompt Injection Guardrail
Detects prompt injection, jailbreak attempts, instruction override, system prompt extraction, role-play attacks, and similar adversarial inputs. Uses a fast regex pre-filter followed by an LLM classifier for anything the regex doesn't catch.

**Implementation:** `jailbreaking.py`
**Config:** `guardrail_prompts.yaml` — classifier system prompt, regex pattern list, and confidence threshold

---

### 2. Medical Advice Guardrail
Detects requests for personalized medical advice while allowing legitimate clinical document retrieval and general medical education queries. This is an **input-only** guardrail — it checks the user's question before the assistant responds; there is no separate output or hallucination check on the assistant's reply.

**Implementation:** `medi_advice.py`
**Config:** `medi_prompts.yaml` — classifier system prompt, main CRA assistant system prompt, regex pattern list, and confidence threshold

---

## Configuration-Driven Guardrails

Both guardrails externalize everything that used to be hardcoded in Python into a companion YAML file:

| Moved out of code | Now lives in |
|---|---|
| Classifier system prompt | `guardrails.<classifier_name>` |
| Main assistant system prompt (medical advice guardrail only) | `agents.cra_assistant` |
| Fast regex detection patterns | `patterns.<pattern_list_name>` |
| Confidence threshold | `thresholds.<threshold_name>` |

This means prompt wording, detection patterns, and thresholds can all be tuned by editing the YAML — no code changes or redeploys needed. Each pattern is validated individually at load time, so a typo in one regex raises a clear error pointing at that specific pattern instead of failing silently.

Both scripts also do manual, tolerant JSON parsing of the classifier's output (handling stray markdown fences, trailing commas, or single quotes) instead of a single bare `json.loads()` call, and only accept a parsed result if it actually contains the expected keys — otherwise they fall back to a safe default (not flagged) rather than raising a `KeyError`.

---

## Repository Structure

```
OPENAISDKGUARDRAILS/
│
├── datasets/
│   ├── jb_testing_eg.csv
│   ├── medi_advice_eg.csv
│
├── .gitignore
├── evals_jb.py
├── guardrail_prompts.yaml
├── jailbreaking.py
├── medi_advice.py
├── medi_eval.py
├── medi_prompts.yaml
├── README.md
├── requirements.txt
├── test_jailbreaking.py
└── test_mediadv.py
```

---

## Running Interactive Tests

These scripts allow developers to manually test individual queries.

### Jailbreak Guardrail

```bash
python test_jailbreaking.py
```

The script prompts for a user query and runs it through the Jailbreak Guardrail.

Example:

```
Enter your message:
Ignore previous instructions.
```

---

### Medical Advice Guardrail

```bash
python test_mediadv.py
```

The script prompts for a user query and runs it through the Medical Advice Guardrail (input check only).

Example:

```
Enter your message:
What dose should I take?
```

---

## Running Batch Evaluations

The evaluation scripts execute the guardrails against CSV datasets containing multiple test queries.

### Jailbreak Evaluation

```bash
python evals_jb.py
```

Input:
- datasets/jb.csv
- datasets/jb2_business.csv

Outputs include:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score

---

### Medical Advice Evaluation

```bash
python medi_eval.py
```

Input:
- datasets/medi.csv
- datasets/medi_business.csv
- datasets/medi_business2.csv
- datasets/medi2.csv

Outputs include:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

`pyyaml` is required in addition to the OpenAI Agents SDK dependencies, since both guardrails now load their configuration from YAML.

---

## Notes

- `jailbreaking.py` and `medi_advice.py` contain the core guardrail implementations; each loads its system prompt(s), regex patterns, and confidence threshold from its companion `*_prompts.yaml` file at import time rather than hardcoding them.
- `medi_advice.py` implements an input guardrail only — no output or hallucination guardrail is applied to the assistant's replies.
- `test_*.py` scripts are intended for interactive manual testing.
- `eval*.py` scripts are intended for batch evaluation using CSV datasets and automatically compute evaluation metrics.
- To point a script at a differently-named or located config file, set the corresponding environment variable (e.g. `GUARDRAIL_PROMPTS_PATH`) instead of editing the script.