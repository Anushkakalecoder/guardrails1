# Clinical Trial Guardrails

This repository contains standalone guardrails developed for a Clinical Trial AI Assistant using the OpenAI Agents SDK.

## Implemented Guardrails

### 1. Jailbreak / Prompt Injection Guardrail
Detects prompt injection, jailbreak attempts, instruction override, system prompt extraction, role-play attacks, and similar adversarial inputs.

**Implementation:** `jailbreaking.py`

---

### 2. Medical Advice Guardrail
Detects requests for personalized medical advice while allowing legitimate clinical document retrieval and general medical education queries.

**Implementation:** `medi_advice.py`

---

## Repository Structure

```
OPENAISDKGUARDRAILS/
│
├── datasets/
│   ├── jb_testing_eg.csv
│   ├── medi_advice_eg.csv
│   
│
├── jailbreaking.py
├── medi_advice.py
│
├── test_jailbreaking.py
├── test_mediadv.py
│
├── evals_jb.py
├── medi_eval.py
│
├── requirements.txt
├── README.md
└── .gitignore
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

The script prompts for a user query and runs it through the Medical Advice Guardrail.

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

---

## Notes

- `jailbreaking.py` and `medi_advice.py` contain the core guardrail implementations.
- `test_*.py` scripts are intended for interactive manual testing.
- `eval*.py` scripts are intended for batch evaluation using CSV datasets and automatically compute evaluation metrics.