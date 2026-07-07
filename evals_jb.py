import asyncio
import pandas as pd

from agents import Agent, Runner
from agents.exceptions import InputGuardrailTripwireTriggered

from jb2 import (
    GUARDRAIL_MODEL,
    jailbreak_guardrail,
)

# --------------------------------------------------
# Test Agent
# --------------------------------------------------

agent = Agent(
    name="Jailbreak Evaluation Agent",
    instructions="You are a helpful assistant.",
    model=GUARDRAIL_MODEL,
    input_guardrails=[jailbreak_guardrail],
)


async def main():

    df = pd.read_csv("input.csv")

    # Create columns if they don't exist
    if "Actual" not in df.columns:
        df["Actual"] = ""

    if "Blocked_By" not in df.columns:
        df["Blocked_By"] = ""

    # Numeric columns
    for col in ["TP", "TN", "FP", "FN"]:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    TP = TN = FP = FN = 0

    for index, row in df.iterrows():

        query = row["Query"]
        expected = str(row["Expected"]).strip().upper()

        print(f"\nRunning Query {index+1}: {query}")

        actual = "SAFE"
        blocked_by = "None"

        try:

            await Runner.run(agent, query)
            await asyncio.sleep(6)

        except InputGuardrailTripwireTriggered:

            actual = "JAILBREAK"

            # Change this if you expose regex/LLM method
            blocked_by = "Regex / LLM"

        # ------------------------
        # Confusion Matrix
        # ------------------------

        tp = tn = fp = fn = 0

        if expected == "JAILBREAK" and actual == "JAILBREAK":
            tp = 1
            TP += 1

        elif expected == "SAFE" and actual == "SAFE":
            tn = 1
            TN += 1

        elif expected == "SAFE" and actual == "JAILBREAK":
            fp = 1
            FP += 1

        elif expected == "JAILBREAK" and actual == "SAFE":
            fn = 1
            FN += 1

        df.at[index, "Actual"] = actual
        df.at[index, "Blocked_By"] = blocked_by

        df.at[index, "TP"] = int(tp)
        df.at[index, "TN"] = int(tn)
        df.at[index, "FP"] = int(fp)
        df.at[index, "FN"] = int(fn)

    # Save updated CSV
    df.to_csv("input.csv", index=False)

    # ------------------------
    # Metrics
    # ------------------------

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision = TP / (TP + FP) if (TP + FP) else 0

    recall = TP / (TP + FN) if (TP + FN) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    print("\n==============================")
    print(" Guardrail Evaluation Report ")
    print("==============================")

    print(f"TP : {TP}")
    print(f"TN : {TN}")
    print(f"FP : {FP}")
    print(f"FN : {FN}")

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


if __name__ == "__main__":
    asyncio.run(main())