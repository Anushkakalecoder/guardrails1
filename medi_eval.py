import asyncio
import pandas as pd

from agents import Agent, Runner
from agents.exceptions import InputGuardrailTripwireTriggered

from medi_advice import (
    GUARDRAIL_MODEL,
    medical_advice_input_guardrail,          # <-- your medical guardrail function
)

# --------------------------------------------------
# Test Agent
# --------------------------------------------------

agent = Agent(
    name="Medical Guardrail Evaluation Agent",
    instructions="You are a helpful assistant.",
    model=GUARDRAIL_MODEL,
    input_guardrails=[medical_advice_input_guardrail],
)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

async def main():
#mention your file
    df = pd.read_csv("file_name/path_name.csv")

    # Create columns if not present
    if "Actual" not in df.columns:
        df["Actual"] = ""

    if "Blocked_By" not in df.columns:
        df["Blocked_By"] = ""

    for col in ["TP", "TN", "FP", "FN"]:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype(int)
            )

    TP = TN = FP = FN = 0

    for index, row in df.iterrows():

        query = row["Query"]
        expected = row["Expected"].strip().upper()

        print(f"\nRunning Query {index+1}")
        print(query)

        actual = "SAFE"
        blocked_by = "None"

        try:

            await Runner.run(agent, query)
            await asyncio.sleep(10)
        except InputGuardrailTripwireTriggered:

            actual = "MEDICAL"

            blocked_by = "Regex / LLM"

        tp = tn = fp = fn = 0

        if expected == "MEDICAL" and actual == "MEDICAL":
            tp = 1
            TP += 1

        elif expected == "SAFE" and actual == "SAFE":
            tn = 1
            TN += 1

        elif expected == "SAFE" and actual == "MEDICAL":
            fp = 1
            FP += 1

        elif expected == "MEDICAL" and actual == "SAFE":
            fn = 1
            FN += 1

        df.at[index, "Actual"] = actual
        df.at[index, "Blocked_By"] = blocked_by

        df.at[index, "TP"] = tp
        df.at[index, "TN"] = tn
        df.at[index, "FP"] = fp
        df.at[index, "FN"] = fn

        await asyncio.sleep(2)

    df.to_csv(
        "output_file_path/name.csv",# your output csv file where you want to save the results
        index=False,
    )

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else 0
    )

    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print("\n==============================")
    print(" Medical Guardrail Evaluation ")
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