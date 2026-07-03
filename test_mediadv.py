import asyncio

from agents import (
    Runner,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)

from guardrails import (
    test_agent,
    MEDICAL_ADVICE_REFUSAL,
)


async def main():

    print("Type 'exit' to quit.\n")

    while True:

        query = input("Enter your message: ").strip()

        if query.lower() == "exit":
            print("\nExiting...")
            break

        try:
            result = await Runner.run(test_agent, query)

            print("\nAGENT:", result.final_output)

        except InputGuardrailTripwireTriggered as exc:

            print("\n🚫 BLOCKED (Input Guardrail)")
            print(exc)

            print("\nResponse:")
            print(MEDICAL_ADVICE_REFUSAL)

        except OutputGuardrailTripwireTriggered as exc:

            print("\n🚫 BLOCKED (Output Guardrail)")
            print(exc)

            print("\nResponse:")
            print(MEDICAL_ADVICE_REFUSAL)

        except Exception as exc:

            print("\nUnexpected Error:")
            print(exc)


if __name__ == "__main__":
    asyncio.run(main())