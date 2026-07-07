import asyncio

from agents import Runner
from agents.exceptions import InputGuardrailTripwireTriggered

from medi_advice import (
    cra_agent,
)


async def main():

    print("Medical Guardrail Test")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("Enter your message: ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            print("\nExiting...")
            break

        try:

            await Runner.run(
                cra_agent,
                query,
            )

            print("✅ ALLOWED")

        except InputGuardrailTripwireTriggered:

            print("🚫 BLOCKED")

        except Exception as e:

            print(f"Unexpected Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())