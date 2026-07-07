import asyncio

from agents import Agent, Runner
from agents.exceptions import InputGuardrailTripwireTriggered

from jailbreaking import (
    GUARDRAIL_MODEL,
    jailbreak_guardrail,
)

agent = Agent(
    name="Test Agent",
    instructions="You are a helpful assistant.",
    model=GUARDRAIL_MODEL,
    input_guardrails=[jailbreak_guardrail],
)


async def main():

    print("Jailbreak Guardrail Test")
    print("Type 'exit' to quit.\n")

    while True:

        prompt = input("Enter Prompt: ").strip()

        if not prompt:
            continue

        if prompt.lower() == "exit":
            print("\nExiting...")
            break

        try:

            await Runner.run(
                agent,
                prompt,
            )

            print("✅ ALLOWED")

        except InputGuardrailTripwireTriggered:

            print("🚫 BLOCKED BY JAILBREAK GUARDRAIL")

        except Exception as e:

            print(f"Unexpected Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())