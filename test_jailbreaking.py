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

    while True:

        prompt = input("\nEnter Prompt : ")

        if prompt.lower() == "exit":
            break

        try:

            result = await Runner.run(
                agent,
                prompt,
            )

            print("\n ALLOWED")
            print(result.final_output)

        except InputGuardrailTripwireTriggered:

            print("\n BLOCKED BY JAILBREAK GUARDRAIL")

        except Exception as e:

            print(e)

asyncio.run(main())