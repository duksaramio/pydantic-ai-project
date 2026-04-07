import os
import asyncio

from dotenv import load_dotenv

from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# Load .env file
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma4:26b")

ollama_model = OpenAIChatModel(
    model_name=OLLAMA_MODEL_NAME,
    provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
)

agent = Agent(ollama_model)

result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.output)

async def main():
    result = await agent.run('What is the capital of France?')
    print(result.output)

    async with agent.run_stream('What is the capital of the UK?') as response:
        async for text in response.stream_text():
            print(text)

    events: list[AgentStreamEvent | AgentRunResultEvent] = []
    async for event in agent.run_stream_events('What is the capital of Mexico?'):
        events.append(event)
    print(events)


if __name__ == '__main__':
    asyncio.run(main())