from pydantic_ai import Agent

# Use ollama model with Pydantic AI
import os
from pathlib import Path

from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

import asyncio

# Load .env file
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma4:26b")

ollama_model = OpenAIChatModel(
    model_name=OLLAMA_MODEL_NAME,
    provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
)
agent = Agent(ollama_model)

async def main():
  async with agent.run_stream('Where does "hello world" come from?') as result:
      async for message in result.stream_text():
          print(message)
          #> The first known
          #> The first known use of "hello,
          #> The first known use of "hello, world" was in
          #> The first known use of "hello, world" was in a 1974 textbook
          #> The first known use of "hello, world" was in a 1974 textbook about the C
          #> The first known use of "hello, world" was in a 1974 textbook about the C programming language.


asyncio.run(main())