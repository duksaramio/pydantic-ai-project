import asyncio
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent
import os
from pathlib import Path

from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

import mlflow

# Note: mlflow.pydantic_ai.autolog() has a version mismatch with pydantic-ai 1.80.0
# The _tool_manager module it expects doesn't exist in this version.
# If you need MLflow tracing, either:
# 1. Wait for a fix in mlflow or pydantic-ai
# 2. Or implement manual logging instead
mlflow.pydantic_ai.autolog()
# Optional: Set a tracking URI and an experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("pydanticai")

# Load .env file
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma4:26b")

ollama_model = OpenAIChatModel(
    model_name=OLLAMA_MODEL_NAME,
    provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
)

@dataclass
class MyDeps:
  api_key: str
  http_client: httpx.AsyncClient


agent = Agent(
  ollama_model,
  deps_type=MyDeps,
)


async def main():
  async with httpx.AsyncClient() as client:
      deps = MyDeps('foobar', client)
      result = await agent.run(
          'Tell me a joke.',
          deps=deps,
      )
      print(result.output)

asyncio.run(main())