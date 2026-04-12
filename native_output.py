from pydantic_ai import Agent, NativeOutput

from tool_output import Fruit, Vehicle

# Use ollama model with Pydantic AI
import os
from pathlib import Path

from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

import mlflow

mlflow.pydantic_ai.autolog()

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

agent = Agent(
  ollama_model,
  output_type=NativeOutput(
      [Fruit, Vehicle],
      name='Fruit_or_vehicle',
      description='Return a fruit or vehicle.'
  ),
)
result = agent.run_sync('What is a Ford Explorer?')
print(repr(result.output))