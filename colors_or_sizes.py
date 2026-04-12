from pydantic_ai import Agent

# Use MiniMax-M2.7 with Pydantic AI

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.anthropic import AnthropicModel

import mlflow

mlflow.pydantic_ai.autolog()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("pydanticai")
# Load .env file
load_dotenv()

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
MINIMAX_MODEL_NAME = os.getenv("MINIMAX_MODEL_NAME", "MiniMax-M2.7")

model = AnthropicModel(
    MINIMAX_MODEL_NAME,
    provider=AnthropicProvider(
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_BASE_URL,
    )
)

agent = Agent(model)

agent = Agent[None, list[str] | list[int]](
  model,
  output_type=list[str] | list[int],  # type: ignore
  instructions='Extract either colors or sizes from the shapes provided.',
)

result = agent.run_sync('red square, blue circle, green triangle')
print(result.output)
#> ['red', 'blue', 'green']

result = agent.run_sync('square size 10, circle size 20, triangle size 30')
print(result.output)
#> [10, 20, 30]