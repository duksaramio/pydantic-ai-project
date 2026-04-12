from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, capture_run_messages
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
agent = Agent(
    ollama_model
)

@agent.tool_plain
def calc_volume(size: int) -> int:
  if size == 42:
      return size**3
  else:
      raise ModelRetry('Please try again.')


with capture_run_messages() as messages:
  try:
      result = agent.run_sync('Please get me the volume of a box with size 6.')
  except UnexpectedModelBehavior as e:
      print('An error occurred:', e)
      #> An error occurred: Tool 'calc_volume' exceeded max retries count of 1
      print('cause:', repr(e.__cause__))
      #> cause: ModelRetry('Please try again.')
      print('messages:', messages)
  else:
      print(result.output)