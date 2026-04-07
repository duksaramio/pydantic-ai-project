import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# Load .env file
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma4:26b")

class CityLocation(BaseModel):
    city: str
    country: str

ollama_model = OpenAIChatModel(
    model_name=OLLAMA_MODEL_NAME,
    provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
)
agent = Agent(
    ollama_model,
    output_type=CityLocation,
    system_prompt=(
        "You are a helpful assistant that extracts location information. "
        "When asked about a city or location, respond ONLY with valid JSON in this exact format: "
        '{"city": "<city_name>", "country": "<country_name>"}. '
        "Do not include any other text, explanations, or formatting."
    ),
)

result = agent.run_sync('Where were the olympics held in 2012?', model_settings={'max_result_retries': 3})
print(result.output)
print(result.usage())
