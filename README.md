# Pydantic Tutorial by Duke

## Ollama + Gemma4 

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run gemma4:26b

uv add pydantic-ai
uv add python-dotenv 
```

```python
import os
from pathlib import Path

from dotenv import load_dotenv

from pydantic_ai import Agent
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
agent = Agent(
    ollama_model,
    system_prompt=(
        "ENTER YOUR SYSTEM PROMPT"
    ),
)
```

## MiniMax-M2.7 using Anthropic Endpoints 

