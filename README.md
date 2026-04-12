# Pydantic Tutorial by Duke

I am currently going through Pydantic AI website examples using Ollama+Gemma4. I use PyCharm with pi ACP as coding Agent on Ubuntu 24. Also running on Nvidia 4090. So these examples are available on Pydantic AI website, but I modified them to use local LLM.

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
agent = Agent(ollama_model)
```

## MiniMax-M2.7 using Anthropic Endpoints 

```python
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.anthropic import AnthropicModel

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
```

## Adding mlflow server to trace

```python
import mlflow

mlflow.pydantic_ai.autolog()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("pydanticai")
```