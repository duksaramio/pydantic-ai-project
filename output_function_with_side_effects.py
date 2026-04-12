import asyncio
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext

# Use MiniMax-M2.7 with Pydantic AI

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.anthropic import AnthropicModel

# # Load .env file
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

class DatabaseRecord(BaseModel):
    name: str
    value: int | None = None  # Make optional to allow partial output


def save_to_database(ctx: RunContext, record: DatabaseRecord) -> DatabaseRecord:
    """Output function with side effect - only save final output to database."""
    if ctx.partial_output:
        # Skip side effects for partial outputs
        return record

    # Only execute side effect for the final output
    print(f'Saving to database: {record.name} = {record.value}')
    #> Saving to database: test = 42
    return record


agent = Agent(model, output_type=save_to_database)

async def main():
    async with agent.run_stream('Create a record with name "test" and value 42') as result:
        async for output in result.stream_output(debounce_by=None):
            print(output)
            #> name='test' value=None
            #> name='test' value=42

asyncio.run(main())