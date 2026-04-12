import os
from pathlib import Path

from dotenv import load_dotenv
import mlflow
from mlflow.genai.scorers import Correctness
from mlflow.genai.optimize.optimizers import GepaPromptOptimizer
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Pydantic AI Optimization")
# Load .env file
load_dotenv()

os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['OPENAI_API_KEY'] = 'not-needed'

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma4:26b")

ollama_model = OpenAIChatModel(
    model_name=OLLAMA_MODEL_NAME,
    provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
)


# If you're inside notebooks, please uncomment the following lines.
# import nest_asyncio
# nest_asyncio.apply()

# Step 1: Register your initial prompts
system_prompt = mlflow.genai.register_prompt(
    name="customer-support-system",
    template="You are a helpful customer support agent for an e-commerce platform. "
    "Assist customers with their questions about orders, returns, and products.",
)

user_prompt = mlflow.genai.register_prompt(
    name="customer-support-query",
    template="Customer inquiry: {{query}}",
)


# Step 2: Create a prediction function that uses Pydantic AI
@mlflow.trace
def predict_fn(query):
    # Load prompts from registry
    system_prompt = mlflow.genai.load_prompt("prompts:/customer-support-system@latest")
    user_prompt = mlflow.genai.load_prompt("prompts:/customer-support-query@latest")

    # Initialize agent with system prompt
    # agent = Agent(
    #     model="openai:gpt-5-mini",
    #     system_prompt=system_prompt.template,
    # )
    agent = Agent(
        ollama_model,
        system_prompt=system_prompt.template,
    )

    # Format user message and run agent
    formatted_query = user_prompt.format(query=query)
    result = agent.run_sync(formatted_query)

    return result.output


# Step 3: Prepare training data
dataset = [
    {
        "inputs": {"query": "Where is my order #12345?"},
        "expectations": {
            "expected_response": "I'd be happy to help you track your order #12345. "
            "Please check your email for a tracking link, or I can look it up for you if you provide your email address."
        },
    },
    {
        "inputs": {"query": "How do I return a defective product?"},
        "expectations": {
            "expected_response": "I'm sorry to hear your product is defective. You can initiate a return "
            "through your account's order history within 30 days of purchase. We'll send you a prepaid shipping label."
        },
    },
    {
        "inputs": {"query": "Do you have this item in blue?"},
        "expectations": {
            "expected_response": "I'd be happy to check product availability for you. "
            "Could you please provide the product name or SKU so I can verify if it's available in blue?"
        },
    },
    # more data...
]

# Step 4: Optimize the prompts
result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=dataset,
    prompt_uris=[system_prompt.uri, user_prompt.uri],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gemma4:26b"),
    scorers=[Correctness(model="openai:/gemma4:26b")],
)

# Step 5: Use the optimized prompts
optimized_system_prompt = result.optimized_prompts[0]
optimized_user_prompt = result.optimized_prompts[1]

print(f"Optimized system prompt URI: {optimized_system_prompt.uri}")
print(f"Optimized system template: {optimized_system_prompt.template}")
print(f"Optimized user prompt URI: {optimized_user_prompt.uri}")
print(f"Optimized user template: {optimized_user_prompt.template}")

# Since your agent already uses @latest, it will automatically use the optimized prompts
predict_fn("Can I get a refund for order #67890?")