import os
import asyncio

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
model_client = OpenAIChatCompletionClient(
    model="qwen/qwen-2.5-72b-instruct",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    model_info={
        "json_output": False,
        "function_calling": True,
        "vision": False,
        "family": "unknown",
    },
)


# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    if "Thailand" in city:
        raise Exception("cannot find result with the given city name, try another")

    return f"The weather in {city} is 73 degrees and Sunny."


# Define an AssistantAgent with the model, tool, system message, and reflection enabled.
# The system message instructs the agent via natural language.
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
    system_message="""You are a helpful assistant. You can access tool to provide real time data to answer user question.
    If Exception is raised, don't try to correct it on your own, just tell the user to try again with different prompt""",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)


# Run the agent and stream the messages to the console.
async def main() -> None:
    await Console(agent.run_stream(task="What is the weather in Bangkok, Thailand?"))


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())
