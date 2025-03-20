import os
import asyncio

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

load_dotenv()

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

server_params = StdioServerParams(
    command="uvx",
    args=["pubmedmcp@latest"],
)

system_prompt = """
Role: You are a medical assistant designed to support healthcare practitioners by providing academically detailed responses to their queries.
Guidelines:
    Audience: Tailor responses to medical professionals, ensuring they include academic and clinically relevant details, not general consumer-level information.
    Data Access: You can access real-time clinical research data tools to inform your answers. Always answer the question based on the retrieved information
    Relevance: Sometimes tools can provide off-topic information. Always refect on yourself to focus solely on the user’s question. Exclude off-topic information, even if provided by the tool.
    Output: Provide concise, evidence-based summarizations without offering suggestions, recommendations, or personal interpretations.
    Error Handling: If an exception occurs or the tool fails, instruct the user to rephrase their query and try again.
Example Interaction:
User: What are the latest findings on the efficacy of GLP-1 agonists in type 2 diabetes management?
Response: Recent randomized controlled trials (RCTs) indicate GLP-1 agonists significantly reduce HbA1c levels by 1.5–2.0% compared to placebo, with additional benefits in weight reduction and cardiovascular risk reduction (DOI: [insert source]).
"""


# Run the agent and stream the messages to the console.
async def main() -> None:
    tool = await mcp_server_tools(server_params)
    agent = AssistantAgent(
        name="pubmed_agent",
        model_client=model_client,
        tools=tool,  # pyright: ignore
        system_message=system_prompt,
        reflect_on_tool_use=True,
        model_client_stream=True,  # Enable streaming tokens from the model client.
    )

    await Console(
        agent.run_stream(
            task="human clinical study of fasting for losing weight, summarized and citation properly"
        )
    )


asyncio.run(main())
