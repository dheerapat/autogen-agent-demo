import os
import asyncio

from dotenv import load_dotenv
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

load_dotenv()

model_client = OpenAIChatCompletionClient(
    model="meta-llama/llama-4-maverick",
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
    args=["pubmedmcp@latest"]
)

system_prompt = """
Role: You are a medical assistant designed to support healthcare practitioners by providing academically rigorous, evidence-based responses to clinical queries.
Guidelines:
    Audience: Tailor responses exclusively for medical professionals, incorporating academic and clinically relevant details. Avoid general consumer-level information.
    Data Access:
        You can access a tool to get abstract from PubMed.
        Utilize real-time clinical research tools to inform answers.
        Retrieve data from 5-10 sources, prioritizing recent publications (sorted by publication date, newest to oldest).
        Use minimal, contextually relevant keywords in tool queries to ensure precision.
    Relevance: Exclude off-topic information, even if provided by tools, and focus solely on the user’s question.
    Output:
        Provide concise, evidence-based summaries without suggestions, recommendations, or personal interpretations.
        If drug names are mentioned, include dosage regimens only if explicitly provided in retrieved data. Do not infer or provide dosage information from external knowledge.
    Error Handling: If tools fail or return insufficient data, instruct the user to rephrase their query and try again.
Example Interaction:
User: What are the latest findings on the efficacy of GLP-1 agonists in type 2 diabetes management?
Response: Recent randomized controlled trials (RCTs) demonstrate that GLP-1 agonists significantly reduce HbA1c levels by 1.5-2.0% compared to placebo, with additional benefits in weight reduction and cardiovascular risk reduction (Smith et al., 2023). Dosage regimens varied across studies, with common examples including liraglutide 1.8 mg daily and semaglutide 1 mg weekly (Johnson et al., 2022; Lee et al., 2023).
"""


# Run the agent and stream the messages to the console.
async def solo_agent() -> None:
    tool = await mcp_server_tools(server_params)
    print(tool)
    agent = AssistantAgent(
        name="pubmed_agent",
        model_client=model_client,
        tools=tool, # type: ignore
        system_message=system_prompt,
        reflect_on_tool_use=True,
        model_client_stream=True,  # Enable streaming tokens from the model client.
    )

    await Console(
        agent.run_stream(
            task="first line therapy for otitis media in children"
        )
    )


async def single_agent_team() -> None:
    tool = await mcp_server_tools(server_params)
    agent = AssistantAgent(
        name="pubmed_agent",
        model_client=model_client,
        tools=tool,  # pyright: ignore
        system_message=system_prompt,
    )

    termination_condition = TextMessageTermination("pubmed_agent")
    team = RoundRobinGroupChat([agent], termination_condition=termination_condition)

    async for message in team.run_stream(
        task="effective treatment for hairloss in men"
    ):
        print(type(message).__name__, message)


asyncio.run(solo_agent())
