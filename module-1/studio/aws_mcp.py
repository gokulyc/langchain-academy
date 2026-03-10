import asyncio
import json
import pathlib

import rich
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.messages import HumanMessage

# from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from asyncer import syncify


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve() / "mcp.json"


async def load_mcp_tools():
    mcp_config = json.loads(SCRIPT_DIR.read_text())
    client = MultiServerMCPClient(
        mcp_config
        # {
        #     "math": {
        #         "transport": "stdio",  # Local subprocess communication
        #         "command": "python",
        #         # Absolute path to your math_server.py file
        #         "args": ["/path/to/math_server.py"],
        #     },
        #     "weather": {
        #         "transport": "http",  # HTTP-based remote server
        #         # Ensure you start your weather server on port 8000
        #         "url": "http://localhost:8000/mcp",
        #     }
        # }
    )

    tools = await client.get_tools()
    return tools

load_mcp_tools_sync = syncify(load_mcp_tools, raise_sync_error=False)

async def main():
    
    tools = await load_mcp_tools()
    rich.print("Tools", tools)
    llm = ChatOpenRouter(model="x-ai/grok-4.1-fast")
    # llm = ChatOpenRouter(model="openai/gpt-5.3-chat")
    llm.bind_tools(tools)
    # agent = create_agent(
    #     "claude-sonnet-4-6",
    #     tools
    # )
    # math_response = await agent.ainvoke(
    #     {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    # )
    # weather_response = await agent.ainvoke(
    #     {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
    # )
    # print(math_response)
    # print(weather_response)

    aws1 = await llm.ainvoke([HumanMessage("What is aws lambda? What are best practices for python aws lambdas?")])
    rich.print(aws1)

    aws2 = await llm.ainvoke(
        [HumanMessage("What is aws ecs? As a python developer, what are best practices for aws ecs?")]
    )

    rich.print(aws2)


if __name__ == "__main__":
    asyncio.run(main())
