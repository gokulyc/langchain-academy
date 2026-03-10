from langchain_core.messages import SystemMessage

# from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from aws_mcp import load_mcp_tools


async def build_graph():
    tools = await load_mcp_tools()

    # Define LLM with bound tools
    llm = ChatOpenRouter(model="x-ai/grok-4.1-fast")
    # llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools)

    # System message
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with providing best practices related to AWS.")

    # Node
    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    graph = builder.compile()
    return graph


async def main():
    from langchain.messages import HumanMessage
    from langchain_core.callbacks.stdout import StdOutCallbackHandler
    from rich import print as rprint

    graph = await build_graph()
    result = await graph.ainvoke(
        input={"messages": [HumanMessage("What is aws lambda? What are best practices for python aws lambdas?")]},
        config={"callbacks": [StdOutCallbackHandler()]},
    )
    rprint(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
