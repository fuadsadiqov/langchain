from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, BaseMessage, HumanMessage, AIMessage

model = ChatOllama(
    model="deepseek-r1:1.5b"
)

@tool
def add(a: int, b: int) -> int:
    return a + b

@tool
def multiple(a: int, b: int) -> int:
    return a * b

tools: list = [add, multiple]
tools_by_name = {tool.name: tool for tool in tools}

model_with_tools: ChatOllama = model.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def call_tool_node(state: AgentState) -> AgentState:
    outputs: list[ToolMessage] = []

    messages: list[BaseMessage] = state['messages']
    if not messages:
        return {"messages": outputs}

    last_message: BaseMessage = messages[-1]
    tool_calls  = getattr(last_message, "tool_calls", [])

    for tool_call in tool_calls:
        tool = tools_by_name.get(tool_call['name'])

        if not tool:
            raise ValueError(f"Tool {tool} not raised")

        tool_args = tool_call["args"]
        tool_result = tool.invoke(tool_args)

    return {"messages": outputs}

def main() -> None:
    print("Hello from lecture-6!")
