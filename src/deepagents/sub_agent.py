from deepagents.prompts import TASK_DESCRIPTION_PREFIX, TASK_DESCRIPTION_SUFFIX
from deepagents.state import DeepAgentState
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from typing import TypedDict
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from typing import Annotated, NotRequired
from langgraph.types import Command

from langgraph.prebuilt import InjectedState


class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]


def _create_task_tool(tools, instructions, subagents: list[SubAgent], model, state_schema):
    # Only allow built-in tools for sub-agents; drop any user tools and 'submit'
    normalized_tools: list[BaseTool] = []
    for tool_ in tools:
        if not isinstance(tool_, BaseTool):
            tool_ = tool(tool_)
        normalized_tools.append(tool_)
    subagent_tools = [t for t in normalized_tools if t.name in {"write_todos", "write_file", "read_file", "ls", "edit_file"}]

    agents = {
        "general-purpose": create_react_agent(
            model, prompt=instructions, tools=subagent_tools, state_schema=state_schema
        )
    }
    for _agent in subagents:
        # Ignore any per-agent custom tool lists; enforce built-ins only
        _tools = subagent_tools
        agents[_agent["name"]] = create_react_agent(
            model, prompt=_agent["prompt"], tools=_tools, state_schema=state_schema
        )

    other_agents_string = [
        f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
    ]

    @tool(
        description=TASK_DESCRIPTION_PREFIX.format(other_agents=other_agents_string)
        + TASK_DESCRIPTION_SUFFIX
    )
    def task(
        description: str,
        subagent_type: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        if subagent_type not in agents:
            return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"
        sub_agent = agents[subagent_type]
        state["messages"] = [{"role": "user", "content": description}]
        result = sub_agent.invoke(state)
        return Command(
            update={
                "files": result.get("files", {}),
                "messages": [
                    ToolMessage(
                        result["messages"][-1].content, tool_call_id=tool_call_id
                    )
                ],
            }
        )

    return task
