from deepagents.sub_agent import _create_task_tool, SubAgent
from deepagents.model import get_default_model
from deepagents.tools import (
    write_todos,
    write_file,
    read_file,
    ls,
    edit_file,
)
from deepagents.prompts import BASE_PROMPT
from deepagents.state import DeepAgentState
from typing import Sequence, Union, Callable, Any, TypeVar, Type, Optional
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike

from langgraph.prebuilt import create_react_agent

StateSchema = TypeVar("StateSchema", bound=DeepAgentState)
StateSchemaType = Type[StateSchema]

# Built-in tools provided by every deep agent
BUILT_IN_TOOLS = [write_todos, write_file, read_file, ls, edit_file]


def _compose_prompt(instructions: str) -> str:
    """Compose the final system prompt.

    - If user_base_prompt is provided, it replaces the default base prompt.
    - Always appends the user instructions after a blank line.
    """
    base = BASE_PROMPT
    prompt = base + "\n\n" + instructions
    return prompt

 

def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
):
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    and then four file editing tools: write_file, ls, read_file, edit_file.

    Args:
        tools: The additional tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
        state_schema: The schema of the deep agent. Should subclass from DeepAgentState
    """

    prompt = _compose_prompt(
        instructions=instructions,
    )

    if model is None:
        model = get_default_model()
    state_schema = state_schema or DeepAgentState
    # Sub-agents should only have access to built-in tools, not user tools
    task_tool = _create_task_tool(
        BUILT_IN_TOOLS,
        instructions,
        subagents or [],
        model,
        state_schema
    )
    # Order: built-in tools, user tools, and the task tool
    all_tools = BUILT_IN_TOOLS + list(tools) + [task_tool]
    return create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
    )
