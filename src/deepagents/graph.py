from pydantic import BaseModel
from deepagents.sub_agent import _create_task_tool, SubAgent
from deepagents.model import get_default_model
from deepagents.tools import (
    write_todos,
    write_file,
    read_file,
    ls,
    edit_file,
    create_submit_tool,
)
from deepagents.prompts import BASE_PROMPT
from deepagents.state import DeepAgentState
from typing import Sequence, Union, Callable, Any, TypeVar, Type, Optional, Dict
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike

from langgraph.prebuilt import create_react_agent

StateSchema = TypeVar("StateSchema", bound=DeepAgentState)
StateSchemaType = Type[StateSchema]

# Built-in tools provided by every deep agent
BUILT_IN_TOOLS = [write_todos, write_file, read_file, ls, edit_file]


def _compose_prompt(instructions: str, user_base_prompt: Optional[str],include_submit_note: bool) -> str:
    """Compose the final system prompt.

    - If user_base_prompt is provided, it replaces the default base prompt.
    - Always appends the user instructions after a blank line.
    - Optionally appends a submit section note when a submit schema is used.
    """
    base = user_base_prompt if user_base_prompt is not None else BASE_PROMPT
    prompt = base + "\n\n" + instructions
    if include_submit_note:
        prompt += (
            "\n\n## `submit`\n\n"
            "As the final step, call the `submit` tool to validate and submit your work in the required schema. "
            "Pass your draft as a single JSON string (NOT a JSON object argument). Use only double quotes and no trailing commas. Do not wrap in Markdown.\n"
            "Rules: Call `submit` exactly once. Provide only the structured content in the string; no prose. ENDS YOUR TURN WHEN YOU CALL SUBMIT."
        )
    return prompt

def _maybe_add_submit_tool(
    submit_schema: Optional[type],
    submit_llm: Optional[LanguageModelLike],
    model: LanguageModelLike,
):
    """Return a list of extra tools (currently just the submit tool if configured).

    Does not echo back the user-provided tools to avoid duplication.
    """
    extra_tools: list[Union[BaseTool, Callable, dict[str, Any]]] = []
    if submit_schema is not None:
        submit_tool = create_submit_tool(
            schema=submit_schema,
            llm=submit_llm or model,
        )
        extra_tools.append(submit_tool)
    return extra_tools

def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    user_base_prompt: Optional[str] = None,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
    submit_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None, 
    submit_llm: Optional[LanguageModelLike] = None,
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
        user_base_prompt=user_base_prompt,
        include_submit_note=submit_schema is not None,
    )

    if model is None:
        model = get_default_model()
    state_schema = state_schema or DeepAgentState
    task_tool = _create_task_tool(
        list(tools) + BUILT_IN_TOOLS,
        instructions,
        subagents or [],
        model,
        state_schema
    )
    # Optionally add a schema-backed submit tool
    submit_tools = _maybe_add_submit_tool(submit_schema, submit_llm, model)

    # Order: built-in tools, user tools, optional submit tool, and the task tool
    all_tools = BUILT_IN_TOOLS + list(tools) + submit_tools + [task_tool]
    return create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
    )
