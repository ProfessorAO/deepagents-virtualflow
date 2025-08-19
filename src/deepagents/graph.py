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
from typing import Sequence, Union, Callable, Any, TypeVar, Type, Optional, Literal
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
    instructions: Optional[str] = None,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: Optional[list[SubAgent]] = None,
    pre_loaded_files: Optional[dict[str, str]] = None,
    state_schema: Optional[StateSchemaType] = None,
    *,
    # Pass-through options to the underlying create_react_agent
    prompt: Optional[Any] = None,
    response_format: Optional[Any] = None,
    pre_model_hook: Optional[Any] = None,
    post_model_hook: Optional[Any] = None,
    context_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Any] = None,
    store: Optional[Any] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
    version: Literal["v1", "v2"] = "v2",
    name: Optional[str] = None,
):
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    and four file editing tools: write_file, ls, read_file, edit_file. It also
    exposes most configuration options supported by the underlying ReAct agent.

    Args:
        tools: Additional tools the agent should have access to.
        instructions: Optional extra instructions appended to the base prompt.
        model: The model to use. If omitted, a default is selected.
        subagents: Optional list of subagent descriptors used by the Task tool.
        pre_loaded_files: Optional dictionary of file paths to file contents to preload
            into the agent's virtual filesystem. These files will be available immediately
            when the agent starts.
        state_schema: Optional schema of the deep agent; should subclass DeepAgentState.

        prompt: Optional prompt override passed directly to the underlying agent. If not
            provided, a prompt is composed from the base prompt and ``instructions`` (if any).
        response_format: Optional structured response schema for the final output.
        pre_model_hook: Optional callable/runnable executed before the LLM call.
        post_model_hook: Optional callable/runnable executed after the LLM call.
        context_schema: Optional schema for run-scoped context.
        checkpointer: Optional checkpointer for persistence.
        store: Optional store for cross-thread persistence.
        interrupt_before: Optional list of node names to interrupt before ("agent", "tools").
        interrupt_after: Optional list of node names to interrupt after ("agent", "tools").
        debug: Enable debug mode.
        version: Agent graph version ("v1" or "v2").
        name: Optional name for the compiled graph.
    """

    # Derive prompt if not explicitly provided
    prompt_to_use = prompt if prompt is not None else (
        _compose_prompt(instructions=instructions or "")
    )

    if model is None:
        model = get_default_model()
    state_schema = state_schema or DeepAgentState

    # Sub-agents should only have access to built-in tools, not user tools
    task_tool = _create_task_tool(
        BUILT_IN_TOOLS,
        instructions or "",
        subagents or [],
        model,
        state_schema,
    )

    # Order: built-in tools, user tools, and the task tool
    all_tools = BUILT_IN_TOOLS + list(tools) + [task_tool]

    # Create the agent
    agent = create_react_agent(
        model,
        tools=all_tools,
        prompt=prompt_to_use,
        response_format=response_format,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
        state_schema=state_schema,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        version=version,
        name=name,
    )

    # If pre_loaded_files are provided, create a wrapper that initializes the state
    if pre_loaded_files:
        original_invoke = agent.invoke
        
        def invoke_with_preloaded_files(inputs):
            # Initialize files in the input if not already present
            if "files" not in inputs:
                inputs["files"] = {}
            
            # Merge pre_loaded_files with any existing files
            inputs["files"].update(pre_loaded_files)
            
            return original_invoke(inputs)
        
        # Replace the invoke method
        agent.invoke = invoke_with_preloaded_files

    return agent
