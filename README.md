<p align="left">
  <img src="virtualflow_logo.png" alt="Virtualflow Technologies" height="48"/>
</p>

# Virtualflow Deep Agents

Production-ready AI agents for planning, tooling, sub-agents, and a virtual filesystem – built on LangGraph/LangChain and inspired by the excellent work in the original deepagents project.

This repository is Virtualflow Technologies’ maintained distribution of deep agents with:
- Default Fireworks models (configurable)
- A robust in-memory virtual filesystem with nested paths
- Comprehensive built-in tools (`ls`, `read_file`, `write_file`, `edit_file`, `write_todos`)
- Optional structured submission via Pydantic + Trustcall

Original project: [hwchase17/deepagents](https://github.com/hwchase17/deepagents)

## Installation

```bash
pip install deepagents
# With Trustcall for structured submissions
pip install "deepagents[structured]"
```

## Quickstart

(To run the example below, will need to `pip install tavily-python`)

```python
import os
from typing import Literal

from tavily import TavilyClient
from deepagents import create_deep_agent


# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    tavily_async_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return tavily_async_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# Prompt prefix to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

# Create the agent
agent = create_deep_agent(
    tools=[internet_search],
    instructions=research_instructions,
)

# Invoke the agent
result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})
```

See [examples/research/research_agent.py](examples/research/research_agent.py) for a more complex example.

The agent created with `create_deep_agent` is just a LangGraph graph - so you can interact with it (streaming, human-in-the-loop, memory, studio)
in the same way you would any LangGraph agent.

## Creating a custom deep agent

There are three parameters you can pass to `create_deep_agent` to create your own custom deep agent.

### `tools` (Required)

The first argument to `create_deep_agent` is `tools`.
This should be a list of functions or LangChain `@tool` objects.
The agent (and any subagents) will have access to these tools.

### `instructions` (Required)

The second argument to `create_deep_agent` is `instructions`.
This will serve as part of the prompt of the deep agent.
Note that there is a [built in system prompt](src/deepagents/prompts.py) as well, so this is not the *entire* prompt the agent will see.

### `subagents` (Optional)

A keyword-only argument to `create_deep_agent` is `subagents`.
This can be used to specify any custom subagents this deep agent will have access to.
You can read more about why you would want to use subagents [here](#sub-agents)

`subagents` should be a list of dictionaries, where each dictionary follow this schema:

```python
class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
```

- **name**: This is the name of the subagent, and how the main agent will call the subagent
- **description**: This is the description of the subagent that is shown to the main agent
- **prompt**: This is the prompt used for the subagent
- **tools**: This is the list of tools that the subagent has access to. By default will have access to all tools passed in, as well as all built-in tools.

To use it looks like:

```python
research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "prompt": sub_research_prompt,
}
subagents = [research_subagent]
agent = create_deep_agent(
    tools,
    prompt,
    subagents=subagents
)
```

### `model` (Optional)

By default, Virtualflow Deep Agents use a Fireworks-hosted model via `langchain-fireworks`:

- Default chat model: `accounts/fireworks/models/kimi-k2-instruct`
- Default structuring model (for Trustcall extraction): `accounts/fireworks/models/llama4-maverick-instruct-basic`
- Required env var: `FIREWORKS_API_KEY`

You can customize this by passing any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

#### Example: Using a Custom Model

Here's how to use a custom model (like OpenAI's `gpt-oss` model via Ollama):

(Requires `pip install langchain` and then `pip install langchain-ollama` for Ollama models)

```python
from deepagents import create_deep_agent

# ... existing agent definitions ...

model = init_chat_model(
    model="ollama:gpt-oss:20b",  
)
agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
    ...
)
```

## Deep Agent Details

The below components are built into `deepagents` and helps make it work for deep tasks off-the-shelf.

### System Prompt

We ship a comprehensive base system prompt in `src/deepagents/prompts.py` that:
- Explains planning expectations and todo usage
- Documents the virtual filesystem and tool usage patterns
- Describes sub-agent usage via the Task tool
- Emphasizes quality, language matching, and artifact integrity

Your `instructions` are appended after the base prompt. The base prompt explicitly tells the model to treat the following project-specific instructions as primary.

### Planning Tool

`deepagents` comes with a built-in planning tool. This planning tool is very simple and is based on ClaudeCode's TodoWrite tool.
This tool doesn't actually do anything - it is just a way for the agent to come up with a plan, and then have that in the context to help keep it on track.

### Virtual Filesystem & Tools

We provide four built-in file system tools: `ls`, `edit_file`, `read_file`, `write_file`.
These operate on a virtual in-memory filesystem stored in the LangGraph State (`state["files"]`), not your host filesystem. This means you can run many agents on the same machine without risk of cross-editing real files.

Nested paths are supported. You can use keys like `src/utils/helpers.py` or `reports/2025/q1.md`—they are stored as string paths in the state map. Directory creation is implicit: writing `a/b/c.txt` creates the path key in the virtual filesystem.

These files can be passed in (and also retrieved) by using the `files` key in the LangGraph State object.

```python
agent = create_deep_agent(...)

result = agent.invoke({
    "messages": ...,
    # Pass in files to the agent using this key
    # "files": {"foo.txt": "foo", ...}
})

# Access any files afterwards like this
result["files"]
```

#### Built-in tools and behavior

- `ls`:
  - Lists the keys of all files currently in the virtual filesystem as path strings (e.g., `"a/b.txt"`).
  - This does not perform hierarchical traversal; it returns the full keys that exist.

- `read_file(file_path, offset=0, limit=2000)`:
  - Reads from the virtual filesystem (`state["files"][file_path]`).
  - Returns contents with line numbers in `cat -n` style: six-character right-aligned line number + tab + content.
  - Supports `offset` (0-based line index) and `limit` (max lines) for partial reads.
  - Long lines are truncated to 2000 chars.
  - If the file does not exist, returns an error message.

- `write_file(file_path, content)`:
  - Writes/overwrites a file at `file_path` in the virtual filesystem.
  - Implicitly creates nested path keys if needed.

- `edit_file(file_path, old_string, new_string, replace_all=False)`:
  - Performs exact string replacement within a single file’s contents.
  - If `replace_all=False`, the edit requires `old_string` to be unique in the file; otherwise it returns a friendly error suggesting `replace_all=True` or providing more specific context.
  - Updates the file in the virtual filesystem and emits a corresponding tool message.

### Submitting structured outputs (Optional)

You can require the agent to submit its final work in a strict Pydantic schema. Pass your schema via `submit_schema` and a `submit` tool will be auto-injected. The tool validates with Pydantic first and falls back to [Trustcall](https://github.com/hinthornw/trustcall?tab=readme-ov-file#complex-schema) to coerce/repair difficult outputs.

Installation options:
- Core package only: `pip install deepagents`
- With Trustcall support: `pip install deepagents[structured]`

```python
from pydantic import BaseModel, Field
from deepagents import create_deep_agent

class AnalysisReport(BaseModel):
    title: str
    summary: str
    items: list[str] = Field(default_factory=list)

agent = create_deep_agent(
    tools=[...],
    instructions="At the end, call the 'submit' tool to submit your work in the required format.",
    submit_schema=AnalysisReport,          # auto-adds a submit tool named 'submit'
    submit_llm=None,                       # optional; defaults to a structuring model via get_default_structuring_model()
)

res = agent.invoke({"messages": [("user", "Do the task and submit") ]})
res["output_submitted"]  # True/False
res["submission"]        # JSON string matching AnalysisReport
```

### Sub Agents

`deepagents` comes with the built-in ability to call sub agents (based on Claude Code).
It has access to a `general-purpose` subagent at all times - this is a subagent with the same instructions as the main agent and all the tools that is has access to.
You can also specify [custom sub agents](#subagents-optional) with their own instructions and tools.

Sub agents are useful for ["context quarantine"](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html#context-quarantine) (to help not pollute the overall context of the main agent)
as well as custom instructions.

## MCP

The `deepagents` library can be ran with MCP tools. This can be achieved by using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

(To run the example below, will need to `pip install langchain-mcp-adapters`)

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent

async def main():
    # Collect MCP tools
    mcp_client = MultiServerMCPClient(...)
    mcp_tools = await mcp_client.get_tools()

    # Create agent
    agent = create_deep_agent(tools=mcp_tools, ....)

    # Stream the agent
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "what is langgraph?"}]},
        stream_mode="values"
    ):
        if "messages" in chunk:
            chunk["messages"][-1].pretty_print()

asyncio.run(main())
```

## Roadmap
- [ ] Allow users to customize full system prompt
- [ ] Code cleanliness (type hinting, docstrings, formating)
- [ ] Allow for more of a robust virtual filesystem
- [ ] Create an example of a deep coding agent built on top of this
- [ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)
- [ ] Add human-in-the-loop support for tools

## Attribution

This distribution is based on and owes a lot to the original work in [hwchase17/deepagents](https://github.com/hwchase17/deepagents). We’ve adapted and expanded it for Virtualflow Technologies’ production needs while keeping the spirit of the original project.
