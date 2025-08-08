from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing import Annotated, Optional, Type
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, ValidationError
import json

from deepagents.prompts import (
    WRITE_TODOS_DESCRIPTION,
    EDIT_DESCRIPTION,
    TOOL_DESCRIPTION,
    LS_DESCRIPTION,
    WRITE_FILE_DESCRIPTION,
)
from deepagents.state import Todo, DeepAgentState
from deepagents.model import get_default_structuring_model
from langchain_core.language_models import LanguageModelLike


@tool(description=WRITE_TODOS_DESCRIPTION)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )

@tool(description=LS_DESCRIPTION)
def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files"""
    return list(state.get("files", {}).keys())


@tool(description=TOOL_DESCRIPTION)
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file."""
    mock_filesystem = state.get("files", {})
    if file_path not in mock_filesystem:
        return f"Error: File '{file_path}' not found"

    # Get file content
    content = mock_filesystem[file_path]

    # Handle empty file
    if not content or content.strip() == "":
        return "System reminder: File exists but has empty contents"

    # Split content into lines
    lines = content.splitlines()

    # Apply line offset and limit
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    # Handle case where offset is beyond file length
    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    # Format output with line numbers (cat -n format)
    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i]

        # Truncate lines longer than 2000 characters
        if len(line_content) > 2000:
            line_content = line_content[:2000]

        # Line numbers start at 1, so add 1 to the index
        line_number = i + 1
        result_lines.append(f"{line_number:6d}\t{line_content}")

    return "\n".join(result_lines)


@tool(description=WRITE_FILE_DESCRIPTION)
def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write to a file."""
    files = state.get("files", {})
    files[file_path] = content
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool(description=EDIT_DESCRIPTION)
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    replace_all: bool = False,
) -> str:
    """Write to a file."""
    mock_filesystem = state.get("files", {})
    # Check if file exists in mock filesystem
    if file_path not in mock_filesystem:
        return f"Error: File '{file_path}' not found"

    # Get current file content
    content = mock_filesystem[file_path]

    # Check if old_string exists in the file
    if old_string not in content:
        return f"Error: String not found in file: '{old_string}'"

    # If not replace_all, check for uniqueness
    if not replace_all:
        occurrences = content.count(old_string)
        if occurrences > 1:
            return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
        elif occurrences == 0:
            return f"Error: String not found in file: '{old_string}'"

    # Perform the replacement
    if replace_all:
        new_content = content.replace(old_string, new_string)
        replacement_count = content.count(old_string)
        result_msg = f"Successfully replaced {replacement_count} instance(s) of the string in '{file_path}'"
    else:
        new_content = content.replace(
            old_string, new_string, 1
        )  # Replace only first occurrence
        result_msg = f"Successfully replaced string in '{file_path}'"

    # Update the mock filesystem
    mock_filesystem[file_path] = new_content
    return Command(
        update={
            "files": mock_filesystem,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )


# Factory to create a structured submission tool using a user-provided Pydantic schema
def create_submit_tool(
    schema: Type[BaseModel],
    llm: Optional[LanguageModelLike] = None,
    tool_name: str = "submit",
    description: Optional[str] = None,
):
    """Create a submit tool that validates output against a Pydantic schema.

    Fast path: validate provided draft as JSON against the schema.
    Fallback: use Trustcall to coerce arbitrary text/JSON into the schema.

    Updates DeepAgentState: sets output_submitted=True and stores JSON string in submission.
    """

    desc = description or (
        "FINAL STEP ONLY. Validate and submit work against a strict schema. "
        "Pass your draft (free text or JSON). The tool will: 1) try fast Pydantic validation; "
        "2) if that fails, invoke Trustcall to coerce it."
    )

    # Lazily initialize Trustcall extractor to avoid hard dependency unless used
    extractor_holder = {"extractor": None}

    def _get_extractor():
        if extractor_holder["extractor"] is not None:
            return extractor_holder["extractor"]
        try:
            from trustcall import create_extractor  # type: ignore
        except Exception as e:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "trustcall is required for submit fallback. Install with `pip install trustcall`."
            ) from e
        _llm = llm or get_default_structuring_model()
        # Tool choice should match schema name
        extractor = create_extractor(
            _llm, tools=[schema], tool_choice=schema.__name__
        )
        extractor_holder["extractor"] = extractor
        return extractor

    def _validate_or_coerce(draft: str) -> BaseModel:
        # Fast path: try JSON validation first
        try:
            return schema.model_validate_json(draft)
        except Exception:
            # Fallback: Trustcall coercion
            extractor = _get_extractor()
            extraction = extractor.invoke({"input": draft})
            responses = extraction.get("responses") or []
            if not responses:
                raise RuntimeError("Trustcall returned no responses for submission.")
            return responses[0]

    def _submit(
        draft: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        try:
            obj = _validate_or_coerce(draft)
            submission_json = (
                obj.model_dump_json(indent=None) if isinstance(obj, BaseModel) else json.dumps(obj)
            )
            return Command(
                update={
                    "output_submitted": True,
                    "submission": submission_json,
                    "messages": [
                        ToolMessage(
                            "Submitted validated output.", tool_call_id=tool_call_id
                        )
                    ],
                }
            )
        except ValidationError as ve:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"Validation failed: {ve}", tool_call_id=tool_call_id
                        )
                    ]
                }
            )
        except Exception as e:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"Error during submission: {e}", tool_call_id=tool_call_id
                        )
                    ]
                }
            )

    return tool(_submit, name=tool_name, description=desc)
