from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing import Annotated, Optional, Type, Union, Dict, Any, List
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, ValidationError, create_model, ConfigDict, TypeAdapter
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
    schema: Union[Type[BaseModel], Dict[str, Any]],
    llm: Optional[LanguageModelLike] = None,
    tool_name: str = "submit",
    description: Optional[str] = None,
):
    """Create a submit tool that validates output against a Pydantic schema.

    Fast path: validate provided draft as JSON against the schema.
    Fallback: use Trustcall to coerce arbitrary text/JSON into the schema.

    Updates DeepAgentState: stores compact JSON string in `submission`.
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
        model_type, tool_choice_name = _ensure_model(schema)
        # Minor debug output for visibility when Trustcall is used
        print("[submit] Trustcall: initializing extractor…")
        # Support multiple Trustcall versions: prefer explicit tool_choice; fallback if unsupported
        try:
            extractor = create_extractor(
                _llm, tools=[model_type], tool_choice=tool_choice_name
            )
        except TypeError:
            # Older Trustcall may not support tool_choice; fall back to minimal signature
            extractor = create_extractor(_llm, tools=[model_type])
        extractor_holder["extractor"] = extractor
        return extractor

    def _adapter_for_schema(schema_or_dict: Union[Type[BaseModel], Dict[str, Any]]):
        """Return a TypeAdapter for root array schemas; otherwise None."""
        if isinstance(schema_or_dict, dict) and schema_or_dict.get("type") == "array":
            # Minimal support: accept any list; stronger typing would recurse on items
            return TypeAdapter(List[Any])
        return None

    def _json_schema_type_to_annotation(spec: Dict[str, Any], parent_name: str = "Item") -> Any:
        """Map a JSON Schema fragment to a Python type annotation (shallow but useful)."""
        t = spec.get("type")
        # Handle union types like ["string", "null"]
        if isinstance(t, list):
            non_null = [x for x in t if x != "null"]
            if not non_null:
                return Any
            primary = non_null[0]
            ann = _json_schema_type_to_annotation({"type": primary, **{k: v for k, v in spec.items() if k != "type"}}, parent_name)
            return Optional[ann]

        if t == "string":
            return str
        if t == "number":
            return float
        if t == "integer":
            return int
        if t == "boolean":
            return bool
        if t == "array":
            items = spec.get("items")
            item_type = _json_schema_type_to_annotation(items, parent_name + "Item") if isinstance(items, dict) else Any
            return List[item_type]
        if t == "object":
            # Nested object; recurse to build a nested model if properties exist
            props = spec.get("properties")
            if isinstance(props, dict):
                nested_model, _ = _build_model_from_schema(spec, name=parent_name.capitalize())
                return nested_model
            return Dict[str, Any]
        # Fallback
        return Any

    def _build_model_from_schema(schema_dict: Dict[str, Any], name: str) -> tuple[Type[BaseModel], str]:
        properties: Dict[str, Any] = schema_dict.get("properties", {})
        required = set(schema_dict.get("required", []))
        field_defs = {}
        for field_name, _spec in properties.items():
            ann = _json_schema_type_to_annotation(_spec, parent_name=field_name)
            default = ... if field_name in required else _spec.get("default", None)
            field_defs[field_name] = (ann, default)

        class _Base(BaseModel):
            model_config = ConfigDict(extra="forbid")

        dynamic_model: Type[BaseModel] = create_model(name, __base__=_Base, **field_defs)  # type: ignore
        return dynamic_model, name

    def _ensure_model(schema_or_dict: Union[Type[BaseModel], Dict[str, Any]]) -> tuple[Type[BaseModel], str]:
        if isinstance(schema_or_dict, type) and issubclass(schema_or_dict, BaseModel):
            return schema_or_dict, schema_or_dict.__name__
        if isinstance(schema_or_dict, dict):
            # Only build a model for object-like schemas; root-array handled via TypeAdapter in validator
            name = schema_or_dict.get("title") or "Submission"
            if schema_or_dict.get("type") == "object" or "properties" in schema_or_dict:
                return _build_model_from_schema(schema_or_dict, name=name)
            # Fallback: no properties, treat as free-form object
            return _build_model_from_schema({"properties": {}}, name=name)
        raise TypeError("submit_schema must be a Pydantic BaseModel subclass or a JSON-schema-like dict")

    def _validate_or_coerce(draft: Any) -> BaseModel | Dict[str, Any]:
        """Validate the draft against the schema, coercing only if needed.

        Accepts:
        - Pydantic model instance (returned as-is)
        - dict/list (validated via model_validate)
        - str (validated via model_validate_json; falls back to Trustcall on failure)
        """
        # Root array support via TypeAdapter
        adapter = _adapter_for_schema(schema)
        if adapter is not None:
            if isinstance(draft, str):
                return adapter.validate_json(draft)
            return adapter.validate_python(draft)

        model_type, _ = _ensure_model(schema)

        # Pydantic model provided
        if isinstance(draft, BaseModel):
            return draft

        # Native Python structure provided
        if isinstance(draft, (dict, list)):
            return model_type.model_validate(draft)

        # String input: attempt JSON fast-path
        if isinstance(draft, str):
            try:
                return model_type.model_validate_json(draft)
            except Exception:
                # Fallback: Trustcall coercion
                print("[submit] Trustcall: attempting coercion for string draft…")
                extractor = _get_extractor()
                extraction = extractor.invoke({"input": draft})
                responses = extraction.get("responses") or []
                if not responses:
                    raise RuntimeError("Trustcall returned no responses for submission.")
                print("[submit] Trustcall: coercion produced a response. Validating with Pydantic…")
                # Extra safety: validate the coerced structure against the schema
                return model_type.model_validate(responses[0])

        # Last resort: try coercion on stringified input
        print("[submit] Trustcall: attempting coercion for non-string draft…")
        extractor = _get_extractor()
        extraction = extractor.invoke({"input": json.dumps(draft)})
        responses = extraction.get("responses") or []
        if not responses:
            raise RuntimeError("Trustcall returned no responses for submission.")
        print("[submit] Trustcall: coercion produced a response. Validating with Pydantic…")
        # Extra safety: validate the coerced structure against the schema
        return model_type.model_validate(responses[0])

    class SubmitArgs(BaseModel):
        draft: str

    def _submit(
        draft: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        try:
            obj = _validate_or_coerce(draft)
            # Compact, unicode-safe JSON
            submission_json = (
                obj.model_dump_json() if isinstance(obj, BaseModel)
                else json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            )
            return Command(
                update={
                    "submission": submission_json,
                    "messages": [
                        ToolMessage(
                            "Submitted validated output.", tool_call_id=tool_call_id
                        )
                    ],
                }
            )
        except ValidationError as ve:
            # Short, actionable error message
            try:
                errs = ve.errors()
                mini = "; ".join([
                    f"{'.'.join(str(x) for x in e.get('loc', []))}: {e.get('msg', '')}"
                    for e in errs[:5]
                ])
            except Exception:
                mini = str(ve)
            mini = (mini[:297] + "...") if len(mini) > 300 else mini
            return Command(update={
                "messages": [ToolMessage(f"Validation failed: {mini}", tool_call_id=tool_call_id)]
            })
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

    # Assign the desired tool name via function __name__ (decorator infers name from func)
    _submit.__name__ = tool_name

    return tool(_submit, description=desc)
