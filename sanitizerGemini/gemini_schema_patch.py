"""
Recursive JSON-Schema sanitizer to make Alpaca MCP tool schemas
compatible with Gemini's function-calling schema (a constrained
subset of OpenAPI/JSON Schema).

Handles four incompatibility patterns found in the Alpaca MCP tool set:

1. anyOf: [{type: X}, {type: 'null'}]  ->  collapsed to {type: X}
   (Pydantic's way of expressing Optional[X]; Gemini has no anyOf/oneOf)

2. enum with integer/number values     ->  stringified enum + type: string
   (Gemini's proto expects enum members to be strings)

3. type as a list, e.g. ['string','null'] -> single string type
   (Gemini requires `type` to be a scalar)

4. additionalProperties: true (open-ended objects) -> converted to a
   plain string param (caller passes JSON; tool parses it internally),
   since Gemini can't represent free-form/open-ended objects.

Usage:
    from gemini_schema_patch import sanitize_tools_for_gemini
    ALPACATools = sanitize_tools_for_gemini(ALPACATools)
    llm_with_tools = llm.bind_tools(tools_list)
"""

import copy
import json
from typing import Any, Dict


def _unwrap_any_of_with_null(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collapse a 2-branch anyOf where one branch is {'type': 'null'}
    into just the non-null branch. Handles nested anyOf branches too
    (e.g. anyOf containing an object with additionalProperties).
    """
    any_of = schema.get("anyOf")
    if not isinstance(any_of, list):
        return schema

    non_null_branches = [b for b in any_of if b.get("type") != "null"]
    had_null = len(non_null_branches) < len(any_of)

    if len(non_null_branches) == 1 and had_null:
        # Merge: keep description/default/etc. from the parent level,
        # pull type/enum/items/properties/additionalProperties from
        # the surviving branch.
        merged = {k: v for k, v in schema.items() if k != "anyOf"}
        branch = non_null_branches[0]
        merged.update(branch)
        return merged

    if len(non_null_branches) > 1:
        # Multiple real (non-null) type options -- Gemini can't express
        # this either. Fall back to treating the field as a string;
        # keep original description so the model still knows the intent.
        merged = {k: v for k, v in schema.items() if k != "anyOf"}
        merged["type"] = "string"
        desc = merged.get("description", "")
        merged["description"] = (
            desc + " (Accepts multiple formats; pass as a string.)"
        ).strip()
        return merged

    return schema


def _normalize_type_list(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert type: ['string', 'null'] (or any list-valued type) into a
    single scalar type. Prefers the first non-null entry.
    """
    t = schema.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        schema = dict(schema)
        schema["type"] = non_null[0] if non_null else "string"
    return schema


def _stringify_numeric_enum(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    If `enum` is present and schema type is integer/number, convert the
    enum values to strings and set type to string, since Gemini expects
    string enum members.
    """
    enum_vals = schema.get("enum")
    if isinstance(enum_vals, list) and schema.get("type") in ("integer", "number"):
        schema = dict(schema)
        schema["enum"] = [str(v) for v in enum_vals]
        schema["type"] = "string"
        desc = schema.get("description", "")
        schema["description"] = (
            desc + " (Provide as a string, e.g. \"1\" not 1.)"
        ).strip()
    return schema


def _flatten_open_object(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gemini can't represent additionalProperties:true / schema-less objects.
    Convert such fields into a plain string param; the tool/agent should
    pass a JSON-encoded string and the underlying tool call parses it.
    """
    if schema.get("type") == "object" and schema.get("additionalProperties") is True:
        desc = schema.get("description", "")
        return {
            "type": "string",
            "description": (
                desc
                + " (Pass this as a JSON-encoded string, e.g. "
                "'{\"key\": \"value\"}', not a nested object.)"
            ).strip(),
        }
    return schema


def _sanitize_node(node: Any) -> Any:
    """Recursively walk a JSON-schema-like structure and patch it in place."""
    if isinstance(node, list):
        return [_sanitize_node(item) for item in node]

    if not isinstance(node, dict):
        return node

    schema = dict(node)

    # Order matters: unwrap anyOf first (may reveal nested object/array
    # branches), then normalize type lists, then enum/object fixes.
    schema = _unwrap_any_of_with_null(schema)
    schema = _normalize_type_list(schema)
    schema = _stringify_numeric_enum(schema)
    schema = _flatten_open_object(schema)

    # Recurse into standard JSON-schema containers.
    if "properties" in schema and isinstance(schema["properties"], dict):
        schema["properties"] = {
            k: _sanitize_node(v) for k, v in schema["properties"].items()
        }

    if "items" in schema:
        schema["items"] = _sanitize_node(schema["items"])

    # In case anyOf survived (e.g. 3+ branches handled above already,
    # but guard for anything unexpected) or array of items schemas.
    if "anyOf" in schema:
        # Should already be resolved by _unwrap_any_of_with_null, but
        # if not (edge case), drop it defensively to a string type.
        schema.pop("anyOf", None)
        schema.setdefault("type", "string")

    return schema


def sanitize_schema_for_gemini(args_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Public entry point: sanitize a single tool's args_schema dict."""
    return _sanitize_node(copy.deepcopy(args_schema))


def sanitize_tools_for_gemini(tools: list) -> list:
    """
    Given a list of LangChain StructuredTool instances (e.g. from
    get_alpaca_tools()), return a new list with each tool's args_schema
    patched to be Gemini-compatible. Mutates a copy, not the originals,
    so this is safe to call once at startup.

    Works whether args_schema is a plain dict or a Pydantic model class
    exposing .schema()/.model_json_schema() -- for MCP-derived tools it's
    typically already a dict, which is the common case here.
    """
    patched_tools = []
    for t in tools:
        schema = getattr(t, "args_schema", None)

        if isinstance(schema, dict):
            t.args_schema = sanitize_schema_for_gemini(schema)
        elif schema is not None and hasattr(schema, "model_json_schema"):
            # Pydantic v2 model class -- patch the JSON schema view.
            # NOTE: bind_tools may still call schema.schema() itself
            # depending on langchain version; if you hit this branch
            # and Gemini still errors, convert args_schema to a plain
            # dict instead of a BaseModel subclass.
            raw = schema.model_json_schema()
            t.args_schema = sanitize_schema_for_gemini(raw)

        patched_tools.append(t)

    return patched_tools


if __name__ == "__main__":
    # Quick smoke test against a few of the problematic shapes you pasted.
    sample = {
        "additionalProperties": False,
        "type": "object",
        "properties": {
            "qty": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "Number of shares.",
            },
            "max_options_trading_level": {
                "enum": [0, 1, 2, 3],
                "type": "integer",
                "description": "Options level.",
            },
            "limit_price": {
                "type": ["string", "null"],
                "description": "Locate fee limit.",
            },
            "advanced_instructions": {
                "anyOf": [
                    {"additionalProperties": True, "type": "object"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Elite router payload.",
            },
            "legs": {
                "anyOf": [
                    {
                        "items": {"additionalProperties": True, "type": "object"},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "Multi-leg order legs.",
            },
        },
        "required": ["qty"],
    }

    print(json.dumps(sanitize_schema_for_gemini(sample), indent=2))