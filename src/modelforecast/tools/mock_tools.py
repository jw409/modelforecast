"""Mock tool schemas for testing probe levels."""

from typing import Any


def get_search_tool_basic() -> dict[str, Any]:
    """T0 Invoke: Basic search tool with only query parameter."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for files in the codebase",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    }
                },
                "required": ["query"],
            },
        },
    }


def get_search_tool_with_limit() -> dict[str, Any]:
    """T1 Schema: Search tool with query (string) and limit (integer) parameters."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for files in the codebase",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return",
                    },
                },
                "required": ["query"],
            },
        },
    }


def get_read_file_tool() -> dict[str, Any]:
    """Tool for reading file contents."""
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a specific file's contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file",
                    }
                },
                "required": ["path"],
            },
        },
    }


def get_list_directory_tool() -> dict[str, Any]:
    """Tool for listing directory contents."""
    return {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path",
                    }
                },
                "required": ["path"],
            },
        },
    }


def get_multi_tool_set() -> list[dict[str, Any]]:
    """T2 Selection: Multiple tools for selection testing."""
    return [
        get_search_tool_basic(),
        get_read_file_tool(),
        get_list_directory_tool(),
    ]
