"""
Message Formatter - Utilities for formatting and displaying chat messages.
Handles markdown rendering, code highlighting, tool call formatting, etc.
"""

from typing import Dict, Any, List
import json
from datetime import datetime


def format_timestamp(iso_timestamp: str) -> str:
    """
    Format ISO timestamp to readable string.

    Args:
        iso_timestamp: ISO format timestamp string

    Returns:
        Formatted timestamp (e.g., "2:30 PM")
    """
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%I:%M %p")
    except:
        return ""


def format_tool_call(tool_call: Dict[str, Any]) -> str:
    """
    Format a tool call for display.

    Args:
        tool_call: Tool call dictionary with id, name, args, result

    Returns:
        Formatted tool call string
    """
    tool_name = tool_call.get('name', 'unknown')
    tool_args = tool_call.get('args', {})
    tool_result = tool_call.get('result')

    # Format args as JSON
    args_str = json.dumps(tool_args, indent=2) if tool_args else "{}"

    # Build formatted string
    formatted = f"**🔧 Tool: {tool_name}**\n\n"
    formatted += f"**Arguments:**\n```json\n{args_str}\n```\n\n"

    if tool_result is not None:
        result_str = str(tool_result)
        # Limit result length for display
        if len(result_str) > 500:
            result_str = result_str[:500] + "..."

        formatted += f"**Result:**\n```\n{result_str}\n```"
    else:
        formatted += "*No result yet*"

    return formatted


def get_role_display_name(role: str) -> str:
    """
    Get display name for message role.

    Args:
        role: Message role

    Returns:
        Display name
    """
    name_map = {
        'user': '👤 You',
        'ai': '🤖 Agent',
        'assistant': '🤖 Agent',
        'system': '⚙️ System'
    }

    return name_map.get(role, role.title())


def truncate_message(content: str, max_length: int = 100) -> str:
    """
    Truncate message content for preview.

    Args:
        content: Message content
        max_length: Maximum length

    Returns:
        Truncated content
    """
    if len(content) <= max_length:
        return content

    return content[:max_length] + "..."
