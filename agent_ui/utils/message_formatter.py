"""
Message Formatter - Utilities for formatting and displaying chat messages.
Handles markdown rendering, code highlighting, tool call formatting, etc.
"""

from typing import Dict, Any, List, Optional
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


def format_message_content(content: str, role: str = "user") -> str:
    """
    Format message content with proper styling.

    Args:
        content: Message content
        role: Message role (user/ai/system)

    Returns:
        Formatted content string
    """
    if not content:
        return ""

    # For system messages, use smaller, italicized text
    if role == "system":
        return f"*{content}*"

    return content


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


def format_tool_calls_list(tool_calls: List[Dict]) -> str:
    """
    Format multiple tool calls for display.

    Args:
        tool_calls: List of tool call dictionaries

    Returns:
        Formatted string with all tool calls
    """
    if not tool_calls:
        return ""

    formatted_calls = []
    for idx, tool_call in enumerate(tool_calls, 1):
        formatted_calls.append(f"### Tool Call {idx}")
        formatted_calls.append(format_tool_call(tool_call))

    return "\n\n".join(formatted_calls)


def get_message_style(role: str) -> str:
    """
    Get CSS style class for message role.

    Args:
        role: Message role (user/ai/system)

    Returns:
        CSS class name
    """
    style_map = {
        'user': 'user-message',
        'ai': 'ai-message',
        'assistant': 'ai-message',
        'system': 'system-message'
    }

    return style_map.get(role, 'default-message')


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


def extract_code_blocks(content: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from message content.

    Args:
        content: Message content with potential code blocks

    Returns:
        List of code blocks with language and code
    """
    import re

    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)

    code_blocks = []
    for lang, code in matches:
        code_blocks.append({
            'language': lang or 'text',
            'code': code.strip()
        })

    return code_blocks


def format_metadata(metadata: Dict[str, Any]) -> str:
    """
    Format message metadata for display.

    Args:
        metadata: Metadata dictionary (token usage, costs, etc.)

    Returns:
        Formatted metadata string
    """
    if not metadata:
        return ""

    formatted_items = []

    # Token usage
    if 'token_usage' in metadata:
        usage = metadata['token_usage']
        total = usage.get('total_tokens', 0)
        prompt = usage.get('prompt_tokens', 0)
        completion = usage.get('completion_tokens', 0)

        formatted_items.append(f"**Tokens:** {total:,} (prompt: {prompt:,}, completion: {completion:,})")

    # Cost
    if 'cost' in metadata:
        cost = metadata['cost']
        formatted_items.append(f"**Cost:** ${cost:.4f}")

    # Execution time
    if 'execution_time' in metadata:
        exec_time = metadata['execution_time']
        formatted_items.append(f"**Time:** {exec_time:.2f}s")

    return " | ".join(formatted_items) if formatted_items else ""


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


def count_messages_by_role(messages: List[Dict]) -> Dict[str, int]:
    """
    Count messages by role.

    Args:
        messages: List of message dictionaries

    Returns:
        Dictionary of role -> count
    """
    counts = {}

    for message in messages:
        role = message.get('role', 'unknown')
        counts[role] = counts.get(role, 0) + 1

    return counts


def get_conversation_summary(messages: List[Dict]) -> str:
    """
    Get a summary of the conversation.

    Args:
        messages: List of message dictionaries

    Returns:
        Summary string
    """
    if not messages:
        return "No messages"

    counts = count_messages_by_role(messages)

    user_count = counts.get('user', 0)
    ai_count = counts.get('ai', 0) + counts.get('assistant', 0)

    summary = f"{len(messages)} messages: {user_count} from you, {ai_count} from agent"

    # Add first message preview
    if messages:
        first_msg = messages[0]
        first_content = truncate_message(first_msg.get('content', ''), 50)
        summary += f"\nStarts with: {first_content}"

    return summary
