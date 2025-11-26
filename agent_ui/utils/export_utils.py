"""
Export Utilities - Export conversations to various formats.
Supports JSON, Markdown, and CSV exports.
"""

import json
from typing import Dict, Any, List
from datetime import datetime
import csv
from io import StringIO


def export_to_json(
    messages: List[Dict],
    thread_id: str = None,
    agent_info: Dict = None,
    metadata: Dict = None
) -> str:
    """
    Export conversation to JSON format.

    Args:
        messages: List of message dictionaries
        thread_id: Optional thread ID
        agent_info: Optional agent information
        metadata: Optional metadata

    Returns:
        JSON string
    """
    export_data = {
        'exported_at': datetime.now().isoformat(),
        'thread_id': thread_id,
        'agent': agent_info,
        'messages': messages,
        'metadata': metadata or {}
    }

    return json.dumps(export_data, indent=2)


def export_to_markdown(
    messages: List[Dict],
    thread_id: str = None,
    agent_info: Dict = None
) -> str:
    """
    Export conversation to Markdown format.

    Args:
        messages: List of message dictionaries
        thread_id: Optional thread ID
        agent_info: Optional agent information

    Returns:
        Markdown string
    """
    lines = []

    # Header
    lines.append("# Conversation Export")
    lines.append("")

    if agent_info:
        lines.append(f"**Agent:** {agent_info.get('name', 'Unknown')}")
        if agent_info.get('description'):
            lines.append(f"**Description:** {agent_info['description']}")
        lines.append("")

    if thread_id:
        lines.append(f"**Thread ID:** `{thread_id}`")
        lines.append("")

    lines.append(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Messages
    for idx, message in enumerate(messages, 1):
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        timestamp = message.get('timestamp', '')

        # Role header
        role_display = {
            'user': '👤 User',
            'ai': '🤖 Agent',
            'assistant': '🤖 Agent',
            'system': '⚙️ System'
        }.get(role, role.title())

        lines.append(f"## Message {idx}: {role_display}")

        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                lines.append(f"*{dt.strftime('%Y-%m-%d %H:%M:%S')}*")
            except:
                pass

        lines.append("")
        lines.append(content)
        lines.append("")

        # Tool calls
        if 'tool_calls' in message:
            lines.append("### Tool Calls")
            lines.append("")

            for tool_call in message['tool_calls']:
                tool_name = tool_call.get('name', 'unknown')
                tool_args = tool_call.get('args', {})
                tool_result = tool_call.get('result')

                lines.append(f"**Tool:** `{tool_name}`")
                lines.append("")
                lines.append("**Arguments:**")
                lines.append("```json")
                lines.append(json.dumps(tool_args, indent=2))
                lines.append("```")
                lines.append("")

                if tool_result is not None:
                    lines.append("**Result:**")
                    lines.append("```")
                    lines.append(str(tool_result))
                    lines.append("```")
                    lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def export_to_csv(messages: List[Dict]) -> str:
    """
    Export conversation to CSV format.

    Args:
        messages: List of message dictionaries

    Returns:
        CSV string
    """
    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(['Index', 'Timestamp', 'Role', 'Content', 'Has Tool Calls', 'Tool Count'])

    # Rows
    for idx, message in enumerate(messages, 1):
        timestamp = message.get('timestamp', '')
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        tool_calls = message.get('tool_calls', [])

        # Clean content for CSV (remove newlines)
        content_cleaned = content.replace('\n', ' ').replace('\r', '')

        writer.writerow([
            idx,
            timestamp,
            role,
            content_cleaned,
            'Yes' if tool_calls else 'No',
            len(tool_calls)
        ])

    return output.getvalue()


def export_all_threads_to_json(threads: Dict[str, Dict], agent_info: Dict = None) -> str:
    """
    Export all threads to a single JSON file.

    Args:
        threads: Dictionary of thread_id -> thread_data
        agent_info: Optional agent information

    Returns:
        JSON string
    """
    export_data = {
        'exported_at': datetime.now().isoformat(),
        'agent': agent_info,
        'threads': threads,
        'thread_count': len(threads)
    }

    return json.dumps(export_data, indent=2)


def import_from_json(json_str: str) -> Dict[str, Any]:
    """
    Import conversation from JSON string.

    Args:
        json_str: JSON string

    Returns:
        Dictionary with imported data

    Raises:
        ValueError: If JSON is invalid
    """
    try:
        data = json.loads(json_str)

        # Validate structure
        if 'messages' not in data:
            raise ValueError("Invalid format: missing 'messages' field")

        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")


def get_export_filename(
    format: str,
    thread_id: str = None,
    agent_name: str = None
) -> str:
    """
    Generate a filename for export.

    Args:
        format: Export format (json/md/csv)
        thread_id: Optional thread ID
        agent_name: Optional agent name

    Returns:
        Filename string
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    parts = ['conversation']

    if agent_name:
        # Clean agent name for filename
        clean_name = agent_name.replace(' ', '_').replace('/', '_')
        parts.append(clean_name)

    if thread_id:
        # Use shortened thread ID
        short_id = thread_id[:8]
        parts.append(short_id)

    parts.append(timestamp)

    filename = '_'.join(parts)

    extension = {
        'json': 'json',
        'markdown': 'md',
        'csv': 'csv'
    }.get(format, 'txt')

    return f"{filename}.{extension}"


def calculate_export_stats(messages: List[Dict]) -> Dict[str, Any]:
    """
    Calculate statistics for export summary.

    Args:
        messages: List of message dictionaries

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_messages': len(messages),
        'user_messages': 0,
        'ai_messages': 0,
        'system_messages': 0,
        'total_tool_calls': 0,
        'total_tokens': 0,
        'total_cost': 0.0
    }

    for message in messages:
        role = message.get('role', 'unknown')

        if role == 'user':
            stats['user_messages'] += 1
        elif role in ['ai', 'assistant']:
            stats['ai_messages'] += 1
        elif role == 'system':
            stats['system_messages'] += 1

        # Count tool calls
        tool_calls = message.get('tool_calls', [])
        stats['total_tool_calls'] += len(tool_calls)

        # Sum tokens and cost if available
        metadata = message.get('metadata', {})
        if 'token_usage' in metadata:
            stats['total_tokens'] += metadata['token_usage'].get('total_tokens', 0)
        if 'cost' in metadata:
            stats['total_cost'] += metadata['cost']

    return stats
