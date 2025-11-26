"""
Session State Manager for Agent UI.
Manages chat sessions, threads, and UI state across pages.
"""

import streamlit as st
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid


def initialize_session_state():
    """Initialize all session state variables for the Agent UI."""

    # API Configuration
    if 'api_base_url' not in st.session_state:
        st.session_state.api_base_url = "http://localhost:8000"

    if 'api_available' not in st.session_state:
        st.session_state.api_available = False

    # Agent Selection
    if 'selected_agent_id' not in st.session_state:
        st.session_state.selected_agent_id = None

    if 'selected_agent_info' not in st.session_state:
        st.session_state.selected_agent_info = None

    if 'selected_agent_config' not in st.session_state:
        st.session_state.selected_agent_config = None

    if 'agent_selected' not in st.session_state:
        st.session_state.agent_selected = False

    # Current Chat State
    if 'current_thread_id' not in st.session_state:
        st.session_state.current_thread_id = None

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Thread Management
    if 'threads' not in st.session_state:
        st.session_state.threads = {}  # thread_id -> {created, messages, metadata, label}

    # UI State
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = False

    if 'waiting_for_response' not in st.session_state:
        st.session_state.waiting_for_response = False

    if 'error_message' not in st.session_state:
        st.session_state.error_message = None

    if 'streaming_enabled' not in st.session_state:
        st.session_state.streaming_enabled = False

    # Context Editor
    if 'context_values' not in st.session_state:
        st.session_state.context_values = {}

    # UI Preferences
    if 'ui_preferences' not in st.session_state:
        st.session_state.ui_preferences = {
            'theme': 'light',
            'message_font_size': 'medium',
            'auto_scroll': True,
            'sound_notifications': False,
            'show_timestamps': True,
            'show_token_usage': True
        }

    # Settings
    if 'auto_deploy_agents' not in st.session_state:
        st.session_state.auto_deploy_agents = True

    if 'default_streaming_mode' not in st.session_state:
        st.session_state.default_streaming_mode = False

    # Reconfiguration State
    if 'reconfigure_dialog_open' not in st.session_state:
        st.session_state.reconfigure_dialog_open = False

    if 'reconfigure_pending_changes' not in st.session_state:
        st.session_state.reconfigure_pending_changes = {}

    if 'reconfigure_result' not in st.session_state:
        st.session_state.reconfigure_result = None

    if 'reconfigure_show_preview' not in st.session_state:
        st.session_state.reconfigure_show_preview = False


def select_agent(agent_id: str, agent_info: Dict, agent_config: Dict):
    """
    Select an agent and store its information in session state.

    Args:
        agent_id: Agent identifier
        agent_info: Agent metadata
        agent_config: Full agent configuration
    """
    st.session_state.selected_agent_id = agent_id
    st.session_state.selected_agent_info = agent_info
    st.session_state.selected_agent_config = agent_config
    st.session_state.agent_selected = True

    # Set streaming enabled based on agent config
    streaming_config = agent_config.get('streaming') or {}
    st.session_state.streaming_enabled = streaming_config.get('enabled', False)


def create_new_thread() -> str:
    """
    Create a new thread and return its ID.

    Returns:
        New thread ID
    """
    thread_id = str(uuid.uuid4())

    st.session_state.threads[thread_id] = {
        'created': datetime.now().isoformat(),
        'messages': [],
        'metadata': {},
        'label': f"Thread {len(st.session_state.threads) + 1}",
        'last_updated': datetime.now().isoformat()
    }

    st.session_state.current_thread_id = thread_id
    st.session_state.messages = []

    return thread_id


def switch_thread(thread_id: str):
    """
    Switch to a different thread.

    Args:
        thread_id: Thread ID to switch to
    """
    if thread_id in st.session_state.threads:
        st.session_state.current_thread_id = thread_id
        st.session_state.messages = st.session_state.threads[thread_id]['messages'].copy()
    else:
        st.error(f"Thread {thread_id} not found")


def save_current_thread():
    """Save current messages to the current thread."""
    if st.session_state.current_thread_id:
        thread_id = st.session_state.current_thread_id

        if thread_id not in st.session_state.threads:
            st.session_state.threads[thread_id] = {
                'created': datetime.now().isoformat(),
                'messages': [],
                'metadata': {},
                'label': f"Thread {len(st.session_state.threads) + 1}",
                'last_updated': datetime.now().isoformat()
            }

        st.session_state.threads[thread_id]['messages'] = st.session_state.messages.copy()
        st.session_state.threads[thread_id]['last_updated'] = datetime.now().isoformat()


def delete_thread(thread_id: str):
    """
    Delete a thread.

    Args:
        thread_id: Thread ID to delete
    """
    if thread_id in st.session_state.threads:
        del st.session_state.threads[thread_id]

        # If deleting current thread, create a new one
        if st.session_state.current_thread_id == thread_id:
            st.session_state.current_thread_id = None
            st.session_state.messages = []


def rename_thread(thread_id: str, new_label: str):
    """
    Rename a thread.

    Args:
        thread_id: Thread ID to rename
        new_label: New label for the thread
    """
    if thread_id in st.session_state.threads:
        st.session_state.threads[thread_id]['label'] = new_label


def get_thread_info(thread_id: str) -> Optional[Dict]:
    """
    Get thread information.

    Args:
        thread_id: Thread ID

    Returns:
        Thread info dict or None if not found
    """
    return st.session_state.threads.get(thread_id)


def get_all_threads() -> Dict[str, Dict]:
    """
    Get all threads.

    Returns:
        Dictionary of thread_id -> thread_info
    """
    return st.session_state.threads


def add_message(
    role: str,
    content: str,
    tool_calls: Optional[List] = None,
    message_id: Optional[str] = None
):
    """
    Add a message to the current conversation with deduplication.

    Args:
        role: Message role (user/ai/system)
        content: Message content
        tool_calls: Optional tool calls list
        message_id: Optional unique message ID for deduplication
    """
    import logging
    logger = logging.getLogger(__name__)

    # Generate ID if not provided
    if not message_id:
        message_id = str(uuid.uuid4())

    # Check for duplicate by ID
    existing_ids = {
        msg.get('id') for msg in st.session_state.messages
        if msg.get('id')
    }

    if message_id in existing_ids:
        logger.debug(f"Skipping duplicate message by ID: {message_id}")
        return

    # Check for content-based duplicates in recent messages (last 10)
    # This catches edge cases where ID wasn't set
    recent_messages = (
        st.session_state.messages[-10:]
        if len(st.session_state.messages) > 10
        else st.session_state.messages
    )

    for existing_msg in recent_messages:
        if (existing_msg.get('role') == role and
            existing_msg.get('content') == content and
            content):  # Don't dedupe empty content
            logger.debug(f"Skipping content-duplicate message for role={role}")
            return

    message = {
        'id': message_id,
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat()
    }

    if tool_calls:
        message['tool_calls'] = tool_calls

    st.session_state.messages.append(message)


def clear_messages():
    """Clear all messages in current conversation."""
    st.session_state.messages = []


def update_context(key: str, value: Any):
    """
    Update a context value.

    Args:
        key: Context key
        value: Context value
    """
    st.session_state.context_values[key] = value


def get_context_values() -> Dict[str, Any]:
    """
    Get all context values.

    Returns:
        Dictionary of context values
    """
    return st.session_state.context_values


def reset_chat():
    """Reset chat state (useful when switching agents)."""
    st.session_state.current_thread_id = None
    st.session_state.messages = []
    st.session_state.input_disabled = False
    st.session_state.waiting_for_response = False
    st.session_state.error_message = None


def reset_all_state():
    """Reset all session state (nuclear option)."""
    keys_to_keep = ['api_base_url']

    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

    initialize_session_state()


def update_preference(key: str, value: Any):
    """
    Update a UI preference.

    Args:
        key: Preference key
        value: Preference value
    """
    if 'ui_preferences' not in st.session_state:
        st.session_state.ui_preferences = {}

    st.session_state.ui_preferences[key] = value


def get_preference(key: str, default: Any = None) -> Any:
    """
    Get a UI preference.

    Args:
        key: Preference key
        default: Default value if not found

    Returns:
        Preference value
    """
    return st.session_state.ui_preferences.get(key, default)


def reload_agent_config(agent_id: str):
    """
    Reload agent configuration from API after reconfiguration.

    This is called after successful agent reconfiguration to ensure
    the UI displays the latest configuration.

    Args:
        agent_id: Agent identifier
    """
    from utils.api_client import get_api_client

    client = get_api_client()
    result = client.get_agent_details(agent_id)

    if result.get('success'):
        data = result.get('data', {})
        st.session_state.selected_agent_config = data

        # Update streaming enabled based on new config
        agent_config = data.get('config', {})
        streaming_config = agent_config.get('streaming') or {}
        st.session_state.streaming_enabled = streaming_config.get('enabled', False)

        return True
    else:
        return False


def reset_reconfigure_state():
    """
    Reset reconfiguration-related session state.

    Call this to clean up reconfiguration state when closing the dialog
    or switching agents.
    """
    st.session_state.reconfigure_dialog_open = False
    st.session_state.reconfigure_pending_changes = {}
    st.session_state.reconfigure_result = None
    st.session_state.reconfigure_show_preview = False
