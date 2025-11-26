"""
Session state management for the agent builder UI.
Handles initialization, updates, and retrieval of form data across all pages.
"""

import streamlit as st
from typing import Any, Dict, Optional
from datetime import datetime

from .constants import DEFAULTS


def initialize_session_state():
    """Initialize all session state variables for the app."""

    # Template selection
    if 'template_selected' not in st.session_state:
        st.session_state.template_selected = False

    if 'template_name' not in st.session_state:
        st.session_state.template_name = None

    # API configuration
    if 'api_base_url' not in st.session_state:
        st.session_state.api_base_url = "http://localhost:8000"

    if 'api_available' not in st.session_state:
        st.session_state.api_available = False

    # Page 1: Basic Info
    if 'page_1_data' not in st.session_state:
        st.session_state.page_1_data = {
            'name': '',
            'description': '',
            'version': DEFAULTS['version'],
            'tags': []
        }

    # Page 2: LLM Config
    if 'page_2_data' not in st.session_state:
        st.session_state.page_2_data = {
            'provider': 'openai',
            'model': 'gpt-4o',
            'temperature': DEFAULTS['temperature'],
            'max_tokens': DEFAULTS['max_tokens'],
            'top_p': None,
            'api_key': None
        }

    # Page 3: Tools
    if 'page_3_data' not in st.session_state:
        st.session_state.page_3_data = {
            'tools': [],
            'custom_tools': [],
            'mcp_servers': []
        }

    # Page 4: Prompts
    if 'page_4_data' not in st.session_state:
        st.session_state.page_4_data = {
            'system_prompt': '',
            'user_template': None,
            'few_shot_examples': []
        }

    # Page 5: Memory
    if 'page_5_data' not in st.session_state:
        st.session_state.page_5_data = {
            'short_term': {
                'enabled': False,
                'type': 'sqlite',
                'path': None,
                'custom_state': {},
                'message_management': 'none'
            },
            'long_term': {
                'enabled': False,
                'type': 'sqlite',
                'path': None,
                'namespaces': [],
                'enable_vector_search': False
            }
        }

    # Page 6: Middleware
    if 'page_6_data' not in st.session_state:
        st.session_state.page_6_data = {
            'middleware': []
        }

    # Page 7: Advanced
    if 'page_7_data' not in st.session_state:
        st.session_state.page_7_data = {
            'streaming': {
                'enabled': DEFAULTS['streaming_enabled'],
                'modes': DEFAULTS['streaming_modes']
            },
            'runtime': {
                'context_schema': []
            },
            'output_formatter': {
                'enabled': False,
                'schema_description': None,
                'pydantic_model': None
            }
        }

    # Page completion tracking
    for i in range(1, 9):
        key = f'page_{i}_complete'
        if key not in st.session_state:
            st.session_state[key] = False

    # Validation errors
    if 'validation_errors' not in st.session_state:
        st.session_state.validation_errors = {}

    # YAML preview
    if 'yaml_preview' not in st.session_state:
        st.session_state.yaml_preview = ''

    # Current page tracking
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1


def get_page_data(page_number: int) -> Dict[str, Any]:
    """
    Get data for a specific page.

    Args:
        page_number: Page number (1-8)

    Returns:
        Dictionary containing page data
    """
    key = f'page_{page_number}_data'
    return st.session_state.get(key, {})


def update_page_data(page_number: int, data: Dict[str, Any]):
    """
    Update page data and trigger YAML regeneration.

    Args:
        page_number: Page number (1-8)
        data: New data dictionary
    """
    key = f'page_{page_number}_data'
    st.session_state[key] = data

    # Regenerate YAML preview
    from .yaml_generator import generate_agent_yaml
    st.session_state.yaml_preview = generate_agent_yaml()


def mark_page_complete(page_number: int, complete: bool = True):
    """
    Mark a page as complete or incomplete.

    Args:
        page_number: Page number (1-8)
        complete: Whether the page is complete
    """
    key = f'page_{page_number}_complete'
    st.session_state[key] = complete


def is_page_complete(page_number: int) -> bool:
    """
    Check if a page is marked as complete.

    Args:
        page_number: Page number (1-8)

    Returns:
        True if page is complete
    """
    key = f'page_{page_number}_complete'
    return st.session_state.get(key, False)


def get_completion_count() -> int:
    """
    Get the number of completed pages.

    Returns:
        Count of completed pages
    """
    return sum(1 for i in range(1, 9) if is_page_complete(i))


def reset_all_state():
    """Reset all session state to initial values."""
    keys_to_keep = ['api_base_url']

    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

    initialize_session_state()


def load_template_data(template_config: Dict[str, Any]):
    """
    Load template configuration into session state.

    Args:
        template_config: Template configuration dictionary
    """
    # Page 1: Basic Info
    if 'name' in template_config:
        st.session_state.page_1_data['name'] = template_config['name']
    if 'description' in template_config:
        st.session_state.page_1_data['description'] = template_config['description']
    if 'version' in template_config:
        st.session_state.page_1_data['version'] = template_config['version']
    if 'tags' in template_config:
        st.session_state.page_1_data['tags'] = template_config['tags']

    # Page 2: LLM Config
    if 'llm' in template_config:
        llm = template_config['llm']
        if 'provider' in llm:
            st.session_state.page_2_data['provider'] = llm['provider']
        if 'model' in llm:
            st.session_state.page_2_data['model'] = llm['model']
        if 'temperature' in llm:
            st.session_state.page_2_data['temperature'] = llm['temperature']
        if 'max_tokens' in llm:
            st.session_state.page_2_data['max_tokens'] = llm['max_tokens']
        if 'top_p' in llm:
            st.session_state.page_2_data['top_p'] = llm['top_p']

    # Page 3: Tools
    if 'tools' in template_config:
        st.session_state.page_3_data['tools'] = template_config['tools']
    if 'mcp_servers' in template_config:
        st.session_state.page_3_data['mcp_servers'] = template_config['mcp_servers']

    # Page 4: Prompts
    if 'prompts' in template_config:
        prompts = template_config['prompts']
        if 'system' in prompts:
            st.session_state.page_4_data['system_prompt'] = prompts['system']
        if 'user_template' in prompts:
            st.session_state.page_4_data['user_template'] = prompts['user_template']
        if 'few_shot_examples' in prompts:
            st.session_state.page_4_data['few_shot_examples'] = prompts['few_shot_examples']

    # Page 5: Memory
    if 'memory' in template_config:
        memory = template_config['memory']
        if 'short_term' in memory:
            st.session_state.page_5_data['short_term'] = {
                'enabled': True,
                **memory['short_term']
            }
        if 'long_term' in memory:
            st.session_state.page_5_data['long_term'] = {
                'enabled': True,
                **memory['long_term']
            }

    # Page 6: Middleware
    if 'middleware' in template_config:
        st.session_state.page_6_data['middleware'] = template_config['middleware']

    # Page 7: Advanced
    if 'streaming' in template_config:
        st.session_state.page_7_data['streaming'] = template_config['streaming']
    if 'runtime' in template_config:
        st.session_state.page_7_data['runtime'] = template_config['runtime']

    # Mark template as selected
    st.session_state.template_selected = True

    # Regenerate YAML
    from .yaml_generator import generate_agent_yaml
    st.session_state.yaml_preview = generate_agent_yaml()


def get_all_data() -> Dict[str, Any]:
    """
    Get all page data combined.

    Returns:
        Dictionary with all configuration data
    """
    return {
        'basic_info': st.session_state.page_1_data,
        'llm': st.session_state.page_2_data,
        'tools': st.session_state.page_3_data,
        'prompts': st.session_state.page_4_data,
        'memory': st.session_state.page_5_data,
        'middleware': st.session_state.page_6_data,
        'advanced': st.session_state.page_7_data
    }


def set_validation_errors(page_number: int, errors: list):
    """
    Set validation errors for a specific page.

    Args:
        page_number: Page number (1-8)
        errors: List of ValidationError objects
    """
    key = f'page_{page_number}_errors'
    st.session_state.validation_errors[key] = errors


def get_validation_errors(page_number: int) -> list:
    """
    Get validation errors for a specific page.

    Args:
        page_number: Page number (1-8)

    Returns:
        List of ValidationError objects
    """
    key = f'page_{page_number}_errors'
    return st.session_state.validation_errors.get(key, [])


def clear_validation_errors(page_number: int):
    """
    Clear validation errors for a specific page.

    Args:
        page_number: Page number (1-8)
    """
    key = f'page_{page_number}_errors'
    if key in st.session_state.validation_errors:
        del st.session_state.validation_errors[key]


# ==================== Chat State Management ====================

def initialize_chat_state():
    """Initialize chat-specific session state variables."""

    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    if 'chat_thread_id' not in st.session_state:
        st.session_state.chat_thread_id = None

    if 'chat_waiting' not in st.session_state:
        st.session_state.chat_waiting = False

    if 'chat_streaming_enabled' not in st.session_state:
        st.session_state.chat_streaming_enabled = False

    if 'test_agent_id' not in st.session_state:
        st.session_state.test_agent_id = None

    if 'chat_context_values' not in st.session_state:
        st.session_state.chat_context_values = {}


def add_chat_message(role: str, content: str, tool_calls: Optional[list] = None):
    """
    Add a message to the chat.

    Args:
        role: Message role (user, ai, assistant)
        content: Message content
        tool_calls: Optional list of tool calls
    """
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.utcnow().isoformat()
    }

    if tool_calls:
        message['tool_calls'] = tool_calls

    st.session_state.chat_messages.append(message)


def clear_chat_messages():
    """Clear all chat messages and reset thread."""
    st.session_state.chat_messages = []
    st.session_state.chat_thread_id = None


def get_chat_messages() -> list:
    """
    Get all chat messages.

    Returns:
        List of message dictionaries
    """
    return st.session_state.get('chat_messages', [])


def set_test_agent(agent_id: str):
    """
    Set the agent being tested.

    Args:
        agent_id: Agent identifier
    """
    st.session_state.test_agent_id = agent_id
    # Clear previous chat when switching agents
    clear_chat_messages()


def get_test_agent() -> Optional[str]:
    """
    Get the current test agent ID.

    Returns:
        Agent ID or None
    """
    return st.session_state.get('test_agent_id')
