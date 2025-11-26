"""
YAML generation from session state data.
"""

import yaml
import streamlit as st
from typing import Dict, Any


def generate_agent_yaml() -> str:
    """
    Generate complete agent YAML configuration from all session state data.

    Returns:
        YAML string
    """
    config = {}

    # Page 1: Basic Info
    page_1 = st.session_state.get('page_1_data') or {}
    if page_1.get('name'):
        config['name'] = page_1['name']
    if page_1.get('version'):
        config['version'] = page_1['version']
    if page_1.get('description'):
        config['description'] = page_1['description']
    if page_1.get('tags'):
        # Defensive handling: ensure tags are always a proper list
        tags = page_1['tags']
        if isinstance(tags, str):
            # Handle comma-separated string
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        elif isinstance(tags, list):
            # Ensure each tag is a string and not empty
            tags = [str(t).strip() for t in tags if str(t).strip()]
        else:
            tags = []
        if tags:  # Only add if we have valid tags
            config['tags'] = tags

    # Page 2: LLM Config
    page_2 = st.session_state.get('page_2_data') or {}
    if page_2.get('provider') and page_2.get('model'):
        llm_config = {
            'provider': page_2['provider'],
            'model': page_2['model'],
            'temperature': page_2.get('temperature', 0.7),
        }
        if page_2.get('max_tokens'):
            llm_config['max_tokens'] = page_2['max_tokens']
        if page_2.get('top_p') is not None:
            llm_config['top_p'] = page_2['top_p']
        config['llm'] = llm_config

    # Page 3: Tools (moved before prompts to match UI flow)
    page_3 = st.session_state.get('page_3_data') or {}
    if page_3.get('tools'):
        config['tools'] = page_3['tools']

    # MCP Servers: combine refs and inline configs
    mcp_servers = []

    # Add references (new format - from saved MCP server configs)
    for ref in (page_3.get('mcp_server_refs') or []):
        ref_entry = {}
        if isinstance(ref, dict):
            ref_entry['ref'] = ref.get('name')
            if ref.get('selected_tools'):
                ref_entry['selected_tools'] = ref['selected_tools']
        else:
            ref_entry['ref'] = ref
        mcp_servers.append(ref_entry)

    # Add inline configs (backwards compatible - manually configured)
    for server in (page_3.get('mcp_servers') or []):
        mcp_servers.append(server)

    if mcp_servers:
        config['mcp_servers'] = mcp_servers

    # Page 4: Prompts (moved after tools to match UI flow)
    page_4 = st.session_state.get('page_4_data') or {}
    if page_4.get('system_prompt'):
        prompts_config = {'system': page_4['system_prompt']}
        if page_4.get('user_template'):
            prompts_config['user_template'] = page_4['user_template']
        if page_4.get('few_shot_examples'):
            prompts_config['few_shot_examples'] = page_4['few_shot_examples']
        config['prompts'] = prompts_config

    # Page 5: Memory
    page_5 = st.session_state.get('page_5_data') or {}
    memory_config = {}

    # Get agent name for default paths
    agent_name = (page_1.get('name') or 'agent')

    if page_5.get('short_term', {}).get('enabled'):
        short_term = page_5['short_term']
        memory_type = short_term.get('type', 'sqlite')
        memory_config['short_term'] = {
            'type': memory_type,
        }

        # Ensure path is included for sqlite type
        if memory_type == 'sqlite':
            path = short_term.get('path') or f'./data/checkpoints/{agent_name}.db'
            memory_config['short_term']['path'] = path
        elif short_term.get('path'):
            memory_config['short_term']['path'] = short_term['path']

        if short_term.get('custom_state'):
            memory_config['short_term']['custom_state'] = short_term['custom_state']
        if short_term.get('message_management') and short_term['message_management'] != 'none':
            memory_config['short_term']['message_management'] = short_term['message_management']

    if page_5.get('long_term', {}).get('enabled'):
        long_term = page_5['long_term']
        memory_type = long_term.get('type', 'sqlite')
        memory_config['long_term'] = {
            'type': memory_type,
        }

        # Ensure path is included for sqlite type
        if memory_type == 'sqlite':
            path = long_term.get('path') or f'./data/stores/{agent_name}.db'
            memory_config['long_term']['path'] = path
        elif long_term.get('path'):
            memory_config['long_term']['path'] = long_term['path']

        if long_term.get('namespaces'):
            memory_config['long_term']['namespaces'] = long_term['namespaces']
        if long_term.get('enable_vector_search'):
            memory_config['long_term']['enable_vector_search'] = True

    if memory_config:
        config['memory'] = memory_config

    # Page 6: Middleware
    page_6 = st.session_state.get('page_6_data') or {}
    if page_6.get('middleware'):
        config['middleware'] = page_6['middleware']

    # Page 7: Advanced
    page_7 = st.session_state.get('page_7_data') or {}

    if page_7.get('streaming', {}).get('enabled'):
        streaming = page_7['streaming']
        config['streaming'] = {
            'enabled': True,
            'modes': streaming.get('modes', ['updates'])
        }

    if page_7.get('runtime', {}).get('context_schema'):
        context_schema = page_7['runtime']['context_schema']
        if context_schema:
            config['runtime'] = {'context_schema': context_schema}

    # Convert to YAML
    try:
        yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return yaml_str
    except Exception as e:
        return f"# Error generating YAML: {str(e)}"


def get_config_dict() -> Dict[str, Any]:
    """
    Get configuration as dictionary (for API submission).

    Returns:
        Configuration dictionary matching AgentConfig schema
    """
    config = {}

    # Page 1: Basic Info
    page_1 = st.session_state.get('page_1_data') or {}
    config['name'] = page_1.get('name', '')
    config['version'] = page_1.get('version', '1.0.0')
    config['description'] = page_1.get('description', '')

    # Defensive handling: ensure tags are always a proper list
    tags = page_1.get('tags', [])
    if isinstance(tags, str):
        # Handle comma-separated string
        tags = [t.strip() for t in tags.split(',') if t.strip()]
    elif isinstance(tags, list):
        # Ensure each tag is a string and not empty
        tags = [str(t).strip() for t in tags if str(t).strip()]
    else:
        tags = []
    config['tags'] = tags

    # Page 2: LLM Config
    page_2 = st.session_state.get('page_2_data') or {}
    config['llm'] = {
        'provider': page_2.get('provider', 'openai'),
        'model': page_2.get('model', 'gpt-4o'),
        'temperature': page_2.get('temperature', 0.7),
    }
    if page_2.get('max_tokens'):
        config['llm']['max_tokens'] = page_2['max_tokens']
    if page_2.get('top_p') is not None:
        config['llm']['top_p'] = page_2['top_p']

    # Page 3: Tools (moved before prompts to match UI flow)
    page_3 = st.session_state.get('page_3_data') or {}
    config['tools'] = page_3.get('tools', [])

    # MCP Servers: combine refs and inline configs
    mcp_servers = []

    # Add references (new format - from saved MCP server configs)
    for ref in (page_3.get('mcp_server_refs') or []):
        ref_entry = {}
        if isinstance(ref, dict):
            ref_entry['ref'] = ref.get('name')
            if ref.get('selected_tools'):
                ref_entry['selected_tools'] = ref['selected_tools']
        else:
            ref_entry['ref'] = ref
        mcp_servers.append(ref_entry)

    # Add inline configs (backwards compatible - manually configured)
    for server in (page_3.get('mcp_servers') or []):
        mcp_servers.append(server)

    if mcp_servers:
        config['mcp_servers'] = mcp_servers

    # Page 4: Prompts (moved after tools to match UI flow)
    page_4 = st.session_state.get('page_4_data') or {}
    config['prompts'] = {'system': page_4.get('system_prompt', '')}
    if page_4.get('user_template'):
        config['prompts']['user_template'] = page_4['user_template']
    if page_4.get('few_shot_examples'):
        config['prompts']['few_shot_examples'] = page_4['few_shot_examples']

    # Page 5: Memory
    page_5 = st.session_state.get('page_5_data') or {}
    memory_config = {}

    # Get agent name for default paths
    agent_name = (page_1.get('name') or 'agent')

    if page_5.get('short_term', {}).get('enabled'):
        short_term = page_5['short_term'].copy()

        # Ensure path is included for sqlite type
        memory_type = short_term.get('type', 'sqlite')
        if memory_type == 'sqlite' and not short_term.get('path'):
            short_term['path'] = f'./data/checkpoints/{agent_name}.db'

        memory_config['short_term'] = short_term

    if page_5.get('long_term', {}).get('enabled'):
        long_term = page_5['long_term'].copy()

        # Ensure path is included for sqlite type
        memory_type = long_term.get('type', 'sqlite')
        if memory_type == 'sqlite' and not long_term.get('path'):
            long_term['path'] = f'./data/stores/{agent_name}.db'

        memory_config['long_term'] = long_term

    if memory_config:
        config['memory'] = memory_config

    # Page 6: Middleware
    page_6 = st.session_state.get('page_6_data') or {}
    config['middleware'] = page_6.get('middleware', [])

    # Page 7: Advanced
    page_7 = st.session_state.get('page_7_data') or {}

    if page_7.get('streaming'):
        config['streaming'] = page_7['streaming']

    if page_7.get('runtime', {}).get('context_schema'):
        config['runtime'] = page_7['runtime']

    return config
