"""
Test Override Panel Component - Runtime override controls for Agent Builder's test chat.
Allows testing with different LLM, tools, and prompt modifications without redeploying.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.api_client import get_api_client
from utils.constants import LLM_PROVIDERS


def _init_override_state():
    """Initialize session state for runtime overrides in test chat."""
    if 'test_override_expanded' not in st.session_state:
        st.session_state.test_override_expanded = False

    if 'test_pending_override' not in st.session_state:
        st.session_state.test_pending_override = None

    if 'test_override_llm_enabled' not in st.session_state:
        st.session_state.test_override_llm_enabled = False

    if 'test_override_tools_enabled' not in st.session_state:
        st.session_state.test_override_tools_enabled = False

    if 'test_override_prompt_enabled' not in st.session_state:
        st.session_state.test_override_prompt_enabled = False


def _fetch_available_tools(agent_id: str) -> Dict[str, Any]:
    """Fetch available tools from API for the deployed agent."""
    if not agent_id:
        return {"builtin_tools": [], "mcp_servers": [], "current_tools": [], "current_mcp_servers": []}

    try:
        client = get_api_client()
        result = client.get_available_tools(agent_id)
        if result.get('success') and result.get('data'):
            return result['data']
    except Exception as e:
        st.error(f"Failed to fetch tools: {e}")

    return {"builtin_tools": [], "mcp_servers": [], "current_tools": [], "current_mcp_servers": []}


def _get_current_override_status(agent_id: str) -> Optional[Dict[str, Any]]:
    """Get current session override status from API."""
    thread_id = st.session_state.get('chat_thread_id')

    if not agent_id or not thread_id:
        return None

    try:
        client = get_api_client()
        result = client.get_session_override(agent_id, thread_id)
        if result.get('success') and result.get('data'):
            return result['data']
    except Exception:
        pass

    return None


def display_test_override_panel(agent_id: str, agent_config: Optional[Dict[str, Any]] = None):
    """
    Display the runtime override panel for test chat.
    Allows users to temporarily override LLM, tools, and prompt.

    Args:
        agent_id: ID of the deployed agent
        agent_config: Optional agent configuration dict (from page data)
    """
    _init_override_state()

    if not agent_id:
        return

    # Use agent config from page data if not provided
    if agent_config is None:
        agent_config = {}

    # Check for existing session override
    override_status = _get_current_override_status(agent_id)
    has_active_override = override_status and override_status.get('has_override', False)

    # Display override controls directly (no expander - header is in Deploy page)
    if has_active_override:
        st.info("🔄 Session override is active")
        if st.button("🗑️ Clear Override", key="test_clear_override_btn", use_container_width=True):
            _clear_session_override(agent_id)
            st.rerun()

    st.caption("Override LLM, tools, or prompt for testing without permanent changes.")

    # Tabs for different override types
    tab_llm, tab_tools, tab_prompt = st.tabs(["🤖 LLM", "🔧 Tools", "📝 Prompt"])

    with tab_llm:
        _display_llm_override(agent_config)

    with tab_tools:
        _display_tools_override(agent_id, agent_config)

    with tab_prompt:
        _display_prompt_override()

    st.markdown("---")

    # Apply button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Apply", key="test_apply_override_btn", use_container_width=True, type="primary"):
            _apply_override()

    with col2:
        if st.button("🔄 Reset", key="test_reset_override_btn", use_container_width=True):
            _reset_override_form()
            st.rerun()


def _display_llm_override(agent_config: Dict[str, Any]):
    """Display LLM override controls."""
    # Get current LLM config from page 2 data (agent being built)
    from utils.state_manager import get_page_data
    page_2_data = get_page_data(2)

    current_provider = page_2_data.get('provider', 'openai')
    current_model = page_2_data.get('model', 'gpt-4o')
    current_temp = page_2_data.get('temperature', 0.7)

    st.caption(f"Current: **{current_provider}/{current_model}**")

    st.session_state.test_override_llm_enabled = st.checkbox(
        "Override LLM",
        value=st.session_state.test_override_llm_enabled,
        key="test_override_llm_checkbox"
    )

    if st.session_state.test_override_llm_enabled:
        # Provider selection
        providers = list(LLM_PROVIDERS.keys())
        provider_idx = providers.index(current_provider) if current_provider in providers else 0

        override_provider = st.selectbox(
            "Provider",
            options=providers,
            index=provider_idx,
            format_func=lambda x: LLM_PROVIDERS[x]["name"],
            key="test_override_provider"
        )

        # Model selection based on provider
        models = LLM_PROVIDERS[override_provider]["models"]
        model_idx = models.index(current_model) if current_model in models else 0
        override_model = st.selectbox(
            "Model",
            options=models,
            index=model_idx if model_idx < len(models) else 0,
            key="test_override_model"
        )

        # Temperature
        override_temp = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(current_temp),
            step=0.1,
            key="test_override_temperature"
        )

        # Max tokens (optional)
        current_max_tokens = page_2_data.get('max_tokens', 0) or 0
        override_max_tokens = st.number_input(
            "Max Tokens (0 = default)",
            min_value=0,
            max_value=128000,
            value=current_max_tokens,
            step=1000,
            key="test_override_max_tokens"
        )

        # Store in session state
        st.session_state.test_override_llm_config = {
            "provider": override_provider,
            "model": override_model,
            "temperature": override_temp,
            "max_tokens": override_max_tokens if override_max_tokens > 0 else None
        }
    else:
        st.session_state.test_override_llm_config = None


def _display_tools_override(agent_id: str, agent_config: Dict[str, Any]):
    """Display tools override controls."""
    # Fetch available tools
    tools_data = _fetch_available_tools(agent_id)

    builtin_tools = tools_data.get('builtin_tools', [])
    mcp_servers = tools_data.get('mcp_servers', [])
    current_tools = tools_data.get('current_tools', [])
    current_mcp_servers = tools_data.get('current_mcp_servers', [])

    st.caption(f"Current: **{len(current_tools)}** built-in, **{len(current_mcp_servers)}** MCP servers")

    st.session_state.test_override_tools_enabled = st.checkbox(
        "Override Tools",
        value=st.session_state.test_override_tools_enabled,
        key="test_override_tools_checkbox"
    )

    if st.session_state.test_override_tools_enabled:
        # Built-in tools multi-select
        if builtin_tools:
            tool_options = []
            tool_labels = {}

            for tool in builtin_tools:
                if isinstance(tool, dict):
                    tool_id = tool.get('tool_id', tool.get('id', ''))
                    tool_name = tool.get('name', tool_id)
                else:
                    tool_id = str(tool)
                    tool_name = tool_id

                if tool_id:
                    tool_options.append(tool_id)
                    tool_labels[tool_id] = tool_name

            selected_builtin = st.multiselect(
                "Built-in Tools",
                options=tool_options,
                default=[t for t in current_tools if t in tool_options],
                format_func=lambda x: tool_labels.get(x, x),
                key="test_override_builtin_tools"
            )
        else:
            selected_builtin = []
            st.caption("No built-in tools available")

        # MCP servers multi-select
        if mcp_servers:
            server_options = []
            server_labels = {}

            for server in mcp_servers:
                if isinstance(server, dict):
                    server_name = server.get('name', '')
                    server_desc = server.get('description', server_name)
                else:
                    server_name = str(server)
                    server_desc = server_name

                if server_name:
                    server_options.append(server_name)
                    server_labels[server_name] = server_desc[:30] + "..." if len(server_desc) > 30 else server_desc

            selected_mcp = st.multiselect(
                "MCP Servers",
                options=server_options,
                default=[s for s in current_mcp_servers if s in server_options],
                format_func=lambda x: server_labels.get(x, x),
                key="test_override_mcp_servers"
            )
        else:
            selected_mcp = []
            st.caption("No MCP servers available")

        # Auto-update prompt checkbox
        auto_update_prompt = st.checkbox(
            "Auto-update prompt with tool docs",
            value=True,
            help="Regenerate tool documentation in system prompt when tools change",
            key="test_override_auto_update_prompt"
        )

        # Store in session state (auto_update_prompt is managed by the checkbox widget)
        st.session_state.test_override_tools_config = {
            "builtin_tools": selected_builtin if selected_builtin else None,
            "mcp_servers": selected_mcp if selected_mcp else None
        }
    else:
        st.session_state.test_override_tools_config = None


def _display_prompt_override():
    """Display prompt override controls (prepend/append)."""
    st.caption("Add text before or after the system prompt")

    st.session_state.test_override_prompt_enabled = st.checkbox(
        "Override Prompt",
        value=st.session_state.test_override_prompt_enabled,
        key="test_override_prompt_checkbox"
    )

    if st.session_state.test_override_prompt_enabled:
        prepend = st.text_area(
            "Prepend (added before)",
            value=st.session_state.get('test_override_prompt_prepend', ''),
            height=80,
            help="Text added at the beginning of the system prompt",
            key="test_override_prompt_prepend"
        )

        append = st.text_area(
            "Append (added after)",
            value=st.session_state.get('test_override_prompt_append', ''),
            height=80,
            help="Text added at the end of the system prompt",
            key="test_override_prompt_append"
        )

        # Store in session state
        st.session_state.test_override_prompt_config = {
            "prepend": prepend if prepend.strip() else None,
            "append": append if append.strip() else None
        }
    else:
        st.session_state.test_override_prompt_config = None


def _apply_override():
    """Build and apply the runtime override."""
    override = {}
    has_changes = False

    # LLM override
    if st.session_state.test_override_llm_enabled and st.session_state.get('test_override_llm_config'):
        override['llm'] = st.session_state.test_override_llm_config
        has_changes = True

    # Tools override
    if st.session_state.test_override_tools_enabled and st.session_state.get('test_override_tools_config'):
        tools_config = st.session_state.test_override_tools_config
        # Only include if at least one is set
        if tools_config.get('builtin_tools') is not None or tools_config.get('mcp_servers') is not None:
            override['tools'] = tools_config
            has_changes = True

    # Prompt override
    if st.session_state.test_override_prompt_enabled and st.session_state.get('test_override_prompt_config'):
        prompt_config = st.session_state.test_override_prompt_config
        if prompt_config.get('prepend') or prompt_config.get('append'):
            override['prompt'] = prompt_config
            has_changes = True

    # Auto-update prompt setting
    override['auto_update_prompt'] = st.session_state.get('test_override_auto_update_prompt', True)

    if has_changes:
        st.session_state.test_pending_override = override
        st.success("Override applied! It will be used for the next message and persist for this session.")
    else:
        st.warning("No overrides configured. Enable at least one override option.")


def _reset_override_form():
    """Reset override form to defaults."""
    st.session_state.test_override_llm_enabled = False
    st.session_state.test_override_tools_enabled = False
    st.session_state.test_override_prompt_enabled = False
    st.session_state.test_override_llm_config = None
    st.session_state.test_override_tools_config = None
    st.session_state.test_override_prompt_config = None
    st.session_state.test_pending_override = None


def _clear_session_override(agent_id: str):
    """Clear the session override from the server."""
    thread_id = st.session_state.get('chat_thread_id')

    if not agent_id or not thread_id:
        st.error("No active session to clear")
        return

    try:
        client = get_api_client()
        result = client.clear_session_override(agent_id, thread_id)
        if result.get('success'):
            st.session_state.test_pending_override = None
            st.success("Session override cleared. Using base agent configuration.")
        else:
            st.error(f"Failed to clear override: {result.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error clearing override: {e}")


def get_test_pending_override() -> Optional[Dict[str, Any]]:
    """
    Get pending runtime override for API calls.

    Returns:
        Override configuration dict or None
    """
    return st.session_state.get('test_pending_override')


def display_test_override_status():
    """Display compact override status indicator for test chat."""
    override = get_test_pending_override()

    if override:
        parts = []
        if override.get('llm'):
            llm = override['llm']
            parts.append(f"🤖 {llm.get('provider', '')}:{llm.get('model', '')}")
        if override.get('tools'):
            tools = override['tools']
            n_builtin = len(tools.get('builtin_tools') or [])
            n_mcp = len(tools.get('mcp_servers') or [])
            parts.append(f"🔧 {n_builtin}+{n_mcp}")
        if override.get('prompt'):
            parts.append("📝 Modified")

        if parts:
            st.caption(f"**Override active:** {' | '.join(parts)}")
