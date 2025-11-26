"""
Agent Reconfiguration Component - UI for dynamically reconfiguring deployed agents.

Allows changing LLM, tools, and MCP servers while preserving middleware and memory.
Conversation threads are maintained across reconfiguration.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.api_client import get_api_client
from agent_builder.utils.constants import LLM_PROVIDERS, BUILTIN_TOOLS, MCP_SERVER_PRESETS


def display_reconfigure_button():
    """
    Display the reconfiguration trigger button in the sidebar.

    Call this in the agent info sidebar to provide access to reconfiguration.
    """
    if 'selected_agent_id' not in st.session_state or not st.session_state.selected_agent_id:
        return

    agent_id = st.session_state.selected_agent_id

    # Check if agent is deployed
    agent_info = st.session_state.get('selected_agent_info', {})
    is_deployed = agent_info.get('deployed', False)

    if not is_deployed:
        st.warning("⚠️ Agent must be deployed before reconfiguring")
        return

    st.markdown("---")

    if st.button("🔧 Reconfigure Agent", key="open_reconfigure_btn", use_container_width=True):
        # Initialize reconfiguration state
        st.session_state.reconfigure_dialog_open = True
        st.session_state.reconfigure_pending_changes = {}
        st.session_state.reconfigure_result = None
        st.session_state.reconfigure_show_preview = False

        # Load current config for editing
        _initialize_reconfigure_state(agent_config)


def display_reconfigure_dialog():
    """
    Display the main reconfiguration dialog (call in main page area).

    This should be called in the chat page when reconfigure_dialog_open is True.
    """
    if not st.session_state.get('reconfigure_dialog_open', False):
        return

    agent_id = st.session_state.selected_agent_id
    agent_config = st.session_state.get('selected_agent_config', {})

    st.markdown("## 🔧 Reconfigure Agent")
    st.caption(f"Modify agent: **{agent_id}**")

    st.info(
        "💡 **Note:** You can change the LLM, tools, and MCP servers. "
        "Middleware and memory configuration will be preserved, maintaining conversation threads."
    )

    # Tabs for different configuration sections
    tab1, tab2, tab3 = st.tabs(["🤖 LLM Configuration", "🔧 Tools", "🔌 MCP Servers"])

    with tab1:
        display_llm_config_editor(agent_config)

    with tab2:
        display_tools_selector(agent_config)

    with tab3:
        display_mcp_editor(agent_config)

    st.markdown("---")

    # Preview changes before applying
    if st.session_state.get('reconfigure_show_preview', False):
        display_change_preview(agent_config)

    # Action buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.button("👁️ Preview Changes", key="preview_changes_btn", use_container_width=True):
            st.session_state.reconfigure_show_preview = True
            st.rerun()

    with col2:
        changes_pending = bool(st.session_state.get('reconfigure_pending_changes', {}))
        if st.button(
            "✅ Apply Changes",
            key="apply_changes_btn",
            use_container_width=True,
            disabled=not changes_pending,
            type="primary"
        ):
            handle_reconfigure_submit(agent_id)

    with col3:
        if st.button("🔄 Reset", key="reset_changes_btn", use_container_width=True):
            _initialize_reconfigure_state(agent_config)
            st.session_state.reconfigure_show_preview = False
            st.rerun()

    with col4:
        if st.button("❌ Cancel", key="cancel_reconfigure_btn", use_container_width=True):
            st.session_state.reconfigure_dialog_open = False
            st.session_state.reconfigure_pending_changes = {}
            st.session_state.reconfigure_result = None
            st.session_state.reconfigure_show_preview = False
            st.rerun()

    # Display result if available
    if st.session_state.get('reconfigure_result'):
        display_reconfigure_result()


def display_llm_config_editor(agent_config: Dict[str, Any]):
    """
    Display LLM configuration editor.

    Args:
        agent_config: Current agent configuration
    """
    st.markdown("### LLM Model Settings")
    st.caption("Change the language model and its parameters")

    # Get current LLM config
    current_config = agent_config.get('config', {})
    current_llm = current_config.get('llm', {})

    # Get initial values from session state or current config
    pending_changes = st.session_state.get('reconfigure_pending_changes', {})
    pending_llm = pending_changes.get('llm', {})

    current_provider = pending_llm.get('provider') or current_llm.get('provider', 'openai')
    current_model = pending_llm.get('model') or current_llm.get('model', 'gpt-4o-mini')
    current_temp = pending_llm.get('temperature') or current_llm.get('temperature', 0.7)
    current_max_tokens = pending_llm.get('max_tokens') or current_llm.get('max_tokens', 4096)

    # Checkbox to enable LLM changes
    change_llm = st.checkbox(
        "Change LLM Configuration",
        value='llm' in pending_changes,
        key="change_llm_checkbox"
    )

    if change_llm:
        col1, col2 = st.columns(2)

        with col1:
            # Provider selection
            provider_options = list(LLM_PROVIDERS.keys())
            try:
                provider_index = provider_options.index(current_provider)
            except ValueError:
                provider_index = 0

            selected_provider = st.selectbox(
                "Provider",
                options=provider_options,
                index=provider_index,
                format_func=lambda x: LLM_PROVIDERS[x]['name'],
                key="reconfigure_provider"
            )

        with col2:
            # Model selection
            available_models = LLM_PROVIDERS.get(selected_provider, {}).get('models', [])
            try:
                model_index = available_models.index(current_model)
            except (ValueError, AttributeError):
                model_index = 0 if available_models else None

            selected_model = st.selectbox(
                "Model",
                options=available_models,
                index=model_index if model_index is not None else 0,
                key="reconfigure_model"
            )

        col3, col4 = st.columns(2)

        with col3:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(current_temp),
                step=0.1,
                help="Controls randomness: 0 = deterministic, 2 = very creative",
                key="reconfigure_temperature"
            )

        with col4:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=128000,
                value=int(current_max_tokens),
                step=100,
                help="Maximum tokens in the response",
                key="reconfigure_max_tokens"
            )

        # Update pending changes
        new_llm_config = {
            "provider": selected_provider,
            "model": selected_model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if 'reconfigure_pending_changes' not in st.session_state:
            st.session_state.reconfigure_pending_changes = {}

        st.session_state.reconfigure_pending_changes['llm'] = new_llm_config

        st.success(f"✅ Will change to: **{LLM_PROVIDERS[selected_provider]['name']} {selected_model}**")

    else:
        # Remove LLM from pending changes if unchecked
        if 'llm' in st.session_state.get('reconfigure_pending_changes', {}):
            del st.session_state.reconfigure_pending_changes['llm']

        st.info(f"Current: **{current_llm.get('provider', 'N/A')} {current_llm.get('model', 'N/A')}**")


def display_tools_selector(agent_config: Dict[str, Any]):
    """
    Display tools selector with categories.

    Args:
        agent_config: Current agent configuration
    """
    st.markdown("### Built-in Tools")
    st.caption("Select tools to enable for this agent")

    # Get current tools
    current_config = agent_config.get('config', {})
    current_tools = current_config.get('tools', [])

    # Get pending tools from session state
    pending_changes = st.session_state.get('reconfigure_pending_changes', {})
    pending_tools = pending_changes.get('tools', current_tools)

    # Checkbox to enable tool changes
    change_tools = st.checkbox(
        "Change Tools",
        value='tools' in pending_changes,
        key="change_tools_checkbox"
    )

    if change_tools:
        # Group tools by category
        tools_by_category = {}
        for tool in BUILTIN_TOOLS:
            category = tool.get('category', 'utility')
            if category not in tools_by_category:
                tools_by_category[category] = []
            tools_by_category[category].append(tool)

        selected_tools = []

        # Display tools by category
        for category, tools in tools_by_category.items():
            category_name = category.replace('_', ' ').title()
            st.markdown(f"**{category_name}**")

            for tool in tools:
                tool_id = tool['id']
                is_selected = tool_id in pending_tools

                col1, col2 = st.columns([1, 4])

                with col1:
                    selected = st.checkbox(
                        "",
                        value=is_selected,
                        key=f"reconfigure_tool_{tool_id}",
                        label_visibility="collapsed"
                    )

                with col2:
                    st.markdown(f"**{tool['name']}**")
                    st.caption(tool['description'])

                if selected:
                    selected_tools.append(tool_id)

        # Update pending changes
        if 'reconfigure_pending_changes' not in st.session_state:
            st.session_state.reconfigure_pending_changes = {}

        st.session_state.reconfigure_pending_changes['tools'] = selected_tools

        st.success(f"✅ Selected: **{len(selected_tools)} tools**")

    else:
        # Remove tools from pending changes if unchecked
        if 'tools' in st.session_state.get('reconfigure_pending_changes', {}):
            del st.session_state.reconfigure_pending_changes['tools']

        st.info(f"Current: **{len(current_tools)} tools** ({', '.join(current_tools[:3])}{'...' if len(current_tools) > 3 else ''})")


def display_mcp_editor(agent_config: Dict[str, Any]):
    """
    Display MCP server editor.

    Args:
        agent_config: Current agent configuration
    """
    st.markdown("### MCP Servers")
    st.caption("Configure Model Context Protocol servers")

    # Get current MCP servers
    current_config = agent_config.get('config', {})
    current_mcp = current_config.get('mcp_servers', [])

    # Get pending MCP from session state
    pending_changes = st.session_state.get('reconfigure_pending_changes', {})

    # Checkbox to enable MCP changes
    change_mcp = st.checkbox(
        "Change MCP Servers",
        value='mcp_servers' in pending_changes,
        key="change_mcp_checkbox"
    )

    if change_mcp:
        st.info("💡 Use presets or add custom MCP server configurations")

        # Preset selection
        st.markdown("**Quick Presets**")

        preset_cols = st.columns(len(MCP_SERVER_PRESETS))
        selected_presets = []

        for idx, (preset_id, preset_info) in enumerate(MCP_SERVER_PRESETS.items()):
            if preset_id == "custom":
                continue  # Skip custom preset

            with preset_cols[idx]:
                if st.checkbox(
                    preset_info['name'],
                    key=f"mcp_preset_{preset_id}",
                    help=preset_info['description']
                ):
                    selected_presets.append(preset_info['server'])

        # Custom MCP server entry
        with st.expander("➕ Add Custom MCP Server", expanded=False):
            st.markdown("**Custom MCP Server**")

            col1, col2 = st.columns(2)

            with col1:
                custom_name = st.text_input(
                    "Server Name",
                    key="custom_mcp_name",
                    placeholder="my_server"
                )
                custom_url = st.text_input(
                    "Server URL",
                    key="custom_mcp_url",
                    placeholder="http://localhost:8000/mcp"
                )

            with col2:
                custom_transport = st.selectbox(
                    "Transport",
                    options=["streamable_http", "http", "sse", "stdio"],
                    key="custom_mcp_transport"
                )
                custom_stateful = st.checkbox(
                    "Stateful",
                    key="custom_mcp_stateful",
                    value=False
                )

            custom_description = st.text_area(
                "Description",
                key="custom_mcp_description",
                placeholder="Server description..."
            )

            if st.button("Add Custom Server", key="add_custom_mcp_btn"):
                if custom_name and custom_url:
                    custom_server = {
                        'name': custom_name,
                        'description': custom_description,
                        'transport': custom_transport,
                        'url': custom_url,
                        'stateful': custom_stateful
                    }
                    selected_presets.append(custom_server)
                    st.success(f"✅ Added custom server: {custom_name}")
                else:
                    st.error("Name and URL are required")

        # Update pending changes
        if 'reconfigure_pending_changes' not in st.session_state:
            st.session_state.reconfigure_pending_changes = {}

        st.session_state.reconfigure_pending_changes['mcp_servers'] = selected_presets

        if selected_presets:
            st.success(f"✅ Configured: **{len(selected_presets)} MCP servers**")
            for server in selected_presets:
                st.caption(f"• {server['name']} ({server['transport']})")
        else:
            st.warning("No MCP servers selected")

    else:
        # Remove MCP from pending changes if unchecked
        if 'mcp_servers' in st.session_state.get('reconfigure_pending_changes', {}):
            del st.session_state.reconfigure_pending_changes['mcp_servers']

        st.info(f"Current: **{len(current_mcp)} MCP servers**")


def display_change_preview(agent_config: Dict[str, Any]):
    """
    Display a preview of changes before applying.

    Args:
        agent_config: Current agent configuration
    """
    st.markdown("### 📋 Change Preview")
    st.caption("Review changes before applying")

    pending_changes = st.session_state.get('reconfigure_pending_changes', {})
    current_config = agent_config.get('config', {})

    if not pending_changes:
        st.info("No changes to preview")
        return

    # LLM changes
    if 'llm' in pending_changes:
        st.markdown("**LLM Configuration**")

        current_llm = current_config.get('llm', {})
        new_llm = pending_changes['llm']

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.caption("**Field**")
            st.text("Provider")
            st.text("Model")
            st.text("Temperature")
            st.text("Max Tokens")

        with col2:
            st.caption("**Current**")
            st.text(current_llm.get('provider', 'N/A'))
            st.text(current_llm.get('model', 'N/A'))
            st.text(current_llm.get('temperature', 'N/A'))
            st.text(current_llm.get('max_tokens', 'N/A'))

        with col3:
            st.caption("**New**")
            st.text(new_llm.get('provider', 'N/A'))
            st.text(new_llm.get('model', 'N/A'))
            st.text(new_llm.get('temperature', 'N/A'))
            st.text(new_llm.get('max_tokens', 'N/A'))

        st.markdown("---")

    # Tools changes
    if 'tools' in pending_changes:
        st.markdown("**Tools**")

        current_tools = set(current_config.get('tools', []))
        new_tools = set(pending_changes['tools'])

        added = new_tools - current_tools
        removed = current_tools - new_tools
        unchanged = current_tools & new_tools

        if added:
            st.success(f"➕ Adding: {', '.join(added)}")

        if removed:
            st.error(f"➖ Removing: {', '.join(removed)}")

        if unchanged:
            st.info(f"✓ Keeping: {len(unchanged)} tools")

        st.markdown("---")

    # MCP changes
    if 'mcp_servers' in pending_changes:
        st.markdown("**MCP Servers**")

        current_mcp = current_config.get('mcp_servers', [])
        new_mcp = pending_changes['mcp_servers']

        current_names = {s.get('name') for s in current_mcp if isinstance(s, dict)}
        new_names = {s.get('name') for s in new_mcp if isinstance(s, dict)}

        added = new_names - current_names
        removed = current_names - new_names
        unchanged = current_names & new_names

        if added:
            st.success(f"➕ Adding: {', '.join(added)}")

        if removed:
            st.error(f"➖ Removing: {', '.join(removed)}")

        if unchanged:
            st.info(f"✓ Keeping: {', '.join(unchanged)}")

        st.markdown("---")

    # Preserved items
    st.markdown("**Preserved Configuration**")
    st.info("✓ Middleware configuration preserved")
    st.info("✓ Memory configuration preserved")
    st.info("✓ Conversation threads maintained")


def handle_reconfigure_submit(agent_id: str):
    """
    Handle the reconfiguration submission.

    Args:
        agent_id: Agent identifier
    """
    pending_changes = st.session_state.get('reconfigure_pending_changes', {})

    if not pending_changes:
        st.warning("No changes to apply")
        return

    # Get API client
    client = get_api_client()

    # Prepare request payload
    llm_config = pending_changes.get('llm')
    tools = pending_changes.get('tools')
    mcp_servers = pending_changes.get('mcp_servers')

    with st.spinner("🔄 Reconfiguring agent..."):
        # Call reconfiguration API
        result = client.reconfigure_agent(
            agent_id=agent_id,
            llm=llm_config,
            tools=tools,
            mcp_servers=mcp_servers,
            preserve_middleware=True,
            preserve_memory=True
        )

        # Store result
        st.session_state.reconfigure_result = result

        # If successful, refresh agent config
        if result.get('success'):
            # Reload agent config
            agent_details = client.get_agent_details(agent_id)
            if agent_details.get('success'):
                st.session_state.selected_agent_config = agent_details.get('data', {})

            # Clear pending changes
            st.session_state.reconfigure_pending_changes = {}
            st.session_state.reconfigure_show_preview = False

        st.rerun()


def display_reconfigure_result():
    """
    Display the result of reconfiguration (success or error).
    """
    result = st.session_state.get('reconfigure_result')

    if not result:
        return

    st.markdown("---")
    st.markdown("### 🎯 Reconfiguration Result")

    if result.get('success'):
        data = result.get('data', {})

        st.success("✅ Agent reconfigured successfully!")

        # Display summary
        summary = data.get('summary', {})
        if summary:
            st.markdown("**Changes Applied:**")
            for key, value in summary.items():
                st.info(f"**{key.replace('_', ' ').title()}:** {value}")

        # Thread continuity confirmation
        if data.get('thread_continuity', True):
            st.success("✓ Conversation threads preserved")

        # Timestamp
        redeployed_at = data.get('redeployed_at')
        if redeployed_at:
            st.caption(f"Redeployed at: {redeployed_at}")

        # Close button
        if st.button("Close", key="close_result_btn", use_container_width=True):
            st.session_state.reconfigure_dialog_open = False
            st.session_state.reconfigure_result = None
            st.rerun()

    else:
        # Error case
        error = result.get('error', {})
        error_message = error if isinstance(error, str) else error.get('message', 'Unknown error')

        st.error(f"❌ Reconfiguration failed: {error_message}")

        # Details if available
        if isinstance(error, dict) and error.get('details'):
            with st.expander("Error Details"):
                st.json(error['details'])

        # Retry button
        if st.button("Try Again", key="retry_reconfigure_btn", use_container_width=True):
            st.session_state.reconfigure_result = None
            st.rerun()


def _initialize_reconfigure_state(agent_config: Dict[str, Any]):
    """
    Initialize reconfiguration state from current agent config.

    Args:
        agent_config: Current agent configuration
    """
    # Start with empty pending changes
    st.session_state.reconfigure_pending_changes = {}

    # No default changes - user must explicitly select what to change
    # This ensures intentional reconfiguration only
