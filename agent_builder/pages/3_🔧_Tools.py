"""
Page 3: Tools - Built-in, custom, and MCP tools.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import initialize_session_state, get_page_data, update_page_data, mark_page_complete
from utils.styling import apply_custom_styles
from utils.constants import BUILTIN_TOOLS, MCP_SERVER_PRESETS, TOOL_CATEGORIES
from utils.api_client import get_api_client
from utils.mcp_tool_discovery import discover_mcp_tools_sync
from components.yaml_preview import display_yaml_preview
from components.navigation import display_page_header
from collections import defaultdict

initialize_session_state()
st.set_page_config(page_title="Tools", page_icon="🔧", layout="wide")
apply_custom_styles()

col1, col2 = st.columns([3, 2])

with col1:
    display_page_header(3, "Tools", "Select built-in tools or create custom ones for your agent.")

    current_data = get_page_data(3)

    # Built-in tools
    st.markdown("### 🔨 Built-in Tools")
    st.caption("Tools are organized by category for easier selection")

    # Show optimization tip if many tools will be available
    total_tools_estimate = len(current_data.get('tools', []))
    if current_data.get('mcp_servers'):
        total_tools_estimate += len(current_data.get('mcp_servers', [])) * 5  # Rough estimate

    if total_tools_estimate >= 5 or current_data.get('mcp_servers'):
        st.info(
            "💡 **Tip**: When using 5+ tools or MCP servers, enable the **'Multi-Tool Optimized'** "
            "middleware preset in the Middleware page for better tool selection and cost savings."
        )

    # Group tools by category
    tools_by_category = defaultdict(list)
    for tool in BUILTIN_TOOLS:
        category = tool.get('category', 'utility')
        tools_by_category[category].append(tool)

    # Display tools grouped by category
    selected_tools = []
    for category_id in sorted(tools_by_category.keys()):
        tools_in_category = tools_by_category[category_id]
        category_name = TOOL_CATEGORIES.get(category_id, category_id.title())

        with st.expander(f"📦 **{category_name}** ({len(tools_in_category)} tools)", expanded=True):
            for tool in tools_in_category:
                if st.checkbox(
                    f"**{tool['name']}**",
                    value=tool['id'] in current_data.get('tools', []),
                    key=f"tool_{tool['id']}",
                    help=tool['description']
                ):
                    selected_tools.append(tool['id'])
                    st.caption(f"  ✓ {tool['description']}")

    # Custom tools placeholder
    st.markdown("---")
    st.markdown("### ✨ Custom Tools")
    st.info("Custom tool generation will be available in the Deploy page where you can test and integrate tools.")

    # MCP Servers
    st.markdown("---")
    st.markdown("### 🌐 MCP Servers")
    st.info("Connect to external MCP servers to access additional tools. Select from saved servers or configure manually.")

    # Section 1: Available MCP Servers (from configs/mcp_servers/)
    st.markdown("#### 📁 Saved MCP Servers")
    st.caption("Select from pre-configured MCP server definitions")

    # Initialize selected_mcp_refs in session state
    if 'selected_mcp_refs' not in st.session_state:
        # Load from saved page data
        saved_refs = current_data.get('mcp_server_refs', [])
        st.session_state.selected_mcp_refs = saved_refs if saved_refs else []

    # Fetch available MCP servers from API
    try:
        api_client = get_api_client()
        response = api_client.get("/mcp-servers/list")

        if response and response.get('success') and response.get('data'):
            available_servers = response['data'].get('servers', [])

            if available_servers:
                # Display available servers with checkboxes
                for server in available_servers:
                    server_name = server.get('name', 'unknown')
                    server_desc = server.get('description', 'No description')
                    server_transport = server.get('transport', 'unknown')

                    # Check if this server is already selected
                    is_selected = any(
                        ref.get('name') == server_name or ref == server_name
                        for ref in st.session_state.selected_mcp_refs
                    )

                    col_check, col_info = st.columns([1, 4])

                    with col_check:
                        new_selection = st.checkbox(
                            f"**{server_name}**",
                            value=is_selected,
                            key=f"mcp_ref_{server_name}",
                        )

                    with col_info:
                        st.caption(f"🔌 {server_transport} | {server_desc}")

                    # Handle selection change
                    if new_selection and not is_selected:
                        # Add to selected refs
                        st.session_state.selected_mcp_refs.append({'name': server_name})
                        st.rerun()
                    elif not new_selection and is_selected:
                        # Remove from selected refs
                        st.session_state.selected_mcp_refs = [
                            ref for ref in st.session_state.selected_mcp_refs
                            if ref.get('name') != server_name and ref != server_name
                        ]
                        st.rerun()

                # Show tool filtering for selected servers
                if st.session_state.selected_mcp_refs:
                    st.markdown("##### 🔧 Tool Selection per Server")
                    st.caption("Optionally filter which tools to use from each server (leave unchecked for all tools)")

                    for ref_idx, ref in enumerate(st.session_state.selected_mcp_refs):
                        ref_name = ref.get('name') if isinstance(ref, dict) else ref
                        with st.expander(f"Tools for **{ref_name}**", expanded=False):
                            # Try to discover tools from this server
                            discovery_key = f"mcp_ref_tools_{ref_name}"
                            if st.button(f"🔍 Discover Tools", key=f"discover_{ref_name}"):
                                try:
                                    # Get server config
                                    server_response = api_client.get(f"/mcp-servers/{ref_name}")
                                    if server_response and server_response.get('success'):
                                        server_config = server_response['data'].get('config', {})
                                        mcp_config = [{
                                            'name': server_config.get('name'),
                                            'transport': server_config.get('transport'),
                                            'url': server_config.get('url'),
                                            'command': server_config.get('command'),
                                        }]
                                        discovered = discover_mcp_tools_sync(mcp_config, timeout=30)
                                        st.session_state[discovery_key] = discovered
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to discover tools: {e}")

                            # Show discovered tools with checkboxes
                            discovered_tools = st.session_state.get(discovery_key)
                            if discovered_tools:
                                current_selection = ref.get('selected_tools') if isinstance(ref, dict) else None

                                updated_selection = []
                                for tool in discovered_tools:
                                    tool_name = tool.get('name', '').replace(f"{ref_name}_", "")
                                    is_tool_selected = current_selection is None or tool_name in (current_selection or [])

                                    if st.checkbox(tool_name, value=is_tool_selected, key=f"ref_tool_{ref_name}_{tool_name}"):
                                        updated_selection.append(tool_name)

                                # Update the reference with selected tools
                                if isinstance(ref, dict):
                                    if len(updated_selection) == len(discovered_tools):
                                        ref['selected_tools'] = None  # All tools
                                    else:
                                        ref['selected_tools'] = updated_selection if updated_selection else None
                            else:
                                st.caption("Click 'Discover Tools' to see available tools")
            else:
                st.caption("No saved MCP servers found. Add some using the API or config files.")
        else:
            st.caption("Could not fetch MCP servers from API. You can still configure manually below.")
    except Exception as e:
        st.caption(f"Could not connect to API: {e}. You can configure servers manually below.")

    st.markdown("---")
    st.markdown("#### ➕ Manual Configuration")
    st.caption("Add custom MCP servers manually (for servers not in configs/mcp_servers/)")

    # Initialize MCP servers list and sync with saved data
    if 'mcp_servers_temp' not in st.session_state:
        mcp_servers = current_data.get('mcp_servers', [])
        st.session_state.mcp_servers_temp = mcp_servers if mcp_servers is not None else []
    else:
        # Ensure it's always a list, never None
        if st.session_state.mcp_servers_temp is None:
            st.session_state.mcp_servers_temp = []
        # Sync with saved data if user returned to this page
        saved_servers = current_data.get('mcp_servers', [])
        saved_servers = saved_servers if saved_servers is not None else []
        # Only reload if saved data is more recent (e.g., after validation fix or template load)
        if len(saved_servers) > 0 and len(st.session_state.mcp_servers_temp) == 0:
            st.session_state.mcp_servers_temp = saved_servers

    # Preset selector (similar to middleware presets)
    st.markdown("#### 📦 Quick Add Presets")
    col_preset, col_add = st.columns([3, 1])

    with col_preset:
        preset = st.selectbox(
            "Choose a preset server",
            options=list(MCP_SERVER_PRESETS.keys()),
            format_func=lambda x: MCP_SERVER_PRESETS[x]['name'],
            key="mcp_preset_select",
            help="Select a pre-configured MCP server or choose Custom to configure manually"
        )
        st.caption(MCP_SERVER_PRESETS[preset]['description'])

    with col_add:
        st.markdown("")  # Spacing to align button
        if st.button("➕ Add Preset", key="add_preset_mcp", use_container_width=True):
            # Ensure mcp_servers_temp is a list before appending
            if st.session_state.mcp_servers_temp is None:
                st.session_state.mcp_servers_temp = []
            preset_config = MCP_SERVER_PRESETS[preset]['server'].copy()
            st.session_state.mcp_servers_temp.append(preset_config)
            st.success(f"Added {MCP_SERVER_PRESETS[preset]['name']}")
            st.rerun()

    st.markdown("#### 🔧 Configured Servers")

    # Display and edit each server
    servers_to_remove = []
    # Ensure mcp_servers_temp is always a list before iterating
    if st.session_state.mcp_servers_temp is None:
        st.session_state.mcp_servers_temp = []
    for idx, server in enumerate(st.session_state.mcp_servers_temp):
        with st.expander(f"**{server.get('name', 'Unnamed Server')}** - {server.get('transport', 'http')}", expanded=not server.get('name')):
            col_s1, col_s2 = st.columns([3, 1])

            with col_s1:
                name = st.text_input(
                    "Server Name*",
                    value=server.get('name', ''),
                    key=f"mcp_name_{idx}",
                    help="Unique identifier for this MCP server"
                )
                server['name'] = name

                description = st.text_input(
                    "Description",
                    value=server.get('description', ''),
                    key=f"mcp_desc_{idx}",
                    help="Brief description of what this server provides"
                )
                server['description'] = description

                transport = st.selectbox(
                    "Transport Type",
                    options=["streamable_http", "stdio", "sse"],
                    index=["streamable_http", "stdio", "sse"].index(server.get('transport', 'streamable_http')),
                    key=f"mcp_transport_{idx}",
                    help="Communication protocol for the MCP server"
                )
                server['transport'] = transport

                if transport in ["streamable_http", "http", "sse"]:
                    url = st.text_input(
                        "Server URL*",
                        value=server.get('url', ''),
                        key=f"mcp_url_{idx}",
                        placeholder="http://localhost:8005/mcp",
                        help="HTTP endpoint for the MCP server"
                    )
                    server['url'] = url
                    server['command'] = None  # Clear command for http transport

                elif transport == "stdio":
                    command = st.text_input(
                        "Command*",
                        value=server.get('command', ''),
                        key=f"mcp_cmd_{idx}",
                        placeholder="python custom_tools/mcp_servers/server.py",
                        help="Command to launch the MCP server as a subprocess"
                    )
                    server['command'] = command
                    server['url'] = None  # Clear URL for stdio transport

                stateful_col1, stateful_col2 = st.columns([1, 3])
                with stateful_col1:
                    stateful = st.checkbox(
                        "Stateful",
                        value=server.get('stateful', False),
                        key=f"mcp_stateful_{idx}",
                        help="Enable stateful sessions (maintains context between calls)"
                    )
                    server['stateful'] = stateful

                # Tool Discovery and Selection
                st.markdown("---")
                st.markdown("#### 🔧 Available Tools")
                
                # Initialize selected_tools in server config if not present
                if 'selected_tools' not in server:
                    server['selected_tools'] = None  # None means all tools selected
                
                # Initialize discovered_tools cache key
                discovery_cache_key = f"mcp_tools_{idx}_{server.get('name', '')}_{transport}_{server.get('url', '')}_{server.get('command', '')}"
                
                # Button to discover tools
                col_discover, col_info = st.columns([2, 3])
                with col_discover:
                    discover_btn = st.button(
                        "🔍 Discover Tools",
                        key=f"discover_tools_{idx}",
                        use_container_width=True,
                        help="Connect to the MCP server and discover available tools"
                    )
                
                # Discover tools when button is clicked or if already discovered
                discovered_tools = st.session_state.get(discovery_cache_key, None)
                
                if discover_btn:
                    # Validate server config before discovery
                    if transport in ["streamable_http", "http", "sse"] and not server.get('url'):
                        st.error("❌ Please provide a server URL first")
                    elif transport == "stdio" and not server.get('command'):
                        st.error("❌ Please provide a command first")
                    else:
                        with st.spinner(f"Discovering tools from {server.get('name', 'server')}..."):
                            try:
                                discovered_tools = discover_mcp_tools_sync(
                                    server_name=server.get('name', ''),
                                    transport=transport,
                                    url=server.get('url') if transport in ["streamable_http", "http", "sse"] else None,
                                    command=server.get('command') if transport == "stdio" else None,
                                    args=server.get('args'),
                                    env=server.get('env')
                                )
                                st.session_state[discovery_cache_key] = discovered_tools
                                if discovered_tools and len(discovered_tools) > 0:
                                    st.success(f"✅ Discovered {len(discovered_tools)} tool(s)")
                                    # Initialize selected_tools to all tools if first discovery
                                    if server.get('selected_tools') is None:
                                        server['selected_tools'] = [tool['name'] for tool in discovered_tools]
                                else:
                                    st.warning("⚠️ No tools found or server not accessible. Make sure:")
                                    st.caption("• The MCP server is running")
                                    st.caption("• The URL/command is correct")
                                    st.caption("• The server is accessible from this machine")
                                    discovered_tools = []
                                    st.session_state[discovery_cache_key] = []
                            except Exception as e:
                                error_msg = str(e)
                                st.error(f"❌ Failed to discover tools: {error_msg}")
                                st.caption(f"Error details: {error_msg}")
                                discovered_tools = []
                                st.session_state[discovery_cache_key] = []
                
                # Display discovered tools with checkboxes
                if discovered_tools:
                    st.caption(f"Select which tools to include from this server:")
                    
                    # Get current selected tools list
                    selected_tools_list = server.get('selected_tools', None)
                    if selected_tools_list is None:
                        # None means all tools - initialize to all
                        selected_tools_list = [tool['name'] for tool in discovered_tools]
                        server['selected_tools'] = selected_tools_list
                    
                    # Display checkboxes for each tool
                    updated_selection = []
                    for tool in discovered_tools:
                        tool_name = tool['name']
                        tool_desc = tool.get('description', 'No description')
                        
                        # Check if tool is currently selected
                        is_selected = tool_name in selected_tools_list
                        
                        checkbox_key = f"mcp_tool_{idx}_{tool_name}"
                        if st.checkbox(
                            f"**{tool_name}**",
                            value=is_selected,
                            key=checkbox_key,
                            help=tool_desc
                        ):
                            updated_selection.append(tool_name)
                        
                        st.caption(f"  {tool_desc}")
                    
                    # Update server config with new selection
                    if len(updated_selection) == len(discovered_tools):
                        # All tools selected - use None to indicate "all"
                        server['selected_tools'] = None
                    else:
                        server['selected_tools'] = updated_selection
                    
                    # Show selection summary
                    selected_count = len(updated_selection)
                    if selected_count == len(discovered_tools):
                        st.info(f"📌 All {selected_count} tool(s) selected")
                    elif selected_count > 0:
                        st.info(f"📌 {selected_count} of {len(discovered_tools)} tool(s) selected")
                    else:
                        st.warning("⚠️ No tools selected - all tools will be included by default")
                elif discovered_tools is not None and len(discovered_tools) == 0:
                    st.info("ℹ️ No tools available from this server")
                else:
                    st.caption("Click 'Discover Tools' to see available tools from this server")

            with col_s2:
                st.markdown("")  # Spacing
                st.markdown("")  # Spacing
                if st.button("🗑️ Remove", key=f"remove_mcp_{idx}", use_container_width=True):
                    servers_to_remove.append(idx)

    # Remove servers marked for deletion
    for idx in reversed(servers_to_remove):
        st.session_state.mcp_servers_temp.pop(idx)
        st.rerun()

    # Navigation
    st.markdown("---")
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        if st.button("⬅️ Previous", use_container_width=True):
            form_data = {
                'tools': selected_tools,
                'custom_tools': [],
                'mcp_server_refs': st.session_state.get('selected_mcp_refs', []),
                'mcp_servers': st.session_state.mcp_servers_temp
            }
            update_page_data(3, form_data)
            st.switch_page("pages/2_🤖_LLM_Config.py")

    with col_b2:
        if st.button("Next ➡️", use_container_width=True, type="primary"):
            form_data = {
                'tools': selected_tools,
                'custom_tools': [],
                'mcp_server_refs': st.session_state.get('selected_mcp_refs', []),
                'mcp_servers': st.session_state.mcp_servers_temp
            }
            update_page_data(3, form_data)
            mark_page_complete(3, True)
            st.switch_page("pages/4_💬_Prompts.py")

with col2:
    st.markdown("### 📄 Configuration Preview")
    display_yaml_preview()

    st.markdown("---")

    # Tools summary
    if selected_tools:
        st.success(f"✅ {len(selected_tools)} built-in tool(s) selected")
    else:
        st.warning("⚠️ No built-in tools selected")

    # MCP servers summary
    mcp_count = len(st.session_state.get('mcp_servers_temp', []))
    if mcp_count > 0:
        st.success(f"✅ {mcp_count} MCP server(s) configured")
        for server in st.session_state.get('mcp_servers_temp', []):
            if server.get('name'):
                st.caption(f"🌐 {server['name']} ({server.get('transport', 'http')})")
