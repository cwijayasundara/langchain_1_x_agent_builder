"""
Agent Selector Component - UI for selecting and displaying agents.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.api_client import get_api_client
from utils.state_manager import select_agent


def display_agent_card(agent: Dict[str, Any], key_prefix: str = ""):
    """
    Display an agent card with information and select button.

    Args:
        agent: Agent information dictionary
        key_prefix: Prefix for widget keys to ensure uniqueness
    """
    agent_id = agent.get('agent_id', 'unknown')
    name = agent.get('name', 'Unnamed Agent')
    description = agent.get('description', 'No description available')
    tags = agent.get('tags', [])
    deployed = agent.get('deployed', False)
    version = agent.get('version', '1.0.0')

    # Card container
    with st.container():
        st.markdown(f"### {name}")
        st.caption(f"Version: {version} | ID: `{agent_id}`")

        # Deployment status
        if deployed:
            st.success("✅ Deployed")
        else:
            st.warning("⚠️ Not Deployed")

        # Description
        st.markdown(description)

        # Tags
        if tags:
            tag_str = " ".join([f"`{tag}`" for tag in tags])
            st.markdown(f"**Tags:** {tag_str}")

        # Select button
        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button(
                "🚀 Start Chat",
                key=f"{key_prefix}_select_{agent_id}",
                use_container_width=True,
                type="primary"
            ):
                # Get full agent details
                api_client = get_api_client()
                details_response = api_client.get_agent_details(agent_id)

                if details_response.get('success', True) and 'agent_info' in details_response.get('data', {}):
                    agent_info = details_response.get('data', {}).get('agent_info')
                    agent_config = details_response.get('data', {}).get('config', {})

                    # Deploy if not deployed and auto-deploy is enabled
                    if not deployed and st.session_state.get('auto_deploy_agents', True):
                        with st.spinner(f"Deploying {name}..."):
                            deploy_response = api_client.deploy_agent(agent_id)

                            if not deploy_response.get('success', True):
                                st.error(f"Failed to deploy agent: {deploy_response.get('error', 'Unknown error')}")
                                return

                    # Select agent
                    select_agent(agent_id, agent_info, agent_config)
                    st.success(f"Selected agent: {name}")
                    st.rerun()
                else:
                    st.error(f"Failed to load agent details: {details_response.get('error', 'Unknown error')}")

        with col2:
            if st.button("ℹ️", key=f"{key_prefix}_info_{agent_id}", help="View details"):
                st.session_state[f'show_details_{agent_id}'] = not st.session_state.get(f'show_details_{agent_id}', False)

        # Expandable details
        if st.session_state.get(f'show_details_{agent_id}', False):
            with st.expander("Agent Details", expanded=True):
                # Get full details
                api_client = get_api_client()
                details_response = api_client.get_agent_details(agent_id)

                if details_response.get('success', True) and 'config' in details_response.get('data', {}):
                    config = details_response.get('data', {}).get('config')

                    # LLM Info
                    llm = config.get('llm', {})
                    st.markdown(f"**LLM:** {llm.get('provider', 'unknown')} / {llm.get('model', 'unknown')}")

                    # Tools
                    tools = config.get('tools', [])
                    if tools:
                        st.markdown(f"**Tools:** {', '.join(tools)}")

                    # Memory
                    memory = config.get('memory', {})
                    short_term = memory.get('short_term', {})
                    long_term = memory.get('long_term', {})

                    memory_info = []
                    if short_term:
                        memory_info.append("Short-term")
                    if long_term:
                        memory_info.append("Long-term")

                    if memory_info:
                        st.markdown(f"**Memory:** {', '.join(memory_info)}")

                    # Streaming
                    streaming = config.get('streaming', {})
                    if streaming.get('enabled'):
                        modes = streaming.get('modes', [])
                        st.markdown(f"**Streaming:** Enabled ({', '.join(modes)})")
                else:
                    st.error("Failed to load agent configuration")

        st.markdown("---")


def display_agent_list(agents: List[Dict], search_query: str = "", tag_filter: List[str] = None):
    """
    Display a list of agents with filtering.

    Args:
        agents: List of agent dictionaries
        search_query: Search query string
        tag_filter: List of tags to filter by
    """
    if not agents:
        st.info("No agents available. Create an agent using the Agent Builder UI first.")
        return

    # Filter agents
    filtered_agents = agents

    if search_query:
        search_lower = search_query.lower()
        filtered_agents = [
            agent for agent in filtered_agents
            if search_lower in agent.get('name', '').lower()
            or search_lower in agent.get('description', '').lower()
            or search_lower in agent.get('agent_id', '').lower()
        ]

    if tag_filter:
        filtered_agents = [
            agent for agent in filtered_agents
            if any(tag in agent.get('tags', []) for tag in tag_filter)
        ]

    # Display count
    st.markdown(f"**{len(filtered_agents)} agent(s) found**")
    st.markdown("")

    # Display agents
    if filtered_agents:
        for idx, agent in enumerate(filtered_agents):
            display_agent_card(agent, key_prefix=f"list_{idx}")
    else:
        st.warning("No agents match your filters.")


def display_agent_selector_with_filters():
    """
    Display agent selector with search and filter options.
    """
    st.markdown("## Select an Agent")
    st.markdown("Choose an agent to start chatting.")
    st.markdown("")

    # Get agents from API
    api_client = get_api_client()

    with st.spinner("Loading agents..."):
        response = api_client.list_agents()

    if not response.get('success', True):
        st.error(f"Failed to load agents: {response.get('error', 'Unknown error')}")
        st.info("Make sure the Agent Builder API is running.")
        return

    agents = response.get('data', {}).get('agents', [])

    if not agents:
        st.info("📝 No agents available yet.")
        st.markdown("""
        **To create an agent:**
        1. Start the Agent Builder UI
        2. Configure your agent
        3. Deploy it
        4. Come back here to chat with it!
        """)
        return

    # Collect all tags
    all_tags = set()
    for agent in agents:
        all_tags.update(agent.get('tags', []))

    # Filters
    col1, col2 = st.columns([3, 2])

    with col1:
        search_query = st.text_input(
            "🔍 Search agents",
            placeholder="Search by name, description, or ID...",
            key="agent_search"
        )

    with col2:
        tag_filter = st.multiselect(
            "🏷️ Filter by tags",
            options=sorted(list(all_tags)),
            key="tag_filter"
        )

    st.markdown("---")

    # Display agents
    display_agent_list(agents, search_query, tag_filter)


def display_deployed_agent_dropdown():
    """
    Display a simple dropdown selector for all available agents.
    Auto-deploys agents when selected if not already deployed.
    """
    st.markdown("## Select an Agent")
    st.markdown("Choose an agent from the dropdown to start chatting. Agents will be deployed automatically when selected.")
    st.markdown("")

    # Get agents from API
    api_client = get_api_client()

    with st.spinner("Loading deployed agents..."):
        response = api_client.list_agents()

    if not response.get('success', True):
        st.error(f"Failed to load agents: {response.get('error', 'Unknown error')}")
        st.info("Make sure the Agent Builder API is running.")
        return

    agents = response.get('data', {}).get('agents', [])

    # Show all available agents (will auto-deploy when selected)
    available_agents = agents

    if not available_agents:
        st.warning("📝 No agents available.")
        st.markdown("""
        **To create an agent:**
        1. Start the Agent Builder API
        2. Create and configure your agent using the Agent Builder UI
        3. Come back here to chat with it!
        """)
        return

    # Create dropdown options with deployment status indicator
    agent_options = {}
    for agent in available_agents:
        name = agent.get('name', 'Unknown')
        agent_id = agent.get('agent_id', '')
        status = "✓" if agent.get('deployed', False) else "○"
        display_name = f"{status} {name} ({agent_id})"
        agent_options[display_name] = agent

    # Display dropdown
    selected_display_name = st.selectbox(
        "Select Agent",
        options=list(agent_options.keys()),
        key="deployed_agent_selector"
    )

    if selected_display_name:
        selected_agent = agent_options[selected_display_name]
        agent_id = selected_agent.get('agent_id', '')
        name = selected_agent.get('name', 'Unknown')
        description = selected_agent.get('description', 'No description')
        version = selected_agent.get('version', '1.0.0')
        tags = selected_agent.get('tags', [])

        st.markdown("---")

        # Display agent details
        st.markdown(f"### {name}")
        st.caption(f"Version: {version} | ID: `{agent_id}`")
        st.markdown(description)

        if tags:
            tag_str = " ".join([f"`{tag}`" for tag in tags])
            st.markdown(f"**Tags:** {tag_str}")

        st.markdown("---")

        # Get full details for selection
        with st.spinner("Loading agent configuration..."):
            details_response = api_client.get_agent_details(agent_id)

        if details_response.get('success', True) and 'agent_info' in details_response.get('data', {}):
            agent_info = details_response.get('data', {}).get('agent_info')
            agent_config = details_response.get('data', {}).get('config', {})

            # Show configuration preview
            with st.expander("📋 Configuration Preview", expanded=False):
                llm = agent_config.get('llm', {})
                tools = agent_config.get('tools', [])
                memory = agent_config.get('memory', {})
                streaming = agent_config.get('streaming', {})

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**LLM:** {llm.get('provider', 'N/A')} / {llm.get('model', 'N/A')}")
                    st.markdown(f"**Tools:** {len(tools)}")

                with col2:
                    memory_types = []
                    if memory.get('short_term'): memory_types.append("Short-term")
                    if memory.get('long_term'): memory_types.append("Long-term")
                    st.markdown(f"**Memory:** {', '.join(memory_types) if memory_types else 'None'}")
                    st.markdown(f"**Streaming:** {'Enabled' if streaming.get('enabled') else 'Disabled'}")

            st.markdown("")

            # Start chat button
            if st.button("🚀 Start Chat", use_container_width=True, type="primary", key="start_chat_btn"):
                # Auto-deploy if not deployed
                if not selected_agent.get('deployed', False):
                    with st.spinner(f"Deploying {name}..."):
                        deploy_response = api_client.deploy_agent(agent_id)
                        if not deploy_response.get('success', True):
                            st.error(f"Failed to deploy agent: {deploy_response.get('error', 'Unknown error')}")
                            return
                        st.success(f"Agent {name} deployed successfully!")

                # Select agent and navigate
                select_agent(agent_id, agent_info, agent_config)
                st.success(f"Selected agent: {name}")
                st.rerun()

        else:
            st.error("Failed to load agent configuration")


def display_selected_agent_info():
    """
    Display information about the currently selected agent.
    Used in sidebar or chat interface.
    """
    if not st.session_state.get('agent_selected'):
        st.warning("No agent selected")
        return

    agent_info = st.session_state.get('selected_agent_info', {})
    agent_config = st.session_state.get('selected_agent_config', {})

    name = agent_info.get('name', 'Unknown')
    description = agent_info.get('description', '')
    agent_id = st.session_state.get('selected_agent_id', '')

    st.markdown(f"### 🤖 {name}")
    st.caption(f"ID: `{agent_id}`")

    if description:
        st.markdown(description)

    st.markdown("---")

    # Quick stats
    llm = agent_config.get('llm') or {}
    tools = agent_config.get('tools', [])
    streaming = agent_config.get('streaming') or {}

    st.markdown(f"**Model:** {llm.get('provider', '')} / {llm.get('model', '')}")
    st.markdown(f"**Tools:** {len(tools)}")
    st.markdown(f"**Streaming:** {'✅' if streaming.get('enabled') else '❌'}")
