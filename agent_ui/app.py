"""
Agent UI - Main Entry Point
Interactive chat interface for deployed LangChain agents.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.state_manager import initialize_session_state, reset_chat
from utils.api_client import check_api_availability
from components.agent_selector import display_deployed_agent_dropdown, display_selected_agent_info

# Page configuration
st.set_page_config(
    page_title="Agent UI - Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Initialize session state
initialize_session_state()

# Custom CSS - Simplified for better visibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }

    /* Main content area - White background with black text */
    .main {
        background-color: #ffffff;
        color: #1a1a1a;
    }

    /* All markdown text - Black on white */
    .stMarkdown,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] * {
        color: #1a1a1a !important;
    }

    /* Dark mode - White text on dark background */
    [data-theme="dark"] .main {
        background-color: #0e1117;
        color: #ffffff;
    }

    [data-theme="dark"] .stMarkdown,
    [data-theme="dark"] [data-testid="stMarkdownContainer"],
    [data-theme="dark"] [data-testid="stMarkdownContainer"] * {
        color: #ffffff !important;
    }

    /* Input fields - Clear contrast */
    .stTextInput input,
    .stSelectbox select,
    .stMultiSelect select {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
        border: 1px solid #d0d0d0;
    }

    [data-theme="dark"] .stTextInput input,
    [data-theme="dark"] .stSelectbox select,
    [data-theme="dark"] .stMultiSelect select {
        color: #ffffff !important;
        background-color: #262730 !important;
        border: 1px solid #404040;
    }

    /* Buttons */
    .stButton button {
        color: #1a1a1a;
    }

    [data-theme="dark"] .stButton button {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""

    # Check API availability
    api_status = check_api_availability()

    # Sidebar
    with st.sidebar:
        st.markdown("### 💬 Agent UI")

        # If agent selected, show info
        if st.session_state.get('agent_selected'):
            display_selected_agent_info()

            st.markdown("---")

            # Navigation to pages
            st.markdown("### Navigation")

            if st.button("💬 Chat", use_container_width=True):
                st.switch_page("pages/1_💬_Chat.py")

            if st.button("📊 Sessions", use_container_width=True):
                st.switch_page("pages/2_📊_Sessions.py")

            if st.button("⚙️ Settings", use_container_width=True):
                st.switch_page("pages/3_⚙️_Settings.py")

            st.markdown("---")

            # Change agent button
            if st.button("🔄 Change Agent", use_container_width=True):
                st.session_state.agent_selected = False
                reset_chat()
                st.rerun()

        else:
            st.info("👈 Select an agent to start")

        st.markdown("---")

        # API Status
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Offline")
            st.caption("Make sure the Agent Builder API is running")

        # Settings
        with st.expander("⚙️ Settings"):
            api_url = st.text_input(
                "API Base URL",
                value=st.session_state.get('api_base_url', 'http://localhost:8000'),
                key='api_url_input'
            )

            if st.button("Update URL"):
                st.session_state.api_base_url = api_url
                st.rerun()

            auto_deploy = st.checkbox(
                "Auto-deploy agents",
                value=st.session_state.get('auto_deploy_agents', True),
                key='auto_deploy_check'
            )
            st.session_state.auto_deploy_agents = auto_deploy

    # Main content
    if not st.session_state.get('agent_selected'):
        # Show agent selector
        st.markdown('<div class="main-header">💬 Agent UI</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Chat with your deployed LangChain agents</div>', unsafe_allow_html=True)

        if not api_status:
            st.error("⚠️ Cannot connect to Agent Builder API")
            st.markdown("""
            **To get started:**
            1. Make sure the Agent Builder API is running:
               ```
               uvicorn agent_api.main:app --reload
               ```
            2. Create and deploy agents using the Agent Builder UI
            3. Come back here to chat with them!
            """)
            return

        # Display agent selector dropdown
        display_deployed_agent_dropdown()

    else:
        # Agent selected - show chat interface prompt
        st.markdown('<div class="main-header">🤖 Ready to Chat</div>', unsafe_allow_html=True)

        agent_info = st.session_state.get('selected_agent_info', {})
        agent_name = agent_info.get('name', 'Unknown')

        st.success(f"✅ Agent **{agent_name}** is ready!")

        st.markdown("""
        ### What would you like to do?

        - **💬 Start Chatting** - Go to the Chat page to have a conversation
        - **📊 View Sessions** - See your conversation history
        - **⚙️ Settings** - Configure your preferences

        Use the sidebar or buttons below to navigate.
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💬 Go to Chat", use_container_width=True, type="primary"):
                st.switch_page("pages/1_💬_Chat.py")

        with col2:
            if st.button("📊 View Sessions", use_container_width=True):
                st.switch_page("pages/2_📊_Sessions.py")

        with col3:
            if st.button("⚙️ Settings", use_container_width=True):
                st.switch_page("pages/3_⚙️_Settings.py")

        # Quick stats
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")

        col_a, col_b, col_c = st.columns(3)

        threads = st.session_state.get('threads', {})
        total_messages = sum(len(t.get('messages', [])) for t in threads.values())

        with col_a:
            st.metric("Threads", len(threads))

        with col_b:
            st.metric("Total Messages", total_messages)

        with col_c:
            agent_config = st.session_state.get('selected_agent_config', {})
            tools = agent_config.get('tools', [])
            st.metric("Tools Available", len(tools))


if __name__ == "__main__":
    main()
