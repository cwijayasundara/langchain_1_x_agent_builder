"""
Page 1: Chat - Main chat interface for interacting with agents.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import initialize_session_state
from components.agent_selector import display_selected_agent_info
from components.chat_interface import display_chat_container, display_chat_input, display_thread_controls
from components.thread_panel import display_thread_panel
from components.context_editor import display_context_summary
from components.agent_reconfiguration import display_reconfigure_button, display_reconfigure_dialog
from components.runtime_override_panel import display_runtime_override_panel, display_override_status

initialize_session_state()
st.set_page_config(
    page_title="Chat",
    page_icon="💬",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Check if agent is selected
if not st.session_state.get('agent_selected'):
    st.warning("⚠️ No agent selected. Please select an agent first.")

    if st.button("← Back to Agent Selection"):
        st.switch_page("app.py")

    st.stop()

# Main layout
col1, col2 = st.columns([4, 1])

with col1:
    # Header
    agent_info = st.session_state.get('selected_agent_info', {})
    agent_name = agent_info.get('name', 'Unknown Agent')

    st.markdown(f"# 💬 Chat with {agent_name}")
    st.markdown("---")

    # Thread controls
    display_thread_controls()

    # Show override status if active (NEW)
    display_override_status()

    st.markdown("---")

    # Reconfiguration dialog (if open)
    if st.session_state.get('reconfigure_dialog_open', False):
        display_reconfigure_dialog()
        st.markdown("---")

    # Chat container (scrollable message area)
    chat_container = st.container()

    with chat_container:
        display_chat_container()

    # Spacer
    st.markdown("")
    st.markdown("")

    # Chat input at bottom
    display_chat_input()

with col2:
    # Sidebar info
    st.markdown("### Agent Info")
    display_selected_agent_info()

    # Reconfiguration button
    display_reconfigure_button()

    st.markdown("---")

    # Runtime override panel (NEW)
    display_runtime_override_panel()

    st.markdown("---")

    # Context editor
    display_context_summary()

    st.markdown("---")

    # Thread panel
    display_thread_panel()

    st.markdown("---")

    # Quick actions
    st.markdown("### Quick Actions")

    if st.button("📊 Sessions", use_container_width=True):
        st.switch_page("pages/2_📊_Sessions.py")

    if st.button("⚙️ Settings", use_container_width=True):
        st.switch_page("pages/3_⚙️_Settings.py")

    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
