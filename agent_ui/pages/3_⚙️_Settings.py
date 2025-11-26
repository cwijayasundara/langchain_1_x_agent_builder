"""
Page 3: Settings - Application settings and preferences.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import initialize_session_state, update_preference, get_preference, reset_all_state
from utils.api_client import get_api_client, check_api_availability

initialize_session_state()
st.set_page_config(
    page_title="Settings",
    page_icon="⚙️",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Header
st.markdown("# ⚙️ Settings")
st.markdown("Configure your Agent UI preferences and settings.")
st.markdown("---")

# Tabs for different setting categories
tab1, tab2, tab3, tab4 = st.tabs(["🌐 API", "🎨 UI Preferences", "⚡ Defaults", "💾 Data"])

# Tab 1: API Configuration
with tab1:
    st.markdown("### API Configuration")
    st.markdown("Configure connection to the Agent Builder API.")

    with st.form("api_settings_form"):
        api_url = st.text_input(
            "API Base URL",
            value=st.session_state.get('api_base_url', 'http://localhost:8000'),
            placeholder="http://localhost:8000"
        )

        timeout = st.number_input(
            "Request Timeout (seconds)",
            min_value=5,
            max_value=120,
            value=30,
            help="Timeout for API requests"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.form_submit_button("💾 Save API Settings", use_container_width=True):
                st.session_state.api_base_url = api_url
                st.success("✅ API settings saved!")

        with col2:
            if st.form_submit_button("🔍 Test Connection", use_container_width=True):
                with st.spinner("Testing connection..."):
                    api_status = check_api_availability()

                if api_status:
                    st.success("✅ API is reachable!")

                    # Get API info
                    api_client = get_api_client()
                    response = api_client.test_connection()

                    if response.get('status') == 'healthy':
                        st.info(f"API Status: {response.get('status')}")
                else:
                    st.error("❌ Cannot connect to API")
                    st.caption("Make sure the Agent Builder API is running at the specified URL")

# Tab 2: UI Preferences
with tab2:
    st.markdown("### UI Preferences")
    st.markdown("Customize the appearance and behavior of the chat interface.")

    with st.form("ui_preferences_form"):
        # Message display
        st.markdown("#### Message Display")

        show_timestamps = st.checkbox(
            "Show timestamps on messages",
            value=get_preference('show_timestamps', True)
        )

        show_token_usage = st.checkbox(
            "Show token usage and costs",
            value=get_preference('show_token_usage', True)
        )

        message_font_size = st.select_slider(
            "Message font size",
            options=["small", "medium", "large"],
            value=get_preference('message_font_size', 'medium')
        )

        st.markdown("---")

        # Behavior
        st.markdown("#### Behavior")

        auto_scroll = st.checkbox(
            "Auto-scroll to latest message",
            value=get_preference('auto_scroll', True)
        )

        sound_notifications = st.checkbox(
            "Sound notifications (when response complete)",
            value=get_preference('sound_notifications', False)
        )

        st.markdown("---")

        # Theme
        st.markdown("#### Theme")

        theme = st.radio(
            "Theme preference",
            options=["light", "dark", "auto"],
            index=["light", "dark", "auto"].index(get_preference('theme', 'light')),
            horizontal=True
        )

        if st.form_submit_button("💾 Save Preferences", use_container_width=True):
            update_preference('show_timestamps', show_timestamps)
            update_preference('show_token_usage', show_token_usage)
            update_preference('message_font_size', message_font_size)
            update_preference('auto_scroll', auto_scroll)
            update_preference('sound_notifications', sound_notifications)
            update_preference('theme', theme)

            st.success("✅ Preferences saved!")

# Tab 3: Defaults
with tab3:
    st.markdown("### Default Settings")
    st.markdown("Configure default values for new conversations.")

    with st.form("defaults_form"):
        # Streaming
        default_streaming = st.checkbox(
            "Enable streaming by default (for agents that support it)",
            value=st.session_state.get('default_streaming_mode', False)
        )

        # Auto-deploy
        auto_deploy = st.checkbox(
            "Auto-deploy agents when selecting them",
            value=st.session_state.get('auto_deploy_agents', True),
            help="Automatically deploy agents if they're not already deployed"
        )

        st.markdown("---")

        # Default context values
        st.markdown("#### Default Context Values")
        st.info("Set default values for runtime context fields that will be pre-filled for new conversations.")

        context_defaults = st.text_area(
            "Context defaults (JSON format)",
            value='{\n  "example_key": "example_value"\n}',
            height=100,
            help="JSON object with default context values"
        )

        if st.form_submit_button("💾 Save Defaults", use_container_width=True):
            st.session_state.default_streaming_mode = default_streaming
            st.session_state.auto_deploy_agents = auto_deploy

            # TODO: Parse and save context defaults

            st.success("✅ Defaults saved!")

# Tab 4: Data Management
with tab4:
    st.markdown("### Data Management")
    st.markdown("Manage your conversation data and application state.")

    st.warning("⚠️ These actions cannot be undone!")

    st.markdown("---")

    # Export all data
    st.markdown("#### Export All Data")
    st.markdown("Export all your conversations and settings to a JSON file.")

    if st.button("📥 Export All Data", use_container_width=True):
        from utils.export_utils import export_all_threads_to_json
        import json

        threads = st.session_state.get('threads', {})
        agent_info = st.session_state.get('selected_agent_info', {})

        export_data = {
            'threads': threads,
            'agent_info': agent_info,
            'preferences': st.session_state.get('ui_preferences', {}),
            'settings': {
                'api_base_url': st.session_state.get('api_base_url'),
                'auto_deploy_agents': st.session_state.get('auto_deploy_agents'),
                'default_streaming_mode': st.session_state.get('default_streaming_mode')
            }
        }

        json_data = json.dumps(export_data, indent=2)

        st.download_button(
            label="⬇️ Download All Data (JSON)",
            data=json_data,
            file_name="agent_ui_backup.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")

    # Import data
    st.markdown("#### Import Data")
    st.markdown("Import conversations and settings from a JSON file.")

    uploaded_file = st.file_uploader(
        "Upload backup file",
        type=["json"],
        key="import_file"
    )

    if uploaded_file:
        if st.button("📤 Import Data", use_container_width=True):
            try:
                import json
                data = json.load(uploaded_file)

                # Import threads
                if 'threads' in data:
                    st.session_state.threads = data['threads']

                # Import preferences
                if 'preferences' in data:
                    st.session_state.ui_preferences = data['preferences']

                # Import settings
                if 'settings' in data:
                    settings = data['settings']
                    if 'api_base_url' in settings:
                        st.session_state.api_base_url = settings['api_base_url']
                    if 'auto_deploy_agents' in settings:
                        st.session_state.auto_deploy_agents = settings['auto_deploy_agents']
                    if 'default_streaming_mode' in settings:
                        st.session_state.default_streaming_mode = settings['default_streaming_mode']

                st.success("✅ Data imported successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Import failed: {str(e)}")

    st.markdown("---")

    # Clear data
    st.markdown("#### Clear Data")
    st.markdown("Remove all conversation threads and reset the application.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear All Threads", use_container_width=True):
            if st.session_state.get('confirm_clear_threads', False):
                st.session_state.threads = {}
                st.session_state.current_thread_id = None
                st.session_state.messages = []
                st.session_state.confirm_clear_threads = False
                st.success("✅ All threads cleared!")
                st.rerun()
            else:
                st.session_state.confirm_clear_threads = True
                st.warning("⚠️ Click again to confirm")

    with col2:
        if st.button("🔄 Reset All Settings", use_container_width=True):
            if st.session_state.get('confirm_reset_all', False):
                reset_all_state()
                st.success("✅ All settings reset!")
                st.rerun()
            else:
                st.session_state.confirm_reset_all = True
                st.warning("⚠️ Click again to confirm")

st.markdown("---")

# Navigation
col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    if st.button("💬 Chat", use_container_width=True):
        st.switch_page("pages/1_💬_Chat.py")

with col_nav2:
    if st.button("📊 Sessions", use_container_width=True):
        st.switch_page("pages/2_📊_Sessions.py")

with col_nav3:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
