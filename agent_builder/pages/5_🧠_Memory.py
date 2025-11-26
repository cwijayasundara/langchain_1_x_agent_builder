"""
Page 5: Memory - Short-term and long-term memory configuration.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import initialize_session_state, get_page_data, update_page_data, mark_page_complete
from utils.styling import apply_custom_styles
from utils.constants import MEMORY_TYPES, MESSAGE_MANAGEMENT
from components.yaml_preview import display_yaml_preview
from components.navigation import display_page_header

initialize_session_state()
st.set_page_config(page_title="Memory", page_icon="🧠", layout="wide")
apply_custom_styles()

col1, col2 = st.columns([3, 2])

with col1:
    display_page_header(5, "Memory", "Configure conversation memory and long-term storage.")

    current_data = get_page_data(5)

    # Get agent name for smart defaults
    page_1_data = get_page_data(1)
    agent_name = page_1_data.get('name', 'agent')
    default_st_path = f'./data/checkpoints/{agent_name}.db'
    default_lt_path = f'./data/stores/{agent_name}.db'

    with st.form("memory_form"):
        # Short-term memory
        st.markdown("### 💭 Short-term Memory (Checkpointer)")
        st.caption("Stores conversation history and agent state for resuming threads.")

        st_enabled = st.checkbox("Enable short-term memory", value=current_data.get('short_term', {}).get('enabled', False))

        st_type = "sqlite"
        st_path = ""
        st_mgmt = "none"

        if st_enabled:
            st_type = st.radio("Type", MEMORY_TYPES, index=MEMORY_TYPES.index(current_data.get('short_term', {}).get('type', 'sqlite')))

            if st_type == "sqlite":
                st_path = st.text_input(
                    "SQLite Path *",
                    value=current_data.get('short_term', {}).get('path', default_st_path),
                    placeholder=default_st_path,
                    help="Path will be auto-generated if left empty"
                )

            st_mgmt = st.selectbox("Message Management", MESSAGE_MANAGEMENT, index=MESSAGE_MANAGEMENT.index(current_data.get('short_term', {}).get('message_management', 'none')))

        st.markdown("---")

        # Long-term memory
        st.markdown("### 🗄️ Long-term Memory (Store)")
        st.caption("Stores memories across sessions in organized namespaces.")

        lt_enabled = st.checkbox("Enable long-term memory", value=current_data.get('long_term', {}).get('enabled', False))

        lt_type = "sqlite"
        lt_path = ""
        lt_namespaces = []

        if lt_enabled:
            lt_type = st.radio("Type ", MEMORY_TYPES, index=MEMORY_TYPES.index(current_data.get('long_term', {}).get('type', 'sqlite')), key="lt_type")

            if lt_type == "sqlite":
                lt_path = st.text_input(
                    "SQLite Path *",
                    value=current_data.get('long_term', {}).get('path', default_lt_path),
                    placeholder=default_lt_path,
                    key="lt_path",
                    help="Path will be auto-generated if left empty"
                )

            namespaces_input = st.text_input(
                "Namespaces (comma-separated)",
                value=', '.join(current_data.get('long_term', {}).get('namespaces', [])),
                placeholder="user_{{user_id}}, session_{{session_id}}"
            )
            lt_namespaces = [ns.strip() for ns in namespaces_input.split(',') if ns.strip()]

        st.markdown("---")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            prev_btn = st.form_submit_button("⬅️ Previous", use_container_width=True)
        with col_b2:
            next_btn = st.form_submit_button("Next ➡️", use_container_width=True, type="primary")

        if prev_btn or next_btn:
            form_data = {
                'short_term': {
                    'enabled': st_enabled,
                    'type': st_type,
                    'path': st_path if st_enabled and st_type == 'sqlite' else None,
                    'custom_state': {},
                    'message_management': st_mgmt if st_enabled else 'none'
                },
                'long_term': {
                    'enabled': lt_enabled,
                    'type': lt_type,
                    'path': lt_path if lt_enabled and lt_type == 'sqlite' else None,
                    'namespaces': lt_namespaces if lt_enabled else [],
                    'enable_vector_search': False
                }
            }

            update_page_data(5, form_data)
            mark_page_complete(5, True)

            if prev_btn:
                st.switch_page("pages/4_💬_Prompts.py")
            else:
                st.switch_page("pages/6_⚙️_Middleware.py")

with col2:
    st.markdown("### 📄 Configuration Preview")
    display_yaml_preview()
