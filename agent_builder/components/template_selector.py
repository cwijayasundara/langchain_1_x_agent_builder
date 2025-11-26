"""
Template selector component for choosing starting configuration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.api_client import get_api_client
from utils.state_manager import load_template_data, initialize_session_state


def display_template_selector():
    """Display template selection interface."""

    st.title("🚀 Agent Builder")
    st.markdown("### Choose how to start building your agent")

    # Initialize state
    initialize_session_state()

    # Check if already selected
    if st.session_state.get('template_selected'):
        return True

    # Option 1: Start from blank
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📄 Start from Blank")
        st.markdown("Create a new agent configuration from scratch.")
        if st.button("Start Blank", use_container_width=True, type="primary"):
            st.session_state.template_selected = True
            st.session_state.template_name = "blank"
            st.rerun()

    with col2:
        st.markdown("#### 📋 Use Template")
        st.markdown("Start with a pre-configured template and customize it.")

    # Fetch templates
    api = get_api_client()
    result = api.get_templates()

    if result.get('error'):
        error_msg = result['error'].get('message', 'Unknown error') if isinstance(result['error'], dict) else str(result['error'])
        st.warning(f"⚠️ Could not fetch templates: {error_msg}")
        st.info("You can still start from blank or work offline.")
    elif result.get('success') and result.get('data'):
        templates_data = result['data']
        templates = templates_data.get('templates', [])

        if templates:
            st.markdown("#### Available Templates:")

            for template in templates:
                with st.expander(f"📋 {template.get('name', 'Unnamed')}"):
                    st.markdown(f"**Description:** {template.get('description', 'No description')}")
                    if template.get('tags'):
                        st.markdown(f"**Tags:** {', '.join(template['tags'])}")

                    if st.button(f"Use {template.get('name')}", key=f"template_{template.get('template_id')}"):
                        # Load template
                        template_detail = api.get_template(template['template_id'])

                        if template_detail.get('success'):
                            template_config = template_detail['data'].get('config', {})
                            load_template_data(template_config)
                            st.session_state.template_name = template.get('name')
                            st.success(f"✅ Loaded template: {template.get('name')}")
                            st.rerun()
                        else:
                            st.error(f"Failed to load template: {template_detail.get('error', 'Unknown error')}")
        else:
            st.info("No templates available. Start from blank.")

    return st.session_state.get('template_selected', False)
