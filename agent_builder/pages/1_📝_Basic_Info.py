"""
Page 1: Basic Info - Agent name, description, version, and tags.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import (
    initialize_session_state,
    get_page_data,
    update_page_data,
    mark_page_complete,
    get_validation_errors,
    clear_validation_errors
)
from utils.validators import validate_page_1
from utils.styling import apply_custom_styles
from components.yaml_preview import display_yaml_preview
from components.navigation import display_page_header

# Initialize
initialize_session_state()

# Page config
st.set_page_config(page_title="Basic Info", page_icon="📝", layout="wide")

# Apply custom styling for better text visibility
apply_custom_styles()

# Layout
col1, col2 = st.columns([3, 2])

with col1:
    display_page_header(
        1,
        "Basic Information",
        "Enter the basic details for your agent including name, description, and version."
    )

    # Get current data
    current_data = get_page_data(1)
    errors = get_validation_errors(1)

    # Display errors
    if errors:
        with st.expander(f"❌ {len(errors)} Validation Error(s)", expanded=True):
            for error in errors:
                st.error(f"**{error.field}**: {error.message}")

    # Form
    with st.form("basic_info_form"):
        st.markdown("### Agent Details")

        # Name
        name = st.text_input(
            "Agent Name *",
            value=current_data.get('name', ''),
            placeholder="my_research_assistant",
            help="Unique identifier for your agent (alphanumeric and underscores only)"
        )

        # Description
        description = st.text_area(
            "Description *",
            value=current_data.get('description', ''),
            placeholder="A helpful AI research assistant with web search capabilities...",
            help="Detailed description of what your agent does",
            height=100
        )

        # Version
        col_v1, col_v2 = st.columns([1, 2])
        with col_v1:
            version = st.text_input(
                "Version",
                value=current_data.get('version', '1.0.0'),
                help="Semantic version (X.Y.Z format)"
            )

        # Tags
        tags_input = st.text_input(
            "Tags (comma-separated)",
            value=', '.join(current_data.get('tags', [])),
            placeholder="research, web-search, analysis",
            help="Tags to categorize your agent"
        )

        # Parse tags
        tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]

        # Submit buttons
        st.markdown("---")
        col_b1, col_b2, col_b3 = st.columns(3)

        with col_b1:
            save_draft = st.form_submit_button("💾 Save Draft", use_container_width=True)

        with col_b2:
            validate_btn = st.form_submit_button("🔍 Validate", use_container_width=True)

        with col_b3:
            next_btn = st.form_submit_button("Next ➡️", use_container_width=True, type="primary")

        # Handle submission
        if save_draft or validate_btn or next_btn:
            form_data = {
                'name': name,
                'description': description,
                'version': version,
                'tags': tags
            }

            # Validate
            validation_errors = validate_page_1(form_data)

            if validation_errors:
                if next_btn:
                    # Store errors and rerun
                    from utils.state_manager import set_validation_errors
                    set_validation_errors(1, validation_errors)
                    st.rerun()
                else:
                    # Just show errors for validate
                    st.error(f"Found {len(validation_errors)} error(s)")
                    for error in validation_errors:
                        st.error(f"**{error.field}**: {error.message}")
            else:
                # Save data
                clear_validation_errors(1)
                update_page_data(1, form_data)
                mark_page_complete(1, True)

                if save_draft:
                    st.success("✅ Draft saved successfully!")
                    st.rerun()
                elif validate_btn:
                    st.success("✅ Validation passed!")
                elif next_btn:
                    st.success("✅ Page complete! Moving to next page...")
                    st.switch_page("pages/2_🤖_LLM_Config.py")

with col2:
    st.markdown("### 📄 Configuration Preview")
    st.markdown("This shows your current configuration in YAML format.")
    display_yaml_preview()
