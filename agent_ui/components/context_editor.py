"""
Context Editor Component - UI for editing runtime context values.
"""

import streamlit as st
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import update_context, get_context_values


def display_context_editor():
    """
    Display context editor based on agent's runtime context schema.
    """
    agent_config = st.session_state.get('selected_agent_config', {})
    runtime_config = agent_config.get('runtime') or {}
    context_schema = runtime_config.get('context_schema', [])

    if not context_schema:
        st.info("This agent doesn't require runtime context.")
        return

    st.markdown("### ⚙️ Runtime Context")
    st.caption("Configure runtime context values for this conversation.")

    current_values = get_context_values()

    with st.form("context_form"):
        new_values = {}

        for field in context_schema:
            field_name = field.get('name', 'unnamed')
            field_type = field.get('type', 'string')
            required = field.get('required', False)
            default = field.get('default')

            # Get current value or default
            current_value = current_values.get(field_name, default)

            # Label
            label = field_name.replace('_', ' ').title()
            if required:
                label += " *"

            # Input based on type
            if field_type == 'string':
                value = st.text_input(
                    label,
                    value=str(current_value) if current_value is not None else "",
                    key=f"ctx_{field_name}"
                )
                new_values[field_name] = value

            elif field_type == 'number':
                value = st.number_input(
                    label,
                    value=float(current_value) if current_value is not None else 0.0,
                    key=f"ctx_{field_name}"
                )
                new_values[field_name] = value

            elif field_type == 'boolean':
                value = st.checkbox(
                    label,
                    value=bool(current_value) if current_value is not None else False,
                    key=f"ctx_{field_name}"
                )
                new_values[field_name] = value

            elif field_type in ['list', 'array']:
                value = st.text_area(
                    label,
                    value=str(current_value) if current_value is not None else "",
                    help="Enter comma-separated values",
                    key=f"ctx_{field_name}"
                )
                # Split by comma and strip whitespace
                new_values[field_name] = [v.strip() for v in value.split(',') if v.strip()]

            else:
                # Default to text input
                value = st.text_input(
                    label,
                    value=str(current_value) if current_value is not None else "",
                    key=f"ctx_{field_name}"
                )
                new_values[field_name] = value

        submitted = st.form_submit_button("Update Context", use_container_width=True)

        if submitted:
            # Validate required fields
            errors = []
            for field in context_schema:
                if field.get('required', False):
                    field_name = field.get('name')
                    if not new_values.get(field_name):
                        errors.append(f"{field_name} is required")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Update context
                for key, value in new_values.items():
                    update_context(key, value)

                st.success("✅ Context updated!")


def display_compact_context_info():
    """
    Display a compact view of current context values.
    """
    context_values = get_context_values()

    if not context_values:
        return

    st.markdown("**Context:**")
    for key, value in context_values.items():
        # Truncate long values
        value_str = str(value)
        if len(value_str) > 30:
            value_str = value_str[:30] + "..."

        st.caption(f"`{key}`: {value_str}")


def display_context_summary():
    """
    Display a summary of context values (for sidebar).
    """
    agent_config = st.session_state.get('selected_agent_config', {})
    runtime_config = agent_config.get('runtime') or {}
    context_schema = runtime_config.get('context_schema', [])

    if not context_schema:
        return

    context_values = get_context_values()

    st.markdown("### Context")

    if context_values:
        for key, value in context_values.items():
            value_str = str(value)
            if len(value_str) > 20:
                value_str = value_str[:20] + "..."
            st.caption(f"**{key}:** {value_str}")
    else:
        st.caption("No context set")

    if st.button("Edit Context", key="edit_context_btn", use_container_width=True):
        st.session_state.show_context_editor = True
