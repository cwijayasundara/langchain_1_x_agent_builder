"""
Page 7: Advanced - Streaming, runtime context, and output formatters.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import initialize_session_state, get_page_data, update_page_data, mark_page_complete
from utils.styling import apply_custom_styles
from utils.constants import STREAMING_MODES, DEFAULTS, PYTHON_TYPES
from components.yaml_preview import display_yaml_preview
from components.navigation import display_page_header
import re

initialize_session_state()
st.set_page_config(page_title="Advanced", page_icon="🚀", layout="wide")
apply_custom_styles()

col1, col2 = st.columns([3, 2])

with col1:
    display_page_header(7, "Advanced Settings", "Configure streaming and runtime options.")

    current_data = get_page_data(7)

    # ===== RUNTIME CONTEXT MANAGEMENT (Outside form for interactivity) =====
    st.markdown("### ⏱️ Runtime Context (Optional)")
    st.markdown("Define context variables that will be provided at runtime. Use them in prompts with `{{variable_name}}`.")

    # Initialize runtime context in session state if not exists
    if 'runtime_context_fields' not in st.session_state:
        # Load from current page data
        existing_fields = current_data.get('runtime', {}).get('context_schema', [])
        st.session_state.runtime_context_fields = existing_fields.copy() if existing_fields else []

    context_fields = st.session_state.runtime_context_fields

    # Display existing fields
    if context_fields:
        st.markdown(f"**Configured Fields ({len(context_fields)}):**")

        fields_to_remove = []
        for i, field in enumerate(context_fields):
            col_field, col_remove = st.columns([5, 1])

            with col_field:
                field_info = f"**{field['name']}** ({field['type']})"
                if not field.get('required', True):
                    field_info += f" - Optional, default: `{field.get('default', 'None')}`"
                else:
                    field_info += " - Required"
                st.markdown(field_info)

            with col_remove:
                if st.button("🗑️", key=f"remove_field_{i}", help="Remove field"):
                    fields_to_remove.append(i)

        # Remove fields (do this after iteration to avoid index issues)
        for idx in sorted(fields_to_remove, reverse=True):
            st.session_state.runtime_context_fields.pop(idx)
            st.rerun()
    else:
        st.caption("No runtime context fields defined yet.")

    # Add new field section
    with st.expander("➕ Add New Context Field", expanded=len(context_fields) == 0):
        st.markdown("**New Field Configuration:**")

        col_name, col_type = st.columns(2)
        with col_name:
            new_field_name = st.text_input(
                "Field Name",
                key="new_field_name",
                placeholder="e.g., user_id, session_id",
                help="Must be a valid Python identifier (letters, numbers, underscores)"
            )

        with col_type:
            new_field_type = st.selectbox(
                "Type",
                options=PYTHON_TYPES,
                key="new_field_type",
                help="Python type for this field"
            )

        col_req, col_default = st.columns(2)
        with col_req:
            new_field_required = st.checkbox(
                "Required",
                value=True,
                key="new_field_required",
                help="If unchecked, you can provide a default value"
            )

        with col_default:
            new_field_default = st.text_input(
                "Default Value",
                key="new_field_default",
                disabled=new_field_required,
                help="Default value when field is not provided"
            )

        if st.button("Add Field", key="add_field_btn", type="primary"):
            # Validation
            errors = []

            if not new_field_name:
                errors.append("Field name is required")
            elif not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', new_field_name):
                errors.append("Field name must be a valid Python identifier (start with letter/underscore, contain only letters, numbers, underscores)")
            elif any(f['name'] == new_field_name for f in context_fields):
                errors.append(f"Field name '{new_field_name}' already exists")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Add field
                new_field = {
                    'name': new_field_name,
                    'type': new_field_type,
                    'required': new_field_required
                }
                if not new_field_required and new_field_default:
                    new_field['default'] = new_field_default

                st.session_state.runtime_context_fields.append(new_field)
                st.success(f"✅ Added field: {new_field_name}")
                st.rerun()

    st.markdown("---")

    # ===== STREAMING & NAVIGATION FORM =====
    with st.form("advanced_form"):
        # Streaming
        st.markdown("### 📡 Streaming")

        streaming_enabled = st.checkbox(
            "Enable Streaming",
            value=current_data.get('streaming', {}).get('enabled', DEFAULTS['streaming_enabled'])
        )

        streaming_modes = []
        if streaming_enabled:
            st.markdown("**Streaming Modes:**")
            for mode in STREAMING_MODES:
                if st.checkbox(
                    f"{mode['label']}",
                    value=mode['value'] in current_data.get('streaming', {}).get('modes', DEFAULTS['streaming_modes']),
                    key=f"stream_{mode['value']}"
                ):
                    streaming_modes.append(mode['value'])
                    st.caption(mode['description'])

        st.markdown("---")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            prev_btn = st.form_submit_button("⬅️ Previous", use_container_width=True)
        with col_b2:
            next_btn = st.form_submit_button("Next ➡️", use_container_width=True, type="primary")

        if prev_btn or next_btn:
            form_data = {
                'streaming': {
                    'enabled': streaming_enabled,
                    'modes': streaming_modes if streaming_enabled else []
                },
                'runtime': {'context_schema': st.session_state.runtime_context_fields},
                'output_formatter': {'enabled': False}
            }

            update_page_data(7, form_data)
            mark_page_complete(7, True)

            if prev_btn:
                st.switch_page("pages/6_⚙️_Middleware.py")
            else:
                st.switch_page("pages/8_✅_Deploy.py")

with col2:
    st.markdown("### 📄 Configuration Preview")
    display_yaml_preview()
