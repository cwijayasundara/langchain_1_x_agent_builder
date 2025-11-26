"""
Page 6: Middleware - Select and configure processing middleware.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import initialize_session_state, get_page_data, update_page_data, mark_page_complete
from utils.styling import apply_custom_styles
from utils.constants import MIDDLEWARE_TYPES, MIDDLEWARE_PRESETS
from components.yaml_preview import display_yaml_preview
from components.navigation import display_page_header

initialize_session_state()
st.set_page_config(page_title="Middleware", page_icon="⚙️", layout="wide")
apply_custom_styles()

col1, col2 = st.columns([3, 2])

with col1:
    display_page_header(6, "Middleware", "Add middleware for monitoring, safety, and optimization.")

    current_data = get_page_data(6)

    # Presets
    st.markdown("### 📦 Presets")
    preset_options = ["None"] + list(MIDDLEWARE_PRESETS.keys())
    preset = st.selectbox(
        "Choose a preset",
        preset_options,
        format_func=lambda x: x.replace('_', ' ').title() if x != "None" else x
    )

    if preset != "None" and st.button("Load Preset"):
        preset_config = MIDDLEWARE_PRESETS[preset]
        update_page_data(6, {'middleware': preset_config['middleware']})
        st.success(f"✅ Loaded preset: {preset_config['name']}")
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Middleware Selection")

    selected_middleware = []

    for middleware in MIDDLEWARE_TYPES:
        with st.container():
            col_m1, col_m2 = st.columns([4, 1])

            with col_m1:
                st.markdown(f"**{middleware['name']}** _{middleware['category']}_")
                st.caption(middleware['description'])  # Always show description

            with col_m2:
                enabled = st.checkbox(
                    "Enable",
                    value=any(m.get('type') == middleware['type'] for m in current_data.get('middleware', [])),
                    key=f"mw_{middleware['type']}"
                )

            if enabled:
                middleware_config = {'type': middleware['type'], 'params': {}, 'enabled': True}

                # Show parameter inputs with tooltips
                if middleware['params']:
                    st.markdown("**Parameters:**")
                    for param_name, param_def in middleware['params'].items():
                        # Get label and help text
                        label = param_def.get('label', param_name.replace('_', ' ').title())
                        help_text = param_def.get('help_text', '')

                        if param_def['type'] == 'number':
                            min_val = param_def.get('min', 0)
                            default_val = param_def.get('default')
                            # Ensure default value is at least equal to min_value
                            if default_val is None:
                                default_val = min_val
                            else:
                                default_val = max(min_val, default_val)

                            value = st.number_input(
                                label,
                                value=default_val,
                                min_value=min_val,
                                max_value=param_def.get('max', 10000),
                                key=f"mw_param_{middleware['type']}_{param_name}",
                                help=help_text  # Add tooltip
                            )

                            # Warning for low run_limit values in model_call_limit middleware
                            if middleware['type'] == 'model_call_limit' and param_name == 'run_limit' and value is not None and 0 < value < 10:
                                st.warning(
                                    f"⚠️ **Low run_limit detected ({value})**: This restricts the agent to only {value} model call(s) per invocation, "
                                    f"which may prevent multi-step reasoning and tool use. Consider:\n"
                                    f"- Setting to `null` (no per-run limit) - recommended for most use cases\n"
                                    f"- Using a higher value (≥ 10) for complex tasks\n"
                                    f"- Relying on `thread_limit` for overall protection"
                                )

                            if value != 0 or param_def['default'] is not None:
                                middleware_config['params'][param_name] = value

                        elif param_def['type'] == 'text':
                            value = st.text_input(
                                label,
                                value=param_def['default'] if param_def['default'] is not None else '',
                                key=f"mw_param_{middleware['type']}_{param_name}",
                                help=help_text  # Add tooltip
                            )
                            if value:
                                middleware_config['params'][param_name] = value

                        elif param_def['type'] == 'textarea':
                            value = st.text_area(
                                label,
                                value=param_def['default'] if param_def['default'] is not None else '',
                                height=150,
                                key=f"mw_param_{middleware['type']}_{param_name}",
                                help=help_text,  # Add tooltip
                                placeholder="Leave empty to use default LangChain prompt"
                            )
                            if value and value.strip():  # Only add if non-empty
                                middleware_config['params'][param_name] = value

                        elif param_def['type'] == 'select':
                            value = st.selectbox(
                                label,
                                options=param_def['options'],
                                index=param_def['options'].index(param_def['default']) if param_def['default'] in param_def['options'] else 0,
                                key=f"mw_param_{middleware['type']}_{param_name}",
                                help=help_text  # Add tooltip
                            )
                            middleware_config['params'][param_name] = value

                        elif param_def['type'] == 'list':
                            value = st.text_input(
                                label,
                                value=', '.join(param_def['default']) if param_def['default'] else '',
                                key=f"mw_param_{middleware['type']}_{param_name}",
                                help=help_text  # Add tooltip
                            )
                            if value:
                                # Convert comma-separated string to list
                                middleware_config['params'][param_name] = [v.strip() for v in value.split(',')]

                selected_middleware.append(middleware_config)

            st.markdown("---")  # Separator between middleware items

    st.markdown("---")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Previous", use_container_width=True):
            update_page_data(6, {'middleware': selected_middleware})
            st.switch_page("pages/5_🧠_Memory.py")

    with col_b2:
        if st.button("Next ➡️", use_container_width=True, type="primary"):
            update_page_data(6, {'middleware': selected_middleware})
            mark_page_complete(6, True)
            st.switch_page("pages/7_🚀_Advanced.py")

with col2:
    st.markdown("### 📄 Configuration Preview")
    display_yaml_preview()

    st.markdown("---")
    if selected_middleware:
        st.success(f"✅ {len(selected_middleware)} middleware configured")
