"""
Page 2: LLM Configuration - Provider, model, and parameters.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import (
    initialize_session_state,
    get_page_data,
    update_page_data,
    mark_page_complete
)
from utils.validators import validate_page_2
from utils.styling import apply_custom_styles
from utils.constants import LLM_PROVIDERS, DEFAULTS
from components.yaml_preview import display_yaml_preview
from components.navigation import display_page_header

initialize_session_state()
st.set_page_config(page_title="LLM Config", page_icon="🤖", layout="wide")
apply_custom_styles()

col1, col2 = st.columns([3, 2])

with col1:
    display_page_header(
        2,
        "LLM Configuration",
        "Select your language model provider and configure parameters."
    )

    current_data = get_page_data(2)

    # Provider & Model Selection (outside form for dynamic updates)
    st.markdown("### Provider & Model")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        provider = st.selectbox(
            "Provider *",
            options=list(LLM_PROVIDERS.keys()),
            format_func=lambda x: LLM_PROVIDERS[x]["name"],
            index=list(LLM_PROVIDERS.keys()).index(current_data.get('provider', 'openai')),
            key="provider_select"
        )

    with col_p2:
        models = LLM_PROVIDERS[provider]["models"]
        current_model = current_data.get('model', '')

        # Validate that current model belongs to selected provider
        if current_model not in models:
            current_model = models[0]

        model_index = models.index(current_model)

        model = st.selectbox(
            "Model *",
            options=models,
            index=model_index,
            key="model_select"
        )

    st.markdown("---")

    # Configuration Form (for parameters only)
    with st.form("llm_config_form"):
        st.markdown("### Parameters")

        # Temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(current_data.get('temperature', DEFAULTS['temperature'])),
            step=0.1,
            help="Controls randomness. Lower is more focused, higher is more creative."
        )

        # Max tokens
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=128000,
            value=current_data.get('max_tokens', DEFAULTS['max_tokens']),
            step=256,
            help="Maximum number of tokens to generate"
        )

        # Top P (optional)
        use_top_p = st.checkbox(
            "Use Top P sampling",
            value=current_data.get('top_p') is not None
        )

        top_p = None
        if use_top_p:
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=float(current_data.get('top_p', 0.9)),
                step=0.05,
                help="Nucleus sampling parameter"
            )

        # API Key override (optional)
        with st.expander("🔑 API Key Override (Optional)"):
            # Get provider from session state
            selected_provider = st.session_state.get('provider_select', current_data.get('provider', 'openai'))
            st.caption(f"Default environment variable: {LLM_PROVIDERS[selected_provider]['env_key']}")
            api_key = st.text_input(
                "API Key",
                value=current_data.get('api_key', '') or '',
                type="password",
                help="Leave blank to use environment variable"
            )

        st.markdown("---")

        col_b1, col_b2 = st.columns([1, 1])

        with col_b1:
            prev_btn = st.form_submit_button("⬅️ Previous", use_container_width=True)

        with col_b2:
            next_btn = st.form_submit_button("Next ➡️", use_container_width=True, type="primary")

        if prev_btn or next_btn:
            # Get provider and model from session state (outside form)
            form_data = {
                'provider': st.session_state.get('provider_select', current_data.get('provider', 'openai')),
                'model': st.session_state.get('model_select', current_data.get('model', '')),
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                'api_key': api_key if api_key else None
            }

            # Validate
            errors = validate_page_2(form_data)

            if errors and next_btn:
                for error in errors:
                    st.error(f"**{error.field}**: {error.message}")
            else:
                # Save
                update_page_data(2, form_data)
                mark_page_complete(2, True)

                if prev_btn:
                    st.switch_page("pages/1_📝_Basic_Info.py")
                elif next_btn:
                    st.success("✅ LLM configuration saved!")
                    st.switch_page("pages/3_🔧_Tools.py")

with col2:
    st.markdown("### 📄 Configuration Preview")
    display_yaml_preview()

    st.markdown("---")
    st.markdown("### 💡 Model Recommendations")

    st.info("""
    **OpenAI**
    • gpt-4o: Best for complex reasoning (128K context)
    • gpt-4o-mini: Fast & cost-effective (128K context)
    • o3-mini: Enhanced reasoning capabilities

    **Anthropic**
    • claude-sonnet-4-5: Best for agents & coding (200K-1M context)
    • claude-haiku-4-5: Fastest with frontier intelligence
    • claude-opus-4-1: Exceptional specialized reasoning

    **Google Gemini**
    • gemini-2.5-pro: State-of-the-art reasoning (1M+ context)
    • gemini-2.5-flash: Best price-performance (1M+ context)

    **Groq**
    • llama-3.3-70b-versatile: 280 tokens/sec, excellent quality
    • groq/compound: Agentic system with built-in tools
    """)
