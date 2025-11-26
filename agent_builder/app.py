"""
Agent Builder UI - Main Entry Point
Multi-page Streamlit application for configuring LangChain agents.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from components.template_selector import display_template_selector
from utils.state_manager import initialize_session_state
from utils.api_client import check_api_availability

# Page configuration
st.set_page_config(
    page_title="Agent Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
initialize_session_state()

# Custom CSS for improved visibility
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
    .stProgress > div > div > div > div {
        background-color: #00c853;
    }

    /* Improve text input visibility */
    .stTextInput > div > div > input {
        color: #000000 !important;
        font-weight: 500 !important;
        background-color: #ffffff !important;
    }

    /* Dark mode text inputs */
    [data-theme="dark"] .stTextInput > div > div > input {
        color: #ffffff !important;
        background-color: #262730 !important;
    }

    /* Text area visibility */
    .stTextArea > div > div > textarea {
        color: #000000 !important;
        font-weight: 500 !important;
        background-color: #ffffff !important;
    }

    /* Dark mode text areas */
    [data-theme="dark"] .stTextArea > div > div > textarea {
        color: #ffffff !important;
        background-color: #262730 !important;
    }

    /* Improve label visibility */
    .stTextInput > label,
    .stTextArea > label,
    .stSelectbox > label,
    .stNumberInput > label,
    .stMultiSelect > label {
        color: #1f1f1f !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* Dark mode labels */
    [data-theme="dark"] .stTextInput > label,
    [data-theme="dark"] .stTextArea > label,
    [data-theme="dark"] .stSelectbox > label,
    [data-theme="dark"] .stNumberInput > label,
    [data-theme="dark"] .stMultiSelect > label {
        color: #fafafa !important;
    }

    /* Placeholder text visibility */
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #666666 !important;
        opacity: 0.7 !important;
    }

    /* Dark mode placeholders */
    [data-theme="dark"] .stTextInput input::placeholder,
    [data-theme="dark"] .stTextArea textarea::placeholder {
        color: #a0a0a0 !important;
    }

    /* Select box visibility */
    .stSelectbox > div > div > div {
        color: #000000 !important;
        font-weight: 500 !important;
    }

    [data-theme="dark"] .stSelectbox > div > div > div {
        color: #ffffff !important;
    }

    /* Number input visibility */
    .stNumberInput > div > div > input {
        color: #000000 !important;
        font-weight: 500 !important;
        background-color: #ffffff !important;
    }

    [data-theme="dark"] .stNumberInput > div > div > input {
        color: #ffffff !important;
        background-color: #262730 !important;
    }

    /* Multi-select visibility */
    .stMultiSelect > div > div > div {
        color: #000000 !important;
    }

    [data-theme="dark"] .stMultiSelect > div > div > div {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Main app logic
def main():
    """Main application entry point."""

    # Check API availability
    api_status = check_api_availability()

    # Sidebar
    with st.sidebar:
        st.markdown("### 🤖 Agent Builder")

        if st.session_state.get('template_selected'):
            template_name = st.session_state.get('template_name', 'Unknown')
            if template_name == 'blank':
                st.info("📄 Starting from blank")
            else:
                st.success(f"📋 Template: {template_name}")

        st.markdown("---")

        # API Status
        col1, col2 = st.columns([3, 1])
        with col1:
            if api_status:
                st.success("✅ API Connected")
            else:
                st.warning("⚠️ API Offline")
                st.caption("Working in offline mode")
        with col2:
            if st.button("🔄", help="Retry API connection", key="retry_api"):
                st.cache_resource.clear()
                st.rerun()

        st.markdown("---")

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

            if st.button("🔄 Reset All"):
                from utils.state_manager import reset_all_state
                reset_all_state()
                st.rerun()

    # Template selection screen
    if not st.session_state.get('template_selected'):
        display_template_selector()
    else:
        # Show navigation to pages
        st.markdown('<div class="main-header">🤖 Agent Builder</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Configure your LangChain agent step by step</div>', unsafe_allow_html=True)

        st.markdown("### 📋 Configuration Steps")
        st.markdown("Use the sidebar to navigate between pages or click below:")

        # Page buttons
        col1, col2, col3, col4 = st.columns(4)

        pages = [
            ("1_📝_Basic_Info.py", "📝 Basic Info", "Name, description, version", 1),
            ("2_🤖_LLM_Config.py", "🤖 LLM Config", "Model and parameters", 2),
            ("3_🔧_Tools.py", "🔧 Tools", "Built-in and custom tools", 3),
            ("4_💬_Prompts.py", "💬 Prompts", "System and user prompts", 4),
            ("5_🧠_Memory.py", "🧠 Memory", "Short and long-term memory", 5),
            ("6_⚙️_Middleware.py", "⚙️ Middleware", "Processing middleware", 6),
            ("7_🚀_Advanced.py", "🚀 Advanced", "Streaming and runtime", 7),
            ("8_✅_Deploy.py", "✅ Deploy", "Review and deploy", 8)
        ]

        for idx, (filename, title, desc, page_num) in enumerate(pages):
            col = [col1, col2, col3, col4][idx % 4]
            with col:
                is_complete = st.session_state.get(f'page_{page_num}_complete', False)
                status = "✅" if is_complete else "⭕"

                if st.button(f"{status} {title}", key=f"nav_{page_num}", use_container_width=True):
                    st.switch_page(f"pages/{filename}")

                st.caption(desc)

        # Quick actions
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")

        action_col1, action_col2 = st.columns(2)

        with action_col1:
            if st.button("📄 View YAML Preview", use_container_width=True):
                from utils.yaml_generator import generate_agent_yaml
                yaml_content = generate_agent_yaml()
                st.code(yaml_content, language='yaml')

        with action_col2:
            if st.button("✅ Go to Deploy", use_container_width=True):
                st.switch_page("pages/8_✅_Deploy.py")


if __name__ == "__main__":
    main()
