"""
Page 8: Deploy - Review, validate, and deploy the agent configuration.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import initialize_session_state, get_page_data, reset_all_state, set_test_agent
from utils.styling import apply_custom_styles
from utils.yaml_generator import generate_agent_yaml, get_config_dict
from utils.api_client import get_api_client
from components.navigation import display_page_header
from components.chat_interface import display_full_chat_interface
from components.test_override_panel import display_test_override_panel, display_test_override_status

initialize_session_state()
st.set_page_config(page_title="Deploy", page_icon="✅", layout="wide")
apply_custom_styles()

# Get page data upfront
page_1_data = get_page_data(1)
page_2_data = get_page_data(2)
page_3_data = get_page_data(3)
page_4_data = get_page_data(4)
page_5_data = get_page_data(5)
page_6_data = get_page_data(6)
page_7_data = get_page_data(7)

# Check if agent is deployed - determines layout
is_deployed = st.session_state.get('deployment_success', False)
agent_id = st.session_state.get('deployed_agent_id')
agent_name = st.session_state.get('deployed_agent_name', 'Your Agent')


# ============================================================================
# YAML Config Dialog
# ============================================================================
@st.dialog("Complete Configuration", width="large")
def show_yaml_config_dialog():
    """Display full YAML configuration in a dialog."""
    yaml_content = generate_agent_yaml()

    # Stats at the top
    st.markdown("### 📊 Configuration Stats")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

    with stats_col1:
        st.metric("Tools", len(page_3_data.get('tools', [])))

    with stats_col2:
        st.metric("Middleware", len(page_6_data.get('middleware', [])))

    with stats_col3:
        memory_count = 0
        if page_5_data.get('short_term', {}).get('enabled'):
            memory_count += 1
        if page_5_data.get('long_term', {}).get('enabled'):
            memory_count += 1
        st.metric("Memory Systems", memory_count)

    with stats_col4:
        streaming_count = len(page_7_data.get('streaming', {}).get('modes', []))
        st.metric("Streaming Modes", streaming_count)

    st.markdown("---")
    st.markdown("### 📄 YAML Configuration")
    st.code(yaml_content, language='yaml', line_numbers=True)

    # Download button
    st.download_button(
        label="⬇️ Download YAML",
        data=yaml_content,
        file_name=f"{page_1_data.get('name', 'agent')}_config.yaml",
        mime="text/yaml",
        use_container_width=True
    )


# ============================================================================
# Main Content - Before Deployment (Full Width)
# ============================================================================
if not is_deployed:
    display_page_header(8, "Deploy & Review", "Review your configuration, validate, and deploy your agent.")

    # Configuration Summary
    st.markdown("### 📋 Configuration Summary")

    # Basic Info
    with st.expander("**📝 Basic Info**", expanded=True):
        st.markdown(f"""
        - **Name:** `{page_1_data.get('name', 'N/A')}`
        - **Description:** {page_1_data.get('description', 'N/A')}
        - **Version:** {page_1_data.get('version', 'N/A')}
        - **Tags:** {', '.join(page_1_data.get('tags', [])) if page_1_data.get('tags') else 'None'}
        """)

    # LLM Config
    with st.expander("**🤖 LLM Configuration**"):
        st.markdown(f"""
        - **Provider:** {page_2_data.get('provider', 'N/A')}
        - **Model:** `{page_2_data.get('model', 'N/A')}`
        - **Temperature:** {page_2_data.get('temperature', 'N/A')}
        - **Max Tokens:** {page_2_data.get('max_tokens', 'Default')}
        """)

    # Tools
    with st.expander("**🔧 Tools**"):
        tools = page_3_data.get('tools', [])
        if tools:
            st.markdown(f"**Selected Tools:** {len(tools)}")
            st.caption(', '.join(tools))
        else:
            st.markdown("No tools selected")

    # Prompts
    with st.expander("**💬 Prompts**"):
        system_prompt_preview = page_4_data.get('system_prompt', 'N/A')
        if len(system_prompt_preview) > 150:
            system_prompt_preview = system_prompt_preview[:150] + "..."
        st.markdown(f"**System Prompt:** {system_prompt_preview}")
        if page_4_data.get('user_template'):
            st.markdown(f"**User Template:** {page_4_data.get('user_template')}")

    # Memory
    with st.expander("**🧠 Memory**"):
        st_enabled = page_5_data.get('short_term', {}).get('enabled', False)
        lt_enabled = page_5_data.get('long_term', {}).get('enabled', False)
        st.markdown(f"""
        - **Short-term Memory:** {'✅ Enabled' if st_enabled else '❌ Disabled'}
        - **Long-term Memory:** {'✅ Enabled' if lt_enabled else '❌ Disabled'}
        """)

    # Middleware
    with st.expander("**⚙️ Middleware**"):
        middleware = page_6_data.get('middleware', [])
        if middleware:
            st.markdown(f"**Active Middleware:** {len(middleware)}")
            for mw in middleware:
                st.caption(f"- {mw.get('type', 'unknown')}")
        else:
            st.markdown("No middleware configured")

    # Advanced
    with st.expander("**🚀 Advanced Settings**"):
        streaming_enabled = page_7_data.get('streaming', {}).get('enabled', False)
        st.markdown(f"**Streaming:** {'✅ Enabled' if streaming_enabled else '❌ Disabled'}")
        if streaming_enabled:
            modes = page_7_data.get('streaming', {}).get('modes', [])
            st.caption(f"Modes: {', '.join(modes) if modes else 'None'}")

    # View Full YAML Config button
    if st.button("📄 View Full YAML Config", use_container_width=False):
        show_yaml_config_dialog()

    st.markdown("---")

    # Validation Section
    st.markdown("### 🔍 Validation")

    col_val1, col_val2 = st.columns(2)

    with col_val1:
        if st.button("🔍 Validate Configuration", use_container_width=True):
            with st.spinner("Validating configuration..."):
                try:
                    api_client = get_api_client()
                    config_dict = get_config_dict()
                    result = api_client.validate_config(config_dict)

                    # Check if API call itself failed
                    if result.get('error'):
                        st.error(f"❌ API Error: {result['error']}")
                        st.session_state.validation_passed = False
                    else:
                        # Extract validation data from APIResponse wrapper
                        data = result.get('data', {})

                        if data.get('valid', False):
                            st.success("✅ Configuration is valid!")

                            # Show warnings if any
                            warnings = data.get('warnings', [])
                            if warnings:
                                with st.expander("⚠️ Warnings", expanded=False):
                                    for warning in warnings:
                                        st.warning(warning)

                            st.session_state.validation_passed = True
                        else:
                            st.error("❌ Configuration validation failed")

                            # Display detailed errors
                            errors = data.get('errors', [])
                            if errors:
                                with st.expander("🔍 Validation Errors", expanded=True):
                                    for error in errors:
                                        if isinstance(error, dict):
                                            field = error.get('field', 'Unknown field')
                                            message = error.get('message', 'Unknown error')
                                            error_type = error.get('type', '')
                                            st.error(f"**{field}**: {message}")
                                            if error_type:
                                                st.caption(f"Error type: {error_type}")
                                        else:
                                            st.error(str(error))
                            else:
                                st.error("Validation failed with no specific errors reported.")

                            st.session_state.validation_passed = False

                except Exception as e:
                    st.error(f"❌ Validation error: {str(e)}")
                    st.session_state.validation_passed = False

    with col_val2:
        yaml_content = generate_agent_yaml()
        st.download_button(
            label="📥 Download YAML",
            data=yaml_content,
            file_name=f"{page_1_data.get('name', 'agent')}_config.yaml",
            mime="text/yaml",
            use_container_width=True
        )

    st.markdown("---")

    # Deployment Section
    st.markdown("### 🚀 Deployment")

    # Show validation status
    if 'validation_passed' in st.session_state:
        if st.session_state.validation_passed:
            st.success("✅ Configuration validated successfully")
        else:
            st.warning("⚠️ Please fix validation errors before deploying")

    deploy_immediately = st.checkbox("Deploy immediately after creation", value=True)

    col_deploy1, col_deploy2 = st.columns(2)

    with col_deploy1:
        if st.button("🚀 Deploy Agent", use_container_width=True, type="primary"):
            # Check if validation passed
            if not st.session_state.get('validation_passed', False):
                st.warning("⚠️ Please validate your configuration first")
            else:
                with st.spinner("Deploying agent..."):
                    try:
                        api_client = get_api_client()
                        config_dict = get_config_dict()
                        result = api_client.create_agent(config_dict, deploy=deploy_immediately)

                        if result.get('success', False):
                            # Extract agent data from APIResponse
                            data = result.get('data', {})
                            agent_id = data.get('agent_id', 'N/A')

                            st.success("🎉 Agent deployed successfully!")
                            st.info(f"**Agent ID:** `{agent_id}`")
                            st.balloons()
                            st.session_state.deployment_success = True
                            st.session_state.deployed_agent_id = agent_id
                            st.session_state.deployed_agent_name = page_1_data.get('name', 'Unknown')

                            # Set test agent for chat
                            set_test_agent(agent_id)
                            st.rerun()
                        else:
                            st.error("❌ Deployment failed")
                            error_msg = result.get('error', 'Unknown error')
                            st.error(f"**Error:** {error_msg}")
                            st.session_state.deployment_success = False

                    except Exception as e:
                        st.error(f"❌ Deployment error: {str(e)}")
                        st.session_state.deployment_success = False

    with col_deploy2:
        if st.button("🔄 Start New Agent", use_container_width=True):
            reset_all_state()
            st.success("✅ Session reset! Starting fresh...")
            st.rerun()

    st.markdown("---")

    # Navigation
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Previous", use_container_width=True):
            st.switch_page("pages/7_🚀_Advanced.py")

    with col_b2:
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("app.py")


# ============================================================================
# Main Content - After Deployment (Two-Column Layout: Chat | Overrides)
# ============================================================================
else:
    # Two-column layout: Chat on left, Runtime Overrides on right
    col_chat, col_overrides = st.columns([3, 1])

    with col_chat:
        st.markdown(f"## 🧪 Test Your Agent: {agent_name}")
        st.caption(f"Agent ID: `{agent_id}`")

        # Show override status if active
        display_test_override_status()

        st.markdown("---")

        # Chat description
        st.markdown("""
        Test your deployed agent with real queries:
        - **Real-time messaging** with streaming support
        - **Tool call visualization** to see agent actions
        - **Runtime overrides** to test different configurations
        """)

        st.markdown("---")

        # Display the full chat interface
        if agent_id:
            display_full_chat_interface(agent_id, agent_name)

        st.markdown("---")

        # Action buttons
        col_action1, col_action2, col_action3 = st.columns(3)

        with col_action1:
            if st.button("📄 View Config", use_container_width=True):
                show_yaml_config_dialog()

        with col_action2:
            if st.button("🔄 New Agent", use_container_width=True):
                reset_all_state()
                st.success("✅ Session reset!")
                st.rerun()

        with col_action3:
            if st.button("🏠 Home", use_container_width=True):
                st.switch_page("app.py")

    with col_overrides:
        st.markdown("### ⚙️ Runtime Overrides")
        st.caption("Test with different configurations without redeploying.")

        # Display the runtime override panel
        if agent_id:
            display_test_override_panel(agent_id)
