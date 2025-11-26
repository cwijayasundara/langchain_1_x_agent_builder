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

initialize_session_state()
st.set_page_config(page_title="Deploy", page_icon="✅", layout="wide")
apply_custom_styles()

col1, col2 = st.columns([3, 2])

with col1:
    display_page_header(8, "Deploy & Review", "Review your configuration, validate, and deploy your agent.")

    # Configuration Summary
    st.markdown("### 📋 Configuration Summary")

    # Basic Info
    page_1_data = get_page_data(1)
    with st.expander("**📝 Basic Info**", expanded=True):
        st.markdown(f"""
        - **Name:** `{page_1_data.get('name', 'N/A')}`
        - **Description:** {page_1_data.get('description', 'N/A')}
        - **Version:** {page_1_data.get('version', 'N/A')}
        - **Tags:** {', '.join(page_1_data.get('tags', [])) if page_1_data.get('tags') else 'None'}
        """)

    # LLM Config
    page_2_data = get_page_data(2)
    with st.expander("**🤖 LLM Configuration**"):
        st.markdown(f"""
        - **Provider:** {page_2_data.get('provider', 'N/A')}
        - **Model:** `{page_2_data.get('model', 'N/A')}`
        - **Temperature:** {page_2_data.get('temperature', 'N/A')}
        - **Max Tokens:** {page_2_data.get('max_tokens', 'Default')}
        """)

    # Tools
    page_3_data = get_page_data(3)
    with st.expander("**🔧 Tools**"):
        tools = page_3_data.get('tools', [])
        if tools:
            st.markdown(f"**Selected Tools:** {len(tools)}")
            st.caption(', '.join(tools))
        else:
            st.markdown("No tools selected")

    # Prompts
    page_4_data = get_page_data(4)
    with st.expander("**💬 Prompts**"):
        system_prompt_preview = page_4_data.get('system_prompt', 'N/A')
        if len(system_prompt_preview) > 150:
            system_prompt_preview = system_prompt_preview[:150] + "..."
        st.markdown(f"**System Prompt:** {system_prompt_preview}")
        if page_4_data.get('user_template'):
            st.markdown(f"**User Template:** {page_4_data.get('user_template')}")

    # Memory
    page_5_data = get_page_data(5)
    with st.expander("**🧠 Memory**"):
        st_enabled = page_5_data.get('short_term', {}).get('enabled', False)
        lt_enabled = page_5_data.get('long_term', {}).get('enabled', False)
        st.markdown(f"""
        - **Short-term Memory:** {'✅ Enabled' if st_enabled else '❌ Disabled'}
        - **Long-term Memory:** {'✅ Enabled' if lt_enabled else '❌ Disabled'}
        """)

    # Middleware
    page_6_data = get_page_data(6)
    with st.expander("**⚙️ Middleware**"):
        middleware = page_6_data.get('middleware', [])
        if middleware:
            st.markdown(f"**Active Middleware:** {len(middleware)}")
            for mw in middleware:
                st.caption(f"- {mw.get('type', 'unknown')}")
        else:
            st.markdown("No middleware configured")

    # Advanced
    page_7_data = get_page_data(7)
    with st.expander("**🚀 Advanced Settings**"):
        streaming_enabled = page_7_data.get('streaming', {}).get('enabled', False)
        st.markdown(f"**Streaming:** {'✅ Enabled' if streaming_enabled else '❌ Disabled'}")
        if streaming_enabled:
            modes = page_7_data.get('streaming', {}).get('modes', [])
            st.caption(f"Modes: {', '.join(modes) if modes else 'None'}")

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
        if st.button("📥 Download YAML", use_container_width=True):
            yaml_content = generate_agent_yaml()
            agent_name = page_1_data.get('name', 'agent')
            st.download_button(
                label="⬇️ Download Configuration",
                data=yaml_content,
                file_name=f"{agent_name}_config.yaml",
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

    # Test Chat Section (only show if agent is deployed)
    if st.session_state.get('deployment_success', False):
        agent_id = st.session_state.get('deployed_agent_id')
        agent_name = st.session_state.get('deployed_agent_name', 'Your Agent')

        if agent_id:
            st.markdown("### 🧪 Test Your Agent")

            # Expandable test interface
            with st.expander("💬 Interactive Chat Test", expanded=True):
                st.markdown("""
                Test your deployed agent with real queries. This is a full-featured chat interface with:
                - **Real-time messaging** (streaming and non-streaming modes)
                - **Conversation history** within this session
                - **Tool call visualization** to see what tools your agent uses
                - **Context support** for runtime variables
                """)

                st.markdown("---")

                # Display the full chat interface
                display_full_chat_interface(agent_id, agent_name)

            st.markdown("---")

    # Navigation
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Previous", use_container_width=True):
            st.switch_page("pages/7_🚀_Advanced.py")

    with col_b2:
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("app.py")

with col2:
    st.markdown("### 📄 Complete Configuration")

    yaml_content = generate_agent_yaml()
    st.code(yaml_content, language='yaml', line_numbers=True)

    # Stats
    st.markdown("---")
    st.markdown("### 📊 Configuration Stats")

    config_dict = get_config_dict()

    stats_col1, stats_col2 = st.columns(2)

    with stats_col1:
        st.metric("Tools", len(page_3_data.get('tools', [])))
        st.metric("Middleware", len(page_6_data.get('middleware', [])))

    with stats_col2:
        memory_count = 0
        if page_5_data.get('short_term', {}).get('enabled'):
            memory_count += 1
        if page_5_data.get('long_term', {}).get('enabled'):
            memory_count += 1
        st.metric("Memory Systems", memory_count)

        streaming_count = len(page_7_data.get('streaming', {}).get('modes', []))
        st.metric("Streaming Modes", streaming_count)

    # Tips
    st.markdown("---")
    st.markdown("### 💡 Deployment Tips")
    st.info("""
    - Always validate before deploying
    - Test with sample queries first
    - Monitor resource usage
    - Keep your configuration backed up
    - Update API keys securely
    """)
