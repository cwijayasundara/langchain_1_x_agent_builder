"""
Page 4: Prompts - System prompt, user template, and examples.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import initialize_session_state, get_page_data, update_page_data, mark_page_complete
from utils.validators import validate_page_4
from utils.styling import apply_custom_styles
from utils.constants import PROMPT_VARIABLES, BUILTIN_TOOLS
from utils.llm_prompt_enhancer import enhance_prompt, estimate_enhancement_cost
from components.yaml_preview import display_yaml_preview
from components.navigation import display_page_header

initialize_session_state()
st.set_page_config(page_title="Prompts", page_icon="💬", layout="wide")
apply_custom_styles()

col1, col2 = st.columns([3, 2])

with col1:
    display_page_header(4, "Prompts", "Configure system prompt, user message template, and examples.")

    current_data = get_page_data(4)

    # AI Prompt Enhancement Section
    st.markdown("### ✨ AI Prompt Enhancement")
    st.caption("Generate an optimized prompt based on your agent's capabilities and selected tools using GPT-4o-mini")

    col_enhance_info, col_enhance_btn = st.columns([3, 1])

    with col_enhance_info:
        # Get data from previous pages to show context
        tools_data = get_page_data(3)  # Tools page
        basic_info = get_page_data(1)
        llm_config = get_page_data(2)

        tools_count = len(tools_data.get('tools', []))
        mcp_count = len(tools_data.get('mcp_servers', []))

        if tools_count > 0 or mcp_count > 0:
            st.info(f"📊 Context: {tools_count} built-in tool(s) + {mcp_count} MCP server(s)")
        else:
            st.warning("⚠️ No tools selected yet. Enhancement will create a general prompt.")

    with col_enhance_btn:
        enhance_btn = st.button("🪄 Enhance", use_container_width=True, type="secondary")

    # Handle enhancement
    if enhance_btn:
        if not basic_info.get('name'):
            st.error("❌ Please complete Basic Info (Page 1) first")
        elif not llm_config.get('model'):
            st.error("❌ Please complete LLM Config (Page 2) first")
        else:
            try:
                with st.spinner("Enhancing prompt with GPT-4o-mini..."):
                    enhanced = enhance_prompt(
                        current_prompt=current_data.get('system_prompt', ''),
                        agent_name=basic_info.get('name', 'agent'),
                        agent_description=basic_info.get('description', ''),
                        selected_tools=tools_data.get('tools', []),
                        mcp_servers=tools_data.get('mcp_servers', []),
                        llm_model=llm_config.get('model', 'unknown'),
                        builtin_tools_metadata=BUILTIN_TOOLS
                    )

                # Store in session state
                st.session_state.enhanced_prompt = enhanced
                st.session_state.show_comparison = True
                st.success("✅ Prompt enhanced successfully!")
                st.rerun()

            except ValueError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                st.error(f"❌ Enhancement failed: {str(e)}")

    # Show comparison if enhancement was just done
    if st.session_state.get('show_comparison') and st.session_state.get('enhanced_prompt'):
        with st.expander("📝 Compare: Original vs Enhanced", expanded=True):
            col_orig, col_enh = st.columns(2)

            with col_orig:
                st.markdown("**Original**")
                original_text = current_data.get('system_prompt', '(No prompt yet)')
                st.text_area(
                    "orig",
                    value=original_text,
                    height=250,
                    disabled=True,
                    label_visibility="collapsed"
                )

            with col_enh:
                st.markdown("**Enhanced**")
                st.text_area(
                    "enh",
                    value=st.session_state.enhanced_prompt,
                    height=250,
                    disabled=True,
                    label_visibility="collapsed"
                )

            col_accept, col_reject = st.columns(2)

            with col_accept:
                if st.button("✅ Use Enhanced Prompt", use_container_width=True, type="primary", key="accept_enhanced"):
                    current_data['system_prompt'] = st.session_state.enhanced_prompt
                    update_page_data(4, current_data)
                    st.session_state.show_comparison = False
                    st.success("✅ Enhanced prompt applied!")
                    st.rerun()

            with col_reject:
                if st.button("❌ Keep Original", use_container_width=True, key="reject_enhanced"):
                    st.session_state.show_comparison = False
                    st.info("Keeping your original prompt")
                    st.rerun()

    st.markdown("---")

    with st.form("prompts_form"):
        st.markdown("### System Prompt *")
        st.caption("Define how your agent should behave. You can use variables like {{agent_name}}, {{date}}, etc.")

        system_prompt = st.text_area(
            "System Prompt",
            value=current_data.get('system_prompt', ''),
            placeholder="You are a helpful AI assistant named {{agent_name}}...",
            height=200,
            label_visibility="collapsed"
        )

        st.markdown("**Available Variables:**")
        for var_info in PROMPT_VARIABLES:
            st.caption(f"`{var_info['var']}` - {var_info['description']}")

        st.markdown("---")
        st.markdown("### User Message Template (Optional)")

        user_template = st.text_area(
            "User Template",
            value=current_data.get('user_template', '') or '',
            placeholder="User query: {{query}}",
            height=100,
            label_visibility="collapsed"
        )

        st.markdown("---")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            prev_btn = st.form_submit_button("⬅️ Previous", use_container_width=True)
        with col_b2:
            next_btn = st.form_submit_button("Next ➡️", use_container_width=True, type="primary")

        if prev_btn or next_btn:
            form_data = {
                'system_prompt': system_prompt,
                'user_template': user_template if user_template else None,
                'few_shot_examples': current_data.get('few_shot_examples', [])
            }

            errors = validate_page_4(form_data)

            if errors and next_btn:
                for error in errors:
                    st.error(f"**{error.field}**: {error.message}")
            else:
                update_page_data(4, form_data)
                mark_page_complete(4, True)

                if prev_btn:
                    st.switch_page("pages/3_🔧_Tools.py")
                else:
                    st.success("✅ Prompts saved!")
                    st.switch_page("pages/5_🧠_Memory.py")

with col2:
    st.markdown("### 📄 Configuration Preview")
    display_yaml_preview()

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    - Be specific about the agent's role
    - Include relevant context
    - Use variables for dynamic content
    - Test different phrasings
    """)
