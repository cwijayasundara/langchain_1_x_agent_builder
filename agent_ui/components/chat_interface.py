"""
Chat Interface Component - Main chat UI container and input handling.
"""

import streamlit as st
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import add_message, save_current_thread, create_new_thread
from utils.api_client import get_api_client
from components.message_renderer import render_message_list, render_loading_message, inject_message_styles


def display_chat_container():
    """
    Display the main chat message container with scrollable history.
    """
    # Inject custom styles
    inject_message_styles()

    # Get preferences
    show_timestamps = st.session_state.get('ui_preferences', {}).get('show_timestamps', True)
    show_metadata = st.session_state.get('ui_preferences', {}).get('show_token_usage', True)

    # Display messages
    render_message_list(
        st.session_state.get('messages', []),
        show_timestamps=show_timestamps,
        show_metadata=show_metadata
    )

    # Show loading indicator if waiting for response
    if st.session_state.get('waiting_for_response', False):
        render_loading_message()


def handle_send_message(user_input: str, streaming: bool = False):
    """
    Handle sending a message to the agent.

    Args:
        user_input: User's message content
        streaming: Whether to use streaming mode
    """
    if not user_input or not user_input.strip():
        return

    # Add user message
    add_message('user', user_input)

    # Save to thread
    save_current_thread()

    # Set waiting state
    st.session_state.waiting_for_response = True
    st.session_state.input_disabled = True

    # Get agent info
    agent_id = st.session_state.get('selected_agent_id')
    if not agent_id:
        st.error("No agent selected")
        return

    # Get context values
    context_values = st.session_state.get('context_values', {})

    # Get current thread ID (or None for first message)
    thread_id = st.session_state.get('current_thread_id')

    # Build messages payload - optimize for thread continuation
    # When thread_id exists, checkpointer already has history - only send new message
    if thread_id:
        messages_payload = [{'role': 'user', 'content': user_input}]
    else:
        # New thread - send full context
        messages_payload = [
            {'role': msg['role'], 'content': msg['content']}
            for msg in st.session_state.messages
        ]

    try:
        api_client = get_api_client()

        if streaming:
            # Streaming mode - handled separately
            st.session_state.streaming_active = True
        else:
            # Non-streaming mode
            with st.spinner("Agent is thinking..."):
                response = api_client.invoke_agent(
                    agent_id=agent_id,
                    messages=messages_payload,
                    thread_id=thread_id,
                    context=context_values if context_values else None
                )

            if response.get('success', True):
                data = response.get('data', {})

                # Update thread ID
                returned_thread_id = data.get('thread_id')
                if returned_thread_id:
                    st.session_state.current_thread_id = returned_thread_id

                # Add AI response messages
                response_messages = data.get('messages', [])
                for msg in response_messages:
                    role = msg.get('role')
                    content = msg.get('content', '')
                    message_id = msg.get('id')

                    # Skip user messages (already have them)
                    if role == 'user':
                        continue

                    # Skip empty AI messages without tool calls
                    if role in ['ai', 'assistant'] and not content and not msg.get('tool_calls'):
                        continue

                    # Add AI/assistant messages
                    if role in ['ai', 'assistant']:
                        add_message(
                            role=role,
                            content=content,
                            tool_calls=msg.get('tool_calls'),
                            message_id=message_id
                        )

                # Save to thread
                save_current_thread()

            else:
                error_msg = response.get('error', 'Unknown error')
                st.error(f"Failed to get response: {error_msg}")

    except Exception as e:
        st.error(f"Error communicating with agent: {str(e)}")

    finally:
        # Reset waiting state
        st.session_state.waiting_for_response = False
        st.session_state.input_disabled = False


def display_chat_input():
    """
    Display the chat input box and send button.
    """
    # Check if input should be disabled
    disabled = st.session_state.get('input_disabled', False)

    # Check if streaming is enabled
    agent_config = st.session_state.get('selected_agent_config', {})
    streaming_config = agent_config.get('streaming') or {}
    streaming_available = streaming_config.get('enabled', False)

    # User input
    col1, col2 = st.columns([6, 1])

    with col1:
        user_input = st.text_input(
            "Message",
            placeholder="Type your message here...",
            disabled=disabled,
            key="user_input_box",
            label_visibility="collapsed"
        )

    with col2:
        send_button = st.button(
            "Send",
            disabled=disabled,
            type="primary",
            use_container_width=True
        )

    # Handle send
    if send_button and user_input:
        streaming_enabled = st.session_state.get('streaming_enabled', False) and streaming_available
        handle_send_message(user_input, streaming=streaming_enabled)
        # Clear input by rerunning
        st.rerun()


def display_thread_controls():
    """
    Display thread control buttons (New Thread, etc.).
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("➕ New Thread", use_container_width=True):
            new_thread_id = create_new_thread()
            st.success(f"Started new thread")
            st.rerun()

    with col2:
        # Thread selector
        threads = st.session_state.get('threads', {})
        if threads:
            thread_options = {
                tid: threads[tid].get('label', f"Thread {idx+1}")
                for idx, tid in enumerate(threads.keys())
            }

            current_thread = st.session_state.get('current_thread_id')

            selected = st.selectbox(
                "Select Thread",
                options=list(thread_options.keys()),
                format_func=lambda tid: thread_options[tid],
                index=list(thread_options.keys()).index(current_thread) if current_thread in thread_options else 0,
                key="thread_selector",
                label_visibility="collapsed"
            )

            if selected != current_thread:
                from utils.state_manager import switch_thread
                switch_thread(selected)
                st.rerun()

    with col3:
        # Streaming toggle (if available)
        agent_config = st.session_state.get('selected_agent_config', {})
        streaming_config = agent_config.get('streaming', {})

        if streaming_config.get('enabled', False):
            streaming_enabled = st.checkbox(
                "⚡ Streaming",
                value=st.session_state.get('streaming_enabled', False),
                key="streaming_toggle",
                help="Enable real-time streaming of responses"
            )

            if streaming_enabled != st.session_state.get('streaming_enabled', False):
                st.session_state.streaming_enabled = streaming_enabled
