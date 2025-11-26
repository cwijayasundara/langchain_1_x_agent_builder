"""
Chat Interface Component - Full-featured chat UI for testing agents in Builder UI.
Includes message rendering, input handling, and streaming support.
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.api_client import get_api_client
from utils.state_manager import (
    add_chat_message,
    clear_chat_messages,
    get_chat_messages,
    initialize_chat_state
)
from utils.websocket_client import StreamingHandler
from utils.message_formatter import get_role_display_name, format_tool_call


def inject_chat_styles():
    """Inject custom CSS styles for chat interface."""
    st.markdown("""
    <style>
    /* Blinking cursor for streaming */
    @keyframes blink {
        0%, 49% { opacity: 1; }
        50%, 100% { opacity: 0; }
    }

    /* Message containers */
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .user-message {
        background-color: #E3F2FD;
        margin-left: 20%;
    }

    .ai-message {
        background-color: #F5F5F5;
        margin-right: 20%;
    }

    .system-message {
        background-color: #FFF3E0;
        text-align: center;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)


def render_message(message: Dict[str, Any]):
    """
    Render a single chat message.

    Args:
        message: Message dictionary with role, content, etc.
    """
    role = message.get('role', 'user')
    content = message.get('content', '')
    tool_calls = message.get('tool_calls', [])

    role_name = get_role_display_name(role)

    if role == 'user':
        # User messages
        with st.container():
            cols = st.columns([1, 4])
            with cols[1]:
                st.markdown(f"**{role_name}**")
                st.markdown(
                    f'<div class="chat-message user-message">{content}</div>',
                    unsafe_allow_html=True
                )

    elif role in ['ai', 'assistant']:
        # AI messages
        with st.container():
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"**{role_name}**")

                if content:
                    st.markdown(
                        f'<div class="chat-message ai-message">{content}</div>',
                        unsafe_allow_html=True
                    )

                # Tool calls
                if tool_calls:
                    with st.expander(f"🔧 Tool Calls ({len(tool_calls)})", expanded=False):
                        for idx, tool_call in enumerate(tool_calls, 1):
                            st.markdown(f"**Tool Call {idx}**")
                            st.markdown(format_tool_call(tool_call))
                            if idx < len(tool_calls):
                                st.markdown("---")

    elif role == 'system':
        # System messages
        st.markdown(
            f'<div class="chat-message system-message">{role_name}: {content}</div>',
            unsafe_allow_html=True
        )


def render_streaming_message(accumulated_text: str):
    """
    Render a message that's currently being streamed.

    Args:
        accumulated_text: Text accumulated so far
    """
    with st.container():
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown("**🤖 Agent** _typing..._")
            st.markdown(
                f'<div class="chat-message ai-message">{accumulated_text}<span style="animation: blink 1s infinite;">▊</span></div>',
                unsafe_allow_html=True
            )


def display_chat_container():
    """Display the main chat message container with scrollable history."""
    inject_chat_styles()

    messages = get_chat_messages()

    if not messages:
        st.info("💬 No messages yet. Start the conversation by typing below!")
        return

    for message in messages:
        render_message(message)
        st.markdown("")  # Spacing


def handle_send_message_non_streaming(agent_id: str, user_input: str):
    """
    Handle sending a message in non-streaming mode.

    Args:
        agent_id: Agent identifier
        user_input: User's message content
    """
    # Add user message
    add_chat_message('user', user_input)

    # Build messages payload
    messages_payload = [
        {'role': msg['role'], 'content': msg['content']}
        for msg in st.session_state.chat_messages
    ]

    # Get thread ID and context
    thread_id = st.session_state.get('chat_thread_id')
    context_values = st.session_state.get('chat_context_values', {})

    try:
        api_client = get_api_client()

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
                st.session_state.chat_thread_id = returned_thread_id

            # Add AI response messages
            response_messages = data.get('messages', [])
            for msg in response_messages:
                if msg.get('role') in ['ai', 'assistant']:
                    add_chat_message(
                        role=msg['role'],
                        content=msg.get('content', ''),
                        tool_calls=msg.get('tool_calls')
                    )

            st.rerun()

        else:
            error_msg = response.get('error', 'Unknown error')
            st.error(f"Failed to get response: {error_msg}")

    except Exception as e:
        st.error(f"Error communicating with agent: {str(e)}")


def handle_send_message_streaming(agent_id: str, user_input: str):
    """
    Handle sending a message in streaming mode.

    Args:
        agent_id: Agent identifier
        user_input: User's message content
    """
    # Add user message
    add_chat_message('user', user_input)

    # Display user message immediately
    st.rerun()

    # Build messages payload
    messages_payload = [
        {'role': msg['role'], 'content': msg['content']}
        for msg in st.session_state.chat_messages
    ]

    # Get thread ID and context
    thread_id = st.session_state.get('chat_thread_id')
    context_values = st.session_state.get('chat_context_values', {})

    # Get base URL
    base_url = st.session_state.get('api_base_url', 'http://localhost:8000')

    # Create streaming handler
    handler = StreamingHandler(base_url)

    # Start streaming
    handler.stream_message(
        agent_id=agent_id,
        messages=messages_payload,
        thread_id=thread_id,
        context=context_values if context_values else None
    )

    # Create placeholder for streaming content
    message_placeholder = st.empty()
    accumulated_text = ""

    # Poll for chunks
    with st.spinner("Connecting..."):
        max_wait = 30  # seconds
        start_time = time.time()

        while not handler.is_complete():
            # Check timeout
            if time.time() - start_time > max_wait:
                st.error("Streaming timeout")
                break

            # Get chunk
            chunk_data = handler.get_chunk(timeout=0.1)

            if chunk_data:
                chunk_type, chunk_value = chunk_data

                if chunk_type == "chunk":
                    # Accumulate text
                    accumulated_text += chunk_value

                    # Update display
                    with message_placeholder.container():
                        render_streaming_message(accumulated_text)

                elif chunk_type == "complete":
                    # Save thread ID
                    st.session_state.chat_thread_id = chunk_value

                    # Add completed message
                    add_chat_message('ai', accumulated_text)

                    # Clear placeholder
                    message_placeholder.empty()

                    st.success("Response complete!")
                    time.sleep(0.5)
                    st.rerun()
                    break

                elif chunk_type == "error":
                    error = chunk_value
                    st.error(f"Streaming error: {error.get('message', 'Unknown error')}")
                    break

            # Small delay
            time.sleep(0.05)

    # Check for errors
    error = handler.get_error()
    if error:
        st.error(f"Streaming error: {error.get('message', 'Unknown error')}")
        st.info("Falling back to non-streaming mode...")
        st.session_state.chat_streaming_enabled = False


def display_chat_input(agent_id: str):
    """
    Display the chat input box and controls.

    Args:
        agent_id: Agent identifier
    """
    # Check if waiting for response
    waiting = st.session_state.get('chat_waiting', False)

    # Get agent config to check streaming availability
    agent_config = st.session_state.get('page_7_data', {})
    streaming_config = agent_config.get('streaming', {})
    streaming_available = streaming_config.get('enabled', False)

    # Chat controls row
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        # Streaming toggle (if available)
        if streaming_available:
            streaming_enabled = st.checkbox(
                "⚡ Enable Streaming",
                value=st.session_state.get('chat_streaming_enabled', False),
                key="chat_streaming_toggle",
                help="Enable real-time streaming of responses"
            )
            if streaming_enabled != st.session_state.get('chat_streaming_enabled', False):
                st.session_state.chat_streaming_enabled = streaming_enabled

    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True, disabled=waiting):
            clear_chat_messages()
            st.rerun()

    with col3:
        thread_id = st.session_state.get('chat_thread_id')
        if thread_id:
            st.caption(f"Thread: {thread_id[:8]}...")

    # Message input
    st.markdown("---")

    col_input, col_send = st.columns([6, 1])

    with col_input:
        user_input = st.text_input(
            "Message",
            placeholder="Type your message here...",
            disabled=waiting,
            key="chat_user_input_box",
            label_visibility="collapsed"
        )

    with col_send:
        send_button = st.button(
            "Send",
            disabled=waiting or not user_input,
            type="primary",
            use_container_width=True
        )

    # Handle send
    if send_button and user_input:
        streaming_enabled = st.session_state.get('chat_streaming_enabled', False) and streaming_available

        if streaming_enabled:
            handle_send_message_streaming(agent_id, user_input)
        else:
            handle_send_message_non_streaming(agent_id, user_input)


def display_full_chat_interface(agent_id: str, agent_name: str):
    """
    Display the complete chat interface for testing an agent.

    Args:
        agent_id: Agent identifier
        agent_name: Agent display name
    """
    # Initialize chat state
    initialize_chat_state()

    # Header
    st.markdown(f"### 💬 Test Chat: {agent_name}")
    st.markdown("Interact with your deployed agent to test its functionality.")
    st.markdown("---")

    # Chat container
    chat_container = st.container()

    with chat_container:
        display_chat_container()

    st.markdown("")
    st.markdown("")

    # Chat input at bottom
    display_chat_input(agent_id)
