"""
Streaming Handler Component - Manages WebSocket streaming for real-time responses.
"""

import streamlit as st
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.websocket_client import StreamingHandler
from utils.state_manager import add_message, save_current_thread
from components.message_renderer import render_streaming_message


def handle_streaming_message(user_input: str):
    """
    Handle sending a message with streaming.

    Args:
        user_input: User's message content
    """
    # Get agent info
    agent_id = st.session_state.get('selected_agent_id')
    if not agent_id:
        st.error("No agent selected")
        return

    # Get API base URL
    base_url = st.session_state.get('api_base_url', 'http://localhost:8000')

    # Get context values
    context_values = st.session_state.get('context_values', {})

    # Build messages payload
    messages_payload = [
        {'role': msg['role'], 'content': msg['content']}
        for msg in st.session_state.messages
    ]

    # Get current thread ID
    thread_id = st.session_state.get('current_thread_id')

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
        import time
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
                    st.session_state.current_thread_id = chunk_value

                    # Add completed message
                    add_message('ai', accumulated_text)
                    save_current_thread()

                    # Clear placeholder and show final message
                    message_placeholder.empty()

                    st.success("Response complete!")
                    break

                elif chunk_type == "error":
                    error = chunk_value
                    st.error(f"Streaming error: {error.get('message', 'Unknown error')}")
                    break

            # Small delay to avoid tight loop
            time.sleep(0.05)

    # Check for errors
    error = handler.get_error()
    if error:
        st.error(f"Streaming error: {error.get('message', 'Unknown error')}")
        st.info("Falling back to non-streaming mode...")
        st.session_state.streaming_enabled = False
