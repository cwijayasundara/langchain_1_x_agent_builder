"""
Message Renderer Component - Renders individual chat messages with styling.
"""

import streamlit as st
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.message_formatter import (
    format_message_content,
    format_tool_call,
    format_timestamp,
    get_role_display_name,
    format_metadata
)


def render_message(message: Dict[str, Any], show_timestamp: bool = True, show_metadata: bool = True):
    """
    Render a single chat message with appropriate styling.

    Args:
        message: Message dictionary with role, content, etc.
        show_timestamp: Whether to show timestamp
        show_metadata: Whether to show metadata (tokens, cost)
    """
    role = message.get('role', 'user')
    content = message.get('content', '')
    tool_calls = message.get('tool_calls', [])
    timestamp = message.get('timestamp', '')
    metadata = message.get('metadata', {})

    # Get role display name
    role_name = get_role_display_name(role)

    # Message container with role-specific styling
    if role == 'user':
        # User messages - right aligned, blue background
        with st.container():
            cols = st.columns([1, 4])
            with cols[1]:
                st.markdown(f"**{role_name}**")
                if show_timestamp and timestamp:
                    st.caption(format_timestamp(timestamp))

                # Message content - render markdown directly to ensure visibility
                st.markdown(content)

    elif role in ['ai', 'assistant']:
        # AI messages - left aligned, with styled container for visibility
        with st.container():
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"**{role_name}**")
                if show_timestamp and timestamp:
                    st.caption(format_timestamp(timestamp))

                # Message content - render markdown directly for proper formatting
                if content:
                    # Use a container with CSS class for styling
                    # This allows proper markdown rendering while maintaining visibility
                    st.markdown(
                        '<div class="ai-message-content">',
                        unsafe_allow_html=True
                    )
                    st.markdown(content)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Tool calls
                if tool_calls:
                    with st.expander(f"🔧 Tool Calls ({len(tool_calls)})", expanded=False):
                        for idx, tool_call in enumerate(tool_calls, 1):
                            st.markdown(f"**Tool Call {idx}**")
                            st.markdown(format_tool_call(tool_call))
                            if idx < len(tool_calls):
                                st.markdown("---")

                # Metadata
                if show_metadata and metadata:
                    metadata_str = format_metadata(metadata)
                    if metadata_str:
                        st.caption(metadata_str)

    elif role == 'system':
        # System messages - centered, small
        with st.container():
            st.markdown(
                f'<div style="text-align: center; color: #888; font-size: 0.9em; padding: 10px;">'
                f'{role_name}: {content}'
                f'</div>',
                unsafe_allow_html=True
            )

    else:
        # Default message rendering
        with st.container():
            st.markdown(f"**{role_name}**")
            if show_timestamp and timestamp:
                st.caption(format_timestamp(timestamp))
            st.markdown(content)


def render_streaming_message(accumulated_text: str, role: str = "ai"):
    """
    Render a message that's currently being streamed.

    Args:
        accumulated_text: Text accumulated so far
        role: Message role
    """
    role_name = get_role_display_name(role)

    with st.container():
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(f"**{role_name}** _typing..._")

            # Streaming content - render markdown directly for proper formatting
            if accumulated_text:
                st.markdown(
                    '<div class="ai-message-content">',
                    unsafe_allow_html=True
                )
                st.markdown(accumulated_text)
                st.markdown('</div>', unsafe_allow_html=True)
            # Add blinking cursor
            st.markdown('<span class="streaming-cursor">▊</span>', unsafe_allow_html=True)


def render_message_list(messages: list, show_timestamps: bool = True, show_metadata: bool = True):
    """
    Render a list of messages in sequence.

    Args:
        messages: List of message dictionaries
        show_timestamps: Whether to show timestamps
        show_metadata: Whether to show metadata
    """
    if not messages:
        st.info("💬 No messages yet. Start the conversation!")
        return

    for message in messages:
        render_message(message, show_timestamps, show_metadata)
        st.markdown("")  # Spacing between messages


def render_error_message(error: str):
    """
    Render an error message in the chat.

    Args:
        error: Error message string
    """
    st.error(f"⚠️ **Error:** {error}")


def render_loading_message(text: str = "Agent is thinking..."):
    """
    Render a loading/thinking message.

    Args:
        text: Loading text to display
    """
    with st.container():
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown("**🤖 Agent**")
            st.markdown(
                f'<div class="ai-message-content" style="font-style: italic; color: #666666;">{text}</div>',
                unsafe_allow_html=True
            )


def inject_message_styles():
    """
    Inject custom CSS styles for messages.
    Call this once at the start of the page.
    """
    st.markdown("""
    <style>
    /* AI message content container */
    .ai-message-content {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }

    [data-theme="dark"] .ai-message-content {
        background-color: #2d2d2d;
    }

    /* Streaming cursor */
    .streaming-cursor {
        animation: blink 1s infinite;
        color: #000000;
    }

    [data-theme="dark"] .streaming-cursor {
        color: #ffffff;
    }

    /* Blinking cursor for streaming */
    @keyframes blink {
        0%, 49% { opacity: 1; }
        50%, 100% { opacity: 0; }
    }

    /* Message container spacing */
    .stContainer {
        margin-bottom: 1rem;
    }

    /* Ensure message content containers have proper spacing */
    .message-content-wrapper {
        display: block;
    }

    /* CRITICAL: Force pure black text (#000000) on ALL markdown containers */
    /* This ensures text is always visible on white/light gray backgrounds */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] * {
        color: #000000 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] div,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em,
    [data-testid="stMarkdownContainer"] a {
        color: #000000 !important;
    }

    /* Dark mode - force pure white text */
    [data-theme="dark"] [data-testid="stMarkdownContainer"],
    [data-theme="dark"] [data-testid="stMarkdownContainer"] * {
        color: #ffffff !important;
    }

    [data-theme="dark"] [data-testid="stMarkdownContainer"] p,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] li,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] span,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] div,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] strong,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] em,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] a {
        color: #ffffff !important;
    }

    /* Force text visibility in ALL markdown containers - highest priority */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] *:not(code):not(pre),
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span:not([class*="code"]),
    [data-testid="stMarkdownContainer"] div:not([class*="code"]),
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em {
        color: #1f1f1f !important;
    }

    /* Dark mode text visibility - force white text */
    [data-theme="dark"] [data-testid="stMarkdownContainer"],
    [data-theme="dark"] [data-testid="stMarkdownContainer"] *:not(code):not(pre),
    [data-theme="dark"] [data-testid="stMarkdownContainer"] p,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] li,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] span:not([class*="code"]),
    [data-theme="dark"] [data-testid="stMarkdownContainer"] div:not([class*="code"]),
    [data-theme="dark"] [data-testid="stMarkdownContainer"] strong,
    [data-theme="dark"] [data-testid="stMarkdownContainer"] em {
        color: #fafafa !important;
    }

    /* Ensure Streamlit markdown text is always visible */
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown div:not([class*="code"]),
    .stMarkdown span:not([class*="code"]) {
        color: #1f1f1f !important;
    }

    [data-theme="dark"] .stMarkdown,
    [data-theme="dark"] .stMarkdown p,
    [data-theme="dark"] .stMarkdown div:not([class*="code"]),
    [data-theme="dark"] .stMarkdown span:not([class*="code"]) {
        color: #fafafa !important;
    }

    /* Additional selector for all text elements */
    .element-container [data-testid="stMarkdownContainer"],
    .element-container [data-testid="stMarkdownContainer"] * {
        color: #1f1f1f !important;
    }

    [data-theme="dark"] .element-container [data-testid="stMarkdownContainer"],
    [data-theme="dark"] .element-container [data-testid="stMarkdownContainer"] * {
        color: #fafafa !important;
    }

    /* Universal text color override for main content area */
    .main .element-container,
    .main .element-container * {
        color: #1f1f1f !important;
    }

    [data-theme="dark"] .main .element-container,
    [data-theme="dark"] .main .element-container * {
        color: #fafafa !important;
    }

    /* Target all markdown content specifically */
    div[data-testid="stMarkdownContainer"] {
        color: #1f1f1f !important;
    }

    [data-theme="dark"] div[data-testid="stMarkdownContainer"] {
        color: #fafafa !important;
    }

    /* Code block styling */
    .stCodeBlock {
        margin: 10px 0;
    }

    /* Tool call expander */
    .streamlit-expanderHeader {
        font-size: 0.9em;
    }

    /* Ensure text input visibility */
    .stTextInput > div > div > input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }

    [data-theme="dark"] .stTextInput > div > div > input {
        color: #ffffff !important;
        background-color: #262730 !important;
    }
    </style>
    """, unsafe_allow_html=True)
