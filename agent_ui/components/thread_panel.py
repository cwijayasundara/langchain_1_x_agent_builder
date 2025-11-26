"""
Thread Panel Component - Sidebar panel for managing conversation threads.
"""

import streamlit as st
from typing import Dict
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import (
    get_all_threads,
    switch_thread,
    delete_thread,
    rename_thread,
    create_new_thread
)
from utils.message_formatter import truncate_message, get_conversation_summary


def display_thread_panel():
    """
    Display the thread management panel in sidebar.
    """
    st.markdown("### 📊 Threads")

    threads = get_all_threads()

    if not threads:
        st.info("No threads yet. Start chatting to create one!")
        return

    current_thread_id = st.session_state.get('current_thread_id')

    # Sort threads by last updated (most recent first)
    sorted_threads = sorted(
        threads.items(),
        key=lambda x: x[1].get('last_updated', ''),
        reverse=True
    )

    st.caption(f"{len(threads)} thread(s)")

    for thread_id, thread_data in sorted_threads:
        is_current = thread_id == current_thread_id

        # Thread container
        with st.container():
            # Thread label
            label = thread_data.get('label', 'Untitled')

            # Highlight current thread
            if is_current:
                st.markdown(f"**▶ {label}**")
            else:
                st.markdown(f"{label}")

            # Thread metadata
            created = thread_data.get('created', '')
            try:
                dt = datetime.fromisoformat(created)
                st.caption(f"Created: {dt.strftime('%m/%d %I:%M %p')}")
            except:
                pass

            # Message count
            message_count = len(thread_data.get('messages', []))
            st.caption(f"{message_count} messages")

            # Preview first message
            messages = thread_data.get('messages', [])
            if messages:
                first_msg = messages[0]
                preview = truncate_message(first_msg.get('content', ''), 50)
                st.caption(f'"{preview}"')

            # Action buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                if not is_current and st.button(
                    "Open",
                    key=f"open_{thread_id}",
                    use_container_width=True
                ):
                    switch_thread(thread_id)
                    st.rerun()

            with col2:
                if st.button(
                    "✏️",
                    key=f"rename_{thread_id}",
                    help="Rename",
                    use_container_width=True
                ):
                    st.session_state[f'renaming_{thread_id}'] = True

            with col3:
                if st.button(
                    "🗑️",
                    key=f"delete_{thread_id}",
                    help="Delete",
                    use_container_width=True
                ):
                    if st.session_state.get(f'confirm_delete_{thread_id}', False):
                        delete_thread(thread_id)
                        st.success("Thread deleted")
                        st.rerun()
                    else:
                        st.session_state[f'confirm_delete_{thread_id}'] = True
                        st.warning("Click again to confirm delete")

            # Rename dialog
            if st.session_state.get(f'renaming_{thread_id}', False):
                new_label = st.text_input(
                    "New label",
                    value=label,
                    key=f"new_label_{thread_id}"
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Save", key=f"save_rename_{thread_id}"):
                        rename_thread(thread_id, new_label)
                        st.session_state[f'renaming_{thread_id}'] = False
                        st.rerun()

                with col_b:
                    if st.button("Cancel", key=f"cancel_rename_{thread_id}"):
                        st.session_state[f'renaming_{thread_id}'] = False
                        st.rerun()

            st.markdown("---")


def display_compact_thread_selector():
    """
    Display a compact thread selector dropdown.
    """
    threads = get_all_threads()

    if not threads:
        st.info("No threads")
        return

    thread_options = {}
    for thread_id, thread_data in threads.items():
        label = thread_data.get('label', 'Untitled')
        message_count = len(thread_data.get('messages', []))
        thread_options[thread_id] = f"{label} ({message_count} msgs)"

    current_thread_id = st.session_state.get('current_thread_id')

    selected = st.selectbox(
        "Thread",
        options=list(thread_options.keys()),
        format_func=lambda tid: thread_options[tid],
        index=list(thread_options.keys()).index(current_thread_id) if current_thread_id in thread_options else 0,
        key="compact_thread_selector"
    )

    if selected != current_thread_id:
        switch_thread(selected)
        st.rerun()
