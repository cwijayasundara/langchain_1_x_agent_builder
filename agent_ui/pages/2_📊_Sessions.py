"""
Page 2: Sessions - Thread/session management and history.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import (
    initialize_session_state,
    get_all_threads,
    switch_thread,
    delete_thread,
    rename_thread
)
from utils.export_utils import (
    export_to_json,
    export_to_markdown,
    export_to_csv,
    export_all_threads_to_json,
    get_export_filename,
    calculate_export_stats
)
from utils.message_formatter import truncate_message, get_conversation_summary

initialize_session_state()
st.set_page_config(
    page_title="Sessions",
    page_icon="📊",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Check if agent is selected
if not st.session_state.get('agent_selected'):
    st.warning("⚠️ No agent selected. Please select an agent first.")

    if st.button("← Back to Agent Selection"):
        st.switch_page("app.py")

    st.stop()

# Header
st.markdown("# 📊 Conversation Sessions")
st.markdown("Manage your conversation threads and history.")
st.markdown("---")

threads = get_all_threads()

if not threads:
    st.info("📝 No conversation threads yet. Start chatting to create one!")

    if st.button("💬 Go to Chat"):
        st.switch_page("pages/1_💬_Chat.py")

    st.stop()

# Stats
col1, col2, col3 = st.columns(3)

total_messages = sum(len(t.get('messages', [])) for t in threads.values())
current_thread_id = st.session_state.get('current_thread_id')

with col1:
    st.metric("Total Threads", len(threads))

with col2:
    st.metric("Total Messages", total_messages)

with col3:
    avg_messages = total_messages // len(threads) if threads else 0
    st.metric("Avg Messages/Thread", avg_messages)

st.markdown("---")

# Bulk actions
st.markdown("### Bulk Actions")

col_a, col_b = st.columns(2)

with col_a:
    if st.button("📥 Export All Threads (JSON)", use_container_width=True):
        agent_info = st.session_state.get('selected_agent_info', {})
        json_data = export_all_threads_to_json(threads, agent_info)

        agent_name = agent_info.get('name', 'agent')
        filename = get_export_filename('json', agent_name=agent_name)

        st.download_button(
            label="⬇️ Download JSON",
            data=json_data,
            file_name=filename,
            mime="application/json",
            use_container_width=True
        )

with col_b:
    if st.button("🗑️ Delete All Threads", use_container_width=True):
        if st.session_state.get('confirm_delete_all', False):
            # Delete all threads
            for thread_id in list(threads.keys()):
                delete_thread(thread_id)

            st.success("All threads deleted!")
            st.session_state.confirm_delete_all = False
            st.rerun()
        else:
            st.session_state.confirm_delete_all = True
            st.warning("⚠️ Click again to confirm deletion of ALL threads!")

st.markdown("---")

# Thread list
st.markdown("### Your Threads")

# Sort threads
sort_option = st.selectbox(
    "Sort by",
    options=["Most Recent", "Oldest First", "Most Messages", "Least Messages"],
    key="sort_threads"
)

sorted_threads = list(threads.items())

if sort_option == "Most Recent":
    sorted_threads.sort(key=lambda x: x[1].get('last_updated', ''), reverse=True)
elif sort_option == "Oldest First":
    sorted_threads.sort(key=lambda x: x[1].get('created', ''))
elif sort_option == "Most Messages":
    sorted_threads.sort(key=lambda x: len(x[1].get('messages', [])), reverse=True)
elif sort_option == "Least Messages":
    sorted_threads.sort(key=lambda x: len(x[1].get('messages', [])))

# Display threads
for thread_id, thread_data in sorted_threads:
    is_current = thread_id == current_thread_id

    # Thread card
    with st.container():
        # Header row
        col_h1, col_h2, col_h3 = st.columns([3, 1, 1])

        with col_h1:
            label = thread_data.get('label', 'Untitled')
            if is_current:
                st.markdown(f"### ▶ {label}")
            else:
                st.markdown(f"### {label}")

        with col_h2:
            message_count = len(thread_data.get('messages', []))
            st.metric("Messages", message_count)

        with col_h3:
            # Created date
            created = thread_data.get('created', '')
            try:
                dt = datetime.fromisoformat(created)
                st.caption(f"Created: {dt.strftime('%m/%d/%y')}")
            except:
                pass

        # Summary
        summary = get_conversation_summary(thread_data.get('messages', []))
        st.caption(summary)

        # Actions
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if not is_current and st.button(
                "📖 Open",
                key=f"sess_open_{thread_id}",
                use_container_width=True
            ):
                switch_thread(thread_id)
                st.switch_page("pages/1_💬_Chat.py")

        with col2:
            if st.button(
                "✏️ Rename",
                key=f"sess_rename_{thread_id}",
                use_container_width=True
            ):
                st.session_state[f'sess_renaming_{thread_id}'] = True

        with col3:
            # Export dropdown
            export_format = st.selectbox(
                "Export",
                options=["JSON", "Markdown", "CSV"],
                key=f"sess_export_format_{thread_id}",
                label_visibility="collapsed"
            )

        with col4:
            if st.button(
                "📥 Download",
                key=f"sess_download_{thread_id}",
                use_container_width=True
            ):
                messages = thread_data.get('messages', [])
                agent_info = st.session_state.get('selected_agent_info', {})

                if export_format == "JSON":
                    data = export_to_json(messages, thread_id, agent_info)
                    mime = "application/json"
                elif export_format == "Markdown":
                    data = export_to_markdown(messages, thread_id, agent_info)
                    mime = "text/markdown"
                else:  # CSV
                    data = export_to_csv(messages)
                    mime = "text/csv"

                agent_name = agent_info.get('name', 'agent')
                filename = get_export_filename(
                    export_format.lower(),
                    thread_id=thread_id,
                    agent_name=agent_name
                )

                st.download_button(
                    label=f"⬇️ {export_format}",
                    data=data,
                    file_name=filename,
                    mime=mime,
                    key=f"sess_dl_btn_{thread_id}"
                )

        with col5:
            if st.button(
                "🗑️ Delete",
                key=f"sess_delete_{thread_id}",
                use_container_width=True
            ):
                if st.session_state.get(f'sess_confirm_delete_{thread_id}', False):
                    delete_thread(thread_id)
                    st.success("Thread deleted!")
                    st.rerun()
                else:
                    st.session_state[f'sess_confirm_delete_{thread_id}'] = True
                    st.warning("Click again to confirm")

        # Rename dialog
        if st.session_state.get(f'sess_renaming_{thread_id}', False):
            with st.form(f"rename_form_{thread_id}"):
                new_label = st.text_input(
                    "New label",
                    value=label,
                    key=f"sess_new_label_{thread_id}"
                )

                col_r1, col_r2 = st.columns(2)

                with col_r1:
                    if st.form_submit_button("Save", use_container_width=True):
                        rename_thread(thread_id, new_label)
                        st.session_state[f'sess_renaming_{thread_id}'] = False
                        st.success("Thread renamed!")
                        st.rerun()

                with col_r2:
                    if st.form_submit_button("Cancel", use_container_width=True):
                        st.session_state[f'sess_renaming_{thread_id}'] = False
                        st.rerun()

        # Show messages preview
        with st.expander("Preview Messages", expanded=False):
            messages = thread_data.get('messages', [])
            if messages:
                for idx, msg in enumerate(messages[:5]):  # Show first 5
                    role = msg.get('role', 'unknown')
                    content = truncate_message(msg.get('content', ''), 100)
                    st.caption(f"**{role}:** {content}")

                if len(messages) > 5:
                    st.caption(f"... and {len(messages) - 5} more messages")
            else:
                st.caption("No messages")

            # Stats
            if messages:
                stats = calculate_export_stats(messages)
                st.markdown(f"**Total messages:** {stats['total_messages']}")
                st.markdown(f"**User:** {stats['user_messages']}, **AI:** {stats['ai_messages']}")
                if stats['total_tool_calls'] > 0:
                    st.markdown(f"**Tool calls:** {stats['total_tool_calls']}")

        st.markdown("---")
