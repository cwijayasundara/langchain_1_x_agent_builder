"""
Navigation component with Previous/Next buttons and progress tracking.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.state_manager import get_completion_count, mark_page_complete


def display_navigation(
    current_page: int,
    total_pages: int = 8,
    show_save: bool = True
):
    """
    Display navigation buttons and progress.

    Args:
        current_page: Current page number (1-8)
        total_pages: Total number of pages
        show_save: Whether to show save draft button
    """

    # Progress bar
    completed = get_completion_count()
    progress = completed / total_pages
    st.progress(progress)
    st.caption(f"Progress: {completed}/{total_pages} pages completed")

    st.markdown("---")

    # Navigation buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if current_page > 1:
            if st.button("⬅️ Previous", use_container_width=True):
                prev_page = current_page - 1
                st.switch_page(f"pages/{prev_page}_{get_page_name(prev_page)}.py")

    with col2:
        if show_save:
            if st.button("💾 Save Draft", use_container_width=True):
                st.success("✅ Draft saved!")
                st.rerun()

    with col3:
        if current_page < total_pages:
            if st.button("Next ➡️", use_container_width=True):
                next_page = current_page + 1
                st.switch_page(f"pages/{next_page}_{get_page_name(next_page)}.py")


def get_page_name(page_number: int) -> str:
    """Get page file name from page number."""
    page_names = {
        1: "📝_Basic_Info",
        2: "🤖_LLM_Config",
        3: "🔧_Tools",
        4: "💬_Prompts",
        5: "🧠_Memory",
        6: "⚙️_Middleware",
        7: "🚀_Advanced",
        8: "✅_Deploy"
    }
    return page_names.get(page_number, "")


def display_page_header(page_number: int, title: str, description: str):
    """
    Display page header with title and description.

    Args:
        page_number: Page number
        title: Page title
        description: Page description
    """
    st.title(f"{get_page_name(page_number).split('_')[0]} {title}")
    st.markdown(description)
    st.markdown("---")
