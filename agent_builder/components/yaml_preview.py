"""
YAML preview component with live updates.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.yaml_generator import generate_agent_yaml


def display_yaml_preview():
    """Display live YAML preview in sidebar or column."""

    st.markdown("### 📄 Live YAML Preview")

    # Generate current YAML
    yaml_content = generate_agent_yaml()

    # Display in code block
    st.code(yaml_content, language='yaml', line_numbers=True)

    # Download button
    st.download_button(
        label="⬇️ Download YAML",
        data=yaml_content,
        file_name="agent_config.yaml",
        mime="text/yaml",
        use_container_width=True
    )

    # Stats
    lines = yaml_content.count('\n') + 1
    st.caption(f"📊 {lines} lines")
