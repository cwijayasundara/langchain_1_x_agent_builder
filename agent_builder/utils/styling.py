"""
Styling utilities for Agent Builder UI.
Provides consistent styling across all pages.
"""

import streamlit as st


def apply_custom_styles():
    """Apply custom CSS for improved text visibility across all input components."""
    st.markdown("""
    <style>
        /* Improve text input visibility */
        .stTextInput > div > div > input {
            color: #000000 !important;
            font-weight: 500 !important;
            background-color: #ffffff !important;
        }

        /* Dark mode text inputs */
        [data-theme="dark"] .stTextInput > div > div > input {
            color: #ffffff !important;
            background-color: #262730 !important;
        }

        /* Text area visibility */
        .stTextArea > div > div > textarea {
            color: #000000 !important;
            font-weight: 500 !important;
            background-color: #ffffff !important;
        }

        /* Dark mode text areas */
        [data-theme="dark"] .stTextArea > div > div > textarea {
            color: #ffffff !important;
            background-color: #262730 !important;
        }

        /* Improve label visibility */
        .stTextInput > label,
        .stTextArea > label,
        .stSelectbox > label,
        .stNumberInput > label,
        .stMultiSelect > label {
            color: #1f1f1f !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }

        /* Dark mode labels */
        [data-theme="dark"] .stTextInput > label,
        [data-theme="dark"] .stTextArea > label,
        [data-theme="dark"] .stSelectbox > label,
        [data-theme="dark"] .stNumberInput > label,
        [data-theme="dark"] .stMultiSelect > label {
            color: #fafafa !important;
        }

        /* Placeholder text visibility */
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #666666 !important;
            opacity: 0.7 !important;
        }

        /* Dark mode placeholders */
        [data-theme="dark"] .stTextInput input::placeholder,
        [data-theme="dark"] .stTextArea textarea::placeholder {
            color: #a0a0a0 !important;
        }

        /* Select box visibility */
        .stSelectbox > div > div > div {
            color: #000000 !important;
            font-weight: 500 !important;
        }

        [data-theme="dark"] .stSelectbox > div > div > div {
            color: #ffffff !important;
        }

        /* Number input visibility */
        .stNumberInput > div > div > input {
            color: #000000 !important;
            font-weight: 500 !important;
            background-color: #ffffff !important;
        }

        [data-theme="dark"] .stNumberInput > div > div > input {
            color: #ffffff !important;
            background-color: #262730 !important;
        }

        /* Multi-select visibility */
        .stMultiSelect > div > div > div {
            color: #000000 !important;
        }

        [data-theme="dark"] .stMultiSelect > div > div > div {
            color: #ffffff !important;
        }

        /* Code block visibility */
        .stCodeBlock {
            background-color: #f5f5f5 !important;
        }

        [data-theme="dark"] .stCodeBlock {
            background-color: #1e1e1e !important;
        }

        /* Better contrast for error/success messages */
        .stAlert {
            font-weight: 500 !important;
        }
    </style>
    """, unsafe_allow_html=True)
