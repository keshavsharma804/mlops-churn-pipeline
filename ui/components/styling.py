import streamlit as st

def apply_theme():
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

    if dark_mode:
        st.markdown("""
            <style>
                body { background-color: #0e1117; color: white; }
                .stButton>button { color:black; background-color:#f5f5f5; }
            </style>
        """, unsafe_allow_html=True)
