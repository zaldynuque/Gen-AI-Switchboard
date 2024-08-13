import streamlit as st

from utils.menu import (
    show_menu,
)

show_menu()

st.switch_page("pages/rag_ai.py")