import streamlit as st
from .constants import menu_text

# "Simplify your life with our AI solutions!"
# "Let our AI solutions ease your tasks!"
# "Enhance efficiency with our AI tools!"
# "Opt for our AI to simplify daily tasks!"
# "Discover ease with our AI solutions!"
# "Our AI solutions, designed for simplicity!"
# "Make life simpler with our AI!"
# "Our AI, your solution for an easier life!"
# "Streamline your tasks with our AI!"

def show_menu():
    # navigation menu
    st.sidebar.markdown(f"### **{menu_text}**")
    st.sidebar.markdown('<hr style="margin-top: 1px; margin-bottom: 1px;"/>', unsafe_allow_html=True)

    # st.sidebar.page_link("streamlit_app.py", label="Home")
    st.sidebar.page_link("pages/rag_ai.py", label="📚 RAG Chat")
    st.sidebar.page_link("pages/df_ai.py", label="📊 Data Insights")
    st.sidebar.page_link("pages/query_ai.py", label="🔗 Query Assistance")