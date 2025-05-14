import streamlit as st

def show_about_page():
    st.title("About this App")
    st.markdown("""
    #### This Streamlit app has been developed as part of a data science project of 'The Build Fellowship Project'
    - Special thanks to mentor Ujwal Gullapalli for his guidance and support throughout the project
    ---
    #### Created by: Fyrooz Anika Khan
    📧 Contact:
    [GitHub](https://github.com/FyroozKhan) | [LinkedIn](https://www.linkedin.com/in/fyroozanika-khan/)
    """)

show_about_page()