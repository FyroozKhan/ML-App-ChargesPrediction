import streamlit as st
from page_1 import show_healthcare_explore_page
from page_2 import show_healthcare_eda_page
from page_3 import show_healthcare_feat_eng_page
from page_4 import show_healthcare_ml_page
from page_5 import show_healthcare_predict_page
from page_about import show_about_page

# Sidebar navigation with vertical layout
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "1. Data Exploration",
    "2. Exploratory Data Analysis (EDA)",
    "3. Feature Engineering",
    "4. ML Model Implementation",
    "5. Charges Prediction",
    "About"
])

# Route to appropriate page
if page.startswith("1"):
    show_healthcare_explore_page()
elif page.startswith("2"):
    show_healthcare_eda_page()
elif page.startswith("3"):
    show_healthcare_feat_eng_page()
elif page.startswith("4"):
    show_healthcare_ml_page()
elif page.startswith("5"):
    show_healthcare_predict_page()
elif page == "About":
    show_about_page()