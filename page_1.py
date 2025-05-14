import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

def show_healthcare_explore_page():
    st.title("Developing HealthCare Cost Predictor: A Data Science Web Application")

    st.write(
        """
        ### Objective
        Develop a data science web application that predicts healthcare costs based on individual factors like age, BMI, region, and more.
        """
    )

    # Robust path handling
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "healthdata.csv")
    df = pd.read_csv(file_path)

    # Data Understanding Section
    st.markdown(
        """
        #### Overview of the Dataset
        - **The link to the data is:** [Healthdata](https://github.com/FyroozKhan/ML-App-ChargesPrediction/blob/main/healthdata.csv)
        - **Rows & Columns:** 1,338 records × 7 columns 
        """
    )

    # Preview Data
    st.subheader("1. Dataset Preview")
    st.write(df.head())
    st.markdown(
        """
        The first five rows give a quick glance at individual records, showing variations in age, BMI, smoking status, and charges.
        The dataset includes the following columns:

        - **Age:** Age in years.  
        - **Sex:** The person's gender (male/female).  
        - **BMI (Body Mass Index):** A measure of body fat based on height and weight.  
          - Formula: weight (kg) / (height (m))²  
          - Normal range: 18.5–24.9  
          - Higher values indicate increased health risks  
        - **Children:** Number of dependents covered by the insurance policy. Range: 0–5.  
        - **Smoker:** Whether the person smokes (yes/no).  
        - **Region:** Residential area in the US (northeast, northwest, southeast, southwest).  
        - **Charges:** The medical insurance cost in USD. This is the target variable we aim to predict.  
        
        These features are used to predict insurance costs.
        """
    )

    # Descriptive Stats
    st.subheader("2. Descriptive Statistics")
    st.write(df.describe())
    st.markdown(
        """
        #### Summary
        - **Average Age:** 39.2 years  
        - **Average BMI:** 30.66 (slightly overweight)  
        - **Charges:**  
            - **Mean:** $13,270  
            - **Min-Max:** $1,122 to $63,770  
            - The wide range in charges suggests influence from features like **smoking** and **age**.
        """
    )



show_healthcare_explore_page()
