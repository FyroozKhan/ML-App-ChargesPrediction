import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data():
    return pd.read_csv(r"C:\\GWU\\Build_Project\\Week8\\healthdata.csv")

def show_healthcare_feat_eng_page():
    st.title("Developing HealthCare Cost Predictor: A Data Science Web Application")
    st.write(
        """
        ## Feature Engineering
        In this section, data is transformed to prepare it for machine learning models.
        """
    )
    # Load data
    df = load_data()
    
    st.subheader("1. One-Hot Encoding (Categorical Features)")
    st.markdown(""" 
    Applied to: `sex`, `smoker`, `region`  
    
    - Transforms categorical data into binary columns (e.g., `smoker_yes`, `region_northwest`)
    - Prevents unintended ordinal relationships in machine learning models.

    **Example After Encoding:**
    ```
    sex_male  smoker_yes  region_northwest  region_southeast  region_southwest
        0.0         1.0              0.0               0.0               1.0
    ```
    """)

    # Categorical columns to encode
    categorical_columns = ['sex', 'smoker', 'region']

    # Initialize encoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(df[categorical_columns])

    # Convert encoded array to DataFrame
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

    # Combine with original dataset
    final_data = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

    st.dataframe(final_data.head())

    st.subheader("2. Standardization (Numerical Features)")
    st.markdown("""
    Applied to: `age`, `bmi`, `children`  

    - Scales values to have mean = 0 and std = 1
    - Ensures balanced feature influence in model training.
    """)

    # Numerical features to scale
    numerical_features = ['age', 'bmi', 'children']

    # Apply standardization
    scaler = StandardScaler()
    final_data[numerical_features] = scaler.fit_transform(final_data[numerical_features])

    st.markdown("""
    **Final Output:**
    
    ```
         age     bmi  children      charges  sex_male  smoker_yes  region_northwest  region_southeast  region_southwest
      -1.0     0.5     -0.9      16884.92       0.0         1.0               0.0               0.0               1.0
    ```
    """)
    st.dataframe(final_data.head())
    
    st.markdown("""            
    A fully numeric dataset is now ready for machine learning models.       
    """)

show_healthcare_feat_eng_page()