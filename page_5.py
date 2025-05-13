import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

def show_healthcare_predict_page():
    # Load model, preprocessor, and feature names
    model = joblib.load('insurance_model.pkl')
    preprocessor = joblib.load('insurance_preprocessor.pkl')
    interaction_feature_names = joblib.load('interaction_feature_names.pkl')

    st.title("Developing HealthCare Cost Predictor: A Data Science Web Application")
    st.markdown("## Insurance Charges Predictor (Optimized Model)")
    st.write("#### Enter the customer details to predict their expected insurance charges:")

    # User inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", options=['male', 'female'])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
    smoker = st.selectbox("Smoker", options=['yes', 'no'])
    region = st.selectbox("Region", options=['northeast', 'northwest', 'southeast', 'southwest'])

    # Group summaries (age and BMI)
    age_group = 'Senior' if age > 56 else ('Middle-aged' if age >= 36 else 'Young')
    if bmi < 18.5:
        bmi_group = 'Underweight'
    elif bmi < 25:
        bmi_group = 'Normal'
    elif bmi < 30:
        bmi_group = 'Overweight'
    else:
        bmi_group = 'Obese'

    st.markdown("### 👤 Customer Summary")
    st.info(f"""
    - **Age**: {age} ({age_group})
    - **Sex**: {sex.capitalize()}
    - **BMI**: {bmi:.2f} ({bmi_group})
    - **Children**: {children}
    - **Smoker**: {smoker.capitalize()}
    - **Region**: {region.capitalize()}
    """)

    def show_gauge(value):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(value, 2),
            number={'valueformat': '$,.2f','font': {'size': 60}},
            title={'text':"<b>Predicted Insurance Charges</b>", 'font': {'size': 30}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {
                    'range': [0, 50000],
                    'tickvals': [0, 10000, 20000, 30000, 40000, 50000],
                    'ticktext': ["$0", "$10k", "$20k", "$30k", "$40k", "$50k"]
                },
                'bar': {'color': "seagreen"},
                'steps': [
                    {'range': [0, 10000], 'color': "#d4f4dd"},
                    {'range': [10000, 25000], 'color': "#f9f871"},
                    {'range': [25000, 40000], 'color': "#fbbf77"},
                    {'range': [40000, 50000], 'color': "#f87171"},
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Predict Charges"):
        # Create input DataFrame
        input_df = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })

        # Preprocess input
        input_processed = preprocessor.transform(input_df)
        input_transformed_df = pd.DataFrame(input_processed, columns=preprocessor.get_feature_names_out())

        # Add interaction features
        input_transformed_df['age_smoker'] = input_transformed_df.get('num__age', 0) * input_transformed_df.get('cat__smoker_yes', 0)
        input_transformed_df['bmi_smoker'] = input_transformed_df.get('num__bmi', 0) * input_transformed_df.get('cat__smoker_yes', 0)
        input_transformed_df['children_smoker'] = input_transformed_df.get('num__children', 0) * input_transformed_df.get('cat__smoker_yes', 0)
        input_transformed_df['age_bmi'] = input_transformed_df.get('num__age', 0) * input_transformed_df.get('num__bmi', 0)

        # Match training features
        input_final = input_transformed_df.reindex(columns=interaction_feature_names, fill_value=0)

        # Predict
        prediction = model.predict(input_final)[0]
        # st.success(f"🎯 Predicted Insurance Charges: ${prediction:,.2f}")

        # Show interactive gauge
        show_gauge(prediction)

        # Interpretation
        if prediction > 30000:
            st.warning("🔎 High predicted charges. Consider smoking cessation and weight management programs.")
        elif prediction > 15000:
            st.info("💡 Moderate charges. Staying healthy can help reduce future premiums.")
        else:
            st.success("✅ Low predicted charges. Great job staying healthy!")

show_healthcare_predict_page()

