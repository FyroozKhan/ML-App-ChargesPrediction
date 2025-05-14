import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os

@st.cache_data
def load_data():
    # Robust path handling
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "healthdata.csv")
    return pd.read_csv(file_path)

def show_healthcare_ml_page():
    st.title("Developing HealthCare Cost Predictor: A Data Science Web Application")
    st.write(
        """
        ## Implementing Machine Learning Models
        In this section, transformed data is used to implement machine learning models. 
        We apply various algorithms, evaluate their performance, and explore feature importance to predict healthcare charges.
        """
    )

    # Load data
    df = load_data()

    # Define Features and Target
    X = df.drop('charges', axis=1)
    y = df['charges']

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['age', 'bmi', 'children']),
        ('cat', OneHotEncoder(drop='first'), ['sex', 'smoker', 'region'])
    ])
    X_processed = preprocessor.fit_transform(X)

    # 1. Baseline Linear Regression Model
    st.subheader("1. Baseline Linear Regression")
    st.markdown("""
    Linear Regression is a simple model used to predict a continuous target variable (charges) based on input features (age, bmi, sex, smoker status, etc.). It establishes a linear relationship between features and the target.
    - Code Explanation:
        The dataset is first preprocessed with StandardScaler for numerical features and OneHotEncoder for categorical features like sex, smoker, and region.
        A linear regression model is then trained and tested, and its performance is evaluated using R² and RMSE.
                """)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performance Evaluation
    st.markdown("""
    The plot below shows the relationship between the actual and predicted charges, which is evaluated using R² and RMSE.
    """)
    
    # Plot
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={'x': 'Actual Charges', 'y': 'Predicted Charges'},
        title='Actual vs Predicted Charges (Baseline)',
        opacity=0.6
    )
    fig.add_shape(
        type='line',
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max(),
        line=dict(color='red', dash='dash')
    )
    fig.update_layout(title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}})
    st.plotly_chart(fig)


    st.markdown("""
    The plot above shows the relationship between actual and predicted charges. A perfect model would have all points on the red diagonal line. This model provides a baseline R² of 0.784 and RMSE of $5,796.28.
    """)

    # 2. Add Interaction Terms
    st.subheader("2. Add Interaction Terms")
    st.markdown("""
    By adding interaction terms between features, we capture more complex relationships that linear models might miss. For example, the interaction between 'age' and 'smoker' might be a strong predictor for charges.
    """)

    # Adding Interaction Features
    X_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
    X_df['age_smoker'] = X_df['num__age'] * X_df['cat__smoker_yes']
    X_df['bmi_smoker'] = X_df['num__bmi'] * X_df['cat__smoker_yes']
    X_df['children_smoker'] = X_df['num__children'] * X_df['cat__smoker_yes']
    X_df['age_bmi'] = X_df['num__age'] * X_df['num__bmi']
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performance Evaluation
    st.write(f"With Interaction R²: {r2_score(y_test, y_pred):.3f}, RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

    # Plot
    fig2 = px.scatter(
        x=y_test,
        y=y_pred,
        labels={'x': 'Actual Charges', 'y': 'Predicted Charges'},
        title='Actual vs Predicted Charges (With Interaction)',
        opacity=0.6
    )
    fig2.add_shape(
        type='line',
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max(),
        line=dict(color='red', dash='dash')
    )
    fig2.update_layout(title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}})
    st.plotly_chart(fig2)


    st.markdown("""
    The interaction terms improve the model, increasing R² to 0.866 and reducing RMSE to $4,567.69. The plot shows that adding interaction terms helps capture the non-linear relationship between features and charges.
    """)

    # 3. Permutation Importance
    st.subheader("3. Permutation Importance")
    st.markdown("""
    Permutation importance measures the impact of each feature on model performance. By shuffling each feature’s values and observing how much the model's accuracy drops, we identify which features are most influential.
    
    - **Code Explanation**: 
        We use `permutation_importance` from `sklearn.inspection` to calculate the drop in model accuracy after permuting each feature. The results are visualized using a bar plot.
    """)

    # Permutation Importance Calculation
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': result.importances_mean
    }).sort_values(by='Importance', ascending=False)

    # Bar Plot
    fig3 = px.bar(
        importance_df.sort_values(by='Importance', ascending=False),
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance using Permutation Importance',
        color='Feature',
        height=500
    )
    fig3.update_layout(
        showlegend=False,
        title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 22}},
    )
    st.plotly_chart(fig3)

    st.markdown("""
    The bar plot shows which features are most important in predicting charges. Features like `cat__smoker_yes` and interaction terms like `bmi_smoker` significantly influence model performance.
    """)

    # 4. Simplified Model with Important Features
    st.subheader("4. Simplified Model with Important Features")
    st.markdown("""
    After identifying the most impactful features using permutation importance, we simplify the model by retaining only those features, which are: `cat__smoker_yes`, `bmi_smoker` and `num_age`  .
    
    - **Code Explanation**: 
        This approach reduces model complexity, enhances interpretability, and sometimes maintains good performance.
    """)

    # Using Only Top Features
    important_features = ['num__age', 'cat__smoker_yes', 'bmi_smoker']
    X_train_imp = X_train[important_features]
    X_test_imp = X_test[important_features]

    model_important = LinearRegression()
    model_important.fit(X_train_imp, y_train)
    y_pred_important = model_important.predict(X_test_imp)

    st.write(f"Simplified Model R²: {r2_score(y_test, y_pred_important):.3f}, RMSE: ${mean_squared_error(y_test, y_pred_important, squared=False):,.2f}")

    # Plot
    fig4 = px.scatter(
        x=y_test,
        y=y_pred_important,
        labels={'x': 'Actual Charges', 'y': 'Predicted Charges'},
        title='Actual vs Predicted Charges (Simplified Model)',
        opacity=0.6,
        color_discrete_sequence=['green']
    )
    fig4.add_shape(
        type='line',
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max(),
        line=dict(color='red', dash='dash')
    )
    fig4.update_layout(title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}})
    st.plotly_chart(fig4)

    st.markdown("""
    This simplified model using just 3 features still performs quite well, showing R² = 0.854 and RMSE around $4,756. Simplifying helps in interpretability without sacrificing much accuracy.
    """)

    # 5. Ridge and Lasso
    st.subheader("5. Ridge and Lasso Regression")
    st.markdown("""
    Ridge and Lasso regression are variations of linear regression that introduce regularization to prevent overfitting. Ridge uses L2 regularization, and Lasso uses L1 regularization. These models are often used when we have multicollinearity or wish to enforce sparsity.
    """)

    # Ridge Model
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    st.write(f"Ridge R²: {r2_score(y_test, ridge.predict(X_test)):.3f}, RMSE: ${mean_squared_error(y_test, ridge.predict(X_test), squared=False):,.2f}")

    # Lasso Model
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    st.write(f"Lasso R²: {r2_score(y_test, lasso.predict(X_test)):.3f}, RMSE: ${mean_squared_error(y_test, lasso.predict(X_test), squared=False):,.2f}")

    st.markdown("""
    Both Ridge and Lasso models improve the baseline model by providing regularization, reducing overfitting. The results show very similar performance with R² = 0.866 and RMSE around $4,564.
    """)

    # 4. Random Forest & XGBoost
    st.subheader("6. Random Forest & XGBoost")
    st.markdown("""
    Random Forest and XGBoost are ensemble models that combine multiple decision trees to improve prediction accuracy. Random Forest creates a collection of decision trees and averages their predictions, while XGBoost uses gradient boosting to optimize performance.
    """)

    # Random Forest Model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    st.write(f"RF R²: {r2_score(y_test, rf.predict(X_test)):.3f}, RMSE: ${mean_squared_error(y_test, rf.predict(X_test), squared=False):,.2f}")

    # XGBoost Model
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    st.write(f"XGB R²: {r2_score(y_test, xgb.predict(X_test)):.3f}, RMSE: ${mean_squared_error(y_test, xgb.predict(X_test), squared=False):,.2f}")

    st.markdown("""
    Both Random Forest and XGBoost outperform the linear models, with R² values around 0.866 and RMSE just over $4,500. These ensemble models capture more complex relationships between the features and target variable.
    """)

    # Model Performance Comparison
    st.write(
    """
    ## Model Performance Comparison
    The plots below summarizes the performance of all models implemented, including R² and RMSE values.
    """
)
    model_results = pd.DataFrame({
        'Model': ['Linear (baseline)', 'Linear + Interaction', 'Important Features',
                'Ridge', 'Lasso', 'Random Forest', 'XGBoost'],
        'R²': [0.784, 0.866, 0.859, 0.866, 0.866, 0.866, 0.850],
        'RMSE': [5796.28, 4567.69, 4679.23, 4564.75, 4567.68, 4567.03, 4833.03]
    })

    # Sort R² descending
    # Sort model results by R² (descending)
    model_results_sorted_r2 = model_results.sort_values(by='R²', ascending=False)

    # Plot R² comparison as horizontal bar chart
    fig4 = px.bar(
        model_results_sorted_r2,
        y='Model',
        x='R²',
        orientation='h',
        title='Model R² Comparison',
        color='Model',
        height=400
    )
    # Update layout for title alignment and font size
    fig4.update_layout(
        title={
            'text': 'Model R² Comparison',
            'x': 0.5,  # Center title horizontally (0 = left, 1 = right)
            'xanchor': 'center',
            'font': {
                'size': 24  # Adjust title font size
            }
        }
    )
    st.plotly_chart(fig4)

    # Sort model results by RMSE (ascending, then reverse for best on top)
    model_results_sorted_rmse = model_results.sort_values(by='RMSE', ascending=True)

    fig5 = px.bar(
        model_results_sorted_rmse,
        y='Model',
        x='RMSE',
        orientation='h',
        title='Model RMSE Comparison',
        color='Model',
        height=400
    )
    fig5.update_layout(
    title={
        'text': 'Model RMSE Comparison',
        'x': 0.5,  # Center title horizontally (0 = left, 1 = right)
        'xanchor': 'center',
        'font': {
            'size': 24  # Adjust title font size
        }
    }
)
    st.plotly_chart(fig5)

    st.markdown("""
    The performance comparison shows how different models perform. Ridge, Lasso, and Random Forest provide the best results with R² values of around 0.866 and RMSE close to $4,500.
    """)

    # Final Feature Importance Plot
    feature_importance = pd.DataFrame({
        'Feature': ['cat__smoker_yes', 'bmi_smoker', 'num__age'],
        'Importance': [1.125, 0.23, 0.169]
    }).sort_values(by='Importance', ascending=True)

    fig6 = px.bar(feature_importance, y='Feature', x='Importance', orientation='h',
                title='Top Predictive Features', text_auto='.2f', height=400)
    fig6.update_layout(title={
    'text': 'Top Predictive Features',
    'x': 0.5,  # Center title horizontally (0 = left, 1 = right)
    'xanchor': 'center',
    'font': {
        'size': 24  # Adjust title font size
    }})
    st.plotly_chart(fig6)

    st.markdown("""
    These features are the most predictive of insurance charges. Being a smoker, BMI-smoker interaction, and age are strong indicators of healthcare costs.
    """)

    ### Exporting the Best Model (XGBoost with Interactions)
    joblib.dump(xgb, 'insurance_model.pkl')  # Best-performing model
    joblib.dump(preprocessor, 'insurance_preprocessor.pkl')  # Original preprocessor used
    joblib.dump(X_train.columns.tolist(), 'interaction_feature_names.pkl')  # Feature names after interaction



show_healthcare_ml_page()
