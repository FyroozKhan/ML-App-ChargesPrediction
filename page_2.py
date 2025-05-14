import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
import os

@st.cache_data
def load_data():
    # Robust path handling
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "healthdata.csv")
    return pd.read_csv(file_path)

def show_healthcare_eda_page():
    st.title("Developing HealthCare Cost Predictor: A Data Science Web Application")
    st.markdown("## Exploratory Data Analysis")

    # Load data
    df = load_data()

    # ---- Sex and Children Distribution ----
    st.subheader("1. Distribution of Sex and Number of Children")
    st.markdown("""
    This plot uses count plots to show the frequency distribution of individuals by sex and number of children.
    Seaborn's `countplot` function is used to visualize the counts of each category.
    """)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    sns.countplot(ax=axes2[0], x='sex', data=df, palette='Set2')
    axes2[0].set_title('Distribution of Sex')
    sns.countplot(ax=axes2[1], x='children', data=df, palette='Set2')
    axes2[1].set_title('Distribution of Children')
    fig2.suptitle("Sex and Children Distribution", fontsize=16, y=1.05)
    st.pyplot(fig2)

    st.markdown("""
    The first plot shows that sex is nearly evenly distributed in the dataset. The second plot indicates most individuals have 0–2 children, with the count decreasing for 3 or more children.
    """)

    # ---- Age, BMI, Charges Distributions ----
    st.subheader("2. Distribution of Charges, Age, and BMI")
    st.markdown("""
    These histograms use Seaborn’s `histplot` to display the distribution of three numerical variables: medical charges, age, and BMI (body mass index).
    KDE curves are added to visualize the underlying probability distribution.
    """)

    fig3, axes3 = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    sns.histplot(df, x="charges", bins=30, kde=True, ax=axes3[0], color='skyblue')
    axes3[0].set_title("Distribution of Charges")
    sns.histplot(df, x="age", bins=15, kde=True, ax=axes3[1], color='salmon')
    axes3[1].set_title("Distribution of Age")
    sns.histplot(df, x="bmi", bins=15, kde=True, ax=axes3[2], color='mediumseagreen')
    axes3[2].set_title("Distribution of BMI")
    fig3.suptitle("Distribution Plots of Charges, Age, and BMI", fontsize=16)
    st.pyplot(fig3)

    st.markdown("""
    The first plot shows that medical charges are right-skewed, indicating that most individuals incur lower costs while a few have very high expenses.
    The second plot reveals that the population is skewed toward younger individuals (age 18–25), with fewer middle-aged entries.
    The third plot shows a roughly normal BMI distribution with a slight skew to the right.
    """)

    # ---- Smoker Distribution + Charges Subplot ----
    st.subheader("3. Smoking Status and Its Effect on Charges")
    st.markdown("""
    This section uses an interactive pie chart to visualize the proportion of smokers and non-smokers, and a static box plot to compare the medical charges between the two groups.
    """)

    col1, col2 = st.columns(2)
    with col1:
        smoker_counts = df['smoker'].value_counts().reset_index()
        smoker_counts.columns = ['smoker', 'count']
        fig_pie = px.pie(smoker_counts, values='count', names='smoker', title='Smoker Status Distribution',
                         color='smoker', color_discrete_map={'Yes': '#E45756', 'No': '#4C78A8'})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_box, ax_box = plt.subplots(figsize=(5, 5))
        sns.boxplot(x='smoker', y='charges', data=df, ax=ax_box)
        ax_box.set_title("Charges by Smoking Status")
        st.pyplot(fig_box)

    st.markdown("""
    The pie chart shows that non-smokers make up nearly 80% of the dataset while smokers comprise about 20%. 
    The box plot clearly shows that smokers incur significantly higher medical charges compared to non-smokers, indicating a strong relationship between smoking and healthcare cost.
    """)

    # ---- Average Charges by Age, Region, BMI ----
    st.subheader("4. Average Charges by Age Group, Region, and BMI Category")
    st.markdown("""
    This plot uses bar charts to show the average medical charges grouped by age category, region, and BMI category. The `age` variable is binned into young, middle-aged, and senior groups. BMI is categorized into standard medical classes: underweight, normal, overweight, and obese.
    """)

    # Create groups
    df['age_group'] = df['age'].apply(lambda x: 'Senior' if x > 56 else ('Middle-aged' if x >= 36 else 'Young'))
    df['bmi_group'] = pd.cut(df['bmi'], [0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    # Averages
    age_group_avg = df.groupby('age_group')['charges'].mean().round(2).reset_index()
    region_avg = df.groupby('region')['charges'].mean().round(2).reset_index()
    bmi_avg = df.groupby('bmi_group')['charges'].mean().round(2).reset_index()

    # Sorting for consistency (optional)
    age_group_avg = age_group_avg.sort_values(by='charges', ascending=False)
    region_avg = region_avg.sort_values(by='charges', ascending=False)
    bmi_avg = bmi_avg.sort_values(by='charges', ascending=False)

    # Plotting
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    # Age Group plot (with y-axis title)
    sns_bar_age = sns.barplot(ax=axes4[0], data=age_group_avg, x='age_group', y='charges', palette='Set2')
    axes4[0].set_title('Avg Charges by Age Group')
    axes4[0].set_ylabel("Average Charges")
    # Apply bar labels to all bars in the first plot
    for container in sns_bar_age.containers:
        axes4[0].bar_label(container, fmt='%.0f', label_type='edge', padding=3)

    # Region plot
    sns_bar_region = sns.barplot(ax=axes4[1], data=region_avg, x='region', y='charges', palette='Set2')
    axes4[1].set_title('Avg Charges by Region')
    axes4[1].set_ylabel(None)
    # Apply bar labels to all bars in the second plot
    for container in sns_bar_region.containers:
        axes4[1].bar_label(container, fmt='%.0f', label_type='edge', padding=3)

    # BMI Group plot
    sns_bar_bmi = sns.barplot(ax=axes4[2], data=bmi_avg, x='bmi_group', y='charges', palette='Set2')
    axes4[2].set_title('Avg Charges by BMI Category')
    axes4[2].set_ylabel(None)
    # Apply bar labels to all bars in the third plot
    for container in sns_bar_bmi.containers:
        axes4[2].bar_label(container, fmt='%.0f', label_type='edge', padding=3)

    # Overall layout
    fig4.suptitle("Average Charges by Groupings", fontsize=16)
    st.pyplot(fig4)

    st.markdown("""
    The first chart shows that seniors have the highest average charges, followed by middle-aged and young individuals. The second chart indicates that the Southeast region experiences the highest average charges compared to other regions. The third chart shows that obese individuals have the highest healthcare costs, followed by overweight and normal BMI groups.
    """)


    # ---- Correlation Heatmap (Plotly Interactive) ----
    st.subheader("5. Correlation Between Numerical Features")
    st.markdown("""
    This heatmap visualizes the Pearson correlation coefficients between numeric features (`age`, `bmi`, `children`, and `charges`).
    An interactive Plotly heatmap is used for enhanced user experience.
    """)

    corr = df[['age', 'bmi', 'children', 'charges']].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='Blues')

    # Manually update layout for title and centering
    fig_corr.update_layout(
        title={
            'text': 'Interactive Correlation Heatmap',
            'x': 0.5,  # Centers the title horizontally
            'xanchor': 'center',
            'font': {'size': 20}
        }
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    This plot shows that age has the highest correlation with medical charges, suggesting that older individuals tend to incur higher costs.
    The correlations of BMI and number of children with charges are relatively weak.
    """)

    # ---- Pairplot by Smoker ----
    st.subheader("6. Pairwise Relationships Colored by Smoker Status")
    st.markdown("""
    This pairplot uses Seaborn’s `pairplot` to visualize scatter and distribution plots for each numeric variable combination, colored by smoker status.
    This helps in observing how the smoker attribute affects the relationship between variables.
    """)

    pair = sns.pairplot(df, hue='smoker', diag_kind='kde', height=2.2, aspect=1)
    pair.fig.suptitle("Pairplot of Features with Smoker Status", fontsize=12, y=0.95)
    pair._legend.set_bbox_to_anchor((0.95, 0.5))
    pair._legend.set_title("Smoker")
    pair.fig.tight_layout()
    pair.fig.subplots_adjust(top=0.9, right=0.85, bottom=0.1)
    st.pyplot(pair.fig)

    st.markdown("""
    The pairplot shows that charges increase with age and are significantly higher for smokers compared to non-smokers across all age ranges.
    There is no strong linear relationship between BMI and charges, but smokers generally have higher costs regardless of BMI.
    """)

show_healthcare_eda_page()
