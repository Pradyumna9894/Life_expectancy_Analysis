import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import os

# Upload dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Load dataset
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df = df[df["Year"].between(2019, 2025)]  # Filter for years 2019-2025
        return df
    else:
        st.error("Please upload a CSV file.")
        return None

df = load_data(uploaded_file)
if df is None:
    st.stop()

# Preprocess data
def preprocess_data(df):
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    label_enc = LabelEncoder()
    df["Status"] = label_enc.fit_transform(df["Status"])
    return df

df = preprocess_data(df)

# UI: Title and Description
st.title("Life Expectancy Analysis & Prediction")
st.write("Explore life expectancy trends and make predictions based on key health indicators.")

# Top Countries with Highest and Lowest Life Expectancy
st.subheader("Top Countries with Highest and Lowest Life Expectancy")

# Top 10 highest life expectancy countries
top_countries = df.groupby("Country")["Life_expect"].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(y=top_countries.index, x=top_countries.values, palette="Blues_r", ax=ax)
ax.set_xlabel("Average Life Expectancy")
ax.set_ylabel("Country")
ax.set_title("Top 10 Countries with Highest Life Expectancy")
st.pyplot(fig)

# Bottom 10 lowest life expectancy countries
bottom_countries = df.groupby("Country")["Life_expect"].mean().sort_values(ascending=True).head(10)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(y=bottom_countries.index, x=bottom_countries.values, palette="Reds_r", ax=ax)
ax.set_xlabel("Average Life Expectancy")
ax.set_ylabel("Country")
ax.set_title("Bottom 10 Countries with Lowest Life Expectancy")
st.pyplot(fig)

# Show dataset
if st.checkbox("Show raw data"):
    st.write(df.head())

# Select Country for Analysis
st.subheader("Life Expectancy Trends Over Time")
selected_country = st.selectbox("Select a country:", df["Country"].unique())
country_data = df[df["Country"] == selected_country]

fig, ax = plt.subplots()
ax.plot(country_data["Year"], country_data["Life_expect"], marker="o", linestyle="-")
ax.set_xlabel("Year")
ax.set_ylabel("Life Expectancy")
ax.set_title(f"Life Expectancy Trend in {selected_country}")
ax.set_xticks(range(2019, 2026))  # Set specific years on x-axis
st.pyplot(fig)

# Additional Visualizations
st.subheader("Data Visualizations")

# Factors Affecting Life Expectancy
st.subheader("Factors Affecting Life Expectancy")
factors = ["Adult_Mortality", "infant_deaths", "Alcohol", "percentage_expenditure", "HepatitisB", "Measles", "BMI", "Total_expenditure", "HIV_AIDS", "GDP", "Population", "Schooling"]
for factor in factors:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df[factor], y=df["Life_expect"], alpha=0.6, ax=ax)
    ax.set_xlabel(factor)
    ax.set_ylabel("Life Expectancy")
    ax.set_title(f"Life Expectancy vs. {factor}")
    st.pyplot(fig)

# Pie Chart - Developed vs Developing Countries
st.subheader("Proportion of Developed vs. Developing Countries")
status_counts = df["Status"].value_counts()
fig, ax = plt.subplots()
ax.pie(status_counts, labels=["Developing", "Developed"], autopct="%1.1f%%", colors=["blue", "orange"], startangle=140)
st.pyplot(fig)

# Line Plot - Global Life Expectancy Trend
st.subheader("Global Life Expectancy Over Time")
global_trend = df.groupby("Year")["Life_expect"].mean()
fig, ax = plt.subplots()
ax.plot(global_trend.index, global_trend.values, marker="o", linestyle="-", color="blue")
ax.set_xlabel("Year")
ax.set_ylabel("Average Life Expectancy")
ax.set_title("Global Life Expectancy Over Time")
ax.set_xticks(range(2019, 2026))  # Set specific years on x-axis
st.pyplot(fig)

# Conclusion
st.subheader("Conclusion")
st.write("The analysis highlights that life expectancy is influenced by multiple socio-economic and health-related factors.")
st.write("Key observations:")
st.write("- Higher GDP and healthcare expenditure generally lead to higher life expectancy.")
st.write("- Infant mortality and HIV/AIDS prevalence negatively impact life expectancy.")
st.write("- Countries with better schooling systems tend to have longer life expectancies.")
st.write("Overall, economic stability, healthcare accessibility, and education are crucial determinants of a nation's life expectancy.")

# Predictive Modeling
st.subheader("Predict Future Life Expectancy")
prediction_method = st.radio("Choose Model:", ["Linear Regression", "SVR"])

# Prepare data for training
features = ['Year', 'Status', 'Adult_Mortality', 'infant_deaths', 'Alcohol', 'percentage_expenditure', 'HepatitisB', 
            'Measles', 'BMI', 'under_five_deaths', 'Polio', 'Total_expenditure', 'Diphtheria', 'HIV_AIDS', 'GDP', 
            'Population', 'thinness_1-19_years', 'thinness_5-9_years', 'Income_composition_of_resources', 'Schooling']

target = "Life_expect"
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train selected model
if prediction_method == "Linear Regression":
    model = LinearRegression()
elif prediction_method == "SVR":
    model = SVR()

model.fit(X_train, y_train)

# Predict for user-selected year
future_year = st.slider("Select future year:", 2019, 2040, 2025)

if st.button("Predict"):
    input_data = country_data[features].iloc[-1:].copy()
    input_data["Year"] = future_year
    input_data = scaler.transform(input_data)
    predicted_life_exp = model.predict(input_data)[0]
    st.write(f"Predicted Life Expectancy in {future_year}: **{predicted_life_exp:.2f} years**")