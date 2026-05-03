import streamlit as st
import pandas as pd

st.title("Public Health Dashboard")

class HealthAnalyzer:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        return self.df.drop_duplicates()

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    analyzer = HealthAnalyzer(df)
    df = analyzer.clean_data()

    st.write(df.head())