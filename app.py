import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Healthcare Analytics Dashboard", layout="wide")

st.title("🏥 Healthcare Patient Analytics Dashboard")

uploaded_file = st.file_uploader(
    "Upload your healthcare dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    st.subheader("Available Columns in Your Dataset")
    st.write(list(df.columns))

    # Helper function to find matching column names
    def find_column(possible_names):
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    admission_col = find_column([
        "Date of Admission",
        "Admission Date",
        "Admission_Date",
        "admission_date"
    ])

    discharge_col = find_column([
        "Discharge Date",
        "Discharge_Date",
        "discharge_date"
    ])

    disease_col = find_column([
        "Medical Condition",
        "Disease",
        "Condition"
    ])

    if admission_col is None or discharge_col is None:
        st.error("Admission or Discharge date column was not found. Check the column names shown above.")
        st.stop()

    df["Admission_Date"] = pd.to_datetime(df[admission_col], errors="coerce")
    df["Discharge_Date"] = pd.to_datetime(df[discharge_col], errors="coerce")

    df["Length of Stay (Days)"] = (
        df["Discharge_Date"] - df["Admission_Date"]
    ).dt.days

    df["Length of Stay (Days)"] = df["Length of Stay (Days)"].fillna(
        df["Length of Stay (Days)"].median()
    )

    if disease_col:
        df["Disease"] = df[disease_col]
    else:
        df["Disease"] = "Unknown"

    if "Billing Amount" in df.columns:
        df["Patient Satisfaction Score"] = (
            df["Billing Amount"].rank(pct=True) * 5
        ).round().clip(1, 5)
    else:
        df["Patient Satisfaction Score"] = 3

    st.subheader("Dataset Preview - Head")
    st.dataframe(df.head())

    st.subheader("Dataset Preview - Tail")
    st.dataframe(df.tail())

    st.subheader("Numeric Summary")
    st.dataframe(
        df[["Age", "Length of Stay (Days)", "Patient Satisfaction Score"]]
        .describe()
    )

    st.subheader("Monthly Admissions Trends")

    df["Admission_Month"] = df["Admission_Date"].dt.to_period("M").astype(str)
    monthly_admissions = df.groupby("Admission_Month").size()

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    monthly_admissions.plot(kind="line", marker="o", ax=ax1)
    ax1.set_title("Monthly Patient Admissions")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Number of Admissions")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("Disease Distribution")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    df["Disease"].value_counts().plot(kind="bar", ax=ax2)
    ax2.set_title("Disease Distribution")
    ax2.set_xlabel("Disease")
    ax2.set_ylabel("Patient Count")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.success("Dashboard completed successfully!")
