# ================================
# PROFESSIONAL STREAMLIT HEALTHCARE DASHBOARD
# Enhanced UI + Visualizations
# ================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Healthcare Analytics Dashboard", layout="wide")

st.title("🏥 Healthcare Patient Analytics Dashboard")
st.markdown("Upload your healthcare dataset for cleaning, analysis, and visualization.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload your healthcare dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # -------------------------------
    # Load Data
    # -------------------------------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # -------------------------------
    # Data Cleaning
    # -------------------------------
    df["Admission_Date"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
    df["Discharge_Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")

    df["Disease"] = df["Medical Condition"]
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Gender"] = df["Gender"]
    df["Patient_Type"] = df["Admission Type"]
    df["Department"] = df["Doctor"].fillna("General")
    df["Outcome"] = df["Test Results"].fillna("Unknown")

    # Calculate Length of Stay
    df["Length of Stay (Days)"] = (
        df["Discharge_Date"] - df["Admission_Date"]
    ).dt.days

    # Fill missing Length of Stay
    median_stay = df["Length of Stay (Days)"].median()
    df["Length of Stay (Days)"] = df["Length of Stay (Days)"].fillna(median_stay)

    # Satisfaction score
    df["Patient Satisfaction Score"] = (
        df["Billing Amount"].rank(pct=True) * 5
    ).round().clip(1, 5)

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing categorical fields
    categorical_cols = ["Disease", "Outcome", "Department", "Gender", "Patient_Type"]
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # -------------------------------
    # Dataset Preview
    # -------------------------------
    st.subheader("📄 Dataset Preview (Head)")
    st.dataframe(df.head())

    st.subheader("📄 Dataset Preview (Tail)")
    st.dataframe(df.tail())

    # -------------------------------
    # Numeric Summary
    # -------------------------------
    st.subheader("📊 Numeric Summary")
    numeric_summary = df[
        ["Age", "Length of Stay (Days)", "Patient Satisfaction Score"]
    ].describe()
    st.dataframe(numeric_summary)

    # -------------------------------
    # Categorical Summary
    # -------------------------------
    st.subheader("📋 Categorical Summary")

    cat_summary = []
    for col in categorical_cols:
        cat_summary.append({
            "Column": col,
            "Unique Values": df[col].nunique(),
            "Most Common Value": df[col].mode()[0]
        })

    cat_summary_df = pd.DataFrame(cat_summary)
    st.dataframe(cat_summary_df)

    # -------------------------------
    # Monthly Admission Trends
    # -------------------------------
    st.subheader("📅 Monthly Admissions Trends")

    df["Admission_Month"] = df["Admission_Date"].dt.to_period("M").astype(str)
    monthly_admissions = df.groupby("Admission_Month").size()

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    monthly_admissions.plot(kind="line", marker="o", ax=ax1)
    ax1.set_title("Monthly Patient Admissions")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Number of Admissions")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # -------------------------------
    # Disease Distribution
    # -------------------------------
    st.subheader("🦠 Disease Distribution")

    disease_counts = df["Disease"].value_counts()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    disease_counts.plot(kind="bar", ax=ax2)
    ax2.set_title("Distribution of Diseases")
    ax2.set_xlabel("Disease")
    ax2.set_ylabel("Patient Count")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # -------------------------------
    # Department Comparison
    # -------------------------------
    st.subheader("🏨 Department Comparison")

    department_counts = df["Department"].value_counts()

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    department_counts.plot(kind="bar", ax=ax3)
    ax3.set_title("Patients by Department")
    ax3.set_xlabel("Department")
    ax3.set_ylabel("Patient Count")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # -------------------------------
    # Final Dataset Shape
    # -------------------------------
    st.subheader("📌 Final Dataset Shape")
    st.write(df.shape)

    st.success("Dashboard analysis completed successfully!")
