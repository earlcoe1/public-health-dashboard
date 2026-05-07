import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Healthcare Analytics Dashboard", layout="wide")

st.title("🏥 Healthcare Patient Analytics Dashboard")
st.markdown("Upload your cleaned healthcare dataset for analysis and visualization.")

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

    # Clean column names
    df.columns = df.columns.str.strip()

    # Display columns
    st.subheader("Available Columns in Your Dataset")
    st.write(list(df.columns))

    # -------------------------------
    # Handle Dates
    # -------------------------------
    if "Admission_Date" in df.columns:
        df["Admission_Date"] = pd.to_datetime(df["Admission_Date"], errors="coerce")

    elif "Year" in df.columns and "Month" in df.columns:
        df["Admission_Date"] = pd.to_datetime(
            df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01",
            errors="coerce"
        )

    else:
        st.error("Admission date information not found.")
        st.stop()

    # -------------------------------
    # Length of Stay
    # -------------------------------
    if "Length_of_Stay" in df.columns:
        df["Length of Stay (Days)"] = pd.to_numeric(
            df["Length_of_Stay"], errors="coerce"
        )

    elif "Length of Stay (Days)" in df.columns:
        df["Length of Stay (Days)"] = pd.to_numeric(
            df["Length of Stay (Days)"], errors="coerce"
        )

    else:
        df["Length of Stay (Days)"] = 0

    # -------------------------------
    # Satisfaction Score
    # -------------------------------
    if "Satisfaction" in df.columns:
        df["Patient Satisfaction Score"] = pd.to_numeric(
            df["Satisfaction"], errors="coerce"
        )

    elif "Patient Satisfaction Score" in df.columns:
        df["Patient Satisfaction Score"] = pd.to_numeric(
            df["Patient Satisfaction Score"], errors="coerce"
        )

    else:
        df["Patient Satisfaction Score"] = 3

    # Fill missing values
    df["Length of Stay (Days)"] = df["Length of Stay (Days)"].fillna(
        df["Length of Stay (Days)"].median()
    )

    df["Patient Satisfaction Score"] = df["Patient Satisfaction Score"].fillna(
        df["Patient Satisfaction Score"].median()
    )

    # -------------------------------
    # Build Numeric Summary
    # -------------------------------
    numeric_cols = []

    if "Age" in df.columns:
        numeric_cols.append("Age")

    numeric_cols.extend([
        "Length of Stay (Days)",
        "Patient Satisfaction Score"
    ])

    # -------------------------------
    # Build Categorical Summary
    # -------------------------------
    categorical_candidates = [
        "Disease",
        "Outcome",
        "Department",
        "Gender",
        "Patient_Type"
    ]

    cat_summary = []

    for col in categorical_candidates:
        if col in df.columns:
            cat_summary.append({
                "Column": col,
                "Unique Values": df[col].nunique(),
                "Most Common Value": df[col].mode()[0]
            })

    # -------------------------------
    # Prepare Charts
    # -------------------------------
    df["Admission_Month"] = df["Admission_Date"].dt.to_period("M").astype(str)
    monthly_admissions = df.groupby("Admission_Month").size()

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    monthly_admissions.plot(kind="line", marker="o", ax=ax1)
    ax1.set_title("Monthly Patient Admissions")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Number of Admissions")
    plt.xticks(rotation=45)

    if "Disease" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        df["Disease"].value_counts().plot(kind="bar", ax=ax2)
        ax2.set_title("Disease Distribution")
        ax2.set_xlabel("Disease")
        ax2.set_ylabel("Patient Count")
        plt.xticks(rotation=45)

    if "Department" in df.columns:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        df["Department"].value_counts().plot(kind="bar", ax=ax3)
        ax3.set_title("Patients by Department")
        ax3.set_xlabel("Department")
        ax3.set_ylabel("Patient Count")
        plt.xticks(rotation=45)

    # -------------------------------
    # PLACEHOLDERS
    # -------------------------------
    head_placeholder = st.empty()
    tail_placeholder = st.empty()
    numeric_placeholder = st.empty()
    categorical_placeholder = st.empty()
    monthly_placeholder = st.empty()
    disease_placeholder = st.empty()
    department_placeholder = st.empty()
    shape_placeholder = st.empty()

    # -------------------------------
    # DISPLAY CONTENT
    # -------------------------------
    with head_placeholder.container():
        st.subheader("📄 Dataset Preview - Head")
        st.dataframe(df.head())

    with tail_placeholder.container():
        st.subheader("📄 Dataset Preview - Tail")
        st.dataframe(df.tail())

    with numeric_placeholder.container():
        st.subheader("📊 Numeric Summary")
        st.dataframe(df[numeric_cols].describe())

    with categorical_placeholder.container():
        st.subheader("📋 Categorical Summary")
        st.dataframe(pd.DataFrame(cat_summary))

    with monthly_placeholder.container():
        st.subheader("📅 Monthly Admissions Trends")
        st.pyplot(fig1)

    if "Disease" in df.columns:
        with disease_placeholder.container():
            st.subheader("🦠 Disease Distribution")
            st.pyplot(fig2)

    if "Department" in df.columns:
        with department_placeholder.container():
            st.subheader("🏨 Department Comparison")
            st.pyplot(fig3)

    with shape_placeholder.container():
        st.subheader("📌 Final Dataset Shape")
        st.write(df.shape)

    st.success("Dashboard analysis completed successfully!")
