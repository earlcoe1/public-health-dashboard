import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Hospital Patient Admission Trends Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Patient Admission Trends Dashboard")
st.write(
    "This dashboard analyzes patient admission trends over time using an uploaded hospital dataset."
)

# 1. Upload hospital dataset
st.header("1. Upload Hospital Dataset")

uploaded_file = st.file_uploader(
    "Upload the hospital dataset file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # Read uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("File uploaded successfully!")

    # 2. Display head and tail
    st.header("2. Preview Head and Tail of Uploaded Data")

    st.subheader("Head of Dataset")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Tail of Dataset")
    st.dataframe(df.tail(), use_container_width=True)

    # 3. Display summary statistics for numerical data
    st.header("3. Summary of Statistical Properties on Numerical Data")

    numeric_df = df.select_dtypes(include=["number"])

    if not numeric_df.empty:
        st.dataframe(numeric_df.describe(), use_container_width=True)
    else:
        st.warning("No numerical columns were found in the dataset.")

    # 4. Calculate and display admissions per month
    st.header("4. Number of Admissions Per Month")

    # Flexible date column detection
    possible_date_columns = [
        "Admission_Date",
        "Date of Admission",
        "D.O.A",
        "DOA",
        "month year",
        "Month Year"
    ]

    date_column = None

    for col in possible_date_columns:
        if col in df.columns:
            date_column = col
            break

    if date_column is not None:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        df = df.dropna(subset=[date_column])

        df["Month-Year"] = df[date_column].dt.to_period("M").astype(str)

        monthly_admissions = (
            df.groupby("Month-Year")
            .size()
            .reset_index(name="Number of Admissions")
        )

        st.dataframe(monthly_admissions, use_container_width=True)

        # 5. Visualize monthly admissions
        st.header("5. Monthly Admissions Visualization")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            monthly_admissions["Month-Year"],
            monthly_admissions["Number of Admissions"],
            marker="o",
            linewidth=2
        )

        ax.set_title("Monthly Patient Admissions Over Time")
        ax.set_xlabel("Month-Year")
        ax.set_ylabel("Number of Admissions")
        plt.xticks(rotation=45)

        st.pyplot(fig)

        st.write(
            "The chart shows monthly patient admission trends over time. "
            "It helps identify periods with high and low hospital admissions, "
            "which can support hospital staffing, resource planning, and operational decision-making."
        )

    else:
        st.error(
            "No valid admission date column was found. "
            "Please ensure your dataset includes one of these columns: "
            "Admission_Date, Date of Admission, D.O.A, DOA, month year, or Month Year."
        )

else:
    st.info("Please upload the hospital dataset to begin.")
