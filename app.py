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

    # 3. Display summary statistics for selected numerical data only
    st.header("3. Summary of Statistical Properties on Numerical Data")

    summary_columns = ["Age", "Length_of_Stay", "Satisfaction"]

    available_summary_columns = [
        col for col in summary_columns if col in df.columns
    ]

    if available_summary_columns:
        st.dataframe(
            df[available_summary_columns].describe().round(2),
            use_container_width=True
        )
    else:
        st.warning(
            "No required numerical columns were found. "
            "Expected columns include Age, Length_of_Stay, and Satisfaction."
        )

    # 4. Calculate and display admissions per month
    st.header("4. Number of Admissions Per Month")

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

        monthly_admissions = (
            df.groupby(df[date_column].dt.to_period("M"))
            .size()
            .reset_index(name="Number of Admissions")
        )

        monthly_admissions[date_column] = monthly_admissions[date_column].astype(str)

        monthly_admissions = monthly_admissions.rename(
            columns={date_column: "Month-Year"}
        )

        # Extra polish: sort monthly admissions by month-year
        monthly_admissions = monthly_admissions.sort_values("Month-Year")

        st.dataframe(
            monthly_admissions.style.format({
                "Number of Admissions": "{:,}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # 5. Visualize monthly admissions
        st.header("5. Monthly Admissions Visualization")

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.bar(
            monthly_admissions["Month-Year"],
            monthly_admissions["Number of Admissions"]
        )

        ax.set_title("Monthly Patient Admissions Trends", fontsize=14)
        ax.set_xlabel("Month-Year")
        ax.set_ylabel("Number of Admissions")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        st.pyplot(fig)

        st.write(
            "The chart shows the number of patient admissions for each month. "
            "It helps identify months with higher or lower hospital admissions, "
            "which can support staffing, resource allocation, and hospital planning."
        )

    else:
        st.error(
            "No valid admission date column was found. "
            "Please ensure your dataset includes one of these columns: "
            "Admission_Date, Date of Admission, D.O.A, DOA, month year, or Month Year."
        )

else:
    st.info("Please upload the hospital dataset to begin.")
