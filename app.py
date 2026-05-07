import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Public Health Dashboard", page_icon="🏥", layout="wide")


class HealthAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    def clean_patient_records(self):
        self.df = self.df.drop_duplicates()

        self.df["Admission_Date"] = pd.to_datetime(
            self.df["Admission_Date"], errors="coerce"
        )

        self.df["Year"] = self.df["Admission_Date"].dt.year
        self.df["Month"] = self.df["Admission_Date"].dt.month

        numeric_cols = ["Age", "Length_of_Stay", "Satisfaction"]
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.df = self.df.dropna(
            subset=[
                "Admission_Date",
                "Disease",
                "Outcome",
                "Department",
                "Age",
                "Gender",
                "Length_of_Stay",
                "Satisfaction",
                "Patient_Type",
            ]
        )

        return self.df


st.title("🏥 Public Health: Patient & Hospital Data Dashboard")

st.write(
    "Analyze hospital or public health data for operational and policy insights."
)

uploaded_file = st.file_uploader(
    "Upload Cleaned Public Health CSV or Excel File",
    type=["csv", "xlsx"],
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    analyzer = HealthAnalyzer(df)
    df = analyzer.clean_patient_records()

    st.success("Dataset uploaded and cleaned successfully.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.dataframe(df.describe(include="all"))

    st.sidebar.header("Interactive Filters")

    gender_filter = st.sidebar.selectbox(
        "Filter by Gender", ["All"] + sorted(df["Gender"].dropna().unique())
    )

    dept_filter = st.sidebar.selectbox(
        "Filter by Department", ["All"] + sorted(df["Department"].dropna().unique())
    )

    outcome_filter = st.sidebar.selectbox(
        "Filter by Outcome", ["All"] + sorted(df["Outcome"].dropna().unique())
    )

    min_age = int(df["Age"].min())
    max_age = int(df["Age"].max())

    if min_age == max_age:
        st.sidebar.info(f"All records have the same age: {min_age}")
        age_filter = (min_age, max_age)
    else:
        age_filter = st.sidebar.slider(
            "Filter by Age",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age),
        )

    filtered_df = df.copy()

    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == gender_filter]

    if dept_filter != "All":
        filtered_df = filtered_df[filtered_df["Department"] == dept_filter]

    if outcome_filter != "All":
        filtered_df = filtered_df[filtered_df["Outcome"] == outcome_filter]

    filtered_df = filtered_df[
        (filtered_df["Age"] >= age_filter[0]) &
        (filtered_df["Age"] <= age_filter[1])
    ]

    st.subheader("Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(filtered_df))
    col2.metric("Departments", filtered_df["Department"].nunique())
    col3.metric("Average Age", round(filtered_df["Age"].mean(), 1))
    col4.metric("Average Satisfaction", round(filtered_df["Satisfaction"].mean(), 2))

    st.divider()

    st.subheader("1. Patient Outcomes by Age")

    fig1, ax1 = plt.subplots()
    for outcome in filtered_df["Outcome"].dropna().unique():
        subset = filtered_df[filtered_df["Outcome"] == outcome]
        ax1.hist(subset["Age"], alpha=0.6, label=str(outcome))

    ax1.set_xlabel("Age")
    ax1.set_ylabel("Number of Records")
    ax1.set_title("Patient Outcomes by Age")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("Patient Outcomes by Gender")
    outcome_gender = pd.crosstab(filtered_df["Gender"], filtered_df["Outcome"])
    st.bar_chart(outcome_gender)

    st.divider()

    st.subheader("2. Admissions Over Time")

    admissions = (
        filtered_df.groupby(["Year", "Month"])
        .size()
        .reset_index(name="Admissions")
    )

    admissions["Date"] = pd.to_datetime(
        admissions["Year"].astype(str)
        + "-"
        + admissions["Month"].astype(str)
        + "-01",
        errors="coerce",
    )

    admissions = admissions.sort_values("Date")

    fig2, ax2 = plt.subplots()
    ax2.plot(admissions["Date"], admissions["Admissions"], marker="o")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Number of Admissions")
    ax2.set_title("Admissions Over Time")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.divider()

    st.subheader("3. Most Common Diseases or Conditions")

    diseases = filtered_df["Disease"].value_counts().head(10)

    fig3, ax3 = plt.subplots()
    diseases.plot(kind="barh", ax=ax3)
    ax3.set_xlabel("Frequency")
    ax3.set_ylabel("Disease / Condition")
    ax3.set_title("Top 10 Most Common Diseases or Conditions")
    ax3.invert_yaxis()
    st.pyplot(fig3)

    st.divider()

    st.subheader("4. Average Length of Stay by Department")

    los = (
        filtered_df.groupby("Department")["Length_of_Stay"]
        .mean()
        .sort_values(ascending=False)
    )

    fig4, ax4 = plt.subplots()
    los.plot(kind="bar", ax=ax4)
    ax4.set_xlabel("Department")
    ax4.set_ylabel("Average Length of Stay")
    ax4.set_title("Length of Stay by Department")
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    st.divider()

    st.subheader("5. Average Service Satisfaction by Department")

    satisfaction = (
        filtered_df.groupby("Department")["Satisfaction"]
        .mean()
        .sort_values(ascending=False)
    )

    fig5, ax5 = plt.subplots()
    satisfaction.plot(kind="bar", ax=ax5)
    ax5.set_xlabel("Department")
    ax5.set_ylabel("Average Satisfaction Rating")
    ax5.set_title("Average Service Satisfaction by Department")
    plt.xticks(rotation=45)
    st.pyplot(fig5)

    st.divider()

    st.subheader("Public Health and Operational Insights")

    st.markdown(
        """
        - Patient outcome analysis helps identify discharge, DAMA, and death trends.
        - Admissions over time supports staffing and resource planning.
        - Disease frequency analysis identifies common public health concerns.
        - Length of stay helps evaluate department efficiency.
        - Satisfaction ratings support quality improvement.
        """
    )

    st.subheader("Policy Recommendations")

    st.markdown(
        """
        1. Increase staffing during high-admission periods.
        2. Prioritize prevention programs for the most common conditions.
        3. Review departments with longer average length of stay.
        4. Improve services in departments with lower satisfaction ratings.
        5. Use dashboard analytics for continuous hospital performance monitoring.
        """
    )

else:
    st.info("Please upload `clean_public_health_data.csv` to begin.")
