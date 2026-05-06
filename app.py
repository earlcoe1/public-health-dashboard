import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Public Health Dashboard",
    page_icon="🏥",
    layout="wide"
)


class HealthAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    def clean_patient_records(self):
        self.df = self.df.drop_duplicates()

        if "Admission_Date" in self.df.columns:
            self.df["Admission_Date"] = pd.to_datetime(
                self.df["Admission_Date"], errors="coerce"
            )
            self.df["Year"] = self.df["Admission_Date"].dt.year
            self.df["Month"] = self.df["Admission_Date"].dt.month
            self.df["Month_Name"] = self.df["Admission_Date"].dt.strftime("%b")

        numeric_cols = ["Age", "Length_of_Stay", "Satisfaction"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        required_cols = ["Age", "Gender", "Department", "Outcome"]
        available_required = [col for col in required_cols if col in self.df.columns]
        self.df = self.df.dropna(subset=available_required)

        return self.df

    def summarize_outcomes(self):
        return self.df["Outcome"].value_counts()

    def aggregate_by_age_gender_department(self):
        return self.df.groupby(["Age", "Gender", "Department"]).size().reset_index(name="Count")

    def admissions_over_time(self):
        return self.df.groupby(["Year", "Month"]).size().reset_index(name="Admissions")

    def common_diseases(self):
        return self.df["Disease"].value_counts().head(10)

    def avg_satisfaction_by_department(self):
        return self.df.groupby("Department")["Satisfaction"].mean().sort_values(ascending=False)

    def avg_length_of_stay_by_department(self):
        return self.df.groupby("Department")["Length_of_Stay"].mean().sort_values(ascending=False)


st.markdown(
    """
    <h1 style='text-align:center;'>🏥 Public Health: Patient & Hospital Data Dashboard</h1>
    <p style='text-align:center; font-size:18px;'>
    An interactive dashboard for analyzing hospital/public health data for operational and policy insights.
    </p>
    """,
    unsafe_allow_html=True
)

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
    df = analyzer.clean_patient_records()

    st.success("Dataset uploaded and cleaned successfully.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(df.describe(include="all"))

    st.sidebar.header("Interactive Filters")

    gender_options = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
    dept_options = ["All"] + sorted(df["Department"].dropna().unique().tolist())
    outcome_options = ["All"] + sorted(df["Outcome"].dropna().unique().tolist())

    gender_filter = st.sidebar.selectbox("Filter by Gender", gender_options)
    dept_filter = st.sidebar.selectbox("Filter by Department", dept_options)
    outcome_filter = st.sidebar.selectbox("Filter by Outcome", outcome_options)

    min_age = int(df["Age"].min())
    max_age = int(df["Age"].max())

    age_filter = st.sidebar.slider(
        "Filter by Age",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
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

    st.subheader("1. Distribution of Patient Outcomes by Age")

    fig1, ax1 = plt.subplots()
    for outcome in filtered_df["Outcome"].dropna().unique():
        subset = filtered_df[filtered_df["Outcome"] == outcome]
        ax1.hist(subset["Age"], alpha=0.6, label=str(outcome))

    ax1.set_xlabel("Age")
    ax1.set_ylabel("Number of Patients / Records")
    ax1.set_title("Patient Outcomes by Age")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("Outcome Distribution by Gender")

    outcome_gender = pd.crosstab(filtered_df["Gender"], filtered_df["Outcome"])
    st.bar_chart(outcome_gender)

    st.divider()

    st.subheader("2. Admissions Over Time")

    admissions = filtered_df.groupby(["Year", "Month"]).size().reset_index(name="Admissions")
    admissions["Date"] = pd.to_datetime(
        admissions["Year"].astype(str) + "-" + admissions["Month"].astype(str) + "-01",
        errors="coerce"
    )
    admissions = admissions.sort_values("Date")

    fig2, ax2 = plt.subplots()
    ax2.plot(admissions["Date"], admissions["Admissions"], marker="o")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Number of Admissions / Records")
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

    los = filtered_df.groupby("Department")["Length_of_Stay"].mean().sort_values(ascending=False)

    fig4, ax4 = plt.subplots()
    los.plot(kind="bar", ax=ax4)
    ax4.set_xlabel("Department")
    ax4.set_ylabel("Average Length of Stay")
    ax4.set_title("Length of Stay by Department")
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    st.divider()

    st.subheader("5. Average Service Satisfaction by Department")

    satisfaction = filtered_df.groupby("Department")["Satisfaction"].mean().sort_values(ascending=False)

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
        - **Patient outcome analysis** helps identify trends in discharge, DAMA, and mortality-related outcomes.
        - **Admissions over time** supports hospital staffing, resource planning, and demand forecasting.
        - **Disease frequency analysis** identifies common public health concerns that require stronger intervention.
        - **Length of stay analysis** helps evaluate department efficiency and patient flow.
        - **Satisfaction ratings** support service quality improvement and patient-centered care.
        """
    )

    st.subheader("Policy Recommendations")

    st.markdown(
        """
        1. Increase staffing and supplies during high-admission periods.
        2. Prioritize prevention and treatment programs for the most frequent conditions.
        3. Investigate departments with longer average length of stay.
        4. Improve patient service processes in departments with lower satisfaction ratings.
        5. Use dashboard analytics to support continuous hospital performance monitoring.
        """
    )

else:
    st.info("Upload your cleaned public health dataset to begin analysis.")
