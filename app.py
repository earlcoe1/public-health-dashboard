import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Hospital & Public Health Insights Dashboard",
    page_icon="🏥",
    layout="wide"
)


class HealthAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    def clean_data(self):
        self.df = self.df.drop_duplicates()

        self.df["Admission_Date"] = pd.to_datetime(
            self.df["Admission_Date"], errors="coerce"
        )

        self.df["Year"] = self.df["Admission_Date"].dt.year
        self.df["Month"] = self.df["Admission_Date"].dt.month

        for col in ["Age", "Length_of_Stay", "Satisfaction"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        required_cols = [
            "Admission_Date", "Disease", "Outcome", "Department",
            "Age", "Gender", "Length_of_Stay", "Satisfaction", "Patient_Type"
        ]

        self.df = self.df.dropna(subset=required_cols)
        self.df = self.df[self.df["Length_of_Stay"] >= 0]

        return self.df

    def admissions_over_time(self, df):
        return df.groupby(["Year", "Month"]).size().reset_index(name="Admissions")

    def disease_counts(self, df):
        return df["Disease"].value_counts().head(10)

    def length_of_stay_by_department(self, df):
        return df.groupby("Department")["Length_of_Stay"].mean().sort_values(ascending=False)

    def satisfaction_by_department(self, df):
        return df.groupby("Department")["Satisfaction"].mean().sort_values(ascending=False)

    def outcome_summary(self, df):
        return df["Outcome"].value_counts()


st.markdown(
    """
    <h1 style='text-align:center;'>🏥 Hospital & Public Health Insights Dashboard</h1>
    <p style='text-align:center; font-size:18px;'>
    Operational, population health, equity, quality, and policy insights for healthcare decision-making.
    </p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload cleaned public health dataset",
    type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)

    analyzer = HealthAnalyzer(raw_df)
    df = analyzer.clean_data()

    if df.empty:
        st.error("The uploaded dataset has no usable records after cleaning.")
        st.stop()

    st.success("Dataset uploaded and cleaned successfully.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Dataset Summary")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Records", len(df))
    s2.metric("Diseases", df["Disease"].nunique())
    s3.metric("Departments", df["Department"].nunique())
    s4.metric("Date Range", f"{int(df['Year'].min())} - {int(df['Year'].max())}")

    st.markdown("### Numeric Summary")
    st.dataframe(
        df[["Age", "Length_of_Stay", "Satisfaction"]].describe(),
        use_container_width=True
    )

    st.markdown("### Categorical Summary")
    categorical_summary = pd.DataFrame({
        "Column": ["Disease", "Outcome", "Department", "Gender", "Patient_Type"],
        "Unique Values": [
            df["Disease"].nunique(),
            df["Outcome"].nunique(),
            df["Department"].nunique(),
            df["Gender"].nunique(),
            df["Patient_Type"].nunique()
        ],
        "Most Common Value": [
            df["Disease"].mode()[0],
            df["Outcome"].mode()[0],
            df["Department"].mode()[0],
            df["Gender"].mode()[0],
            df["Patient_Type"].mode()[0]
        ]
    })
    st.dataframe(categorical_summary, use_container_width=True)

    st.sidebar.header("Interactive Filters")

    gender_filter = st.sidebar.selectbox(
        "Gender",
        ["All"] + sorted(df["Gender"].dropna().unique().tolist())
    )

    department_filter = st.sidebar.multiselect(
        "Departments",
        sorted(df["Department"].dropna().unique().tolist()),
        default=sorted(df["Department"].dropna().unique().tolist())
    )

    disease_filter = st.sidebar.multiselect(
        "Diseases",
        sorted(df["Disease"].dropna().unique().tolist()),
        default=sorted(df["Disease"].dropna().unique().tolist())
    )

    outcome_filter = st.sidebar.selectbox(
        "Outcome",
        ["All"] + sorted(df["Outcome"].dropna().unique().tolist())
    )

    equity_dimension = st.sidebar.selectbox(
        "Equity Dimension",
        ["Gender", "Patient_Type", "Department"]
    )

    min_age = int(df["Age"].min())
    max_age = int(df["Age"].max())

    if min_age == max_age:
        st.sidebar.info(f"All records have the same age: {min_age}")
        age_filter = (min_age, max_age)
    else:
        age_filter = st.sidebar.slider(
            "Age Range",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )

    if st.sidebar.button("Reset Filters"):
        st.rerun()

    filtered_df = df.copy()

    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == gender_filter]

    if department_filter:
        filtered_df = filtered_df[filtered_df["Department"].isin(department_filter)]

    if disease_filter:
        filtered_df = filtered_df[filtered_df["Disease"].isin(disease_filter)]

    if outcome_filter != "All":
        filtered_df = filtered_df[filtered_df["Outcome"] == outcome_filter]

    filtered_df = filtered_df[
        (filtered_df["Age"] >= age_filter[0]) &
        (filtered_df["Age"] <= age_filter[1])
    ]

    if filtered_df.empty:
        st.warning("No records match the selected filters. Please adjust the filters.")
        st.stop()

    st.subheader("Executive KPI Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(filtered_df))
    col2.metric("Avg. Length of Stay", round(filtered_df["Length_of_Stay"].mean(), 2))
    col3.metric("Avg. Satisfaction", round(filtered_df["Satisfaction"].mean(), 2))
    col4.metric("Active Diseases", filtered_df["Disease"].nunique())

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Operations View",
            "Population Health View",
            "Equity View",
            "Quality & Outcomes View",
            "Policy Planning View"
        ]
    )

    with tab1:
        st.header("Operations View")
        st.write("Hospital utilization and operational performance indicators.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Emergency Cases", len(filtered_df[filtered_df["Patient_Type"] == "Emergency"]))
        c2.metric("Avg. Length of Stay", round(filtered_df["Length_of_Stay"].mean(), 2))
        c3.metric("Departments", filtered_df["Department"].nunique())

        st.subheader("Admissions Over Time")

        admissions = analyzer.admissions_over_time(filtered_df)
        admissions["Date"] = pd.to_datetime(
            admissions["Year"].astype(str) + "-" +
            admissions["Month"].astype(str) + "-01",
            errors="coerce"
        )
        admissions = admissions.sort_values("Date")

        fig, ax = plt.subplots()
        ax.plot(admissions["Date"], admissions["Admissions"], marker="o")
        ax.set_title("Admissions Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Admissions")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Average Length of Stay by Department")

        los = analyzer.length_of_stay_by_department(filtered_df)

        fig, ax = plt.subplots()
        los.plot(kind="bar", ax=ax)
        ax.set_title("Average Length of Stay by Department")
        ax.set_xlabel("Department")
        ax.set_ylabel("Average Days")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with tab2:
        st.header("Population Health View")
        st.write("Disease burden, case counts, and prevention-related insights.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Cases", len(filtered_df))
        c2.metric("Most Common Disease", filtered_df["Disease"].mode()[0])
        c3.metric("Mortality Count", len(filtered_df[filtered_df["Outcome"] == "Death"]))

        st.subheader("Most Common Diseases or Conditions")

        disease_counts = analyzer.disease_counts(filtered_df)

        fig, ax = plt.subplots()
        disease_counts.plot(kind="barh", ax=ax)
        ax.set_title("Top Diseases / Conditions")
        ax.set_xlabel("Case Count")
        ax.set_ylabel("Disease")
        ax.invert_yaxis()
        st.pyplot(fig)

        st.subheader("Disease Trends Over Time")

        disease_trend = (
            filtered_df.groupby(["Year", "Disease"])
            .size()
            .reset_index(name="Cases")
        )

        pivot = disease_trend.pivot(index="Year", columns="Disease", values="Cases").fillna(0)
        st.line_chart(pivot)

    with tab3:
        st.header("Equity View")
        st.write("Stratified outcomes by selected demographic or operational dimension.")

        st.subheader(f"Outcome Distribution by {equity_dimension}")

        equity_table = pd.crosstab(
            filtered_df[equity_dimension],
            filtered_df["Outcome"]
        )

        st.dataframe(equity_table, use_container_width=True)
        st.bar_chart(equity_table)

        st.subheader(f"Average Satisfaction by {equity_dimension}")

        satisfaction_equity = (
            filtered_df.groupby(equity_dimension)["Satisfaction"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(satisfaction_equity)

    with tab4:
        st.header("Quality & Outcomes View")
        st.write("Quality indicators, outcome patterns, and alert monitoring.")

        outcome_summary = analyzer.outcome_summary(filtered_df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Discharged", int(outcome_summary.get("Discharged", 0)))
        c2.metric("DAMA", int(outcome_summary.get("DAMA", 0)))
        c3.metric("Deaths", int(outcome_summary.get("Death", 0)))

        st.subheader("Outcome Distribution")

        fig, ax = plt.subplots()
        outcome_summary.plot(kind="bar", ax=ax)
        ax.set_title("Patient Outcomes")
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.subheader("Service Satisfaction by Department")

        satisfaction = analyzer.satisfaction_by_department(filtered_df)

        fig, ax = plt.subplots()
        satisfaction.plot(kind="bar", ax=ax)
        ax.set_title("Average Service Satisfaction by Department")
        ax.set_xlabel("Department")
        ax.set_ylabel("Average Rating")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Active Alerts")

        if filtered_df["Length_of_Stay"].mean() > 20:
            st.error("Critical Alert: Average length of stay is above expected operational threshold.")
        else:
            st.success("Length of stay is within acceptable range.")

        if filtered_df["Satisfaction"].mean() < 3:
            st.warning("Warning: Average satisfaction score is below quality target.")
        else:
            st.info("Satisfaction scores are stable.")

    with tab5:
        st.header("Policy Planning View")
        st.write("Actionable recommendations for decision-makers.")

        most_common_disease = filtered_df["Disease"].mode()[0]

        highest_los_department = (
            filtered_df.groupby("Department")["Length_of_Stay"]
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )

        lowest_satisfaction_department = (
            filtered_df.groupby("Department")["Satisfaction"]
            .mean()
            .sort_values()
            .index[0]
        )

        st.subheader("Policy Insights")

        st.markdown(
            f"""
            ### 1. Resource Allocation
            **Finding:** The department with the highest average length of stay is **{highest_los_department}**.  
            **Recommendation:** Review staffing, discharge planning, and patient flow processes in this department.

            ### 2. Disease Prevention Priority
            **Finding:** The most common condition in the dataset is **{most_common_disease}**.  
            **Recommendation:** Prioritize prevention, screening, and education programs targeting this condition.

            ### 3. Service Quality Improvement
            **Finding:** The department with the lowest satisfaction score is **{lowest_satisfaction_department}**.  
            **Recommendation:** Conduct service quality review and patient feedback analysis.

            ### 4. Capacity Planning
            **Finding:** Admissions trends show changing demand across months and years.  
            **Recommendation:** Use admission trends to guide staffing, bed planning, and supply allocation.
            """
        )

        st.subheader("Filtered Dataset Preview")
        st.dataframe(filtered_df.head(20), use_container_width=True)

else:
    st.info("Upload `clean_public_health_data.csv` to begin dashboard analysis.")
