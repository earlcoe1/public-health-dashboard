# ==============================
# UPDATED STREAMLIT APP.PY
# Placeholder Dropdown Filters Version
# ==============================

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

    # ============================
    # FIXED DATASET SUMMARY
    # ============================
    st.subheader("Dataset Summary")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Records", len(df))
    s2.metric("Diseases", df["Disease"].nunique())
    s3.metric("Departments", df["Department"].nunique())
    s4.metric("Date Range", f"{int(df['Year'].min())} - {int(df['Year'].max())}")

    st.markdown("### Numeric Summary")
    numeric_summary = df[["Age", "Length_of_Stay", "Satisfaction"]].describe()
    st.dataframe(numeric_summary, use_container_width=True)

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

    # ============================
    # SIDEBAR FILTERS (UPDATED)
    # ============================
    st.sidebar.header("Interactive Filters")

    gender_filter = st.sidebar.selectbox(
        "Gender",
        ["All"] + sorted(df["Gender"].dropna().unique().tolist())
    )

    department_filter = st.sidebar.selectbox(
        "Department",
        ["All"] + sorted(df["Department"].dropna().unique().tolist())
    )

    disease_filter = st.sidebar.selectbox(
        "Disease",
        ["All"] + sorted(df["Disease"].dropna().unique().tolist())
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
        age_filter = (min_age, max_age)
    else:
        age_filter = st.sidebar.slider(
            "Age Range",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )

    filtered_df = df.copy()

    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == gender_filter]

    if department_filter != "All":
        filtered_df = filtered_df[filtered_df["Department"] == department_filter]

    if disease_filter != "All":
        filtered_df = filtered_df[filtered_df["Disease"] == disease_filter]

    if outcome_filter != "All":
        filtered_df = filtered_df[filtered_df["Outcome"] == outcome_filter]

    filtered_df = filtered_df[
        (filtered_df["Age"] >= age_filter[0]) &
        (filtered_df["Age"] <= age_filter[1])
    ]

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    # ============================
    # KPI SUMMARY
    # ============================
    st.subheader("Executive KPI Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(filtered_df))
    col2.metric("Avg. Length of Stay", round(filtered_df["Length_of_Stay"].mean(), 2))
    col3.metric("Avg. Satisfaction", round(filtered_df["Satisfaction"].mean(), 2))
    col4.metric("Active Diseases", filtered_df["Disease"].nunique())

    # ============================
    # VISUALS
    # ============================

    st.header("1. Patient Outcomes Distribution by Age")

    fig, ax = plt.subplots()
    for outcome in filtered_df["Outcome"].unique():
        subset = filtered_df[filtered_df["Outcome"] == outcome]
        ax.hist(subset["Age"], alpha=0.5, label=outcome)

    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

    st.header("2. Admissions Over Time")

    admissions = analyzer.admissions_over_time(filtered_df)
    admissions["Date"] = pd.to_datetime(
        admissions["Year"].astype(str) + "-" +
        admissions["Month"].astype(str) + "-01"
    )

    fig, ax = plt.subplots()
    ax.plot(admissions["Date"], admissions["Admissions"], marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Admissions")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.header("3. Most Common Diseases or Conditions")

    disease_counts = analyzer.disease_counts(filtered_df)

    fig, ax = plt.subplots()
    disease_counts.plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    st.pyplot(fig)

    st.header("4. Average Length of Stay by Department")

    los = analyzer.length_of_stay_by_department(filtered_df)

    fig, ax = plt.subplots()
    los.plot(kind="bar", ax=ax)
    ax.set_ylabel("Days")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.header("5. Average Service Satisfaction by Department")

    satisfaction = analyzer.satisfaction_by_department(filtered_df)

    fig, ax = plt.subplots()
    satisfaction.plot(kind="bar", ax=ax)
    ax.set_ylabel("Rating")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Policy Recommendations")

    st.markdown(f"""
    - Focus preventive resources on **{filtered_df['Disease'].mode()[0]}**
    - Improve patient experience in low-satisfaction departments
    - Optimize staffing for high-admission periods
    - Reduce long stays in departments with operational inefficiencies
    """)

else:
    st.info("Upload `clean_public_health_data.csv` to begin dashboard analysis.")
