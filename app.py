import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Project 6: Public Health Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# CSS DESIGN
# ============================================================

st.markdown("""
<style>
.dashboard-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #0f172a;
}
.dashboard-subtitle {
    text-align: center;
    font-size: 18px;
    color: #475569;
    margin-bottom: 25px;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.10);
    text-align: center;
    border-left: 6px solid #2563eb;
}
.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #1e293b;
}
.metric-label {
    font-size: 14px;
    color: #64748b;
}
.insight-box {
    background-color: #eff6ff;
    border-left: 6px solid #2563eb;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 12px;
}
.warning-box {
    background-color: #fff7ed;
    border-left: 6px solid #f97316;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 12px;
}
.success-box {
    background-color: #ecfdf5;
    border-left: 6px solid #059669;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# OOP CLASS
# ============================================================

class HealthAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    def standardize_columns(self):
        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )

        rename_map = {
            "Date_of_Admission": "Admission_Date",
            "Discharge_Date": "Discharge_Date",
            "Medical_Condition": "Disease",
            "Condition": "Disease",
            "Diagnosis": "Disease",
            "Test_Results": "Outcome",
            "Admission_Type": "Patient_Type",
            "Hospital": "Department",
            "Ward": "Department",
            "Patient_Satisfaction": "Satisfaction",
            "Satisfaction_Score": "Satisfaction",
            "Feedback_Rating": "Satisfaction"
        }

        self.df.rename(
            columns={old: new for old, new in rename_map.items() if old in self.df.columns},
            inplace=True
        )

    def clean_data(self):
        self.standardize_columns()

        self.df = self.df.drop_duplicates()

        required_defaults = {
            "Admission_Date": pd.NaT,
            "Disease": "Unknown",
            "Outcome": "Discharged",
            "Department": "General",
            "Age": np.nan,
            "Gender": "Unknown",
            "Length_of_Stay": np.nan,
            "Satisfaction": np.nan,
            "Patient_Type": "Unknown"
        }

        for col, default in required_defaults.items():
            if col not in self.df.columns:
                self.df[col] = default

        self.df["Admission_Date"] = pd.to_datetime(
            self.df["Admission_Date"], errors="coerce"
        )

        self.df["Age"] = pd.to_numeric(self.df["Age"], errors="coerce")
        self.df["Length_of_Stay"] = pd.to_numeric(
            self.df["Length_of_Stay"], errors="coerce"
        )
        self.df["Satisfaction"] = pd.to_numeric(
            self.df["Satisfaction"], errors="coerce"
        )

        self.df["Age"] = self.df["Age"].fillna(self.df["Age"].median())
        self.df["Length_of_Stay"] = self.df["Length_of_Stay"].fillna(
            self.df["Length_of_Stay"].median()
        )
        self.df["Satisfaction"] = self.df["Satisfaction"].fillna(
            self.df["Satisfaction"].mean()
        )

        self.df["Disease"] = self.df["Disease"].fillna("Unknown")
        self.df["Outcome"] = self.df["Outcome"].fillna("Discharged")
        self.df["Department"] = self.df["Department"].fillna("General")
        self.df["Gender"] = self.df["Gender"].fillna("Unknown")
        self.df["Patient_Type"] = self.df["Patient_Type"].fillna("Unknown")

        self.df["Outcome"] = self.df["Outcome"].replace({
            "Discharge": "Discharged",
            "discharge": "Discharged",
            "Discharged": "Discharged",
            "DAMA": "DAMA",
            "dama": "DAMA",
            "Death": "Death",
            "Died": "Death",
            "death": "Death",
            "Recovered": "Discharged",
            "Normal": "Discharged",
            "Abnormal": "DAMA",
            "Inconclusive": "DAMA"
        })

        self.df = self.df[self.df["Age"] >= 0]
        self.df = self.df[self.df["Length_of_Stay"] >= 0]
        self.df = self.df.dropna(subset=["Admission_Date"])

        self.df["Year"] = self.df["Admission_Date"].dt.year
        self.df["Month"] = self.df["Admission_Date"].dt.month
        self.df["Admission_Period"] = self.df["Admission_Date"].dt.to_period("M").astype(str)

        return self.df

    def kpi_summary(self, df):
        return {
            "Total Patients": len(df),
            "Diseases": df["Disease"].nunique(),
            "Departments": df["Department"].nunique(),
            "Avg Age": round(df["Age"].mean(), 1),
            "Avg Stay": round(df["Length_of_Stay"].mean(), 1),
            "Avg Satisfaction": round(df["Satisfaction"].mean(), 1)
        }

    def outcome_summary(self, df):
        return df["Outcome"].value_counts()

    def admissions_over_time(self, df):
        return df.groupby("Admission_Period").size().reset_index(name="Admissions")

    def disease_counts(self, df):
        return df["Disease"].value_counts().head(10)

    def length_of_stay_by_department(self, df):
        return df.groupby("Department")["Length_of_Stay"].mean().sort_values(ascending=False)

    def length_of_stay_by_patient_type(self, df):
        return df.groupby("Patient_Type")["Length_of_Stay"].mean().sort_values(ascending=False)

    def satisfaction_by_department(self, df):
        return df.groupby("Department")["Satisfaction"].mean().sort_values(ascending=False)

    def outcome_by_gender(self, df):
        return df.groupby(["Gender", "Outcome"]).size().reset_index(name="Count")


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="dashboard-title">🏥 Project 6: Public Health Dashboard</div>
<div class="dashboard-subtitle">
Patient and hospital data dashboard for operational, policy, quality, and public health insights.
</div>
""", unsafe_allow_html=True)

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload hospital or public health dataset",
    type=["csv", "xlsx"]
)

if not uploaded_file:
    st.info("Upload your CSV or Excel dataset to begin.")
    st.stop()

if uploaded_file.name.endswith(".csv"):
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = pd.read_excel(uploaded_file)

analyzer = HealthAnalyzer(raw_df)
df = analyzer.clean_data()

if df.empty:
    st.error("No usable records found after cleaning.")
    st.stop()

st.success("Dataset uploaded, cleaned, and prepared successfully.")

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Interactive Filters")

gender_filter = st.sidebar.multiselect(
    "Gender",
    sorted(df["Gender"].unique()),
    default=sorted(df["Gender"].unique())
)

department_filter = st.sidebar.multiselect(
    "Department",
    sorted(df["Department"].unique()),
    default=sorted(df["Department"].unique())
)

disease_filter = st.sidebar.multiselect(
    "Disease / Condition",
    sorted(df["Disease"].unique()),
    default=sorted(df["Disease"].unique())
)

outcome_filter = st.sidebar.multiselect(
    "Outcome",
    sorted(df["Outcome"].unique()),
    default=sorted(df["Outcome"].unique())
)

patient_type_filter = st.sidebar.multiselect(
    "Patient Type",
    sorted(df["Patient_Type"].unique()),
    default=sorted(df["Patient_Type"].unique())
)

age_range = st.sidebar.slider(
    "Age Range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max()))
)

filtered_df = df[
    (df["Gender"].isin(gender_filter)) &
    (df["Department"].isin(department_filter)) &
    (df["Disease"].isin(disease_filter)) &
    (df["Outcome"].isin(outcome_filter)) &
    (df["Patient_Type"].isin(patient_type_filter)) &
    (df["Age"] >= age_range[0]) &
    (df["Age"] <= age_range[1])
]

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# ============================================================
# KPI SUMMARY
# ============================================================

st.subheader("Executive KPI Summary")

kpis = analyzer.kpi_summary(filtered_df)
cols = st.columns(6)

for col, key in zip(cols, kpis.keys()):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{kpis[key]}</div>
            <div class="metric-label">{key}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Preparation",
    "Patient Outcomes",
    "Admissions Trend",
    "Disease Burden",
    "Department Analysis",
    "Downloads & Insights"
])

# ============================================================
# TAB 1: DATA PREPARATION
# ============================================================

with tab1:
    st.header("Step 1: Data Preparation and Cleaning")

    st.markdown("""
    This dashboard performs the following cleaning steps:

    - Loads CSV or Excel files using Pandas.
    - Removes duplicate records.
    - Standardizes column names.
    - Converts admission dates to proper date format.
    - Handles missing numeric values using median or mean.
    - Handles missing categorical values using default labels.
    - Standardizes patient outcomes such as Discharged, DAMA, and Death.
    - Creates year, month, and admission period fields for trend analysis.
    """)

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(filtered_df.head(30), use_container_width=True)

    st.subheader("Numeric Summary")
    st.dataframe(
        filtered_df[["Age", "Length_of_Stay", "Satisfaction"]].describe(),
        use_container_width=True
    )

    st.subheader("Categorical Summary")
    cat_summary = pd.DataFrame({
        "Column": ["Disease", "Outcome", "Department", "Gender", "Patient_Type"],
        "Unique Values": [
            filtered_df["Disease"].nunique(),
            filtered_df["Outcome"].nunique(),
            filtered_df["Department"].nunique(),
            filtered_df["Gender"].nunique(),
            filtered_df["Patient_Type"].nunique()
        ],
        "Most Common Value": [
            filtered_df["Disease"].mode()[0],
            filtered_df["Outcome"].mode()[0],
            filtered_df["Department"].mode()[0],
            filtered_df["Gender"].mode()[0],
            filtered_df["Patient_Type"].mode()[0]
        ]
    })

    st.dataframe(cat_summary, use_container_width=True)

# ============================================================
# TAB 2: PATIENT OUTCOMES
# Requirement: distribution of outcomes by age or gender
# Chart: Histogram patient outcomes by age
# ============================================================

with tab2:
    st.header("Patient Outcomes Analysis")

    st.subheader("Outcome Distribution")

    outcome_counts = analyzer.outcome_summary(filtered_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=outcome_counts.index, y=outcome_counts.values, ax=ax)
    ax.set_title("Patient Outcome Distribution")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Number of Patients")
    st.pyplot(fig)

    st.subheader("Required Chart: Histogram of Patient Outcomes by Age")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=filtered_df,
        x="Age",
        hue="Outcome",
        multiple="stack",
        bins=20,
        ax=ax
    )
    ax.set_title("Patient Outcomes by Age")
    ax.set_xlabel("Patient Age")
    ax.set_ylabel("Number of Patients")
    st.pyplot(fig)

    st.subheader("Patient Outcomes by Gender")

    outcome_gender = analyzer.outcome_by_gender(filtered_df)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=outcome_gender,
        x="Gender",
        y="Count",
        hue="Outcome",
        ax=ax
    )
    ax.set_title("Patient Outcomes by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of Patients")
    st.pyplot(fig)

# ============================================================
# TAB 3: ADMISSIONS TREND
# Requirement: admissions per month or year
# Chart: Line chart admissions over time
# ============================================================

with tab3:
    st.header("Admissions Over Time")

    admissions = analyzer.admissions_over_time(filtered_df)

    st.subheader("Required Chart: Line Chart of Admissions Over Time")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(
        data=admissions,
        x="Admission_Period",
        y="Admissions",
        marker="o",
        ax=ax
    )
    ax.set_title("Monthly Admissions Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Admissions")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.dataframe(admissions, use_container_width=True)

# ============================================================
# TAB 4: DISEASE BURDEN
# Requirement: most common diseases or conditions
# ============================================================

with tab4:
    st.header("Disease Burden Analysis")

    disease_counts = analyzer.disease_counts(filtered_df)

    st.subheader("Top 10 Most Common Diseases or Conditions")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=disease_counts.values,
        y=disease_counts.index,
        ax=ax
    )
    ax.set_title("Top 10 Most Common Diseases or Conditions")
    ax.set_xlabel("Number of Cases")
    ax.set_ylabel("Disease")
    st.pyplot(fig)

    st.dataframe(
        disease_counts.reset_index().rename(
            columns={"index": "Disease", "Disease": "Cases"}
        ),
        use_container_width=True
    )

# ============================================================
# TAB 5: DEPARTMENT ANALYSIS
# Requirements:
# - Length of stay by department or patient type
# - Satisfaction by department
# - Bar chart average service satisfaction by department
# ============================================================

with tab5:
    st.header("Department and Patient Type Analysis")

    st.subheader("Length of Stay by Department")

    los_department = analyzer.length_of_stay_by_department(filtered_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=los_department.index,
        y=los_department.values,
        ax=ax
    )
    ax.set_title("Average Length of Stay by Department")
    ax.set_xlabel("Department")
    ax.set_ylabel("Average Length of Stay")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Length of Stay by Patient Type")

    los_patient_type = analyzer.length_of_stay_by_patient_type(filtered_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=los_patient_type.index,
        y=los_patient_type.values,
        ax=ax
    )
    ax.set_title("Average Length of Stay by Patient Type")
    ax.set_xlabel("Patient Type")
    ax.set_ylabel("Average Length of Stay")
    plt.xticks(rotation=30)
    st.pyplot(fig)

    st.subheader("Required Chart: Average Service Satisfaction by Department")

    satisfaction = analyzer.satisfaction_by_department(filtered_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=satisfaction.index,
        y=satisfaction.values,
        ax=ax
    )
    ax.set_title("Average Service Satisfaction by Department")
    ax.set_xlabel("Department")
    ax.set_ylabel("Average Satisfaction Rating")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ============================================================
# TAB 6: INSIGHTS AND DOWNLOADS
# ============================================================

with tab6:
    st.header("Key Findings and Actionable Insights")

    top_disease = filtered_df["Disease"].mode()[0]
    top_outcome = filtered_df["Outcome"].mode()[0]
    highest_los_dept = analyzer.length_of_stay_by_department(filtered_df).index[0]
    lowest_satisfaction_dept = analyzer.satisfaction_by_department(filtered_df).index[-1]
    highest_admission_month = analyzer.admissions_over_time(filtered_df).sort_values(
        by="Admissions",
        ascending=False
    ).iloc[0]["Admission_Period"]

    st.markdown(f"""
    <div class="insight-box">
    <b>1. Patient Outcomes:</b> The most frequent patient outcome is 
    <b>{top_outcome}</b>. This helps summarize the general hospital outcome pattern.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
    <b>2. Disease Burden:</b> The most common disease or condition is 
    <b>{top_disease}</b>. Hospital leaders can use this to prioritize preventive care and resource planning.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="warning-box">
    <b>3. Length of Stay:</b> The department with the highest average length of stay is 
    <b>{highest_los_dept}</b>. This may indicate higher case complexity or operational delays.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="success-box">
    <b>4. Satisfaction:</b> The department with the lowest average satisfaction rating is 
    <b>{lowest_satisfaction_dept}</b>. This area may need service quality improvement.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
    <b>5. Admissions Trend:</b> The highest number of admissions occurred in 
    <b>{highest_admission_month}</b>. This can support staffing and budget planning.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Download Cleaned and Filtered Dataset")

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned Dataset CSV",
        data=csv,
        file_name="project6_cleaned_public_health_data.csv",
        mime="text/csv"
    )

    summary_report = pd.DataFrame({
        "Metric": list(kpis.keys()),
        "Value": list(kpis.values())
    })

    summary_csv = summary_report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download KPI Summary Report",
        data=summary_csv,
        file_name="project6_kpi_summary_report.csv",
        mime="text/csv"
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Project 6: Public Health Patient & Hospital Data Dashboard | Python, OOP, Streamlit, Pandas, Matplotlib, and Seaborn"
)
