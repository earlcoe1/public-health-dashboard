import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Hospital & Public Health Insights Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 1rem;
}

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
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    text-align: center;
    border-left: 6px solid #2563eb;
}

.metric-value {
    font-size: 30px;
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
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 15px;
}

.warning-box {
    background-color: #fff7ed;
    border-left: 6px solid #f97316;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 15px;
}

.success-box {
    background-color: #ecfdf5;
    border-left: 6px solid #059669;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 15px;
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
            "Admission_Type": "Patient_Type",
            "Test_Results": "Outcome",
            "Hospital": "Department",
            "Patient_Satisfaction": "Satisfaction",
            "Satisfaction_Score": "Satisfaction"
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
        self.df["Length_of_Stay"] = pd.to_numeric(self.df["Length_of_Stay"], errors="coerce")
        self.df["Satisfaction"] = pd.to_numeric(self.df["Satisfaction"], errors="coerce")

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
            "death": "Death"
        })

        self.df = self.df[self.df["Length_of_Stay"] >= 0]
        self.df = self.df[self.df["Age"] >= 0]

        self.df["Year"] = self.df["Admission_Date"].dt.year
        self.df["Month"] = self.df["Admission_Date"].dt.month
        self.df["Month_Name"] = self.df["Admission_Date"].dt.strftime("%b")
        self.df["Admission_Period"] = self.df["Admission_Date"].dt.to_period("M").astype(str)

        self.df = self.df.dropna(subset=["Admission_Date"])

        return self.df

    def kpi_summary(self, df):
        return {
            "Total Patients": len(df),
            "Diseases": df["Disease"].nunique(),
            "Departments": df["Department"].nunique(),
            "Avg. Age": round(df["Age"].mean(), 1),
            "Avg. Stay": round(df["Length_of_Stay"].mean(), 1),
            "Avg. Satisfaction": round(df["Satisfaction"].mean(), 1)
        }

    def admissions_over_time(self, df):
        return df.groupby("Admission_Period").size().reset_index(name="Admissions")

    def disease_counts(self, df):
        return df["Disease"].value_counts().head(10)

    def length_of_stay_by_department(self, df):
        return df.groupby("Department")["Length_of_Stay"].mean().sort_values(ascending=False)

    def satisfaction_by_department(self, df):
        return df.groupby("Department")["Satisfaction"].mean().sort_values(ascending=False)

    def outcome_summary(self, df):
        return df["Outcome"].value_counts()

    def outcome_by_gender(self, df):
        return df.groupby(["Gender", "Outcome"]).size().reset_index(name="Count")

    def patient_type_summary(self, df):
        return df["Patient_Type"].value_counts()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="dashboard-title">🏥 Hospital & Public Health Insights Dashboard</div>
<div class="dashboard-subtitle">
Operational, population health, patient outcome, quality, and policy insights for healthcare decision-making.
</div>
""", unsafe_allow_html=True)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your public health dataset",
    type=["csv", "xlsx"]
)

if not uploaded_file:
    st.info("Upload your CSV or Excel file to begin dashboard analysis.")
    st.stop()

if uploaded_file.name.endswith(".csv"):
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = pd.read_excel(uploaded_file)


# ============================================================
# CLEAN DATA
# ============================================================

analyzer = HealthAnalyzer(raw_df)
df = analyzer.clean_data()

if df.empty:
    st.error("The uploaded dataset has no usable records after cleaning.")
    st.stop()

st.success("Dataset uploaded and cleaned successfully.")


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Interactive Filters")

gender_options = sorted(df["Gender"].dropna().unique().tolist())
department_options = sorted(df["Department"].dropna().unique().tolist())
disease_options = sorted(df["Disease"].dropna().unique().tolist())
outcome_options = sorted(df["Outcome"].dropna().unique().tolist())
patient_type_options = sorted(df["Patient_Type"].dropna().unique().tolist())

selected_gender = st.sidebar.multiselect(
    "Gender",
    gender_options,
    default=gender_options
)

selected_department = st.sidebar.multiselect(
    "Department",
    department_options,
    default=department_options
)

selected_disease = st.sidebar.multiselect(
    "Disease / Condition",
    disease_options,
    default=disease_options
)

selected_outcome = st.sidebar.multiselect(
    "Patient Outcome",
    outcome_options,
    default=outcome_options
)

selected_patient_type = st.sidebar.multiselect(
    "Patient Type",
    patient_type_options,
    default=patient_type_options
)

min_age = int(df["Age"].min())
max_age = int(df["Age"].max())

age_range = st.sidebar.slider(
    "Age Range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

filtered_df = df[
    (df["Gender"].isin(selected_gender)) &
    (df["Department"].isin(selected_department)) &
    (df["Disease"].isin(selected_disease)) &
    (df["Outcome"].isin(selected_outcome)) &
    (df["Patient_Type"].isin(selected_patient_type)) &
    (df["Age"] >= age_range[0]) &
    (df["Age"] <= age_range[1])
]

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("Executive KPI Summary")

kpis = analyzer.kpi_summary(filtered_df)

k1, k2, k3, k4, k5, k6 = st.columns(6)

for col, label in zip(
    [k1, k2, k3, k4, k5, k6],
    kpis.keys()
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{kpis[label]}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Outcomes",
    "Admissions",
    "Departments",
    "Data & Downloads"
])


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(20), use_container_width=True)

    st.subheader("Numeric Summary")
    st.dataframe(
        filtered_df[["Age", "Length_of_Stay", "Satisfaction"]].describe(),
        use_container_width=True
    )

    st.subheader("Categorical Summary")

    categorical_summary = pd.DataFrame({
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

    st.dataframe(categorical_summary, use_container_width=True)


# ============================================================
# TAB 2: OUTCOMES
# ============================================================

with tab2:
    st.subheader("Patient Outcomes Distribution")

    outcome_counts = analyzer.outcome_summary(filtered_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    outcome_counts.plot(kind="bar", ax=ax)
    ax.set_title("Patient Outcomes Distribution")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Number of Patients")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    st.subheader("Patient Outcomes Distribution by Age")

    fig, ax = plt.subplots(figsize=(9, 5))

    for outcome in filtered_df["Outcome"].unique():
        subset = filtered_df[filtered_df["Outcome"] == outcome]
        ax.hist(subset["Age"], alpha=0.5, label=outcome)

    ax.set_title("Patient Outcomes by Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Patients")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Patient Outcomes by Gender")

    gender_outcome = analyzer.outcome_by_gender(filtered_df)

    pivot_gender = gender_outcome.pivot(
        index="Gender",
        columns="Outcome",
        values="Count"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot_gender.plot(kind="bar", ax=ax)
    ax.set_title("Patient Outcomes by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of Patients")
    plt.xticks(rotation=0)
    st.pyplot(fig)


# ============================================================
# TAB 3: ADMISSIONS
# ============================================================

with tab3:
    st.subheader("Admissions Over Time")

    admissions = analyzer.admissions_over_time(filtered_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(admissions["Admission_Period"], admissions["Admissions"], marker="o")
    ax.set_title("Monthly Admissions Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Admissions")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Most Common Diseases or Conditions")

    disease_counts = analyzer.disease_counts(filtered_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    disease_counts.plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Top 10 Most Common Diseases")
    ax.set_xlabel("Number of Cases")
    ax.set_ylabel("Disease")
    st.pyplot(fig)


# ============================================================
# TAB 4: DEPARTMENTS
# ============================================================

with tab4:
    st.subheader("Average Length of Stay by Department")

    los = analyzer.length_of_stay_by_department(filtered_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    los.plot(kind="bar", ax=ax)
    ax.set_title("Average Length of Stay by Department")
    ax.set_xlabel("Department")
    ax.set_ylabel("Average Days")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Average Service Satisfaction by Department")

    satisfaction = analyzer.satisfaction_by_department(filtered_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    satisfaction.plot(kind="bar", ax=ax)
    ax.set_title("Average Satisfaction by Department")
    ax.set_xlabel("Department")
    ax.set_ylabel("Average Satisfaction Score")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Patient Type Distribution")

    patient_type_counts = analyzer.patient_type_summary(filtered_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    patient_type_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    ax.set_title("Patient Type Distribution")
    st.pyplot(fig)


# ============================================================
# TAB 5: DATA & DOWNLOADS
# ============================================================

with tab5:
    st.subheader("Cleaned and Filtered Dataset")
    st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Dataset as CSV",
        data=csv,
        file_name="project6_filtered_public_health_data.csv",
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
# AUTOMATED INSIGHTS
# ============================================================

st.subheader("Key Findings and Policy Recommendations")

top_disease = filtered_df["Disease"].mode()[0]
top_department_los = analyzer.length_of_stay_by_department(filtered_df).index[0]
lowest_satisfaction_department = analyzer.satisfaction_by_department(filtered_df).index[-1]
highest_admission_month = analyzer.admissions_over_time(filtered_df).sort_values(
    by="Admissions",
    ascending=False
).iloc[0]["Admission_Period"]

st.markdown(f"""
<div class="insight-box">
<b>1. Disease Burden:</b> The most common disease or condition is <b>{top_disease}</b>. 
This suggests that preventive care, education, and clinical resources should focus on this condition.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="warning-box">
<b>2. Operational Efficiency:</b> The department with the highest average length of stay is 
<b>{top_department_los}</b>. This may require review of discharge planning, staffing, or case complexity.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="success-box">
<b>3. Service Quality:</b> The department with the lowest average satisfaction score is 
<b>{lowest_satisfaction_department}</b>. Hospital leadership may need to investigate patient feedback in this area.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="insight-box">
<b>4. Admission Trend:</b> The highest number of admissions occurred in 
<b>{highest_admission_month}</b>. This can help administrators plan staffing and resource allocation.
</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Project 6: Public Health Patient & Hospital Data Dashboard | Built with Python, OOP, Streamlit, Pandas, and Matplotlib"
)
