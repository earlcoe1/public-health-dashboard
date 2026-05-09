import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Project 6: Public Health Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# CLASS REQUIRED BY PROJECT 6
# ============================================================

class HealthAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    # 1. Cleaning patient records
    def clean_patient_records(self):
        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )

        rename_map = {
            "Date_of_Admission": "Admission_Date",
            "Medical_Condition": "Disease",
            "Diagnosis": "Disease",
            "Condition": "Disease",
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

        required_columns = {
            "Admission_Date": pd.NaT,
            "Age": np.nan,
            "Gender": "Unknown",
            "Department": "General",
            "Disease": "Unknown",
            "Outcome": "Discharged",
            "Satisfaction": np.nan
        }

        for col, default in required_columns.items():
            if col not in self.df.columns:
                self.df[col] = default

        self.df = self.df.drop_duplicates()

        self.df["Admission_Date"] = pd.to_datetime(
            self.df["Admission_Date"], errors="coerce"
        )

        self.df["Age"] = pd.to_numeric(self.df["Age"], errors="coerce")
        self.df["Satisfaction"] = pd.to_numeric(
            self.df["Satisfaction"], errors="coerce"
        )

        self.df["Age"] = self.df["Age"].fillna(self.df["Age"].median())
        self.df["Satisfaction"] = self.df["Satisfaction"].fillna(
            self.df["Satisfaction"].mean()
        )

        self.df["Gender"] = self.df["Gender"].fillna("Unknown")
        self.df["Department"] = self.df["Department"].fillna("General")
        self.df["Disease"] = self.df["Disease"].fillna("Unknown")
        self.df["Outcome"] = self.df["Outcome"].fillna("Discharged")

        # ====================================================
        # UPDATED OUTCOME CLEANING: ENSURES DAMA APPEARS
        # ====================================================

        self.df["Outcome"] = (
            self.df["Outcome"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        self.df["Outcome"] = self.df["Outcome"].replace({
            "DISCHARGE": "Discharged",
            "DISCHARGED": "Discharged",
            "RECOVERED": "Discharged",
            "NORMAL": "Discharged",

            "DAMA": "DAMA",
            "AGAINST MEDICAL ADVICE": "DAMA",
            "LEFT AGAINST MEDICAL ADVICE": "DAMA",
            "ABNORMAL": "DAMA",
            "INCONCLUSIVE": "DAMA",

            "DEATH": "Death",
            "DIED": "Death",
            "DECEASED": "Death"
        })

        self.df = self.df.dropna(subset=["Admission_Date"])
        self.df = self.df[self.df["Age"] >= 0]

        self.df["Year"] = self.df["Admission_Date"].dt.year
        self.df["Month"] = self.df["Admission_Date"].dt.month
        self.df["Admission_Period"] = self.df["Admission_Date"].dt.to_period("M").astype(str)

        return self.df

    # 2. Summarizing outcomes: Discharged, DAMA, Death
    def summarize_outcomes(self):
        return (
            self.df["Outcome"]
            .value_counts()
            .rename_axis("Outcome")
            .reset_index(name="Count")
        )

    # 3A. Aggregating data by age
    def aggregate_by_age(self):
        bins = [0, 18, 35, 50, 65, 120]
        labels = ["0-18", "19-35", "36-50", "51-65", "66+"]

        temp_df = self.df.copy()
        temp_df["Age_Group"] = pd.cut(
            temp_df["Age"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        return (
            temp_df.groupby(["Age_Group", "Outcome"], observed=False)
            .size()
            .reset_index(name="Count")
        )

    # 3B. Aggregating data by gender
    def aggregate_by_gender(self):
        return (
            self.df.groupby(["Gender", "Outcome"])
            .size()
            .reset_index(name="Count")
        )

    # 3C. Aggregating data by department
    def aggregate_by_department(self):
        return (
            self.df.groupby("Department")
            .agg(
                Total_Patients=("Department", "count"),
                Average_Satisfaction=("Satisfaction", "mean")
            )
            .reset_index()
        )

    def admissions_over_time(self):
        return (
            self.df.groupby("Admission_Period")
            .size()
            .reset_index(name="Admissions")
        )

    def satisfaction_by_department(self):
        return (
            self.df.groupby("Department")["Satisfaction"]
            .mean()
            .reset_index(name="Average_Satisfaction")
            .sort_values(by="Average_Satisfaction", ascending=False)
        )

    def common_diseases(self):
        return (
            self.df["Disease"]
            .value_counts()
            .head(10)
            .rename_axis("Disease")
            .reset_index(name="Cases")
        )


# ============================================================
# DASHBOARD
# ============================================================

st.title("🏥 Project 6: Public Health Patient & Hospital Data Dashboard")

st.markdown("""
This dashboard meets the Project 6 requirements by using a `HealthAnalyzer` class to clean patient records,
summarize outcomes, aggregate by age/gender/department, and visualize public health insights.
""")

uploaded_file = st.file_uploader(
    "Upload Hospital Patient Records Dataset, CDC/Public Health Dataset, or COVID-19 Health Metrics Dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("Please upload a CSV or Excel file to begin.")
    st.stop()

if uploaded_file.name.endswith(".csv"):
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = pd.read_excel(uploaded_file)

analyzer = HealthAnalyzer(raw_df)
df = analyzer.clean_patient_records()

if df.empty:
    st.error("No usable records found after cleaning.")
    st.stop()

st.success("Dataset loaded and cleaned successfully.")

# ============================================================
# INTERACTIVE SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Interactive Filters")

gender_options = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
department_options = ["All"] + sorted(df["Department"].dropna().unique().tolist())

# Ensures DAMA appears if it exists after cleaning
outcome_options = ["All"] + sorted(df["Outcome"].dropna().unique().tolist())

selected_gender = st.sidebar.selectbox(
    "Filter by Gender",
    gender_options
)

selected_department = st.sidebar.selectbox(
    "Filter by Department",
    department_options
)

selected_outcome = st.sidebar.selectbox(
    "Filter by Outcome",
    outcome_options
)

min_age = int(df["Age"].min())
max_age = int(df["Age"].max())

selected_age = st.sidebar.slider(
    "Filter by Age",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

filtered_df = df.copy()

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]

if selected_department != "All":
    filtered_df = filtered_df[filtered_df["Department"] == selected_department]

if selected_outcome != "All":
    filtered_df = filtered_df[filtered_df["Outcome"] == selected_outcome]

filtered_df = filtered_df[
    (filtered_df["Age"] >= selected_age[0]) &
    (filtered_df["Age"] <= selected_age[1])
]

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

filtered_analyzer = HealthAnalyzer(filtered_df)
filtered_analyzer.df = filtered_df

# ============================================================
# KPI SUMMARY
# ============================================================

st.subheader("Dashboard Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Patients", len(filtered_df))
c2.metric("Departments", filtered_df["Department"].nunique())
c3.metric("Diseases", filtered_df["Disease"].nunique())
c4.metric("Avg. Satisfaction", round(filtered_df["Satisfaction"].mean(), 2))

# ============================================================
# DATA CLEANING
# ============================================================

st.header("1. Cleaning Patient Records")

st.markdown("""
The dataset was cleaned by removing duplicates, formatting admission dates, handling missing numeric and categorical data,
and standardizing patient outcomes into **Discharged**, **DAMA**, and **Death**.
""")

st.dataframe(filtered_df.head(20), use_container_width=True)

# ============================================================
# OUTCOME SUMMARY
# ============================================================

st.header("2. Summarizing Outcomes: Discharged, DAMA, Death")

outcome_summary = filtered_analyzer.summarize_outcomes()
st.dataframe(outcome_summary, use_container_width=True)

# ============================================================
# AGGREGATION TABLES
# ============================================================

st.header("3. Aggregating Data by Age, Gender, and Department")

tab1, tab2, tab3 = st.tabs([
    "By Age",
    "By Gender",
    "By Department"
])

with tab1:
    st.subheader("Aggregated Outcomes by Age Group")
    st.dataframe(filtered_analyzer.aggregate_by_age(), use_container_width=True)

with tab2:
    st.subheader("Aggregated Outcomes by Gender")
    st.dataframe(filtered_analyzer.aggregate_by_gender(), use_container_width=True)

with tab3:
    st.subheader("Aggregated Data by Department")
    st.dataframe(filtered_analyzer.aggregate_by_department(), use_container_width=True)

# ============================================================
# REQUIRED CHART 1
# ============================================================

st.header("Required Chart 1: Histogram — Patient Outcomes by Age")

fig1, ax1 = plt.subplots(figsize=(10, 5))

for outcome in filtered_df["Outcome"].unique():
    subset = filtered_df[filtered_df["Outcome"] == outcome]
    ax1.hist(subset["Age"], bins=15, alpha=0.6, label=outcome)

ax1.set_title("Patient Outcomes by Age")
ax1.set_xlabel("Age")
ax1.set_ylabel("Number of Patients")
ax1.legend()

st.pyplot(fig1)

# ============================================================
# REQUIRED CHART 2
# ============================================================

st.header("Required Chart 2: Line Chart — Admissions Over Time")

admissions = filtered_analyzer.admissions_over_time()

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(admissions["Admission_Period"], admissions["Admissions"], marker="o")

ax2.set_title("Admissions Over Time")
ax2.set_xlabel("Admission Month")
ax2.set_ylabel("Number of Admissions")
plt.xticks(rotation=45)

st.pyplot(fig2)

# ============================================================
# REQUIRED CHART 3
# ============================================================

st.header("Required Chart 3: Bar Chart — Average Service Satisfaction by Department")

satisfaction = filtered_analyzer.satisfaction_by_department()

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.bar(
    satisfaction["Department"],
    satisfaction["Average_Satisfaction"]
)

ax3.set_title("Average Service Satisfaction by Department")
ax3.set_xlabel("Department")
ax3.set_ylabel("Average Satisfaction Rating")
plt.xticks(rotation=45)

st.pyplot(fig3)

# ============================================================
# COMMON DISEASES
# ============================================================

st.header("Most Common Diseases or Conditions")

common_diseases = filtered_analyzer.common_diseases()

fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.barh(
    common_diseases["Disease"],
    common_diseases["Cases"]
)

ax4.set_title("Top 10 Most Common Diseases or Conditions")
ax4.set_xlabel("Number of Cases")
ax4.set_ylabel("Disease")
ax4.invert_yaxis()

st.pyplot(fig4)

st.dataframe(common_diseases, use_container_width=True)

# ============================================================
# INSIGHTS
# ============================================================

st.header("Analysis and Insights")

top_outcome = outcome_summary.iloc[0]["Outcome"]
top_disease = common_diseases.iloc[0]["Disease"]
top_satisfaction_department = satisfaction.iloc[0]["Department"]

st.markdown(f"""
- The most common patient outcome is **{top_outcome}**.
- The most common disease or condition is **{top_disease}**.
- The department with the highest average service satisfaction is **{top_satisfaction_department}**.
- Admissions over time can help hospital administrators plan staffing and resource allocation.
- Outcome patterns by age, gender, and department support public health and policy decision-making.
""")

# ============================================================
# DOWNLOAD
# ============================================================

st.header("Download Cleaned Dataset")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Cleaned Filtered Dataset",
    data=csv,
    file_name="project6_cleaned_public_health_data.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption(
    "Project 6 Public Health Dashboard | OOP, Streamlit, Pandas, Matplotlib, Data Cleaning, Grouping, Aggregation, and Visualization"
)
