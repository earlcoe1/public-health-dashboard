import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Project 6: Public Health Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# HEALTH ANALYZER CLASS
# ============================================================

class HealthAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    # Requirement 1: Cleaning patient records
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
            "Admission_Type": "Patient_Type",
            "Test_Results": "Outcome",
            "Patient_Satisfaction": "Satisfaction",
            "Satisfaction_Score": "Satisfaction",
            "Hospital": "Department",
            "Ward": "Department"
        }

        self.df.rename(
            columns={old: new for old, new in rename_map.items() if old in self.df.columns},
            inplace=True
        )

        required_columns = [
            "Admission_Date",
            "Age",
            "Gender",
            "Department",
            "Disease",
            "Outcome",
            "Satisfaction"
        ]

        for col in required_columns:
            if col not in self.df.columns:
                if col == "Admission_Date":
                    self.df[col] = pd.NaT
                elif col in ["Age", "Satisfaction"]:
                    self.df[col] = np.nan
                elif col == "Outcome":
                    self.df[col] = "Discharged"
                else:
                    self.df[col] = "Unknown"

        self.df = self.df.drop_duplicates()

        self.df["Admission_Date"] = pd.to_datetime(
            self.df["Admission_Date"],
            errors="coerce"
        )

        self.df["Age"] = pd.to_numeric(self.df["Age"], errors="coerce")
        self.df["Satisfaction"] = pd.to_numeric(
            self.df["Satisfaction"],
            errors="coerce"
        )

        self.df["Age"] = self.df["Age"].fillna(self.df["Age"].median())
        self.df["Satisfaction"] = self.df["Satisfaction"].fillna(
            self.df["Satisfaction"].mean()
        )

        self.df["Gender"] = self.df["Gender"].fillna("Unknown")
        self.df["Department"] = self.df["Department"].fillna("Unknown")
        self.df["Disease"] = self.df["Disease"].fillna("Unknown")
        self.df["Outcome"] = self.df["Outcome"].fillna("Discharged")

        self.df["Outcome"] = self.df["Outcome"].replace({
            "Discharge": "Discharged",
            "discharge": "Discharged",
            "Discharged": "Discharged",
            "Recovered": "Discharged",
            "DAMA": "DAMA",
            "dama": "DAMA",
            "Death": "Death",
            "Died": "Death",
            "death": "Death"
        })

        self.df = self.df.dropna(subset=["Admission_Date"])
        self.df = self.df[self.df["Age"] >= 0]

        self.df["Year"] = self.df["Admission_Date"].dt.year
        self.df["Month"] = self.df["Admission_Date"].dt.month
        self.df["Admission_Period"] = self.df["Admission_Date"].dt.to_period("M").astype(str)

        return self.df

    # Requirement 2: Summarizing outcomes
    def summarize_outcomes(self):
        return self.df["Outcome"].value_counts().reset_index().rename(
            columns={
                "index": "Outcome",
                "Outcome": "Count"
            }
        )

    # Requirement 3: Aggregating by age
    def aggregate_by_age(self):
        bins = [0, 18, 35, 50, 65, 100]
        labels = ["0-18", "19-35", "36-50", "51-65", "66+"]

        self.df["Age_Group"] = pd.cut(
            self.df["Age"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        return (
            self.df.groupby(["Age_Group", "Outcome"])
            .size()
            .reset_index(name="Count")
        )

    # Requirement 4: Aggregating by gender
    def aggregate_by_gender(self):
        return (
            self.df.groupby(["Gender", "Outcome"])
            .size()
            .reset_index(name="Count")
        )

    # Requirement 5: Aggregating by department
    def aggregate_by_department(self):
        return (
            self.df.groupby("Department")
            .agg(
                Total_Patients=("Department", "count"),
                Average_Satisfaction=("Satisfaction", "mean")
            )
            .reset_index()
        )

    # Required chart data: Admissions over time
    def admissions_over_time(self):
        return (
            self.df.groupby("Admission_Period")
            .size()
            .reset_index(name="Admissions")
        )

    # Required chart data: Satisfaction by department
    def satisfaction_by_department(self):
        return (
            self.df.groupby("Department")["Satisfaction"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

    # Extra project question: Most common diseases
    def common_diseases(self):
        return (
            self.df["Disease"]
            .value_counts()
            .head(10)
            .reset_index()
            .rename(columns={"index": "Disease", "Disease": "Cases"})
        )


# ============================================================
# DASHBOARD TITLE
# ============================================================

st.title("🏥 Project 6: Public Health Patient & Hospital Data Dashboard")

st.markdown("""
This dashboard analyzes hospital or public health data for operational and policy insights.
It includes patient outcome analysis, admissions trends, disease frequency, and service satisfaction.
""")


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Hospital Patient Records Dataset, CDC/Public Health Dataset, or COVID-19 Health Metrics Dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("Please upload a CSV or Excel dataset to begin.")
    st.stop()

if uploaded_file.name.endswith(".csv"):
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = pd.read_excel(uploaded_file)


# ============================================================
# CLEAN DATA USING OOP
# ============================================================

analyzer = HealthAnalyzer(raw_df)
df = analyzer.clean_patient_records()

if df.empty:
    st.error("No usable records found after cleaning.")
    st.stop()

st.success("Dataset loaded and cleaned successfully.")


# ============================================================
# INTERACTIVE FILTERS
# ============================================================

st.sidebar.header("Interactive Filters")

min_age = int(df["Age"].min())
max_age = int(df["Age"].max())

age_filter = st.sidebar.slider(
    "Filter by Age",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

gender_filter = st.sidebar.multiselect(
    "Filter by Gender",
    options=sorted(df["Gender"].unique()),
    default=sorted(df["Gender"].unique())
)

department_filter = st.sidebar.multiselect(
    "Filter by Department",
    options=sorted(df["Department"].unique()),
    default=sorted(df["Department"].unique())
)

filtered_df = df[
    (df["Age"] >= age_filter[0]) &
    (df["Age"] <= age_filter[1]) &
    (df["Gender"].isin(gender_filter)) &
    (df["Department"].isin(department_filter))
]

if filtered_df.empty:
    st.warning("No records match your selected filters.")
    st.stop()

filtered_analyzer = HealthAnalyzer(filtered_df)
filtered_analyzer.df = filtered_df


# ============================================================
# KPI SUMMARY
# ============================================================

st.subheader("Dashboard Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Patients", len(filtered_df))
col2.metric("Total Departments", filtered_df["Department"].nunique())
col3.metric("Total Diseases", filtered_df["Disease"].nunique())
col4.metric("Average Satisfaction", round(filtered_df["Satisfaction"].mean(), 2))


# ============================================================
# DATA CLEANING OUTPUT
# ============================================================

st.subheader("Data Cleaning and Handling Missing/Categorical Data")

st.markdown("""
The dataset was cleaned by removing duplicates, formatting admission dates, handling missing numeric values,
handling missing categorical values, and standardizing patient outcomes into Discharged, DAMA, and Death.
""")

st.dataframe(filtered_df.head(20), use_container_width=True)


# ============================================================
# OUTCOME SUMMARY
# ============================================================

st.subheader("Summarized Patient Outcomes")

outcome_summary = filtered_analyzer.summarize_outcomes()
st.dataframe(outcome_summary, use_container_width=True)


# ============================================================
# AGGREGATION OUTPUTS
# ============================================================

st.subheader("Grouping and Aggregating Outcomes")

tab_age, tab_gender, tab_department = st.tabs([
    "Aggregate by Age",
    "Aggregate by Gender",
    "Aggregate by Department"
])

with tab_age:
    age_summary = filtered_analyzer.aggregate_by_age()
    st.dataframe(age_summary, use_container_width=True)

with tab_gender:
    gender_summary = filtered_analyzer.aggregate_by_gender()
    st.dataframe(gender_summary, use_container_width=True)

with tab_department:
    department_summary = filtered_analyzer.aggregate_by_department()
    st.dataframe(department_summary, use_container_width=True)


# ============================================================
# REQUIRED CHART 1:
# HISTOGRAM - PATIENT OUTCOMES BY AGE
# ============================================================

st.header("Required Chart 1: Histogram — Patient Outcomes by Age")

fig1, ax1 = plt.subplots(figsize=(10, 5))

for outcome in filtered_df["Outcome"].unique():
    subset = filtered_df[filtered_df["Outcome"] == outcome]
    ax1.hist(
        subset["Age"],
        bins=15,
        alpha=0.6,
        label=outcome
    )

ax1.set_title("Patient Outcomes by Age")
ax1.set_xlabel("Age")
ax1.set_ylabel("Number of Patients")
ax1.legend()

st.pyplot(fig1)


# ============================================================
# REQUIRED CHART 2:
# LINE CHART - ADMISSIONS OVER TIME
# ============================================================

st.header("Required Chart 2: Line Chart — Admissions Over Time")

admissions = filtered_analyzer.admissions_over_time()

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(
    admissions["Admission_Period"],
    admissions["Admissions"],
    marker="o"
)

ax2.set_title("Admissions Over Time")
ax2.set_xlabel("Admission Month")
ax2.set_ylabel("Number of Admissions")
plt.xticks(rotation=45)

st.pyplot(fig2)


# ============================================================
# REQUIRED CHART 3:
# BAR CHART - AVERAGE SERVICE SATISFACTION BY DEPARTMENT
# ============================================================

st.header("Required Chart 3: Bar Chart — Average Service Satisfaction by Department")

satisfaction = filtered_analyzer.satisfaction_by_department()

fig3, ax3 = plt.subplots(figsize=(10, 5))

ax3.bar(
    satisfaction["Department"],
    satisfaction["Satisfaction"]
)

ax3.set_title("Average Service Satisfaction by Department")
ax3.set_xlabel("Department")
ax3.set_ylabel("Average Satisfaction Rating")
plt.xticks(rotation=45)

st.pyplot(fig3)


# ============================================================
# EXTRA PROJECT QUESTION:
# MOST COMMON DISEASES
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


# ============================================================
# BASIC PUBLIC HEALTH INSIGHTS
# ============================================================

st.subheader("Analysis and Insights")

top_outcome = filtered_df["Outcome"].mode()[0]
top_disease = filtered_df["Disease"].mode()[0]
highest_satisfaction_department = satisfaction.iloc[0]["Department"]

st.markdown(f"""
- The most common patient outcome is **{top_outcome}**.
- The most common disease or condition is **{top_disease}**.
- The department with the highest average service satisfaction is **{highest_satisfaction_department}**.
- Admissions over time can help hospital administrators plan staffing, beds, and resources.
- Patient outcome patterns by age and gender can support public health planning and policy decisions.
""")


# ============================================================
# DOWNLOAD CLEANED DATA
# ============================================================

st.subheader("Download Output")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Cleaned Filtered Dataset",
    data=csv,
    file_name="project6_cleaned_public_health_data.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Project 6 Public Health Dashboard | Streamlit, Pandas, Matplotlib, OOP, Data Cleaning, Grouping, Aggregation, and Visualization"
)
