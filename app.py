import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="AMTH Public Health Dashboard",
    page_icon="🏥",
    layout="wide"
)

class HealthAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    def clean_patient_records(self):
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
            "Discharge_Date": pd.NaT,
            "Age": np.nan,
            "Gender": "Unknown",
            "Department": "General",
            "Disease": "Unknown",
            "Outcome": "Discharged",
            "Satisfaction": np.nan,
            "Patient_Type": "Unknown",
            "Length_of_Stay": np.nan
        }

        for col, default in required_columns.items():
            if col not in self.df.columns:
                self.df[col] = default

        self.df = self.df.drop_duplicates()

        self.df["Admission_Date"] = pd.to_datetime(self.df["Admission_Date"], errors="coerce")
        self.df["Discharge_Date"] = pd.to_datetime(self.df["Discharge_Date"], errors="coerce")

        self.df["Age"] = pd.to_numeric(self.df["Age"], errors="coerce")
        self.df["Satisfaction"] = pd.to_numeric(self.df["Satisfaction"], errors="coerce")
        self.df["Length_of_Stay"] = pd.to_numeric(self.df["Length_of_Stay"], errors="coerce")

        if self.df["Length_of_Stay"].isna().all() and "Discharge_Date" in self.df.columns:
            self.df["Length_of_Stay"] = (
                self.df["Discharge_Date"] - self.df["Admission_Date"]
            ).dt.days

        self.df["Age"] = self.df["Age"].fillna(self.df["Age"].median())
        self.df["Satisfaction"] = self.df["Satisfaction"].fillna(self.df["Satisfaction"].mean())
        self.df["Length_of_Stay"] = self.df["Length_of_Stay"].fillna(
            self.df["Length_of_Stay"].median()
        )

        self.df["Gender"] = self.df["Gender"].fillna("Unknown")
        self.df["Department"] = self.df["Department"].fillna("General")
        self.df["Disease"] = self.df["Disease"].fillna("Unknown")
        self.df["Outcome"] = self.df["Outcome"].fillna("Discharged")
        self.df["Patient_Type"] = self.df["Patient_Type"].fillna("Unknown")

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
        self.df = self.df[self.df["Length_of_Stay"] >= 0]

        self.df["Year"] = self.df["Admission_Date"].dt.year
        self.df["Month"] = self.df["Admission_Date"].dt.month
        self.df["Admission_Period"] = self.df["Admission_Date"].dt.to_period("M").astype(str)

        return self.df

    def summarize_outcomes(self):
        required_outcomes = ["Discharged", "DAMA", "Death"]

        summary = (
            self.df["Outcome"]
            .value_counts()
            .reindex(required_outcomes, fill_value=0)
            .rename_axis("Outcome")
            .reset_index(name="Count")
        )

        return summary

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

    def aggregate_by_gender(self):
        return (
            self.df.groupby(["Gender", "Outcome"])
            .size()
            .reset_index(name="Count")
        )

    def aggregate_by_department(self):
        return (
            self.df.groupby("Department")
            .agg(
                Total_Patients=("Department", "count"),
                Average_Satisfaction=("Satisfaction", "mean"),
                Average_Length_of_Stay=("Length_of_Stay", "mean")
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

    def length_of_stay_by_department(self):
        return (
            self.df.groupby("Department")["Length_of_Stay"]
            .mean()
            .reset_index(name="Average_Length_of_Stay")
            .sort_values(by="Average_Length_of_Stay", ascending=False)
        )

    def length_of_stay_by_patient_type(self):
        return (
            self.df.groupby("Patient_Type")["Length_of_Stay"]
            .mean()
            .reset_index(name="Average_Length_of_Stay")
            .sort_values(by="Average_Length_of_Stay", ascending=False)
        )

    def common_diseases(self):
        return (
            self.df["Disease"]
            .value_counts()
            .head(10)
            .rename_axis("Disease")
            .reset_index(name="Cases")
        )


st.title("🏥 AMTH Public Health Patient & Hospital Data Dashboard")

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

st.sidebar.header("Interactive Filters")

gender_options = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
department_options = ["All"] + sorted(df["Department"].dropna().unique().tolist())

outcome_options = ["All", "DAMA", "Death", "Discharged"]

selected_gender = st.sidebar.selectbox("Filter by Gender", gender_options)
selected_department = st.sidebar.selectbox("Filter by Department", department_options)
selected_outcome = st.sidebar.selectbox("Filter by Outcome", outcome_options)

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

st.subheader("Dashboard Summary")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Patients", len(filtered_df))
c2.metric("Departments", filtered_df["Department"].nunique())
c3.metric("Diseases", filtered_df["Disease"].nunique())
c4.metric("Avg. Satisfaction", round(filtered_df["Satisfaction"].mean(), 2))
c5.metric("Avg. Length of Stay", round(filtered_df["Length_of_Stay"].mean(), 2))

st.header("Patient Records")

st.markdown("""
The dataset was cleaned by removing duplicates, formatting admission dates, handling missing numeric and categorical data,
calculating or cleaning length of stay, and standardizing patient outcomes into **Discharged**, **DAMA**, and **Death**.
""")

st.dataframe(filtered_df.head(20), use_container_width=True)

st.header("Summary Outcomes: Discharged, DAMA, Death")

outcome_summary = filtered_analyzer.summarize_outcomes()
st.dataframe(outcome_summary, use_container_width=True)

st.header("Aggregated Data by Age, Gender, and Department")

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

st.header("Patient Outcomes by Age")

fig1, ax1 = plt.subplots(figsize=(10, 5))

for outcome in ["Discharged", "DAMA", "Death"]:
    subset = filtered_df[filtered_df["Outcome"] == outcome]
    if not subset.empty:
        ax1.hist(subset["Age"], bins=15, alpha=0.6, label=outcome)

ax1.set_title("Patient Outcomes by Age")
ax1.set_xlabel("Age")
ax1.set_ylabel("Number of Patients")
ax1.legend()

st.pyplot(fig1)

st.header("Admissions Over Time")

admissions = filtered_analyzer.admissions_over_time()

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(admissions["Admission_Period"], admissions["Admissions"], marker="o")

ax2.set_title("Admissions Over Time")
ax2.set_xlabel("Admission Month")
ax2.set_ylabel("Number of Admissions")
plt.xticks(rotation=45)

st.pyplot(fig2)

st.header("Average Service Satisfaction by Department")

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
# LENGTH OF STAY ANALYSIS ADDED
# ============================================================

st.header("Length of Stay Analysis")

los_tab1, los_tab2 = st.tabs([
    "By Department",
    "By Patient Type"
])

with los_tab1:
    st.subheader("Average Length of Stay by Department")

    los_department = filtered_analyzer.length_of_stay_by_department()

    fig_los1, ax_los1 = plt.subplots(figsize=(10, 5))
    ax_los1.bar(
        los_department["Department"],
        los_department["Average_Length_of_Stay"]
    )

    ax_los1.set_title("Average Length of Stay by Department")
    ax_los1.set_xlabel("Department")
    ax_los1.set_ylabel("Average Length of Stay")
    plt.xticks(rotation=45)

    st.pyplot(fig_los1)
    st.dataframe(los_department, use_container_width=True)

with los_tab2:
    st.subheader("Average Length of Stay by Patient Type")

    los_patient_type = filtered_analyzer.length_of_stay_by_patient_type()

    fig_los2, ax_los2 = plt.subplots(figsize=(8, 5))
    ax_los2.bar(
        los_patient_type["Patient_Type"],
        los_patient_type["Average_Length_of_Stay"]
    )

    ax_los2.set_title("Average Length of Stay by Patient Type")
    ax_los2.set_xlabel("Patient Type")
    ax_los2.set_ylabel("Average Length of Stay")
    plt.xticks(rotation=30)

    st.pyplot(fig_los2)
    st.dataframe(los_patient_type, use_container_width=True)

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

st.header("Analysis and Insights")

top_outcome = outcome_summary.sort_values(by="Count", ascending=False).iloc[0]["Outcome"]
top_disease = common_diseases.iloc[0]["Disease"]
top_satisfaction_department = satisfaction.iloc[0]["Department"]
highest_los_department = filtered_analyzer.length_of_stay_by_department().iloc[0]["Department"]

st.markdown(f"""
- The most common patient outcome is **{top_outcome}**.
- The outcome summary includes **Discharged**, **DAMA**, and **Death**, even when one category has zero records.
- The most common disease or condition is **{top_disease}**.
- The department with the highest average service satisfaction is **{top_satisfaction_department}**.
- The department with the highest average length of stay is **{highest_los_department}**.
- Admissions over time can help hospital administrators plan staffing and resource allocation.
- Outcome patterns by age, gender, and department support public health and policy decision-making.
""")

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
    "Project 6 Public Health Dashboard | OOP, Streamlit, Pandas, Matplotlib, Data Cleaning, Grouping, Aggregation, Length of Stay, and Visualization"
)
