import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Public Health Patient & Hospital Data Dashboard",
    page_icon="🏥",
    layout="wide"
)


class HealthAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    def clean_data(self):
        self.df = self.df.drop_duplicates()

        date_columns = ["Admission_Date", "Date of Admission", "D.O.A", "DOA", "month year", "Month Year"]
        self.date_column = None

        for col in date_columns:
            if col in self.df.columns:
                self.date_column = col
                break

        if self.date_column:
            self.df[self.date_column] = pd.to_datetime(self.df[self.date_column], errors="coerce")

        for col in ["Age", "Length_of_Stay", "Satisfaction"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        if "Outcome" in self.df.columns:
            self.df["Outcome"] = self.df["Outcome"].replace({
                "Discharge": "Discharged",
                "Died": "Death",
                "Dead": "Death"
            })

        self.df = self.df.dropna(how="all")
        return self.df

    def numeric_summary(self):
        summary_cols = ["Age", "Length_of_Stay", "Satisfaction"]
        available_cols = [col for col in summary_cols if col in self.df.columns]
        return self.df[available_cols].describe().round(2) if available_cols else pd.DataFrame()

    def admissions_per_month(self):
        if not self.date_column:
            return pd.DataFrame()

        monthly = (
            self.df.dropna(subset=[self.date_column])
            .groupby(self.df[self.date_column].dt.to_period("M"))
            .size()
            .reset_index(name="Number of Admissions")
        )

        monthly[self.date_column] = monthly[self.date_column].astype(str)
        monthly = monthly.rename(columns={self.date_column: "Month-Year"})
        monthly = monthly.sort_values("Month-Year")

        if len(monthly) > 1:
            last_value = monthly["Number of Admissions"].iloc[-1]
            avg_previous = monthly["Number of Admissions"].iloc[:-1].mean()
            if last_value < avg_previous * 0.5:
                monthly = monthly.iloc[:-1]

        return monthly

    def disease_counts(self):
        if "Disease" in self.df.columns:
            return self.df["Disease"].value_counts().head(10)
        return pd.Series(dtype=int)

    def outcome_summary(self):
        if "Outcome" in self.df.columns:
            return self.df["Outcome"].value_counts()
        return pd.Series(dtype=int)

    def satisfaction_by_department(self):
        if "Department" in self.df.columns and "Satisfaction" in self.df.columns:
            return self.df.groupby("Department")["Satisfaction"].mean().sort_values(ascending=False)
        return pd.Series(dtype=float)

    def length_of_stay_by_department(self):
        if "Department" in self.df.columns and "Length_of_Stay" in self.df.columns:
            return self.df.groupby("Department")["Length_of_Stay"].mean().sort_values(ascending=False)
        return pd.Series(dtype=float)


st.title("🏥 Public Health: Patient & Hospital Data Dashboard")

st.write(
    "This dashboard analyzes hospital patient records to support operational, quality, "
    "and public health decision-making."
)

st.header("1. Upload Hospital Dataset")

uploaded_file = st.file_uploader(
    "Upload hospital dataset file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)

    analyzer = HealthAnalyzer(raw_df)
    df = analyzer.clean_data()

    st.success("Dataset uploaded and cleaned successfully!")

    st.header("2. Preview Head and Tail of Uploaded Data")

    st.subheader("Head of Dataset")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Tail of Dataset")
    st.dataframe(df.tail(), use_container_width=True)

    st.header("3. Summary of Statistical Properties on Numerical Data")

    numeric_summary = analyzer.numeric_summary()

    if not numeric_summary.empty:
        st.dataframe(numeric_summary, use_container_width=True)
    else:
        st.warning("No numerical summary columns were found.")

    st.sidebar.header("Interactive Filters")

    filtered_df = df.copy()

    if "Gender" in df.columns:
        gender = st.sidebar.selectbox("Filter by Gender", ["All"] + sorted(df["Gender"].dropna().unique().tolist()))
        if gender != "All":
            filtered_df = filtered_df[filtered_df["Gender"] == gender]

    if "Department" in df.columns:
        department = st.sidebar.selectbox("Filter by Department", ["All"] + sorted(df["Department"].dropna().unique().tolist()))
        if department != "All":
            filtered_df = filtered_df[filtered_df["Department"] == department]

    if "Disease" in df.columns:
        disease = st.sidebar.selectbox("Filter by Disease", ["All"] + sorted(df["Disease"].dropna().unique().tolist()))
        if disease != "All":
            filtered_df = filtered_df[filtered_df["Disease"] == disease]

    if "Outcome" in df.columns:
        outcome = st.sidebar.selectbox("Filter by Outcome", ["All"] + sorted(df["Outcome"].dropna().unique().tolist()))
        if outcome != "All":
            filtered_df = filtered_df[filtered_df["Outcome"] == outcome]

    filtered_analyzer = HealthAnalyzer(filtered_df)
    filtered_df = filtered_analyzer.clean_data()

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    st.header("4. Executive KPI Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Patient Records", len(filtered_df))

    if "Age" in filtered_df.columns:
        c2.metric("Average Age", round(filtered_df["Age"].mean(), 2))
    else:
        c2.metric("Average Age", "N/A")

    if "Length_of_Stay" in filtered_df.columns:
        c3.metric("Average Length of Stay", round(filtered_df["Length_of_Stay"].mean(), 2))
    else:
        c3.metric("Average Length of Stay", "N/A")

    if "Satisfaction" in filtered_df.columns:
        c4.metric("Average Satisfaction", round(filtered_df["Satisfaction"].mean(), 2))
    else:
        c4.metric("Average Satisfaction", "N/A")

    st.header("5. Number of Admissions Per Month")

    monthly_admissions = filtered_analyzer.admissions_per_month()

    if not monthly_admissions.empty:
        st.dataframe(
            monthly_admissions.style.format({"Number of Admissions": "{:,}"}),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Monthly Admissions Over Time")

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            monthly_admissions["Month-Year"],
            monthly_admissions["Number of Admissions"],
            marker="o",
            linewidth=2
        )
        ax.set_title("Monthly Patient Admissions Over Time")
        ax.set_xlabel("Month-Year")
        ax.set_ylabel("Number of Admissions")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No valid date column was found for monthly admissions.")

    st.header("6. Patient Outcomes Distribution by Age")

    if "Outcome" in filtered_df.columns and "Age" in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))

        for outcome_name in filtered_df["Outcome"].dropna().unique():
            subset = filtered_df[filtered_df["Outcome"] == outcome_name]
            ax.hist(subset["Age"], alpha=0.5, label=str(outcome_name))

        ax.set_title("Patient Outcomes by Age")
        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Patients")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Outcome and Age columns are required for this chart.")

    st.header("7. Most Common Diseases or Conditions")

    diseases = filtered_analyzer.disease_counts()

    if not diseases.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        diseases.plot(kind="barh", ax=ax)
        ax.set_title("Top 10 Most Common Diseases or Conditions")
        ax.set_xlabel("Number of Cases")
        ax.set_ylabel("Disease")
        ax.invert_yaxis()
        st.pyplot(fig)
    else:
        st.warning("Disease column not found.")

    st.header("8. Average Length of Stay by Department")

    los = filtered_analyzer.length_of_stay_by_department()

    if not los.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        los.plot(kind="bar", ax=ax)
        ax.set_title("Average Length of Stay by Department")
        ax.set_xlabel("Department")
        ax.set_ylabel("Average Length of Stay")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Department and Length_of_Stay columns are required.")

    st.header("9. Average Service Satisfaction by Department")

    satisfaction = filtered_analyzer.satisfaction_by_department()

    if not satisfaction.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        satisfaction.plot(kind="bar", ax=ax)
        ax.set_title("Average Service Satisfaction by Department")
        ax.set_xlabel("Department")
        ax.set_ylabel("Average Satisfaction Rating")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Department and Satisfaction columns are required.")

    st.header("10. Key Findings and Actionable Insights")

    most_common_disease = (
        filtered_df["Disease"].mode()[0]
        if "Disease" in filtered_df.columns and not filtered_df["Disease"].mode().empty
        else "N/A"
    )

    st.markdown(f"""
    - The dataset contains **{len(filtered_df):,} patient records** after applying selected filters.
    - The most common disease or condition in the filtered data is **{most_common_disease}**.
    - Monthly admissions trends can help hospital administrators plan staffing and resource allocation.
    - Length-of-stay analysis can help identify departments where operational efficiency may be improved.
    - Satisfaction ratings can support patient experience improvement initiatives.
    """)

else:
    st.info("Please upload the hospital dataset to begin.")
