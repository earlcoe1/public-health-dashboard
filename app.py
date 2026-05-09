import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

st.set_page_config(
    page_title="Public Health Patient & Hospital Dashboard",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Premium CSS Styling
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 2rem;
}
.dashboard-title {
    font-size: 42px;
    font-weight: 800;
    color: #1f2937;
}
.dashboard-subtitle {
    font-size: 18px;
    color: #4b5563;
    margin-bottom: 25px;
}
.kpi-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
    border-left: 6px solid #2563eb;
}
.kpi-label {
    font-size: 14px;
    color: #6b7280;
}
.kpi-value {
    font-size: 30px;
    font-weight: 800;
    color: #111827;
}
.section-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}
.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 40px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# OOP Class
# -----------------------------
class HealthAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.date_column = None

    def clean_data(self):
        self.df = self.df.drop_duplicates()

        date_columns = [
            "Admission_Date", "Date of Admission", "D.O.A",
            "DOA", "month year", "Month Year"
        ]

        for col in date_columns:
            if col in self.df.columns:
                self.date_column = col
                break

        if self.date_column:
            self.df[self.date_column] = pd.to_datetime(
                self.df[self.date_column],
                errors="coerce"
            )
            self.df["Year"] = self.df[self.date_column].dt.year
            self.df["Month"] = self.df[self.date_column].dt.month

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
        cols = ["Age", "Length_of_Stay", "Satisfaction"]
        available = [col for col in cols if col in self.df.columns]
        if available:
            return self.df[available].describe().round(2)
        return pd.DataFrame()

    def admissions_per_month(self):
        if not self.date_column:
            return pd.DataFrame(columns=["Month-Year", "Number of Admissions"])

        monthly = (
            self.df.dropna(subset=[self.date_column])
            .groupby(self.df[self.date_column].dt.to_period("M"))
            .size()
            .reset_index(name="Number of Admissions")
        )

        monthly[self.date_column] = monthly[self.date_column].astype(str)
        monthly = monthly.rename(columns={self.date_column: "Month-Year"})
        monthly = monthly.sort_values("Month-Year")

        # Remove incomplete final month if it is unusually low
        if len(monthly) > 1:
            last_value = monthly["Number of Admissions"].iloc[-1]
            avg_previous = monthly["Number of Admissions"].iloc[:-1].mean()

            if last_value < avg_previous * 0.5:
                monthly = monthly.iloc[:-1]

        return monthly

    def disease_counts(self):
        if "Disease" in self.df.columns:
            disease_df = (
                self.df["Disease"]
                .value_counts()
                .head(10)
                .reset_index()
            )

            disease_df.columns = ["Disease", "Cases"]
            return disease_df

        return pd.DataFrame(columns=["Disease", "Cases"])

    def satisfaction_by_department(self):
        if "Department" in self.df.columns and "Satisfaction" in self.df.columns:
            return (
                self.df.groupby("Department")["Satisfaction"]
                .mean()
                .round(2)
                .reset_index()
                .sort_values("Satisfaction", ascending=False)
            )

        return pd.DataFrame(columns=["Department", "Satisfaction"])

    def length_of_stay_by_department(self):
        if "Department" in self.df.columns and "Length_of_Stay" in self.df.columns:
            return (
                self.df.groupby("Department")["Length_of_Stay"]
                .mean()
                .round(2)
                .reset_index()
                .sort_values("Length_of_Stay", ascending=False)
            )

        return pd.DataFrame(columns=["Department", "Length_of_Stay"])


# -----------------------------
# Helper Functions
# -----------------------------
def kpi_card(label, value):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Filtered_Data")
    return output.getvalue()


# -----------------------------
# Dashboard Header
# -----------------------------
st.markdown(
    """
    <div class="dashboard-title">🏥 Public Health Patient & Hospital Data Dashboard</div>
    <div class="dashboard-subtitle">
    Executive healthcare analytics dashboard for patient outcomes, admissions trends,
    disease burden, length of stay, and satisfaction insights.
    </div>
    """,
    unsafe_allow_html=True
)

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

    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    st.sidebar.title("🔎 Dashboard Filters")

    filtered_df = df.copy()

    if "Gender" in df.columns:
        gender_options = sorted(df["Gender"].dropna().unique())
        gender = st.sidebar.multiselect(
            "Gender",
            options=gender_options,
            default=gender_options
        )
        filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]

    if "Department" in df.columns:
        department_options = sorted(df["Department"].dropna().unique())
        department = st.sidebar.multiselect(
            "Department",
            options=department_options,
            default=department_options
        )
        filtered_df = filtered_df[filtered_df["Department"].isin(department)]

    if "Disease" in df.columns:
        disease_options = sorted(df["Disease"].dropna().unique())
        disease = st.sidebar.multiselect(
            "Disease",
            options=disease_options,
            default=disease_options
        )
        filtered_df = filtered_df[filtered_df["Disease"].isin(disease)]

    if "Outcome" in df.columns:
        outcome_options = sorted(df["Outcome"].dropna().unique())
        outcome = st.sidebar.multiselect(
            "Outcome",
            options=outcome_options,
            default=outcome_options
        )
        filtered_df = filtered_df[filtered_df["Outcome"].isin(outcome)]

    if "Age" in df.columns and not df["Age"].dropna().empty:
        min_age = int(df["Age"].min())
        max_age = int(df["Age"].max())

        age_range = st.sidebar.slider(
            "Age Range",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )

        filtered_df = filtered_df[
            (filtered_df["Age"] >= age_range[0]) &
            (filtered_df["Age"] <= age_range[1])
        ]

    filtered_analyzer = HealthAnalyzer(filtered_df)
    filtered_df = filtered_analyzer.clean_data()

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    # -----------------------------
    # KPI Cards
    # -----------------------------
    st.header("Executive KPI Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        kpi_card("Total Patients", f"{len(filtered_df):,}")

    with col2:
        avg_age = round(filtered_df["Age"].mean(), 2) if "Age" in filtered_df.columns else "N/A"
        kpi_card("Average Age", avg_age)

    with col3:
        avg_los = round(filtered_df["Length_of_Stay"].mean(), 2) if "Length_of_Stay" in filtered_df.columns else "N/A"
        kpi_card("Avg. Stay Days", avg_los)

    with col4:
        avg_sat = round(filtered_df["Satisfaction"].mean(), 2) if "Satisfaction" in filtered_df.columns else "N/A"
        kpi_card("Avg. Satisfaction", avg_sat)

    with col5:
        if "Outcome" in filtered_df.columns:
            death_count = (filtered_df["Outcome"] == "Death").sum()
            mortality_rate = round((death_count / len(filtered_df)) * 100, 2)
            kpi_card("Mortality Rate", f"{mortality_rate}%")
        else:
            kpi_card("Mortality Rate", "N/A")

    # -----------------------------
    # Dataset Preview
    # -----------------------------
    st.header("Dataset Preview")

    with st.expander("View Head and Tail of Dataset"):
        st.subheader("Head of Dataset")
        st.dataframe(filtered_df.head(), use_container_width=True)

        st.subheader("Tail of Dataset")
        st.dataframe(filtered_df.tail(), use_container_width=True)

    # -----------------------------
    # Numerical Summary
    # -----------------------------
    st.header("Numerical Summary")

    numeric_summary = filtered_analyzer.numeric_summary()

    if not numeric_summary.empty:
        st.dataframe(numeric_summary, use_container_width=True)
    else:
        st.warning("No numerical summary columns were found.")

    # -----------------------------
    # Downloadable Reports
    # -----------------------------
    st.header("Download Filtered Report")

    csv_data = convert_df_to_csv(filtered_df)
    excel_data = convert_df_to_excel(filtered_df)

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv_data,
            file_name="filtered_hospital_report.csv",
            mime="text/csv"
        )

    with d2:
        st.download_button(
            label="Download Filtered Data as Excel",
            data=excel_data,
            file_name="filtered_hospital_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # -----------------------------
    # Admissions Trend Analysis
    # -----------------------------
    st.header("Admissions Trend Analysis")

    monthly_admissions = filtered_analyzer.admissions_per_month()

    if not monthly_admissions.empty:
        st.dataframe(
            monthly_admissions.style.format({"Number of Admissions": "{:,}"}),
            use_container_width=True,
            hide_index=True
        )

        fig = px.line(
            monthly_admissions,
            x="Month-Year",
            y="Number of Admissions",
            markers=True,
            title="Monthly Patient Admissions Over Time"
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="Month-Year",
            yaxis_title="Number of Admissions",
            title_x=0.5
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No valid date column was found for monthly admissions.")

    # -----------------------------
    # Patient Demographic Analysis
    # -----------------------------
    st.header("Patient Demographic Analysis")

    if "Age" in filtered_df.columns:
        fig = px.histogram(
            filtered_df,
            x="Age",
            nbins=20,
            title="Patient Age Distribution",
            labels={"Age": "Patient Age"}
        )

        fig.update_layout(
            title_x=0.5,
            xaxis_title="Age",
            yaxis_title="Number of Patients"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Age column not found for demographic analysis.")

    # -----------------------------
    # Patient Outcomes by Age
    # -----------------------------
    if "Outcome" in filtered_df.columns and "Age" in filtered_df.columns:
        fig = px.histogram(
            filtered_df,
            x="Age",
            color="Outcome",
            nbins=20,
            barmode="overlay",
            title="Patient Outcomes Distribution by Age"
        )

        fig.update_layout(
            title_x=0.5,
            xaxis_title="Age",
            yaxis_title="Number of Patients"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Outcome and Age columns are required for outcomes-by-age analysis.")

    # -----------------------------
    # Disease Burden Analysis
    # -----------------------------
    st.header("Disease Burden Analysis")

    diseases = filtered_analyzer.disease_counts()

    if not diseases.empty and "Disease" in diseases.columns and "Cases" in diseases.columns:

        fig = px.bar(
            diseases,
            x="Cases",
            y="Disease",
            orientation="h",
            title="Top 10 Most Common Diseases or Conditions",
            text="Cases"
        )

        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_title="Number of Cases",
            yaxis_title="Disease",
            title_x=0.5
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.warning("Disease data is not available for this chart.")

    # -----------------------------
    # Operational Efficiency Analysis
    # -----------------------------
    st.header("Operational Efficiency Analysis")

    los = filtered_analyzer.length_of_stay_by_department()

    if not los.empty:
        fig = px.bar(
            los,
            x="Department",
            y="Length_of_Stay",
            title="Average Length of Stay by Department",
            text="Length_of_Stay"
        )

        fig.update_layout(
            title_x=0.5,
            xaxis_title="Department",
            yaxis_title="Average Length of Stay"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Department and Length_of_Stay columns are required.")

    # -----------------------------
    # Satisfaction Analysis
    # -----------------------------
    st.header("Patient Satisfaction Analysis")

    satisfaction = filtered_analyzer.satisfaction_by_department()

    if not satisfaction.empty:
        fig = px.bar(
            satisfaction,
            x="Department",
            y="Satisfaction",
            title="Average Service Satisfaction by Department",
            text="Satisfaction"
        )

        fig.update_layout(
            title_x=0.5,
            xaxis_title="Department",
            yaxis_title="Average Satisfaction Rating"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Department and Satisfaction columns are required.")

    # -----------------------------
    # Patient Outcome Summary
    # -----------------------------
    st.header("Patient Outcome Summary")

    if "Outcome" in filtered_df.columns:
        outcome_df = (
            filtered_df["Outcome"]
            .value_counts()
            .reset_index()
        )
        outcome_df.columns = ["Outcome", "Count"]

        fig = px.pie(
            outcome_df,
            names="Outcome",
            values="Count",
            title="Patient Outcome Distribution"
        )

        fig.update_layout(
            title_x=0.5
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Outcome column not found.")

    # -----------------------------
    # Key Findings
    # -----------------------------
    st.header("Key Findings and Actionable Insights")

    most_common_disease = (
        filtered_df["Disease"].mode()[0]
        if "Disease" in filtered_df.columns and not filtered_df["Disease"].mode().empty
        else "N/A"
    )

    st.markdown(f"""
    <div class="section-card">
    <ul>
        <li>The filtered dataset contains <b>{len(filtered_df):,}</b> patient records.</li>
        <li>The average patient age is <b>{avg_age}</b>.</li>
        <li>The average length of stay is <b>{avg_los}</b> days.</li>
        <li>The average satisfaction rating is <b>{avg_sat}</b>.</li>
        <li>The most common disease or condition is <b>{most_common_disease}</b>.</li>
        <li>Admission trends can support staffing, bed planning, and resource allocation.</li>
        <li>Length-of-stay analysis can identify departments needing operational improvement.</li>
        <li>Satisfaction trends can guide patient experience improvement initiatives.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="footer">
        Developed by Earl Nimley | Public Health Analytics Dashboard | BUIS 305 / INSS 405
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.info("Please upload the hospital dataset to begin.")
