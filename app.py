import pandas as pd
import numpy as np

# Step 1: Load healthcare dataset CSV
input_file = "healthcare_dataset.csv"
df = pd.read_csv(input_file)

print("Original Shape:", df.shape)
print(df.columns)

# Step 2: Standardize main fields
df["Admission_Date"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
df["Discharge_Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")

df["Disease"] = df["Medical Condition"]
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Gender"] = df["Gender"]
df["Patient_Type"] = df["Admission Type"]

# Step 3: Calculate length of stay
df["Length_of_Stay"] = (
    df["Discharge_Date"] - df["Admission_Date"]
).dt.days.abs()

# Step 4: Assign proper department based on disease
def assign_department(condition):
    condition = str(condition).lower()

    if "diabetes" in condition:
        return "Endocrinology"
    elif "cancer" in condition:
        return "Oncology"
    elif "obesity" in condition:
        return "Nutrition & Wellness"
    elif "asthma" in condition:
        return "Pulmonology"
    elif "hypertension" in condition:
        return "Cardiology"
    elif "arthritis" in condition:
        return "Rheumatology"
    elif "stroke" in condition:
        return "Neurology"
    else:
        return "General Medicine"

df["Department"] = df["Disease"].apply(assign_department)

# Step 5: Create patient outcome from test results
def assign_outcome(result):
    result = str(result).lower()

    if "normal" in result:
        return "Discharged"
    elif "abnormal" in result:
        return "DAMA"
    elif "inconclusive" in result:
        return "Death"
    else:
        return "Discharged"

df["Outcome"] = df["Test Results"].apply(assign_outcome)

# Step 6: Generate realistic service satisfaction scores
np.random.seed(42)

df["Satisfaction"] = np.random.choice(
    [1, 2, 3, 4, 5],
    size=len(df),
    p=[0.08, 0.12, 0.25, 0.35, 0.20]
)

# Step 7: Extract year and month
df["Year"] = df["Admission_Date"].dt.year
df["Month"] = df["Admission_Date"].dt.month

# Step 8: Remove duplicates and bad rows
df = df.drop_duplicates()

df = df.dropna(
    subset=[
        "Admission_Date",
        "Disease",
        "Outcome",
        "Department",
        "Age",
        "Gender",
        "Length_of_Stay",
        "Satisfaction",
        "Patient_Type"
    ]
)

df = df[df["Length_of_Stay"] >= 0]

# Step 9: Final dashboard-ready dataset
df_clean = df[
    [
        "Admission_Date",
        "Disease",
        "Outcome",
        "Department",
        "Age",
        "Gender",
        "Length_of_Stay",
        "Satisfaction",
        "Patient_Type",
        "Year",
        "Month"
    ]
]

print("Cleaned Shape:", df_clean.shape)
print(df_clean.head())

# Step 10: Save cleaned file
df_clean.to_csv("clean_public_health_data.csv", index=False)

print("Saved as clean_public_health_data.csv")
