import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# 1. CREATE REQUIRED DIRECTORIES
# ---------------------------------------------------------

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)


# ---------------------------------------------------------
# 2. DATA ACQUISITION
# ---------------------------------------------------------

print("=" * 60)
print("DATA ACQUISITION")
print("=" * 60)

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]

# Read training data
train_data = pd.read_csv(
    "data/raw/adult.data",
    names=columns,
    skipinitialspace=True,
    na_values="?"
)

# Read test data
test_data = pd.read_csv(
    "data/raw/adult.test",
    names=columns,
    skipinitialspace=True,
    na_values="?",
    skiprows=1
)

# Combine training and test data
df = pd.concat(
    [train_data, test_data],
    ignore_index=True
)

print("\nDataset successfully loaded from local files.")
print("Dataset shape:", df.shape)

# Save raw combined dataset
df.to_csv(
    "data/raw/adult_raw.csv",
    index=False
)

print("Raw dataset saved to: data/raw/adult_raw.csv")


# ---------------------------------------------------------
# 3. INITIAL DATA EXPLORATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("INITIAL DATA EXPLORATION")
print("=" * 60)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe(include="all").transpose())


# ---------------------------------------------------------
# 4. CLEAN COLUMN NAMES
# ---------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace("-", "_")
    .str.replace(" ", "_")
)

print("\nCleaned column names:")
print(df.columns.tolist())


# ---------------------------------------------------------
# 5. CLEAN TEXT VALUES
# ---------------------------------------------------------

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].astype("string").str.strip()


# ---------------------------------------------------------
# 6. STANDARDIZE MISSING VALUES
# ---------------------------------------------------------

missing_values = [
    "?",
    "NA",
    "N/A",
    "na",
    "n/a",
    "None",
    "none",
    ""
]

df = df.replace(missing_values, np.nan)

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False))


# ---------------------------------------------------------
# 7. MISSING VALUE ANALYSIS
# ---------------------------------------------------------

missing_summary = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (
        df.isnull().sum() / len(df)
    ) * 100
})

missing_summary = missing_summary[
    missing_summary["Missing Count"] > 0
].sort_values(
    "Missing Count",
    ascending=False
)

print("\nMissing value summary:")
print(missing_summary)


# ---------------------------------------------------------
# 8. HANDLE MISSING VALUES
# ---------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].median()
        )


categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


print("\nMissing values after treatment:")
print(
    df.isnull().sum()
    .sort_values(ascending=False)
    .head(10)
)


# ---------------------------------------------------------
# 9. DUPLICATE RECORD ANALYSIS
# ---------------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate rows:", duplicate_count)

if duplicate_count > 0:

    df = df.drop_duplicates()

print(
    "Shape after duplicate removal:",
    df.shape
)


# ---------------------------------------------------------
# 10. STANDARDIZE INCOME TARGET
# ---------------------------------------------------------

df["income"] = (
    df["income"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.strip()
)

df["income"] = df["income"].replace({
    "<=50K": "<=50K",
    ">50K": ">50K"
})

print("\nIncome distribution:")
print(df["income"].value_counts())


# ---------------------------------------------------------
# 11. OUTLIER DETECTION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("OUTLIER DETECTION")
print("=" * 60)

outlier_columns = [
    "age",
    "fnlwgt",
    "education_num",
    "hours_per_week"
]

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"\n{column}")
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower bound:", lower_bound)
    print("Upper bound:", upper_bound)
    print("Number of outliers:", len(outliers))


# ---------------------------------------------------------
# 12. OUTLIER TREATMENT
# ---------------------------------------------------------

columns_to_cap = [
    "age",
    "fnlwgt",
    "hours_per_week"
]

for column in columns_to_cap:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = df[column].clip(
        lower=lower_bound,
        upper=upper_bound
    )


print("\nOutlier treatment completed.")


# ---------------------------------------------------------
# 13. DATA TYPE VALIDATION
# ---------------------------------------------------------

numeric_columns = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ---------------------------------------------------------
# 14. FINAL MISSING VALUE CHECK
# ---------------------------------------------------------

print("\nFinal missing-value check:")

print(
    df.isnull()
    .sum()
    .sort_values(ascending=False)
    .head(10)
)


# ---------------------------------------------------------
# 15. SAVE CLEAN DATASET
# ---------------------------------------------------------

processed_path = (
    "data/processed/adult_cleaned.csv"
)

df.to_csv(
    processed_path,
    index=False
)

print("\nClean dataset saved to:")
print(processed_path)

print("\nFinal dataset shape:")
print(df.shape)


# ---------------------------------------------------------
# 16. VISUALIZATION — INCOME DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="income"
)

plt.title("Income Distribution")
plt.xlabel("Income Category")
plt.ylabel("Number of Individuals")

plt.tight_layout()

plt.savefig(
    "outputs/plots/income_distribution.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 17. VISUALIZATION — AGE DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "outputs/plots/age_distribution.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 18. VISUALIZATION — HOURS PER WEEK
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="hours_per_week",
    bins=30,
    kde=True
)

plt.title("Working Hours per Week")
plt.xlabel("Hours per Week")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "outputs/plots/hours_per_week.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 19. CORRELATION MATRIX
# ---------------------------------------------------------

plt.figure(figsize=(10, 7))

correlation = df.select_dtypes(
    include=np.number
).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/plots/correlation_matrix.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# PROJECT COMPLETE
# ---------------------------------------------------------

print("\nAll visualizations generated successfully.")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)