import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# STEP 1: Load cleaned datasets
# ==============================
cases_monthly = pd.read_csv("C:\\Users\\lalith\\Downloads\\dataset\\cases\\cases_monthly.csv")
vacc_monthly = pd.read_csv("C:\\Users\\lalith\\Downloads\\dataset\\vaccination\\vaccinations_monthly.csv")
tests_monthly = pd.read_csv("C:\\Users\\lalith\\Downloads\\dataset\\test\\tests_monthly.csv")

cases_weekly = pd.read_csv("C:\\Users\\lalith\\Downloads\\dataset\\cases\\cases_weekly.csv")
vacc_weekly = pd.read_csv("C:\\Users\\lalith\\Downloads\\dataset\\vaccination\\vaccinations_weekly.csv")
tests_weekly = pd.read_csv("C:\\Users\\lalith\\Downloads\\dataset\\test\\tests_weekly.csv")

# ==============================
# STEP 2: Normalize column names
# ==============================
for df in [cases_monthly, vacc_monthly, tests_monthly,
           cases_weekly, vacc_weekly, tests_weekly]:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ==============================
# STEP 3: Parse date columns
# ==============================
for df, col in [(cases_monthly, "year_month"),
                (vacc_monthly, "year_month"),
                (tests_monthly, "year_month"),
                (cases_weekly, "year_week"),
                (vacc_weekly, "year_week"),
                (tests_weekly, "year_week")]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# ==============================
# STEP 4: Ensure 'people_fully_vaccinated' exists
# ==============================
for df in [vacc_monthly, vacc_weekly]:
    if "people_fully_vaccinated" not in df.columns:
        if "people_fully_vaccinated_per_hundred" in df.columns and "total_vaccinations" in df.columns:
            df["people_fully_vaccinated"] = (df["people_fully_vaccinated_per_hundred"]/100) * df["total_vaccinations"]
        else:
            df["people_fully_vaccinated"] = pd.NA

# ==============================
# STEP 5: Merge datasets
# ==============================
def integrate_data(cases, vacc, tests, time_key):
    df = cases.merge(vacc, on=["location", time_key], how="outer") \
              .merge(tests, on=["location", time_key], how="outer")
    return df

monthly_df = integrate_data(cases_monthly, vacc_monthly, tests_monthly, "year_month")
weekly_df = integrate_data(cases_weekly, vacc_weekly, tests_weekly, "year_week")
# --- Fix merged column names ---
for df in [monthly_df, weekly_df]:
    if "people_fully_vaccinated_y" in df.columns:
        df["people_fully_vaccinated"] = df["people_fully_vaccinated_y"]
    elif "people_fully_vaccinated_x" in df.columns:
        df["people_fully_vaccinated"] = df["people_fully_vaccinated_x"]

    if "people_vaccinated_y" in df.columns:
        df["people_vaccinated"] = df["people_vaccinated_y"]
    elif "people_vaccinated_x" in df.columns:
        df["people_vaccinated"] = df["people_vaccinated_x"]

    if "total_vaccinations_y" in df.columns:
        df["total_vaccinations"] = df["total_vaccinations_y"]
    elif "total_vaccinations_x" in df.columns:
        df["total_vaccinations"] = df["total_vaccinations_x"]

# --- Fix test column names ---
for df in [monthly_df, weekly_df]:
    if "short-term_tests_per_case" in df.columns:
        df["tests_per_case"] = df["short-term_tests_per_case"]
    if "Short-term positive rate" in df.columns:
        df["positivity_rate"] = df["Short-term positive rate"]

# ==============================
# STEP 6: Save integrated datasets
# ==============================
monthly_df.to_csv("covid_integrated_monthly.csv", index=False)
weekly_df.to_csv("covid_integrated_weekly.csv", index=False)

print("✅ Integrated datasets saved.")
print("Monthly dataset shape:", monthly_df.shape)
print("Weekly dataset shape:", weekly_df.shape)
print("Monthly columns:", monthly_df.columns.tolist())
print("Weekly columns:", weekly_df.columns.tolist())


# ==============================
# STEP 4: Visualizations
# ==============================

# --- 1. Cases & Deaths Trend ---
plt.figure(figsize=(12,6))
for country in ["India", "United States", "France", "Germany", "Japan"]:
    country_data = monthly_df[monthly_df["location"] == country]
    plt.plot(country_data["year_month"], country_data["new_cases"], label=f"{country} Cases")
plt.xticks(rotation=45)
plt.title("Monthly COVID-19 New Cases Trend")
plt.xlabel("Month")
plt.ylabel("New Cases")
plt.legend()
plt.tight_layout()
plt.show()

# --- 2. Vaccination Progress ---
plt.figure(figsize=(12,6))
for country in ["India", "United States", "France", "Germany", "Japan"]:
    country_data = monthly_df[monthly_df["location"] == country]
    plt.plot(country_data["year_month"], country_data["people_fully_vaccinated"], label=f"{country}")
plt.xticks(rotation=45)
plt.title("Monthly Vaccination Progress (Fully Vaccinated)")
plt.xlabel("Month")
plt.ylabel("People Fully Vaccinated")
plt.legend()
plt.tight_layout()
plt.show()

# --- 3. Testing Efficiency (Tests per Case) ---
plt.figure(figsize=(12,6))
sns.barplot(data=monthly_df, x="location", y="tests_per_case", ci=None)
plt.xticks(rotation=45)
plt.title("Testing Efficiency (Tests per Case)")
plt.ylabel("Tests per Case")
plt.tight_layout()
plt.show()

# --- 4. Correlation Heatmap ---
plt.figure(figsize=(10,6))
corr = monthly_df[["new_cases","new_deaths","total_cases","total_deaths",
                   "people_vaccinated","people_fully_vaccinated",
                   "cumulative_tests","positivity_rate","tests_per_case"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between COVID-19 Indicators")
plt.tight_layout()
plt.show()

# --- 5. CFR vs Vaccination Coverage (Risk View) ---
plt.figure(figsize=(10,6))
sns.scatterplot(data=monthly_df, x="vaccination_coverage", y="CFR", hue="location")
plt.title("Risk Assessment: Vaccination Coverage vs Case Fatality Rate")
plt.xlabel("Vaccination Coverage (%)")
plt.ylabel("Case Fatality Rate (%)")
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.show()
