import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os

# === OUTPUT FOLDER ===
os.makedirs("outputs/module1/weekly", exist_ok=True)

# === LOAD CLEANED DATASETS ===
cases_w = pd.read_csv("dataset/cases/cases_weekly.csv")
vacc_w  = pd.read_csv("dataset/vaccination/vaccinations_weekly.csv")
tests_w = pd.read_csv("dataset/test/tests_weekly.csv")

# Ensure types are strings
for df in (cases_w, vacc_w, tests_w):
    if "year_week" in df.columns:
        df["year_week"] = df["year_week"].astype(str)
    if "location" in df.columns:
        df["location"] = df["location"].astype(str)

# === MERGE ===
df = pd.merge(cases_w, vacc_w, on=["location", "year_week"], how="outer")
df = pd.merge(df, tests_w, on=["location", "year_week"], how="outer")

# Sort
df["week_sort_key"] = pd.to_datetime(df["year_week"].str.replace("-", "") + "1", errors="coerce", format="%G%V%w")
df = df.sort_values(["location", "week_sort_key"]).reset_index(drop=True)

plot_df = df.copy()
numeric_cols = [c for c in plot_df.columns if c not in ["location", "year_week", "week_sort_key"]]
for c in numeric_cols:
    plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")

# === HEATMAP: Weekly new cases ===
heat_pivot = plot_df.pivot_table(index="location", columns="year_week", values="new_cases", aggfunc="sum")
heat_mat = heat_pivot.fillna(0).to_numpy()

plt.figure(figsize=(14, 6))
im = plt.imshow(heat_mat, aspect="auto")
plt.colorbar(im)
plt.title("Heatmap of Weekly New COVID-19 Cases by Country")
weeks = list(heat_pivot.columns.astype(str))
step = max(1, len(weeks) // 20)
plt.xticks(ticks=np.arange(0, len(weeks), step), labels=[weeks[i] for i in range(0, len(weeks), step)], rotation=45, ha="right")
plt.yticks(ticks=np.arange(len(heat_pivot.index)), labels=heat_pivot.index)
plt.tight_layout()
plt.savefig("outputs/module1/weekly/weekly_heatmap_cases.png", dpi=200)
plt.close()

# === DISTRIBUTION CURVES ===
plt.figure(figsize=(12, 7))
for country in plot_df["location"].dropna().unique():
    vals = plot_df.loc[plot_df["location"] == country, "new_cases"].dropna().astype(float)
    if len(vals) < 5:
        continue
    density, bins = np.histogram(vals, bins=30, density=True)
    centers = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(centers, density, alpha=0.8, linewidth=1.2, label=country)

plt.title("Weekly New Cases Distribution by Country (Density Curves)")
plt.xlabel("Weekly New Cases")
plt.ylabel("Density")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig("outputs/module1/weekly/weekly_cases_distribution_kde.png", dpi=200)
plt.close()

# === CHOROPLETH: Latest Vaccination Snapshot ===
latest_per_country = plot_df.sort_values("week_sort_key").dropna(subset=["people_vaccinated"]).groupby("location").tail(1)
latest_per_country["people_vaccinated"] = latest_per_country["people_vaccinated"].fillna(0)

fig = px.choropleth(
    latest_per_country,
    locations="location",
    locationmode="country names",
    color="people_vaccinated",
    hover_name="location",
    title="COVID-19 People Vaccinated (Latest Weekly Snapshot)",
    color_continuous_scale="Viridis"
)
fig.write_html("outputs/module1/weekly/weekly_vaccination_choropleth_latest.html")

# === TREND LINES ===
plt.figure(figsize=(14, 7))
for country in plot_df["location"].dropna().unique():
    cdf = plot_df[plot_df["location"] == country]
    plt.plot(cdf["week_sort_key"], cdf["new_cases"], linewidth=1.2, label=country)

plt.title("Weekly New COVID-19 Cases Trend by Country")
plt.xlabel("Week")
plt.ylabel("New Cases")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig("outputs/module1/weekly/weekly_trend_cases.png", dpi=200)
plt.close()

print("✅ Module 1 Weekly Visualizations saved in outputs/module1/weekly/")
