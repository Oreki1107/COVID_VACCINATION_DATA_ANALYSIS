# m1.py — Module 1 (Interactive, no heatmap) 
# Weekly COVID-19: Animated Bubble, Choropleth, Distribution, Violin, Trend, Risk

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ========= 1) LOAD & MERGE WEEKLY DATASETS =========
CASES_W = r"C:\Users\vigne\Downloads\dataset\cases\cases_weekly.csv"
VACC_W  = r"C:\Users\vigne\Downloads\dataset\vaccination\vaccinations_weekly.csv"
TESTS_W = r"C:\Users\vigne\Downloads\dataset\test\tests_weekly.csv"

cases_w = pd.read_csv(CASES_W)
vacc_w  = pd.read_csv(VACC_W)
tests_w = pd.read_csv(TESTS_W)

# types
for d in (cases_w, vacc_w, tests_w):
    if "year_week" in d.columns:
        d["year_week"] = d["year_week"].astype(str)
    if "location" in d.columns:
        d["location"] = d["location"].astype(str)

# merge (outer to preserve info)
df = pd.merge(
    cases_w, vacc_w,
    on=["location", "year_week"],
    how="outer",
    suffixes=("", "_vacc"),
    validate="one_to_one"   # ✅ Ensures unique join
)

df = pd.merge(
    df, tests_w,
    on=["location", "year_week"],
    how="outer",
    suffixes=("", "_tests"),
    validate="one_to_one"   # ✅ Ensures unique join
)
# robust week sort key (fallback to coerce)
# we try ISO week first for better accuracy; fall back to a permissive parse
def _parse_week_safe(s):
    try:
        # ISO week: %G (ISO year), %V (ISO week), %u (ISO weekday 1-7)
        return pd.to_datetime(s + "-1", format="%G-%V-%u", errors="raise")
    except Exception:
        return pd.to_datetime(s + "-1", errors="coerce")

df["week_sort_key"] = df["year_week"].apply(_parse_week_safe)
df = df.sort_values(["location", "week_sort_key"]).reset_index(drop=True)

# numeric coercion
for c in df.columns:
    if c not in ["location", "year_week", "week_sort_key"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ========= 2) PICK BEST-FIT METRICS (WITH FALLBACKS) =========
# Y-axis for most charts
Y_CASES = "new_cases" if "new_cases" in df.columns else None

# Testing intensity (X for bubble)
TEST_INT_CANDIDATES = [
    # your cleaned weekly tests may have some of these:
    "tests_per_thousand",
    "seven_day_smoothed_daily_change_per_thousand",
    "7-day smoothed daily change per thousand",
    "Cumulative total per thousand",
    "Daily change in cumulative total per thousand"
]
X_TESTS = next((c for c in TEST_INT_CANDIDATES if c in df.columns), None)

# Vaccination size metric (bubble size & choropleth scale)
VACC_PCT_COL = "people_vaccinated_per_hundred" if "people_vaccinated_per_hundred" in df.columns else None
VACC_RAW_COL = "people_vaccinated" if "people_vaccinated" in df.columns else None
SIZE_COL = VACC_PCT_COL if VACC_PCT_COL else VACC_RAW_COL

# Positivity for violin/box/risk
POS_COL = "positivity_rate" if "positivity_rate" in df.columns else None

# ========= 3) REPLACEMENT: ANIMATED BUBBLE CHART =========
if X_TESTS and Y_CASES and SIZE_COL:
    df_plot = df.copy()
    for col in [X_TESTS, Y_CASES, SIZE_COL]:
        df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce").clip(lower=0)
    df_plot[SIZE_COL] = df_plot[SIZE_COL].fillna(0)

    size_max = 45
    if SIZE_COL == VACC_PCT_COL:
        size_col_for_plot = SIZE_COL
        df_plot[size_col_for_plot] = df_plot[size_col_for_plot].clip(lower=0, upper=100)
    else:
        size_col_for_plot = "_size_sqrt_vacc"
        df_plot[size_col_for_plot] = np.sqrt(df_plot[SIZE_COL].fillna(0))

    # Global axis ranges (fix blinking)
    x_min, x_max = df_plot[X_TESTS].min(skipna=True), df_plot[X_TESTS].max(skipna=True)
    y_min, y_max = df_plot[Y_CASES].min(skipna=True), df_plot[Y_CASES].max(skipna=True)

    fig_bubble = px.scatter(
        df_plot,
        x=X_TESTS, y=Y_CASES,
        animation_frame="year_week",
        animation_group="location",
        size=size_col_for_plot,
        color="location",
        hover_name="location",
        title="Weekly COVID-19 Timelapse — Testing vs New Cases (Bubble size shows vaccination)",
        size_max=size_max
    )
    fig_bubble.update_layout(
        xaxis=dict(title=f"Testing intensity ({X_TESTS})", range=[x_min, x_max]),
        yaxis=dict(title="Weekly New Cases", range=[y_min, y_max]),
        legend_title_text="Country",
        margin=dict(l=40, r=20, t=60, b=40)
    )
    fig_bubble.write_html("weekly_bubble_timelapse.html")


# ========= ENHANCED VIOLIN PLOT (INTERACTIVE) =========
# Weekly new cases distribution per country
if "new_cases" not in df_plot.columns or "location" not in df_plot.columns:
    print("Skipping violin plot: Required columns (new_cases, location) not found in df_plot")
else:
    # Existing violin plot code
    fig_violin = px.violin(
    df_plot,  # Changed from plot_df to df_plot
    x="location",
    y="new_cases",
    color="location",
    box=True,                # adds mini boxplot inside
    points="all",            # show all points (jittered)
    hover_data=df_plot.columns,  # Changed from plot_df to df_plot
    title="Weekly New Cases Distribution by Country (Violin Plot)"
)

fig_violin.update_layout(
    xaxis_title="Country",
    yaxis_title="Weekly New Cases",
    xaxis_tickangle=-45,
    margin=dict(l=40, r=20, t=60, b=40),
    legend_title="Country"
)

fig_violin.write_html("weekly_violin_distribution.html")



# ========= 5) VIOLIN (POSITIVITY) =========
if POS_COL:
    fig_violin = px.violin(
        df, x="location", y=POS_COL, box=True, points="all",
        title="Weekly Testing Positivity Rate — Distribution per Country"
    )
    fig_violin.update_layout(xaxis_tickangle=-45, yaxis_title="Positivity Rate")
    fig_violin.write_html("weekly_positivity_violin.html")

# ========= 6) ENHANCED CHOROPLETH (VACCINATION) =========
if SIZE_COL:
    vacc_col = VACC_PCT_COL if VACC_PCT_COL else VACC_RAW_COL
    title = "People Vaccinated (% of Population) Over Time (Weekly)" if VACC_PCT_COL else "People Vaccinated (Counts) Over Time (Weekly)"

    # Stable color range across frames:
    if VACC_PCT_COL:
        color_range = [0, 100]
        colorbar_title = "% Vaccinated"
    else:
        # global max for consistent scale
        vmax = float(df[vacc_col].max(skipna=True) or 0)
        color_range = [0, vmax] if vmax > 0 else None
        colorbar_title = "People Vaccinated"

    # Keep only rows with a valid location to avoid noisy frames
    df_choro = df[df["location"].notna()].copy()
    # Use the parsed week for correct frame sort
    df_choro = df_choro.sort_values("week_sort_key")

    fig_vacc = px.choropleth(
        df_choro,
        locations="location",
        locationmode="country names",
        color=vacc_col,
        hover_name="location",
        animation_frame="year_week",
        title=title,
        color_continuous_scale="Viridis",
        range_color=color_range
    )
    fig_vacc.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
        coloraxis_colorbar=dict(title=colorbar_title),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    # hover extras: show cases and tests if present
    hover_cols = []
    if Y_CASES: hover_cols.append(Y_CASES)
    if X_TESTS: hover_cols.append(X_TESTS)
    if POS_COL: hover_cols.append(POS_COL)
    # build custom hover template if any extras available
    if hover_cols:
        # add to figure via hovertemplate for first trace (applies to frames too)
        fig_vacc.update_traces(
            hovertemplate=(
    "<b>%{location}</b><br>" +
    f"{vacc_col}: %{{z:.2f}}<br>" +   # ✅ Now correctly references z from Plotly, not Python
    "<br>".join([f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols)]) +
    "<extra></extra>"
),
customdata=df_choro[hover_cols].to_numpy()
)


    try:
        fig_vacc.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
        fig_vacc.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 400
    except Exception:
        pass

    fig_vacc.write_html("weekly_vaccination_choropleth.html")

# ========= 7) INTERACTIVE TREND LINES (NEW CASES) =========
if Y_CASES:
    fig_trend = px.line(
        df.sort_values("week_sort_key"),
        x="week_sort_key", y=Y_CASES, color="location",
        title="Weekly New COVID-19 Cases — Trends by Country"
    )
    fig_trend.update_layout(xaxis_title="Week", yaxis_title="New Cases")
    fig_trend.write_html("weekly_trend_cases.html")

# ========= 8) RISK ASSESSMENT (BAR) =========
risk = df.copy()

# convert required cols
for col in [Y_CASES, POS_COL, VACC_RAW_COL]:
    if col and col in risk.columns:
        risk[col] = pd.to_numeric(risk[col], errors="coerce")

def _norm_by_country(d, col):
    vals = d[col].astype(float)
    if vals.notna().sum() < 3:
        return pd.Series(np.nan, index=d.index)
    vmin, vmax = np.nanpercentile(vals, 5), np.nanpercentile(vals, 95)
    rng = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    return (vals - vmin) / rng

if Y_CASES:
    risk["cases_norm"] = risk.groupby("location", group_keys=False).apply(lambda d: _norm_by_country(d, Y_CASES))
else:
    risk["cases_norm"] = 0

if POS_COL:
    def _safe_z(s):
        std = s.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            std = 1.0
        mean = s.mean()
        return (s - mean) / std
    risk["positivity_z"] = risk.groupby("location")[POS_COL].transform(_safe_z)
else:
    risk["positivity_z"] = 0

if VACC_RAW_COL:
    risk["vacc_norm"] = risk.groupby("location", group_keys=False).apply(lambda d: _norm_by_country(d, VACC_RAW_COL))
else:
    risk["vacc_norm"] = 0

# weights
w_cases, w_pos, w_vacc = 0.5, 0.35, 0.15
risk["risk_score"] = (w_cases * risk["cases_norm"].fillna(0) +
                      w_pos   * risk["positivity_z"].fillna(0) -
                      w_vacc  * risk["vacc_norm"].fillna(0))

risk_recent = risk.sort_values("week_sort_key").groupby("location").tail(12)
risk_avg = (
    risk_recent.groupby("location", as_index=False)["risk_score"]
    .mean()
    .sort_values("risk_score", ascending=False)
)

fig_risk = px.bar(
    risk_avg,
    x="location",
    y="risk_score",
    color="risk_score",
    color_continuous_scale="RdYlGn_r",  # green=low, red=high
    title="Average Weekly Risk Score by Country (Recent 12 Weeks)",
    hover_data=["risk_score"]
)

fig_risk.update_layout(
    xaxis_title="Country",
    yaxis_title="Risk Score",
    xaxis_tickangle=-45,
    margin=dict(l=40, r=20, t=60, b=40),
    coloraxis_colorbar=dict(title="Risk Level")
)

fig_risk.write_html("weekly_risk_bar.html")

# ========= 9) DONE =========
print("✅ Module 1 complete. Interactive HTML files saved:")
print(" - weekly_bubble_timelapse.html")
print(" - weekly_distribution.html")
print(" - weekly_positivity_violin.html" if POS_COL else " - (violin skipped: positivity not found)")
print(" - weekly_vaccination_choropleth.html" if SIZE_COL else " - (choropleth skipped: vaccination columns missing)")
print(" - weekly_trend_cases.html" if Y_CASES else " - (trend skipped: new_cases missing)")
print(" - weekly_risk_bar.html")
