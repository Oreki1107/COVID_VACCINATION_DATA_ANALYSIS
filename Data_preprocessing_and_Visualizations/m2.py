import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

# =====================
# Load Dataset
# =====================
file_path = "C:\\Users\\vigne\\Downloads\\dataset\\risk_analysis_weekly.csv"
data = pd.read_csv(file_path)

# Convert year_week (like "2020-05") into a datetime (using Monday of that week)
# Extract the first date in the year_week range (everything before "/")
data["Date"] = pd.to_datetime(data["year_week"].str.split("/").str[0])


# Columns in your dataset:
# location, year_week, new_cases, positivity_rate, people_vaccinated, 
# cases_norm, positivity_z, vacc_norm, risk_score, risk_level

# =====================
# Radar Chart per Country
# =====================
metrics = ["risk_score", "cases_norm", "vacc_norm", "positivity_rate"]
countries = data["location"].unique()

radar_fig = go.Figure()

for country in countries:
    df_country = data[data["location"] == country]
    avg_values = [
        df_country["risk_score"].mean(),
        df_country["cases_norm"].mean(),
        df_country["vacc_norm"].mean(),
        df_country["positivity_rate"].mean()
    ]
    
    radar_fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=["Risk Score", "Infection Rate", "Vaccination Rate", "Positivity Rate"],
        fill='toself',
        name=country,
        hovertemplate='<b>%{theta}</b>: %{r:.2f}<extra>' + country + '</extra>'
    ))

radar_fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="Radar Chart - Country Comparison",
    showlegend=True
)
radar_fig.write_html("radar_chart.html")
print("Radar chart saved: radar_chart.html")



# --- utilities ---
def parse_week_start(week_str: str) -> pd.Timestamp:
    """
    Robustly parse the start date of a week from strings like:
      - 'YYYY-MM-DD/YYYY-MM-DD'  (preferred)
      - or fallback to 'YYYY-WW' (Monday as start), if such format exists
    """
    if isinstance(week_str, str) and "/" in week_str:
        # format 'YYYY-MM-DD/YYYY-MM-DD': take left date as the week start
        left = week_str.split("/")[0]
        return pd.to_datetime(left, errors="coerce")
    # Fallbacks: try to parse other patterns if they appear
    # e.g., '2024-37' interpreted as ISO week. Adjust if your data ever looks like this.
    # If unsure, coerce to NaT to avoid breaking the plot
    try:
        # Try ISO format if needed
        return pd.to_datetime(week_str, errors="coerce")
    except Exception:
        return pd.NaT

def parse_week_end(week_str: str) -> pd.Timestamp:
    """
    Extract the end-of-week date if 'YYYY-MM-DD/YYYY-MM-DD' is present,
    else fall back to the same start date (safe default).
    """
    if isinstance(week_str, str) and "/" in week_str:
        right = week_str.split("/")[1]
        return pd.to_datetime(right, errors="coerce")
    # If we lack an explicit end date, use start-of-week as a safe fallback
    return parse_week_start(week_str)

# --- assume `data` is your DataFrame already loaded with columns:
#     ['location', 'year_week', 'risk_score', ...]
#     If column names differ, adjust accordingly.

# Build a proper Date column (start-of-week) once and for all
data["Date"] = data["year_week"].apply(parse_week_start)
# Drop rows with invalid dates to avoid mixing types
data = data.dropna(subset=["Date"]).copy()

# Make sure data types are right
data["Date"] = pd.to_datetime(data["Date"], utc=False)

# Sort by Date for time-consistent plotting
data = data.sort_values(["location", "Date"])

# ----------------------------
# CUMULATIVE (EVOLVING) FRAMES
# ----------------------------
# We want the animation to "grow" over time.
# For each year_week (frame), include all rows with Date <= that week's end date.

# Map frame -> frame_end_date
frame_meta = (
    data[["year_week"]].drop_duplicates().assign(frame_end=lambda d: d["year_week"].apply(parse_week_end))
)
frame_meta = frame_meta.dropna(subset=["frame_end"]).sort_values("frame_end")

# Build an expanded DataFrame that accumulates data up to each frame's end
frames_expanded = []
for _, row in frame_meta.iterrows():
    frame_label = row["year_week"]
    frame_end = row["frame_end"]
    # include all history up to this end date
    subset = data.loc[data["Date"] <= frame_end].copy()
    subset["anim_frame"] = frame_label
    frames_expanded.append(subset)

data_anim = pd.concat(frames_expanded, ignore_index=True)

# Plotly Express line chart with animation on 'anim_frame'
line_fig = px.line(
    data_anim,
    x="Date",                    # <-- always datetime for x
    y="risk_score",
    color="location",
    animation_frame="anim_frame",
    hover_data={"location": True, "year_week": True, "Date": True, "risk_score": True},
    title="Multi-line Time Series of Risk Score (Evolving)",
)

# Improve visuals & interactions
line_fig.update_traces(mode="lines")  # or "lines+markers" if you like markers
line_fig.update_layout(
    xaxis=dict(
        type="date",             # force date axis
        title="Date",
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ]
        ),
    ),
    yaxis=dict(title="Risk Score", automargin=True),
    legend_title_text="Country",
    updatemenus=[  # play/pause controls
        dict(
            type="buttons",
            direction="left",
            x=0.1, y=0, xanchor="right", yanchor="top",
            pad=dict(r=10, t=70),
            showactive=False,
            buttons=[
                dict(
                    label="▶",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 500, "redraw": False},
                        "transition": {"duration": 500, "easing": "linear"},
                        "fromcurrent": True,
                        "mode": "immediate",
                    }],
                ),
                dict(
                    label="■",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "transition": {"duration": 0},
                        "mode": "immediate",
                    }],
                ),
            ],
        )
    ],
    sliders=[dict(
        currentvalue={"prefix": "week="},
        pad={"t": 60, "b": 10},
        len=0.9,
    )],
)

line_fig.write_html("multi_line_timeseries.html", include_plotlyjs="cdn")
print("Multi-line time series saved: multi_line_timeseries.html")
