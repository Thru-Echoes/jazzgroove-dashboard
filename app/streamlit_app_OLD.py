\
# ------------------------------------------------------------
# Import core libraries for I/O and computation.
# ------------------------------------------------------------
import os
# ------------------------------------------------------------
# Import pandas for data wrangling and CSV I/O.
# ------------------------------------------------------------
import pandas as pd
# ------------------------------------------------------------
# Import numpy for numeric utilities.
# ------------------------------------------------------------
import numpy as np
# ------------------------------------------------------------
# Import yaml to read config values.
# ------------------------------------------------------------
import yaml
# ------------------------------------------------------------
# Import plotly for interactive charts in Streamlit.
# ------------------------------------------------------------
import plotly.express as px
# ------------------------------------------------------------
# Import datetime for date arithmetic.
# ------------------------------------------------------------
from datetime import datetime
# ------------------------------------------------------------
# Import Streamlit for the dashboard UI.
# ------------------------------------------------------------
import streamlit as st

# ------------------------------------------------------------
# Define a helper to read YAML config and return as a dict.
# ------------------------------------------------------------
def load_config(path: str) -> dict:
    # Open and parse the YAML file to a Python dictionary.
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ------------------------------------------------------------
# Define a helper to safely load a CSV with optional date parsing.
# ------------------------------------------------------------
def load_csv(path: str, parse_dates=None) -> pd.DataFrame:
    # If the file exists at path, load it; otherwise return empty DataFrame.
    return pd.read_csv(path, parse_dates=parse_dates) if os.path.exists(path) else pd.DataFrame()

# ------------------------------------------------------------
# Define a function to compute a dynamic watchlist as of an arbitrary date.
# This mirrors the logic in your analysis notebooks.
# ------------------------------------------------------------
def compute_dynamic_watchlist(events_df: pd.DataFrame, hazard_df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    # If required inputs are missing, return an empty DataFrame immediately.
    if events_df.empty or hazard_df.empty:
        return pd.DataFrame()
    # Identify members who are active on the as_of date.
    active_mask = (events_df["Start"] <= as_of_date) & (events_df["End"] >= as_of_date)
    # Build the current roster from those active intervals.
    active = events_df.loc[active_mask, ["Name_or_Email", "Plan", "Start"]].copy()
    # Find each member's earliest record to derive Entry_Plan and First_Start.
    first_rows = events_df.sort_values(["Name_or_Email", "Start"]).groupby("Name_or_Email", as_index=False).first()
    # Rename columns to clear identifiers used throughout the app.
    first_rows = first_rows.rename(columns={"Plan": "Entry_Plan", "Start": "First_Start"})
    # Merge entry attributes onto the active roster.
    active = active.merge(first_rows[["Name_or_Email", "Entry_Plan", "First_Start"]], on="Name_or_Email", how="left")
    # Compute 1-indexed months since first start as of the selected date.
    active["Months_Since_First"] = (
        (as_of_date.year - active["First_Start"].dt.year) * 12
        + (as_of_date.month - active["First_Start"].dt.month)
        + 1
    ).clip(lower=1).astype(int)
    # Compute the hazard month (next month relative to current tenure).
    active["Hazard_Month"] = active["Months_Since_First"] + 1
    # Prepare the hazard table for joining by renaming Month to Hazard_Month.
    hz = hazard_df.rename(columns={"Month": "Hazard_Month"}).copy()
    # Defensively add a Plan column if missing (single-segment hazard).
    if "Plan" not in hz.columns:
        hz["Plan"] = "All Plans"
    # Ensure the base hazard_df we use for 3‑month joins also has Plan.
    if "Plan" not in hazard_df.columns:
        hazard_df = hazard_df.assign(Plan="All Plans")
    # Join hazards by entry plan and hazard month.
    out = active.merge(hz[["Plan", "Hazard_Month", "Avg_Hazard"]],
                       left_on=["Entry_Plan", "Hazard_Month"],
                       right_on=["Plan", "Hazard_Month"],
                       how="left").drop(columns=["Plan"])
    # Sort by hazard descending so riskiest members appear first.
    out = out.sort_values("Avg_Hazard", ascending=False)
    # ---- 3-month combined hazard (t, t+1, t+2) ----
    hz0 = hazard_df.rename(columns={"Month": "Hazard_Month_0", "Avg_Hazard": "h0"}).copy()
    hz1 = hazard_df.rename(columns={"Month": "Hazard_Month_1", "Avg_Hazard": "h1"}).copy()
    hz2 = hazard_df.rename(columns={"Month": "Hazard_Month_2", "Avg_Hazard": "h2"}).copy()
    out["Hazard_Month_0"] = out["Hazard_Month"].astype(int)
    out["Hazard_Month_1"] = out["Hazard_Month_0"] + 1
    out["Hazard_Month_2"] = out["Hazard_Month_0"] + 2
    out = out.merge(hz0[["Plan", "Hazard_Month_0", "h0"]], left_on=["Entry_Plan", "Hazard_Month_0"], right_on=["Plan", "Hazard_Month_0"], how="left").drop(columns=["Plan"])
    out = out.merge(hz1[["Plan", "Hazard_Month_1", "h1"]], left_on=["Entry_Plan", "Hazard_Month_1"], right_on=["Plan", "Hazard_Month_1"], how="left").drop(columns=["Plan"])
    out = out.merge(hz2[["Plan", "Hazard_Month_2", "h2"]], left_on=["Entry_Plan", "Hazard_Month_2"], right_on=["Plan", "Hazard_Month_2"], how="left").drop(columns=["Plan"])
    out[["h0", "h1", "h2"]] = out[["h0", "h1", "h2"]].fillna(0.0)
    out["Pred_3mo_Risk"] = 1.0 - (1.0 - out["h0"]) * (1.0 - out["h1"]) * (1.0 - out["h2"])
    # Return the full watchlist now enriched with 3-month predicted risk.
    return out

# ------------------------------------------------------------
# Define a helper to label actual churn within a multi-month window [eval_start, eval_end].
# We check whether the active interval that covers as_of ends inside the evaluation window.
# ------------------------------------------------------------
def label_churn_in_window(events_df: pd.DataFrame, as_of_date: pd.Timestamp, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> pd.DataFrame:
    # Identify rows where the member is active at the as_of date.
    active_mask = (events_df["Start"] <= as_of_date) & (events_df["End"] >= as_of_date)
    # Pull the End date of that active interval for each member.
    act_rows = events_df.loc[active_mask, ["Name_or_Email", "End"]].copy()
    # Flag whether that End date falls in the evaluation window (inclusive).
    act_rows["actual_churned_in_window"] = (act_rows["End"] >= eval_start) & (act_rows["End"] <= eval_end)
    # Keep the end date as Cancel_Date_in_window only if churn is True, else NA.
    act_rows["Cancel_Date_in_window"] = act_rows["End"].where(act_rows["actual_churned_in_window"], pd.NaT)
    # Collapse to one row per member by taking the earliest End (defensive if duplicates exist).
    act_rows = act_rows.sort_values(["Name_or_Email", "End"]).groupby("Name_or_Email", as_index=False).first()
    # Return the label table keyed by Name_or_Email.
    return act_rows[["Name_or_Email", "actual_churned_in_window", "Cancel_Date_in_window"]]

# ------------------------------------------------------------
# Define a function to run a multi-month backtest in-memory for the UI.
# ------------------------------------------------------------
def backtest_multimonth(events_df: pd.DataFrame,
                        hazard_df: pd.DataFrame,
                        hazard_as_of: pd.Timestamp,
                        eval_end: pd.Timestamp,
                        min_hazard: float = 0.0,
                        top_n: int = 1000000):
    # Compute the first day of the month following hazard_as_of.
    eval_start = (hazard_as_of + pd.offsets.MonthBegin(1)).normalize()
    # Build the watchlist for the chosen as_of month.
    wl = compute_dynamic_watchlist(events_df, hazard_df, hazard_as_of)
    # Apply a minimum hazard threshold if the user specified one.
    if min_hazard > 0:
        wl = wl[wl["Avg_Hazard"].fillna(0) >= float(min_hazard)]
    # Keep the first top_n rows as requested by the user.
    wl = wl.head(int(top_n)).copy()
    # Label which watchlist members actually churned in [eval_start, eval_end].
    labs = label_churn_in_window(events_df, hazard_as_of, eval_start, eval_end)
    # Merge labels into the watchlist so we can compute metrics and show details.
    res = wl.merge(labs, on="Name_or_Email", how="left")
    # Compute time-to-cancel in days (only for those who churned).
    res["days_to_cancel_from_as_of"] = (res["Cancel_Date_in_window"] - hazard_as_of).dt.days
    # Compute precision/recall by building the eligible population (active at as_of).
    active_mask = (events_df["Start"] <= hazard_as_of) & (events_df["End"] >= hazard_as_of)
    # Pull eligible End dates for those active at as_of.
    eligible = events_df.loc[active_mask, ["Name_or_Email", "End"]].copy()
    # Flag whether each eligible member churned within the window.
    eligible["churn_in_window"] = (eligible["End"] >= eval_start) & (eligible["End"] <= eval_end)
    # Collapse to one row per member using 'any' to aggregate multiple rows defensively.
    elig = eligible.groupby("Name_or_Email", as_index=False)["churn_in_window"].any()
    # Compute counts for metrics.
    total_churn = int(elig["churn_in_window"].sum())
    # Count true positives on the watchlist.
    tp = int(res["actual_churned_in_window"].fillna(False).sum())
    # Count flagged members.
    flagged = int(len(res))
    # Calculate precision safely.
    precision = (tp / flagged) if flagged > 0 else np.nan
    # Calculate recall safely.
    recall = (tp / total_churn) if total_churn > 0 else np.nan
    # Return the results DataFrame and a metrics dict for display.
    return res, {"eval_start": eval_start, "eval_end": eval_end, "flagged": flagged, "tp": tp, "total_churn": total_churn, "precision": precision, "recall": recall}

# ------------------------------------------------------------
# Set page configuration for a tidy, exec-friendly layout.
# ------------------------------------------------------------
st.set_page_config(page_title="Jazz Retention Dashboard", layout="wide")

# ------------------------------------------------------------
# Load configuration and the key CSV assets once at startup.
# ------------------------------------------------------------
cfg = load_config("config/config.yaml")
# ------------------------------------------------------------
# Resolve the expected file paths from config.
# ------------------------------------------------------------
events_path = cfg["paths"]["raw_events"]
# ------------------------------------------------------------
# Load the normalized events CSV with parsed date columns.
# ------------------------------------------------------------
events = load_csv(events_path, parse_dates=["Start", "End"])
# ------------------------------------------------------------
# Load survival curves for plotting survival tab; ignore if missing.
# ------------------------------------------------------------
surv = load_csv(cfg["paths"]["survival_csv"])
# ------------------------------------------------------------
# Load the pooled hazard table for heatmap + watchlist scoring.
# ------------------------------------------------------------
haz = load_csv(cfg["paths"]["hazard_csv"])

# ------------------------------------------------------------
# Header and quick sanity indicator for data readiness.
# ------------------------------------------------------------
st.title("🎷 Retention Rhythm Dashboard")
# ------------------------------------------------------------
# Show a small status badge if data is missing to guide setup.
# ------------------------------------------------------------
if events.empty or haz.empty:
    # Explain what is missing in a friendly way for non-technical users.
    st.warning("Data not ready. Please run the prep script first: `python scripts/01_prep_survival_and_risk.py`")
# ------------------------------------------------------------
# Build tabs for Survival, Cancellation Risk, and Backtesting.
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Survival", "⚠️ Cancellation Risk", "🧪 Backtest"])

# ------------------------------------------------------------
# Survival tab content: show survival curves if available.
# ------------------------------------------------------------
with tab1:
    # If survival curves exist, render a line chart with plan and cohort filters.
    if not surv.empty:
        # Provide simple selectors for plan and (optionally) cohort year.
        plans = sorted(surv["Plan"].dropna().unique()) if "Plan" in surv.columns else ["All"]
        # Let users choose one or more plans to display.
        sel_plans = st.multiselect("Select Plans", plans, default=plans[:2] if plans else [])
        # Filter the DataFrame based on selected plans (if the column exists).
        splot = surv[surv["Plan"].isin(sel_plans)] if "Plan" in surv.columns else surv.copy()
        # Create an interactive Plotly line figure for Retention over Month.
        fig = px.line(splot, x="Month", y="Retention", color="Plan", facet_col="Cohort Year", facet_col_wrap=3, title="Monthly Survival by Cohort")
        # Display the chart in the Streamlit app.
        st.plotly_chart(fig, use_container_width=True)
    # If survival data is missing, provide a gentle note.
    else:
        st.info("Survival curves will appear here after the prep script is run.")

# ------------------------------------------------------------
# Cancellation Risk tab content: dynamic watchlist for any chosen month.
# ------------------------------------------------------------
with tab2:
    # Draw a date input for the as-of date, defaulting to the config param.
    as_of_default = pd.to_datetime(cfg["params"]["as_of"]).date() if "as_of" in cfg["params"] else pd.to_datetime("2025-03-31").date()
    # Render the calendar widget for selecting the risk as-of date.
    sel_date = st.date_input("As‑of date for risk (pick month‑end)", value=as_of_default)
    # Convert the date to pandas Timestamp for downstream functions.
    as_of_ts = pd.to_datetime(sel_date)
    # Draw sliders/inputs for hazard threshold and top-N cap.
    col_a, col_b = st.columns(2)
    # Provide a slider for minimum hazard; default 0.0 (no filter).
    min_hz = col_a.slider("Minimum hazard to include", min_value=0.0, max_value=0.50, value=0.0, step=0.01, format="%.0f%%")
    # Provide a numeric input for top-N watchlist length.
    top_n = int(col_b.number_input("Top N", min_value=10, max_value=100000, value=500))
    # Compute the dynamic watchlist for the chosen date.
    wl = compute_dynamic_watchlist(events, haz, as_of_ts)
    # Apply filters from the UI.
    wl = wl[wl["Avg_Hazard"].fillna(0) >= float(min_hz)]
    # Keep only the top-N rows.
    wl = wl.head(top_n).copy()
    # Show a title that updates with the chosen month.
    st.subheader(f"High‑risk for {as_of_ts.strftime('%B %Y')}")
    # Display the table to users; hide the index for cleanliness.
    st.dataframe(wl[["Name_or_Email", "Entry_Plan", "Months_Since_First", "Hazard_Month", "Avg_Hazard", "Pred_3mo_Risk"]])

# ------------------------------------------------------------
# Backtest tab content: supports a custom multi-month window (e.g., Feb–Apr 2025).
# ------------------------------------------------------------
with tab3:
    # Draw a two-column layout for inputs.
    c1, c2, c3 = st.columns([1, 1, 1])
    # The hazard-as-of date defaults to the config value but can be changed.
    hazard_as_of_default = pd.to_datetime(cfg["params"]["as_of"]).date() if "as_of" in cfg["params"] else pd.to_datetime("2025-01-31").date()
    # Create the date picker for hazard-as-of.
    hazard_as_of = c1.date_input("Hazard as‑of (choose month‑end)", value=hazard_as_of_default)
    # Create the date picker for evaluation window end date.
    eval_end = c2.date_input("Evaluation end (inclusive)", value=pd.to_datetime("2025-04-30").date())
    # Provide optional thresholds to focus on the riskiest members.
    min_hz_bt = float(c3.number_input("Min hazard (0–1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01))
    # Provide a top-N cap for the backtest watchlist.
    top_n_bt = int(st.number_input("Top N to evaluate", min_value=10, max_value=200000, value=5000, step=10))
    # Add a button to run the backtest to prevent constant recomputation.
    if st.button("Run backtest"):
        # Convert input dates to pandas Timestamps.
        haz_ts = pd.to_datetime(hazard_as_of)
        # Convert evaluation end to Timestamp as well.
        eval_end_ts = pd.to_datetime(eval_end)
        # Execute the backtest using in-memory events and hazards.
        bt_res, bt_metrics = backtest_multimonth(events, haz, haz_ts, eval_end_ts, min_hazard=min_hz_bt, top_n=top_n_bt)
        # Lay out headline metrics in three columns for quick scanning.
        m1, m2, m3, m4 = st.columns(4)
        # Show precision among flagged in percentage format.
        m1.metric("Precision (flagged)", f"{bt_metrics['precision']:.1%}" if not np.isnan(bt_metrics['precision']) else "—")
        # Show recall across all churners in the window.
        m2.metric("Recall (window)", f"{bt_metrics['recall']:.1%}" if not np.isnan(bt_metrics['recall']) else "—")
        # Show total flagged count.
        m3.metric("Flagged", f"{bt_metrics['flagged']:,}")
        # Show total churners observed in the evaluation window.
        m4.metric("Churners in window", f"{bt_metrics['total_churn']:,}")
        # Add a header for the detailed results table.
        st.subheader(f"Results • as_of {haz_ts.date()} • window [{bt_metrics['eval_start'].date()} → {bt_metrics['eval_end'].date()}]")
        # Display the detailed table with whether each flagged member churned and when.
        st.dataframe(bt_res[["Name_or_Email", "Entry_Plan", "Months_Since_First", "Avg_Hazard", "actual_churned_in_window", "Cancel_Date_in_window", "days_to_cancel_from_as_of"]])
        # Build a simple calibration plot: observed churn rate by hazard bin.
        # First, create bins and labels on the fly.
        bins = [0.0, 0.05, 0.10, 0.20, 1.0]
        labels = ["0–5%", "5–10%", "10–20%", "20%+"]
        bt_res["hazard_bin"] = pd.cut(bt_res["Pred_3mo_Risk"].fillna(0.0), bins=bins, labels=labels, include_lowest=True, right=True)
        calib = (bt_res.groupby("hazard_bin")["actual_churned_in_window"].mean().reindex(labels).reset_index())
        # Create a bar chart for the calibration curve.
        fig_c = px.bar(calib, x="hazard_bin", y="actual_churned_in_window", title="Calibration: observed churn rate by hazard bin", labels={"hazard_bin": "Hazard bin", "actual_churned_in_window": "Observed churn rate"})
        # Render the chart.
        st.plotly_chart(fig_c, use_container_width=True)
        # Offer CSV downloads for the results and the calibration table.
        st.download_button("Download detailed results CSV", data=bt_res.to_csv(index=False), file_name=f"backtest_{haz_ts.date()}_{bt_metrics['eval_end'].date()}_results.csv", mime="text/csv")
        # Provide a separate download for the calibration data.
        st.download_button("Download calibration CSV", data=calib.to_csv(index=False), file_name=f"backtest_{haz_ts.date()}_{bt_metrics['eval_end'].date()}_calibration.csv", mime="text/csv")
