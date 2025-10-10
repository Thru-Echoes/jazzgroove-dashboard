
# ------------------------------------------------------------
# Import core libraries for I/O and computation.
# ------------------------------------------------------------
import os
import math 
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

# Above each line: explain the line below
# Define a function that takes the survival dataframe currently being plotted
def compute_thinning_month(surv_df, min_at_risk=30):
    # Ensure we have the columns we need
    if not {"Month", "At_Risk"}.issubset(surv_df.columns):
        # If columns are missing, return None (no fence)
        return None
    # Group by Month and sum At_Risk across cohorts/plans currently filtered
    month_at_risk = (
        surv_df.groupby("Month", as_index=True)["At_Risk"]
               .sum()
               .sort_index()
    )
    # Find the last month where At_Risk is still adequate
    supported = month_at_risk[month_at_risk >= min_at_risk]
    # If nothing is supported, skip
    if supported.empty:
        return None
    # Place the fence just after the last adequately supported month
    # (e.g., if last supported is 16, the fence appears between 16 and 17)
    return int(supported.index.max()) + 0.5

# -------------------------------
# Helper: draw the fence on a Plotly figure (line chart or heatmap)
# -------------------------------

# Define a function to add a dashed red vline and soft shading to a Plotly figure
def add_coverage_fence(fig, thinning_x, max_x):
    # If there is no fence to draw, return immediately
    if thinning_x is None:
        return fig
    # Add a red dashed vertical line at the thinning boundary
    fig.add_vline(
        x=thinning_x,
        line_width=2,
        line_dash="dash",
        line_color="red"
    )
    # Add a light red shaded region beyond the boundary to the chart end
    fig.add_vrect(
        x0=thinning_x, x1=max_x + 0.5,
        fillcolor="red",
        opacity=0.06,
        line_width=0,
    )
    # Annotate the fence so viewers know what it means
    fig.add_annotation(
        x=thinning_x,
        y=1.04,            # position at top of plotting area
        yref="paper",      # reference the vertical paper coordinate
        text="Data thins beyond here (Apr 29, 2025 cutoff)",
        showarrow=False,
        font=dict(color="red", size=11),
        xanchor="left"
    )
    # Return the modified figure
    return fig

# Draw fixed per‑cohort vertical "coverage fences" across a faceted figure
def add_fixed_cohort_fences_OLD(fig, cohort_order, fence_map, max_x, cols=3, shade=True):
    """
    fig          : Plotly figure created with px.line(..., facet_col='Cohort Year', facet_col_wrap=cols)
    cohort_order : ordered list of Cohort Year values *in the same dtype the figure uses*
    fence_map    : dict {year_int: month_int or None}  e.g., {2021: 41, 2022: 29, ...}
    max_x        : right limit to shade to (usually max Month in the plotted data)
    cols         : number of facet columns (must match facet_col_wrap used in px.line)
    shade        : if True, light red shading appears to the right of each fence
    """
    if not cohort_order:
        return fig

    for i, label in enumerate(cohort_order):
        # Convert label to int for lookup; ignore if not convertible (defensive)
        try:
            yr = int(label)
        except Exception:
            continue

        x = fence_map.get(yr)
        if x is None:
            continue  # no fence for this cohort

        # Compute subplot position (row/col) given facet_col_wrap=cols
        row = (i // cols) + 1
        col = (i % cols) + 1

        fig.add_vline(
            x=x,
            line_width=2,
            line_dash="dash",
            line_color="red",
            row=row,
            col=col,
        )
        if shade:
            fig.add_vrect(
                x0=x, x1=max_x + 0.5,
                fillcolor="red",
                opacity=0.06,
                line_width=0,
                row=row,
                col=col,
            )

    # A single annotation at the top of the full figure
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=1.08,
        #text="Red dashed lines mark where results may thin due to Apr 29, 2025 cutoff.",

        text = "",
        showarrow=False,
        font=dict(color="red", size=11),
    )
    return fig

def add_fixed_cohort_fences(fig, fence_map, max_x, shade=True, facet_col_name="Cohort Year"):
    """
    Draw fixed, per‑cohort vertical fences on a faceted figure made by px.line(..., facet_col=facet_col_name).

    fig           : Plotly figure
    fence_map     : dict {year_int: month_int or None}
                    e.g. {2021: 41, 2022: 29, 2023: 17, 2024: 5, 2025: 1}
                    Use None for cohorts that should have no fence.
    max_x         : right limit to shade to (e.g., max tenure month shown)
    shade         : if True, lightly shade to the right of the fence
    facet_col_name: the name of the facet column (default "Cohort Year")
    """
    # Find the facet title annotations like "Cohort Year=2025"
    anns = [
        a for a in getattr(fig.layout, "annotations", []) 
        if isinstance(getattr(a, "text", None), str)
        and a.text.startswith(f"{facet_col_name}=")
    ]
    if not anns:
        return fig  # no facets found, nothing to draw

    # Determine the grid by the actual positions of the titles (paper coords)
    xs = sorted({float(a.x) for a in anns})
    ys = sorted({float(a.y) for a in anns}, reverse=True)  # top row has largest y

    # Map the *label* shown in the title to its (row, col)
    title_to_rowcol = {}
    for a in anns:
        label_str = a.text.split("=", 1)[1].strip()  # "2025"
        col = xs.index(float(a.x)) + 1
        row = ys.index(float(a.y)) + 1
        title_to_rowcol[label_str] = (row, col)

    # Add a fence for each cohort that has an entry in fence_map
    for yr, x in fence_map.items():
        if x is None:
            continue
        key = str(yr)  # facet titles are strings
        if key not in title_to_rowcol:
            continue  # this cohort panel isn't present (filtered out)
        row, col = title_to_rowcol[key]
        fig.add_vline(x=x, line_width=2, line_dash="dash", line_color="red", row=row, col=col)
        if shade:
            fig.add_vrect(
                x0=x, x1=max_x + 0.5,
                fillcolor="red", opacity=0.06, line_width=0,
                row=row, col=col,
            )

    # One overall note to explain the fences
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=1.08,
        #text="Red dashed lines mark where results thin due to Apr 29, 2025 cutoff.",
        text = "",
        showarrow=False, font=dict(color="red", size=11),
    )
    return fig



# ------------------------------------------------------------
# Define a function to compute a dynamic watchlist as of an arbitrary date.
# This mirrors the logic in your analysis notebooks.
# ------------------------------------------------------------
def compute_dynamic_watchlist(events_df: pd.DataFrame, hazard_df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Computes a dynamic watchlist for an arbitrary calendar month. 
        Robust to missing dates and uses nullable Int64 dtypes."""
    
    # If required inputs are missing, return an empty DataFrame immediately.
    if events_df.empty or hazard_df.empty:
        return pd.DataFrame()
    
    # Ensure only allowed plans are considered.
    if "Plan" in events_df.columns:
        events_df = events_df[events_df["Plan"].isin(ALLOWED_PLANS)].copy()
    if "Plan" in hazard_df.columns:
        hazard_df = hazard_df[hazard_df["Plan"].isin(ALLOWED_PLANS)].copy()

    # Guarantee datetime types and drop rows with missing or malformed dates early.
    events_df["Start"] = pd.to_datetime(events_df["Start"], errors="coerce")
    events_df["End"]   = pd.to_datetime(events_df["End"],   errors="coerce")
    events_df = events_df.dropna(subset=["Start", "End"])

    # Filter to members active at the selected as_of_date (Start <= as_of <= End).
    active_mask = (events_df["Start"] <= as_of_date) & (events_df["End"] >= as_of_date)
    active = events_df.loc[active_mask, ["Name_or_Email", "Plan", "Start"]].copy()

    # If nobody is active as of that date, return empty to avoid downstream errors.
    if active.empty:
        return pd.DataFrame()

    # Build entry plan and first start by taking the earliest record per member.
    # Sort to make .first() deterministic.
    first_rows = (
        events_df.sort_values(["Name_or_Email", "Start"])
                 .groupby("Name_or_Email", as_index=False)
                 .first()
                 .rename(columns={"Plan": "Entry_Plan", "Start": "First_Start"})
    )

    # Merge entry plan + first start onto the active roster.
    active = active.merge(first_rows[["Name_or_Email", "Entry_Plan", "First_Start"]],
                          on="Name_or_Email", how="left")

    # Drop rows where First_Start is missing (cannot compute tenure without it).
    active = active.dropna(subset=["First_Start"])

    # FIXME: remove debugger once in production mode 
    #import pdb 
    #pdb.set_trace()

    # Convert First_Start to datetime explicitly (defensive; should already be datetime).
    active["First_Start"] = pd.to_datetime(active["First_Start"], errors="coerce")

    # Compute 1-indexed months since first start using monthly periods (robust to calendar edge cases).
    # as_of_period - first_start_period gives integer month diff; +1 to make it 1-indexed.
    #as_of_period = pd.Period(as_of_date, freq="M")
    #first_period = active["First_Start"].dt.to_period("M")
    #active["Months_Since_First"] = (as_of_period - first_period).astype("Int64") + 1

    # --- robust month difference without Period subtraction ---
    # 1) extract year and month from as_of and First_Start
    asof_y = as_of_date.year
    asof_m = as_of_date.month

    # 2) compute 1-indexed months since first start
    msf = (
        (asof_y - active["First_Start"].dt.year) * 12
        + (asof_m - active["First_Start"].dt.month)
        + 1
    )

    # 3) coerce to nullable Int64 and clamp to ≥ 1
    active["Months_Since_First"] = (
        pd.to_numeric(msf, errors="coerce")
        .astype("Int64")
        .clip(lower=1)
    )


    # Clamp months to be at least 1 (in case of any negative artifacts).
    #active["Months_Since_First"] = active["Months_Since_First"].clip(lower=1)

    # Next-month hazard lookup index (nullable Int64 to avoid casting errors).
    active["Hazard_Month"] = active["Months_Since_First"] + 1
    active["Hazard_Month"] = active["Hazard_Month"].astype("Int64")

    # Prepare hazard table: ensure it has a Plan column for joins; if not, create a dummy "All Plans".
    hz = hazard_df.rename(columns={"Month": "Hazard_Month"}).copy()
    if "Plan" not in hz.columns:
        hz["Plan"] = "All Plans"
        # Also label the active side to match this single-segment hazard.
        active["Entry_Plan"] = "All Plans"

    # RENAME right-hand side 'Plan' column before merging (since there will then be 2x "Plan" columns):
    hz = hz.rename(columns = {"Plan" : "Hazard_Plan"})

    # FIXME: remove debugger after done with dev 
    #import pdb 
    #pdb.set_trace()

    # Join the *one-month* hazard by (Entry_Plan, Hazard_Month).
    out = active.merge(hz[["Hazard_Plan", "Hazard_Month", "Avg_Hazard"]],
                       left_on=["Entry_Plan", "Hazard_Month"],
                       right_on=["Hazard_Plan", "Hazard_Month"],
                       how="left", validate = "many_to_one") # .drop(columns=["Plan"], errors = "ignore")

    # ---- 3-month combined hazard (t, t+1, t+2) for multi-month backtests and prioritization ----
    # Build three offset months for the consecutive windows.
    out["Hazard_Month_0"] = out["Hazard_Month"]
    out["Hazard_Month_1"] = out["Hazard_Month_0"] + 1
    out["Hazard_Month_2"] = out["Hazard_Month_0"] + 2

    # Create hazard copies keyed by (Plan, Month) and rename Avg_Hazard to h0/h1/h2.
    hz0 = hazard_df.rename(columns={"Month": "Hazard_Month_0", "Avg_Hazard": "h0", "Plan" : "Hazard_Plan"}).copy()
    hz1 = hazard_df.rename(columns={"Month": "Hazard_Month_1", "Avg_Hazard": "h1", "Plan" : "Hazard_Plan"}).copy()
    hz2 = hazard_df.rename(columns={"Month": "Hazard_Month_2", "Avg_Hazard": "h2", "Plan" : "Hazard_Plan"}).copy()

    # Ensure Plan exists on those tables; if not, align them to the "All Plans" label.
    #if "Plan" not in hz0.columns:
    #    hz0["Plan"] = "All Plans"
    #if "Plan" not in hz1.columns:
    #    hz1["Plan"] = "All Plans"
    #if "Plan" not in hz2.columns:
    #    hz2["Plan"] = "All Plans"
    #if "Entry_Plan" not in out.columns:
    #    out["Entry_Plan"] = "All Plans"

    # If Plan missing, create a single-segment label on each table (and align left)
    for hzX in (hz0, hz1, hz2):
        if "Hazard_Plan" not in hzX.columns:
            hzX["Hazard_Plan"] = "All Plans"
    if "Entry_Plan" not in out.columns:
        out["Entry_Plan"] = "All Plans"

    
    # FIXME: remove debugger after done with dev 
    #import pdb 
    #pdb.set_trace()

    # If the right tables came from a hazard CSV without plan segmentation,
    # add a single-segment label so the joins can proceed.
    for hzX in (hz0, hz1, hz2):
        if "Hazard_Plan" not in hzX.columns:
            hzX["Hazard_Plan"] = "All Plans"

    # If the left side doesn't have Entry_Plan yet, align it too:
    if "Entry_Plan" not in out.columns:
        out["Entry_Plan"] = "All Plans"


    # --- ensure no leftover right-key column in `out` before first h-merge ---
    if "Hazard_Plan" in out.columns:
        out = out.drop(columns=["Hazard_Plan"])

    # --- h0 (t) ---
    hz0_ = hz0.rename(columns={"Hazard_Plan": "HzPlan0"})  # give right key a unique name
    out = (
        out.merge(
            hz0_[["HzPlan0", "Hazard_Month_0", "h0"]],
            left_on=["Entry_Plan", "Hazard_Month_0"],
            right_on=["HzPlan0", "Hazard_Month_0"],
            how="left",
            validate="many_to_one",
        )
        .drop(columns=["HzPlan0"])  # drop the unique right key we just used
    )

    # --- h1 (t+1) ---
    hz1_ = hz1.rename(columns={"Hazard_Plan": "HzPlan1"})
    out = (
        out.merge(
            hz1_[["HzPlan1", "Hazard_Month_1", "h1"]],
            left_on=["Entry_Plan", "Hazard_Month_1"],
            right_on=["HzPlan1", "Hazard_Month_1"],
            how="left",
            validate="many_to_one",
        )
        .drop(columns=["HzPlan1"])
    )

    # --- h2 (t+2) ---
    hz2_ = hz2.rename(columns={"Hazard_Plan": "HzPlan2"})
    out = (
        out.merge(
            hz2_[["HzPlan2", "Hazard_Month_2", "h2"]],
            left_on=["Entry_Plan", "Hazard_Month_2"],
            right_on=["HzPlan2", "Hazard_Month_2"],
            how="left",
            validate="many_to_one",
        )
        .drop(columns=["HzPlan2"])
    )



    # Fill missing hazards with 0.0 to keep the product well-defined.
    out[["Avg_Hazard"]] = out[["Avg_Hazard"]].fillna(0.0)
    out[["h0", "h1", "h2"]] = out[["h0", "h1", "h2"]].fillna(0.0)

    # Compute the 3-month cumulative risk.
    out["Pred_3mo_Risk"] = 1.0 - (1.0 - out["h0"]) * (1.0 - out["h1"]) * (1.0 - out["h2"])

    # Sort by one-month hazard (primary) and then three-month risk (secondary).
    out = out.sort_values(["Avg_Hazard", "Pred_3mo_Risk"], ascending=False)

    # Return the enriched watchlist for the chosen as_of date.
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
# Restrict all analysis to these plans only (as requested).
# ------------------------------------------------------------
ALLOWED_PLANS = {"premium-plus", "premium-plus-monthly"}
# ------------------------------------------------------------
# Resolve the expected file paths from config.
# ------------------------------------------------------------
events_path = cfg["paths"]["raw_events"]
# ------------------------------------------------------------
# Load the normalized events CSV with parsed date columns.
# ------------------------------------------------------------
events = load_csv(events_path, parse_dates=["Start", "End"])
# ------------------------------------------------------------
# Filter events to only allowed plans.
# ------------------------------------------------------------
if not events.empty and "Plan" in events.columns:
    events = events[events["Plan"].isin(ALLOWED_PLANS)].copy()
# ------------------------------------------------------------
# Load survival curves for plotting survival tab; ignore if missing.
# ------------------------------------------------------------
surv = load_csv(cfg["paths"]["survival_csv"]) 
# ------------------------------------------------------------
# Filter survival to only allowed plans (if Plan column exists).
# ------------------------------------------------------------
if not surv.empty and "Plan" in surv.columns:
    surv = surv[surv["Plan"].isin(ALLOWED_PLANS)].copy()
# ------------------------------------------------------------
# Load the pooled hazard table for heatmap + watchlist scoring.
# ------------------------------------------------------------
haz = load_csv(cfg["paths"]["hazard_csv"]) 
# ------------------------------------------------------------
# Filter hazard to only allowed plans (if Plan column exists).
# ------------------------------------------------------------
if not haz.empty and "Plan" in haz.columns:
    haz = haz[haz["Plan"].isin(ALLOWED_PLANS)].copy()

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
        # Show an info banner about data coverage (optional but recommended)
        #st.info("Data coverage ends on **Apr 29, 2025**. Lines stop where follow-up is incomplete (right-censored).")
        st.warning("This is a **beta (development)** dashboard. Expect rough edges. The goal is to quickly surface patterns \
so we can decide what’s real customer behavior vs. analysis artifacts—and whether this is worth taking to production.")

        #st.markdown(
        #"""
        #**What the lines mean & why late months can look worse**  
        #Our data stops on **April 29, 2025**. After that, results are **incomplete**, not extra cancellations.  
        #We draw lines up to 48 months to compare cohorts (2019–2025), but later months—especially for people who joined in 2024–2025—can look lower simply because we can’t observe them beyond April 2025. Read those later points as “what we’ve seen so far” rather than final outcomes.  
        #Each cohort (e.g., “2023”) includes everyone who joined that year; a value like **0.38 at Month 20** means **38% of 2023 joiners were still active 20 months after they joined**.

        #**Why drops can appear near Month 14, and what we discovered**  
        #We count the **start month as Month 1**. If someone starts in Oct‑2023, they’re active through Oct‑2024 (Month 13). If they don’t renew then, the drop shows in **Month 14** (Nov‑2024).  
        #The heatmap shows the same story at a glance: **darker cells = bigger month‑to‑month drop**.  
        #Early insight from this beta: **annual (Premium‑Plus) lines dip before Month 12** in every cohort—sometimes around **Month 3**. That means some “annual” members are leaving early (e.g., failed payments, plan changes/refunds, or data mapping quirks). This likely went unnoticed and is a high‑value area to investigate.
        #    """
        #)

         # Short, non-technical intro to the Survival tab
        st.markdown(
            """
            ### What you’re seeing
            - Lines show **what share of members remain active** after Month 1, 2, 3, …
            - Lines **stop/fade** when we don’t have enough months of data.

            ### How to read it
            - **Higher lines = better retention**.
            - **Drops** mark sensitive moments (e.g., trial end, renewal).

            ### How to use it
            - Use **Plan** and **Cohort Year** filters to compare groups.
            - The **hazard heatmap** below shows **where churn is hottest** (darker = higher monthly drop-off).
            """
        )


        # Provide simple selectors for plan and (optionally) cohort year.
        plans = sorted(surv["Plan"].dropna().unique()) if "Plan" in surv.columns else ["All"]
        # Let users choose one or more plans to display.
        sel_plans = st.multiselect("Select Plans", plans, default=plans[:2] if plans else [])
        # Filter the DataFrame based on selected plans (if the column exists).
        splot = surv[surv["Plan"].isin(sel_plans)] if "Plan" in surv.columns else surv.copy()

        # Create an interactive Plotly line figure for Retention over Month.
        # We’ll keep the facet order stable but let the fence helper discover positions.
        if pd.api.types.is_numeric_dtype(surv["Cohort Year"]):
            cohort_order = (
                splot["Cohort Year"].dropna().astype(int).sort_values().unique().tolist()
            )
        else:
            cohort_order = sorted(
                splot["Cohort Year"].dropna().astype(str).unique().tolist(),
                key=lambda x: int(x)
            )

        fig = px.line(
            splot,
            x="Month",
            y="Retention",
            color="Plan",
            facet_col="Cohort Year",
            facet_col_wrap=3,
            category_orders={"Cohort Year": cohort_order},
            title="Monthly Survival by Cohort",
        )

        # Your fixed per‑cohort fences (exactly as requested)
        FENCE_BY_COHORT = {
            2019: None,  # no line
            2020: None,  # no line
            2021: 41,
            2022: 29,
            2023: 17,
            2024: 5,
            #2025: 1,
            2025: None
        }

        # Determine a reasonable right bound for the shading
        max_x = int(splot["Month"].max()) if ("Month" in splot.columns and not splot.empty) else 48

        # Add the fences into the *correct* facet panels by reading the figure’s titles
        fig = add_fixed_cohort_fences(
            fig,
            fence_map=FENCE_BY_COHORT,
            max_x=max_x,
            shade=True,
            facet_col_name="Cohort Year",
        )
        fence_x = 1  # month where you want the dashed line for 2025
        fig.add_vline(x=fence_x, line_width=2, line_dash="dash", line_color="red",
                    row=1, col=1)
        fig.add_vrect(x0=fence_x, x1=int(splot["Month"].max()) + 0.5,
                    fillcolor="red", opacity=0.06, line_width=0,
                    row=1, col=1)
        
        fence_x = 41  # month where you want the dashed line for 2021
        fig.add_vline(x=fence_x, line_width=2, line_dash="dash", line_color="red",
                    row=3, col=3)
        fig.add_vrect(x0=fence_x, x1=int(splot["Month"].max()) + 0.5,
                    fillcolor="red", opacity=0.06, line_width=0,
                    row=3, col=3)


        # ADD RED DASHED LINES HERE?? 

        # Let the user decide how thin is "too thin" (N people still under observation)
        
        # Display the chart in the Streamlit app.
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning("Red dashed lines mark where results may thin due to Apr 29, 2025 cutoff.")
        # ADD TEXT HERE 
        st.info("Data coverage ends on **Apr 29, 2025**. Lines stop where follow-up is incomplete (right-censored; red dashed lines).")
        st.markdown(
            """
            This issue has to do with how Chargebee delivers and stores data. Oliver had to cutoff all subscriptions by April 29, 2025 in order to reliably analyze our data.
            These plots go up to 48 months as a way to start looking at long-term patterns. However, late-reaching months (especially for the end of cohort 2021, latter portion of cohort 2022, second half of 2023, most of cohort 2024, and all of cohort 2025) will look significantly lower simply because we cannot keep following them past April 2025.

            """
        )

        # ADD TEXT HERE...
        st.markdown("**NOTE:** The first month someone started, let's say October 2022, is their month 1. If that person has a subscription until October 2023, they have 13 months of a subscription. If that person does not renew the next month, *i.e. November 2023*, they have churned at month 14.")
        
        st.warning("Bottom line: long-term brand loyalty is strong.")
        st.markdown(
            """
            When we compare long-term loyalty, we see that generally 44 - 50% of premium-plus (annual) members renew after their first year (i.e. at month 14). However, another 12 months later (i.e. at month 26) that number usually only drops 5 - 7% to 39 - 45% retention. That means just shy of half of premium-plus members are expected to renew after 2 years of their subscriptions. That is a very strong signal.
            """
        )

        st.warning("Bottom line: this dashboard shows us previously undetected patterns in our data that are likely going to lead to high-value fixes / decisions.")
        st.markdown(
            """
            We see that each cohort has premium-plus (annual) subscribers that are churning before their first 12 months of a membership. That is unexpected and is a consistent pattern that has been going unnoticed for years. They may had had payment failures, unexpected plan changes, or is an artifact of data mapping issues at the Chargebee level). This is something that requires further investigation and is a high-value fix.
This dashboard is in beta mode (development mode) and likely has unpolished issues here and there. It is meant to show initial patterns in the data and generate discussion about the pros, cons, and features desired for this app to yield the highest value.
            """
        )

        # ----------------------------

        # Also show a hazard heatmap by tenure month if hazard data is present.
        if 'Month' in haz.columns and 'Avg_Hazard' in haz.columns:
            import plotly.express as px
            hsel = haz.copy()
            hsel = hsel[hsel['Plan'].isin(sel_plans)] if 'Plan' in hsel.columns else hsel
            fig_hz = px.density_heatmap(hsel, x='Month', y='Plan', z='Avg_Hazard', color_continuous_scale='Reds',
                                        title='Hazard Heatmap by Tenure Month', nbinsx=24)
            
            # Reuse the survival-based thinning boundary so both plots align
            #fig_hz = add_coverage_fence(fig_hz, thinning_x, max_x)
            st.plotly_chart(fig_hz, use_container_width=True)
    # If survival data is missing, provide a gentle note.
    else:
        st.info("Survival curves will appear here after the prep script is run.")

# ------------------------------------------------------------
# Cancellation Risk tab content: dynamic watchlist for any chosen month.
# ------------------------------------------------------------
with tab2:
    # Short, non-technical intro to the Risk tab
    st.markdown(
        """
        ### What you’re seeing
        - A **ranked list** of current members who look most at risk of canceling.
        - **Hazard (1-mo)** = chance of canceling **next month**.
        - **3-month risk** = chance of canceling **in any of the next 3 months**.

        ### How to read it
        - Bigger numbers = **higher risk**, prioritize outreach there.
        - **Months since first** = how long they’ve been with us.

        ### How to use it
        - Pick the **as-of date** (the month you care about).
        - Use **Min 1-mo hazard** and **Min 3-mo risk** to focus on the riskiest members.
        - **Top N** limits the list to a manageable outreach target.
        - Click **Download CSV** to hand to the outreach team.
        """
    )

    # Draw a date input for the as-of date, defaulting to the config param.
    as_of_default = pd.to_datetime(cfg["params"]["as_of"]).date() if "as_of" in cfg["params"] else pd.to_datetime("2025-03-31").date()
    # Render the calendar widget for selecting the risk as-of date.
    sel_date = st.date_input("As‑of date for risk (pick month‑end)", value=as_of_default)
    # Convert the date to pandas Timestamp for downstream functions.
    as_of_ts = pd.to_datetime(sel_date)
    # Draw sliders/inputs for hazard thresholds and top-N cap.
    col_a, col_b, col_c = st.columns(3)
    # Provide a slider for minimum *one-month* hazard; default 0.0 (no filter).
    min_hz = col_a.slider("Min 1-mo hazard", min_value=0.0, max_value=0.50, value=0.0, step=0.01, format="%.0f%%")
    # Provide a slider for minimum *3-month* risk; default 0.0 (no filter).
    min_risk3 = col_b.slider("Min 3-mo risk", min_value=0.0, max_value=0.80, value=0.0, step=0.01, format="%.0f%%")
    # Provide a numeric input for top-N watchlist length.
    top_n = int(col_c.number_input("Top N", min_value=10, max_value=100000, value=500))
    # Compute the dynamic watchlist for the chosen date.
    wl = compute_dynamic_watchlist(events, haz, as_of_ts)
    # Apply filters from the UI.
    wl = wl[wl["Avg_Hazard"].fillna(0) >= float(min_hz)]
    wl = wl[wl["Pred_3mo_Risk"].fillna(0) >= float(min_risk3)]
    # Keep only the top-N rows.
    wl = wl.head(top_n).copy()
    # Show KPIs: active base and expected cancels next month among flagged.
    k1, k2, k3 = st.columns(3)
    active_base = int(((events["Start"] <= as_of_ts) & (events["End"] >= as_of_ts)).sum()) if not events.empty else 0
    expected_next_month = float(wl["Avg_Hazard"].fillna(0).sum())
    k1.metric("Active base (as-of)", f"{active_base:,}")
    k2.metric("Flagged shown", f"{len(wl):,}")
    k3.metric("Expected cancels next month (sum h)", f"{expected_next_month:.1f}")
    # Show a title that updates with the chosen month.
    st.subheader(f"High‑risk for {as_of_ts.strftime('%B %Y')}")
    # Display the table to users; hide the index for cleanliness.
    st.dataframe(wl[["Name_or_Email", "Entry_Plan", "Months_Since_First", "Hazard_Month", "Avg_Hazard", "Pred_3mo_Risk"]])
    # Offer a CSV download.
    st.download_button("Download high‑risk CSV", data=wl.to_csv(index=False), file_name=f"high_risk_{as_of_ts.date()}.csv", mime="text/csv")

# ------------------------------------------------------------
# Backtest tab content: supports a custom multi-month window (e.g., Feb–Apr 2025).
# ------------------------------------------------------------
with tab3:
    st.warning("IMPORTANT! The calculations in this tab have slightly incomplete code. As such, results are not final. View this tab as a rough prototype used to determine if the final version of this tool would likely be of high-value.")
    # Short, non-technical intro to the Backtest tab
    st.markdown(
        """
        ### What you’re seeing
        - We freeze the risk list **as of a past month**, then check who **actually canceled** afterward.
        - **Precision** = of those we flagged, how many actually canceled.
        - **Recall** = of everyone who canceled, how many we flagged.
        - The bar chart checks that **higher risk bins** canceled more often.

        ### How to use it
        - Choose **Hazard as-of** and an **evaluation end** (usually ~3 months later, not past Apr 29, 2025).
        - Adjust **Min 1-mo hazard / Min 3-mo risk / Top N** to test stricter or looser lists.
        - Download the **results CSV** if you want to review specific members.
        """
    )

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
    # Add a min 3-month risk filter too (optional).
    min_risk3_bt = float(st.number_input("Min 3-mo risk (0–1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01))
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
        # Optionally filter results by 3-month risk threshold before reporting.
        if min_risk3_bt > 0:
            bt_res = bt_res[bt_res["Pred_3mo_Risk"].fillna(0) >= min_risk3_bt]
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
