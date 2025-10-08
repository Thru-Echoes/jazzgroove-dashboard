# ------------------------------------------------------------
# Import standard libraries for argument parsing and filesystem.
# ------------------------------------------------------------
import argparse
# ------------------------------------------------------------
# Import datetime for date computations.
# ------------------------------------------------------------
from datetime import datetime
# ------------------------------------------------------------
# Import pandas for CSV I/O and data manipulation.
# ------------------------------------------------------------
import pandas as pd
# ------------------------------------------------------------
# Import numpy for numerical utilities.
# ------------------------------------------------------------
import numpy as np
# ------------------------------------------------------------
# Import yaml to read configuration values (paths/params).
# ------------------------------------------------------------
import yaml
# ------------------------------------------------------------
# Import os for path manipulations.
# ------------------------------------------------------------
import os
# Restrict all analysis to only these plans
ALLOWED_PLANS = {"premium-plus", "premium-plus-monthly"}

# ------------------------------------------------------------
# Define a small helper to parse YYYY-MM-DD strings safely.
# ------------------------------------------------------------
def _parse_date(s: str) -> pd.Timestamp:
    # Convert the input string into a pandas Timestamp for consistency.
    return pd.to_datetime(s)

# ------------------------------------------------------------
# Define a helper to load YAML configuration once.
# ------------------------------------------------------------
def load_config(path: str) -> dict:
    # Open the YAML file and convert its contents into a Python dictionary.
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ------------------------------------------------------------
# Define a robust CSV loader that optionally parses date columns.
# ------------------------------------------------------------
def load_csv(path: str, parse_dates=None) -> pd.DataFrame:
    # If the file exists at the given path, read and return it as a DataFrame.
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=parse_dates)
    # Otherwise, return an empty DataFrame to keep callers safe.
    return pd.DataFrame()

# ------------------------------------------------------------
# Define a utility to compute a dynamic watchlist for an arbitrary as_of date.
# This mirrors the logic used in the Streamlit app and prior notebooks.
# ------------------------------------------------------------
def compute_dynamic_watchlist(events_df: pd.DataFrame, hazard_df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    # If either the events or hazard tables are empty, we cannot compute a watchlist.
    if events_df.empty or hazard_df.empty:
        return pd.DataFrame()
    # Determine which members are active at the as_of date by checking Start <= as_of <= End.
    active_mask = (events_df["Start"] <= as_of_date) & (events_df["End"] >= as_of_date)
    # Extract the active roster columns for downstream joins and scoring.
    active = events_df.loc[active_mask, ["Name_or_Email", "Plan", "Start"]].copy()
    # Build entry plan and first start by taking the earliest record per member.
    first_rows = events_df.sort_values(["Name_or_Email", "Start"]).groupby("Name_or_Email", as_index=False).first()
    # Rename for clarity: initial plan becomes Entry_Plan and earliest Start is First_Start.
    first_rows = first_rows.rename(columns={"Plan": "Entry_Plan", "Start": "First_Start"})
    # Merge entry attributes into the active roster.
    active = active.merge(first_rows[["Name_or_Email", "Entry_Plan", "First_Start"]], on="Name_or_Email", how="left")
    
    # remove rows with NAs 
    active = active.dropna()

    # FIXME: remove debugger for production 
    #import pdb 
    #pdb.set_trace() 
    
    
    # Compute months since first start (1-indexed).
    active["Months_Since_First"] = (
        (as_of_date.year - active["First_Start"].dt.year) * 12
        + (as_of_date.month - active["First_Start"].dt.month)
        + 1
    ).clip(lower=1).astype(int)
    # Compute the "hazard month" index as next month relative to Months_Since_First.
    active["Hazard_Month"] = active["Months_Since_First"] + 1
    # Prepare the hazard table for joining by renaming Month to Hazard_Month.
    hz = hazard_df.rename(columns={"Month": "Hazard_Month"}).copy()
    # If the hazard table lacks a Plan column, create a single "All Plans" value to allow joining.
    if "Plan" not in hz.columns:
        hz["Plan"] = "All Plans"
    # Also ensure the base hazard_df we use for 3-month merges has a Plan column.
    if "Plan" not in hazard_df.columns:
        hazard_df = hazard_df.assign(Plan="All Plans")

    # FIXME: remove debugger for production 
    import pdb 
    #pdb.set_trace() 



    # Join the hazard value based on (Entry_Plan, Hazard_Month) to align with entry cohorts.
    try:
        out = active.merge(hz[["Plan", "Hazard_Month", "Avg_Hazard"]],
                       left_on=["Entry_Plan", "Hazard_Month"],
                       right_on=["Plan", "Hazard_Month"],
                       how="left").drop(columns=["Plan"])
    except Exception:
        out = active.merge(hz[["Plan", "Hazard_Month", "Avg_Hazard"]],
                       left_on=["Entry_Plan", "Hazard_Month"],
                       right_on=["Plan", "Hazard_Month"],
                       how="left")
    
    # Sort descending by predicted hazard so highest-risk rise to the top.
    out = out.sort_values("Avg_Hazard", ascending=False)
    # ---- 3-month combined hazard (next 3 months after as_of) ----
    # Make copies of the hazard table keyed by Plan and Month so we can join offsets efficiently.
    hz0 = hazard_df.copy()
    hz1 = hazard_df.copy()
    hz2 = hazard_df.copy()
    # Name columns to reflect offsets (t, t+1, t+2 months ahead).
    hz0 = hz0.rename(columns={"Month": "Hazard_Month_0", "Avg_Hazard": "h0"})
    hz1 = hz1.rename(columns={"Month": "Hazard_Month_1", "Avg_Hazard": "h1"})
    hz2 = hz2.rename(columns={"Month": "Hazard_Month_2", "Avg_Hazard": "h2"})
    # Compute offset month indices for each active member.
    out["Hazard_Month_0"] = out["Hazard_Month"].astype(int)
    out["Hazard_Month_1"] = out["Hazard_Month_0"] + 1
    out["Hazard_Month_2"] = out["Hazard_Month_0"] + 2

    # Join hazards for the three consecutive months based on Entry_Plan.
    try:
        out = out.merge(hz0[["Plan", "Hazard_Month_0", "h0"]], left_on=["Entry_Plan", "Hazard_Month_0"], right_on=["Plan", "Hazard_Month_0"], how="left").drop(columns=["Plan"])
        out = out.merge(hz1[["Plan", "Hazard_Month_1", "h1"]], left_on=["Entry_Plan", "Hazard_Month_1"], right_on=["Plan", "Hazard_Month_1"], how="left").drop(columns=["Plan"])
        out = out.merge(hz2[["Plan", "Hazard_Month_2", "h2"]], left_on=["Entry_Plan", "Hazard_Month_2"], right_on=["Plan", "Hazard_Month_2"], how="left").drop(columns=["Plan"])
    except Exception:
        out = out.merge(hz0[["Plan_x", "Hazard_Month_0", "h0"]], left_on=["Entry_Plan", "Hazard_Month_0"], right_on=["Plan", "Hazard_Month_0"], how="left").drop(columns=["Plan"])
        out = out.merge(hz1[["Plan_x", "Hazard_Month_1", "h1"]], left_on=["Entry_Plan", "Hazard_Month_1"], right_on=["Plan", "Hazard_Month_1"], how="left").drop(columns=["Plan"])
        out = out.merge(hz2[["Plan_x", "Hazard_Month_2", "h2"]], left_on=["Entry_Plan", "Hazard_Month_2"], right_on=["Plan", "Hazard_Month_2"], how="left").drop(columns=["Plan"])
    
    
    # Replace missing hazards with 0.0 so the product is well-defined.
    out[["h0", "h1", "h2"]] = out[["h0", "h1", "h2"]].fillna(0.0)
    # Compute 3-month churn probability assuming conditional independence.
    out["Pred_3mo_Risk"] = 1.0 - (1.0 - out["h0"]) * (1.0 - out["h1"]) * (1.0 - out["h2"])
    # Return the fully-scored watchlist, now with a 3-month risk as well.
    return out

# ------------------------------------------------------------
# Define a function that flags whether a member actually churned in a multi-month window.
# We treat a member as "churned" if the active interval covering as_of ends within [eval_start, eval_end].
# ------------------------------------------------------------
def label_churn_in_window(events_df: pd.DataFrame, as_of_date: pd.Timestamp, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> pd.DataFrame:
    # Identify the event rows where members are active at as_of.
    active_mask = (events_df["Start"] <= as_of_date) & (events_df["End"] >= as_of_date)
    # Select the rows corresponding to active members with their interval End dates.
    act_rows = events_df.loc[active_mask, ["Name_or_Email", "End"]].copy()
    # Compute whether the current interval End falls inside the evaluation window.
    act_rows["actual_churned_in_window"] = (act_rows["End"] >= eval_start) & (act_rows["End"] <= eval_end)
    # Keep the end date where applicable so we can compute time-to-cancel later.
    act_rows["Cancel_Date_in_window"] = act_rows["End"].where(act_rows["actual_churned_in_window"], pd.NaT)
    # For safety, drop duplicate Name_or_Email keeping the earliest End per member.
    act_rows = act_rows.sort_values(["Name_or_Email", "End"]).groupby("Name_or_Email", as_index=False).first()
    # Return the churn label table keyed by Name_or_Email.
    return act_rows[["Name_or_Email", "actual_churned_in_window", "Cancel_Date_in_window"]]

# ------------------------------------------------------------
# Define the main backtest routine: build watchlist at hazard_as_of and evaluate churn in a window.
# ------------------------------------------------------------
def run_backtest(events_path: str,
                 hazard_path: str,
                 hazard_as_of: pd.Timestamp,
                 eval_end: pd.Timestamp,
                 out_results: str,
                 out_metrics: str,
                 min_hazard: float = 0.0,
                 top_n: int = 1000000) -> None:
    # Load the events CSV and parse the Start/End columns as datetimes.
    events = load_csv(events_path, parse_dates=["Start", "End"])
    # Keep only allowed plans in events.
    if not events.empty and "Plan" in events.columns:
        events = events[events["Plan"].isin(ALLOWED_PLANS)].copy()
    # Load the hazard CSV (no dates to parse needed).
    hazard = load_csv(hazard_path)
    # Keep only allowed plans in hazard if present.
    if not hazard.empty and "Plan" in hazard.columns:
        hazard = hazard[hazard["Plan"].isin(ALLOWED_PLANS)].copy()
    # If either table is missing, raise an informative error.
    if events.empty:
        raise FileNotFoundError(f"Missing or empty events at: {events_path}")
    if hazard.empty:
        raise FileNotFoundError(f"Missing or empty hazard table at: {hazard_path}")
    # Compute the first day of the next month to define the evaluation start.
    eval_start = (hazard_as_of + pd.offsets.MonthBegin(1)).normalize()
    # Build a dynamic watchlist for the hazard_as_of date using events + hazard.
    wl = compute_dynamic_watchlist(events, hazard, hazard_as_of)
    # Apply the minimum hazard threshold if desired.
    if min_hazard > 0:
        wl = wl[wl["Avg_Hazard"].fillna(0) >= float(min_hazard)]
    # Keep only the top_n rows as requested (default is effectively "all").
    wl = wl.head(int(top_n)).copy()
    # Label which of the active-as-of members actually churned in the window [eval_start, eval_end].
    churn_labs = label_churn_in_window(events, hazard_as_of, eval_start, eval_end)
    # Join the churn labels onto the watchlist.
    res = wl.merge(churn_labs, on="Name_or_Email", how="left")
    # Compute time-to-cancel in days for members who churned within the window.
    res["days_to_cancel_from_as_of"] = (res["Cancel_Date_in_window"] - hazard_as_of).dt.days
    # Compute basic metrics: precision and recall.

    # True positives are flagged members who actually churned.
    tp = int(res["actual_churned_in_window"].fillna(False).sum())
    # Flagged count is simply the number of rows on the watchlist after filters.
    flagged = int(len(res))
    # Build the universe of eligible members (active at as_of) for recall denominator.
    active_mask = (events["Start"] <= hazard_as_of) & (events["End"] >= hazard_as_of)
    # Determine how many of those eligible members churned in the evaluation window.
    eligible = events.loc[active_mask, ["Name_or_Email", "End"]].copy()
    # Compute churn within window for each eligible member by checking its interval end.
    eligible["churn_in_window"] = (eligible["End"] >= eval_start) & (eligible["End"] <= eval_end)
    # Collapse to a unique per-member view using "any" over multiple rows (defensive).
    elig_churners = (eligible.groupby("Name_or_Email")["churn_in_window"].any()).reset_index()
    # Count the number of members who actually churned in window among the eligible population.
    total_churn = int(elig_churners["churn_in_window"].sum())
    # Precision: among those we flagged, how many churned.
    precision = (tp / flagged) if flagged > 0 else np.nan
    # Recall: among all churners in the window, how many were flagged.
    recall = (tp / total_churn) if total_churn > 0 else np.nan

    # Build calibration bins for interpretability over the watchlist subset only.
    # Define fixed bins (you can tune these to quantiles if preferred).
    bins = [0.0, 0.05, 0.10, 0.20, 1.0]
    # Build string labels for the bins for readability in charts.
    labels = ["0–5%", "5–10%", "10–20%", "20%+"]
    # Cut the 3-month predicted risk values into the specified bins.
    res["hazard_bin"] = pd.cut(res["Pred_3mo_Risk"].fillna(0.0), bins=bins, labels=labels, include_lowest=True, right=True)
    # Compute observed churn rates per hazard bin on the watchlist.
    calib = (res.groupby("hazard_bin")["actual_churned_in_window"]
               .mean()
               .reindex(labels)  # ensure logical bin order
               .reset_index(name="observed_churn_rate"))
    # Ensure the output directory exists for saving CSVs.
    for p in [out_results, out_metrics]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    # Save the detailed row-by-row results to CSV.
    res.to_csv(out_results, index=False)
    # Build a metrics table to CSV for dashboard consumption.
    metrics_rows = [
        {"metric": "hazard_as_of", "value": hazard_as_of.strftime("%Y-%m-%d")},
        {"metric": "eval_start", "value": eval_start.strftime("%Y-%m-%d")},
        {"metric": "eval_end", "value": eval_end.strftime("%Y-%m-%d")},
        {"metric": "flagged", "value": flagged},
        {"metric": "total_churn_in_window", "value": total_churn},
        {"metric": "true_positives", "value": tp},
        {"metric": "precision_among_flagged", "value": precision},
        {"metric": "recall_flagged_over_all_window_churners", "value": recall},
    ]
    # Append calibration rows with a metric name that includes the bin for easy plotting.
    for _, row in calib.iterrows():
        metrics_rows.append({
            "metric": f"calibration_bin_{row['hazard_bin']}",
            "value": row["observed_churn_rate"]
        })
    # Convert metrics list to a DataFrame for saving.
    metrics_df = pd.DataFrame(metrics_rows)
    # Save metrics to CSV.
    metrics_df.to_csv(out_metrics, index=False)
    # Print a succinct summary to stdout so the CLI user sees results immediately.
    print(f"[OK] Backtest complete. as_of={hazard_as_of.date()}  window=[{eval_start.date()} → {eval_end.date()}]")
    print(f"     Flagged={flagged:,}  TP={tp:,}  Churners={total_churn:,}  Prec={precision:.2%}  Rec={recall:.2%}")

# ------------------------------------------------------------
# Define a convenience function to wire up CLI arguments cleanly.
# ------------------------------------------------------------
def parse_args():
    # Create the argument parser with a helpful description.
    ap = argparse.ArgumentParser(description="Backtest multi-month churn using a watchlist built at a chosen as-of date.")
    # Add an optional path to a YAML config file; defaults to config/config.yaml.
    ap.add_argument("--config", default="config/config.yaml", help="Path to YAML config with default paths/params.")
    # Add a hazard-as-of override; if omitted, we take it from the config.
    ap.add_argument("--hazard-as-of", default=None, help="YYYY-MM-DD for watchlist as-of (use month-end).")
    # Add an evaluation end date override; if omitted, we take it from the config or derive from window months.
    ap.add_argument("--eval-end", default=None, help="YYYY-MM-DD for evaluation end (inclusive).")
    # Add an optional minimum hazard threshold for the watchlist.
    ap.add_argument("--min-hazard", type=float, default=0.0, help="Minimum Avg_Hazard to include (e.g., 0.05 for 5%).")
    # Add an optional top-N cap for the watchlist length.
    ap.add_argument("--top-n", type=int, default=1000000, help="Max rows from the watchlist to evaluate.")
    # Add an optional events CSV path override.
    ap.add_argument("--events", default=None, help="Path to normalized events CSV (Start/End).")
    # Add an optional hazard CSV path override.
    ap.add_argument("--hazard", default=None, help="Path to pooled hazard CSV.")
    # Add an optional results CSV path override.
    ap.add_argument("--out-results", default=None, help="Where to write the detailed backtest results CSV.")
    # Add an optional metrics CSV path override.
    ap.add_argument("--out-metrics", default=None, help="Where to write the summary metrics CSV.")
    # Return the parsed arguments to the caller.
    return ap.parse_args()

# ------------------------------------------------------------
# Provide a main section that reads config, applies overrides, and runs the backtest.
# ------------------------------------------------------------
if __name__ == "__main__":
    # Parse command-line arguments first.
    args = parse_args()
    # Load the YAML configuration to get defaults for paths/params.
    cfg = load_config(args.config)
    # Resolve the events CSV path, preferring CLI override then config.
    events_path = args.events or os.path.join(os.path.dirname(args.config), "..", cfg["paths"]["raw_events"])
    # Resolve the hazard CSV path the same way.
    hazard_path = args.hazard or os.path.join(os.path.dirname(args.config), "..", cfg["paths"]["hazard_csv"])
    # Determine the hazard-as-of date: CLI override or config.params.as_of.
    hazard_as_of_str = args.hazard_as_of or cfg["params"].get("as_of")
    # Parse the hazard-as-of date string into a Timestamp.
    hazard_as_of = _parse_date(hazard_as_of_str)
    # Determine the evaluation end date: CLI override or config.params.backtest_eval_end.
    eval_end_str = args.eval_end or cfg["params"].get("backtest_eval_end")
    # If not provided, default to 3 calendar months after as_of (inclusive window end).
    if not eval_end_str:
        eval_end = (hazard_as_of + pd.offsets.MonthEnd(3))
    else:
        eval_end = _parse_date(eval_end_str)
    # Resolve output file paths using CLI overrides or reasonable defaults near the configured folder.
    out_results = args.out_results or os.path.join(os.path.dirname(args.config), "..", "data", "processed", f"backtest_{hazard_as_of.date()}_to_{eval_end.date()}_results.csv")
    out_metrics = args.out_metrics or os.path.join(os.path.dirname(args.config), "..", "data", "processed", f"backtest_{hazard_as_of.date()}_to_{eval_end.date()}_metrics.csv")
    # Run the backtest with the chosen parameters.
    run_backtest(events_path=events_path,
                 hazard_path=hazard_path,
                 hazard_as_of=hazard_as_of,
                 eval_end=eval_end,
                 out_results=out_results,
                 out_metrics=out_metrics,
                 min_hazard=float(args.min_hazard),
                 top_n=int(args.top_n))
