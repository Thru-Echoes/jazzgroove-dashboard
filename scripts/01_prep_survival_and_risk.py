# ------------------------------------------------------------
# Import os for building file paths relative to this script.
# ------------------------------------------------------------
import os
# ------------------------------------------------------------
# Import pandas for DataFrame operations and CSV I/O.
# ------------------------------------------------------------
import pandas as pd
# ------------------------------------------------------------
# Import yaml to read configuration from config.yaml.
# ------------------------------------------------------------
import yaml

# ------------------------------------------------------------
# Import your retention helper functions.
# ------------------------------------------------------------
from retention_lib import (
    build_gap_adjusted_intervals,
    expand_intervals_to_records,
    map_entry_plan,
    survival_from_records,
    compute_average_monthly_hazard,
)

# ------------------------------------------------------------
# Load YAML configuration helper.
# ------------------------------------------------------------
def load_config(path: str) -> dict:
    # Open the YAML file and parse its contents into a dictionary.
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ------------------------------------------------------------
# Schema shim: ensure retention_lib-compatible columns exist.
# ------------------------------------------------------------
def ensure_retention_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy so we do not mutate the caller's DataFrame in-place.
    d = df.copy()
    # If the expected 'Activation Date' column is missing, but 'Start' exists, alias it.
    if "Activation Date" not in d.columns and "Start" in d.columns:
        d["Activation Date"] = d["Start"]
    # If the expected 'Cancellation Date' column is missing, but 'End' exists, alias it.
    if "Cancellation Date" not in d.columns and "End" in d.columns:
        d["Cancellation Date"] = d["End"]

    if "End Date" not in d.columns and "End" in d.columns:
        d["End Date"] = d["End"]
    # Return the DataFrame with both naming conventions present so retention_lib is happy.
    return d

# ------------------------------------------------------------
# Build survival by entry plan, using the schema shim first.
# ------------------------------------------------------------
def brand_survival_by_entry_plan_on_df(df_cut: pd.DataFrame, grace_months: int = 1, max_months: int = 24, cap_date: str = "2025-12-31") -> pd.DataFrame:
    # Apply the schema shim to guarantee columns that retention_lib expects.
    df_for_lib = ensure_retention_schema(df_cut)
    # Build gap-adjusted intervals (handles short gaps as a single continuous subscription).
    intervals_b = build_gap_adjusted_intervals(
        df=df_for_lib,
        grace_months=grace_months,
        cap_date=cap_date,
        merge_across_plans=True,
        plan_col="Plan",
    )
    # Expand each interval into per-month records so we can compute hazards/survival.
    rec = expand_intervals_to_records(intervals_b, max_months=max_months)
    # Map each member to the first plan they ever had (their "entry" cohort).
    entry = map_entry_plan(df_for_lib, plan_col="Plan")[["Name_or_Email", "Entry_Plan"]]
    # Attach the entry plan to the expanded records for grouping.
    rec = rec.merge(entry, on="Name_or_Email", how="left")
    # Drop the in-interval plan since we want to group by entry plan.
    rec = rec.drop(columns=["Plan"])
    # Rename Entry_Plan to Plan so downstream code is consistent.
    rec = rec.rename(columns={"Entry_Plan": "Plan"})
    # Compute survival metrics from the monthly records.
    surv = survival_from_records(rec)
    # Return the survival DataFrame.
    return surv

# ------------------------------------------------------------
# Main execution: read config and events CSV, produce outputs.
# ------------------------------------------------------------
if __name__ == "__main__":
    # Resolve the path to the config.yaml (one level up from scripts/).
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    # Load configuration values.
    cfg = load_config(cfg_path)
    # Resolve path to normalized events CSV.
    events_csv = os.path.join(os.path.dirname(__file__), "..", cfg["paths"]["raw_events"])
    # Read events CSV with parsed date columns.
    df_events = pd.read_csv(events_csv, parse_dates=["Start", "End"])
    # Extract parameters from config.
    grace = int(cfg["params"]["grace_months"])
    max_months = int(cfg["params"]["max_months"])
    cap_date = str(cfg["params"]["cap_date"])
    as_of = str(cfg["params"]["as_of"])
    top_n = int(cfg["params"]["top_n_watchlist"])

    # Compute survival curves by entry plan.
    surv = brand_survival_by_entry_plan_on_df(df_events, grace_months=grace, max_months=max_months, cap_date=cap_date)
    # Compute pooled hazards across cohorts.
    pooled_hazard = compute_average_monthly_hazard(surv)

    # --- Build active roster as-of the chosen cutoff (with Entry_Plan-aware join) ---
    # Convert as_of string to Timestamp.
    as_of_ts = pd.to_datetime(as_of)
    # Determine which intervals contain as_of (i.e., active at as_of).
    active_mask = (df_events["Start"] <= as_of_ts) & (df_events["End"] >= as_of_ts)
    # Subset the events for active members.
    active = df_events.loc[active_mask, ["Name_or_Email", "Plan", "Start"]].copy()
    # Compute the very first Start per member to get "first start" and "entry plan".
    first_rows = df_events.sort_values(["Name_or_Email", "Start"]).groupby("Name_or_Email", as_index=False).first()
    # Rename columns for clarity.
    first_rows = first_rows.rename(columns={"Plan": "Entry_Plan", "Start": "First_Start"})
    # Attach entry plan/first start to the active roster.
    active = active.merge(first_rows[["Name_or_Email", "Entry_Plan", "First_Start"]], on="Name_or_Email", how="left")

    active = active.dropna()
    #import pdb 
    #pdb.set_trace()

    # WE ONLY WANT premium-plus and premium-plus-monthly members: 
    

    # Compute Months_Since_First as 1-indexed integer months.
    active["Months_Since_First"] = (
        (as_of_ts.year - active["First_Start"].dt.year) * 12
        + (as_of_ts.month - active["First_Start"].dt.month)
        + 1
    ).clip(lower=1).astype(int)
    # Next-month hazard is evaluated at this month + 1.
    active["Hazard_Month"] = active["Months_Since_First"] + 1

    # Prepare hazard table for join: try to join on both Entry_Plan and Hazard_Month if possible.
    hz = pooled_hazard.rename(columns={"Month": "Hazard_Month"})
    # If pooled_hazard has a Plan column, do a plan-aware join; otherwise fall back to month-only join.
    if "Plan" in hz.columns:  
        # Join by (Entry_Plan, Hazard_Month) to get the right hazard for the member's entry cohort.
        active = active.merge(hz[["Plan", "Hazard_Month", "Avg_Hazard"]], left_on=["Entry_Plan", "Hazard_Month"], right_on=["Plan", "Hazard_Month"], how="left")
        # Drop the hazard table's Plan label to avoid confusion.
        
        #import pdb 
        #pdb.set_trace()

        try:
            active = active.drop(columns=["Plan"])
        except Exception:
            active = active.drop(columns = ["Plan_x"])
        
    else:
        # Month-only join as a safe fallback.
        active = active.merge(hz[["Hazard_Month", "Avg_Hazard"]], on="Hazard_Month", how="left")

    # Sort by hazard descending to form the watchlist.
    watchlist = active.sort_values("Avg_Hazard", ascending=False)

    # Resolve output paths and ensure directories exist.
    out_surv = os.path.join(os.path.dirname(__file__), "..", cfg["paths"]["survival_csv"])
    out_hz = os.path.join(os.path.dirname(__file__), "..", cfg["paths"]["hazard_csv"])
    out_wl = os.path.join(os.path.dirname(__file__), "..", cfg["paths"]["watchlist_csv"])
    os.makedirs(os.path.dirname(out_surv), exist_ok=True)
    # Write outputs as CSVs.
    surv.to_csv(out_surv, index=False)
    pooled_hazard.to_csv(out_hz, index=False)
    watchlist.to_csv(out_wl, index=False)
    # Print confirmations.
    print("Saved:")
    print(" -", out_surv)
    print(" -", out_hz)
    print(" -", out_wl)
