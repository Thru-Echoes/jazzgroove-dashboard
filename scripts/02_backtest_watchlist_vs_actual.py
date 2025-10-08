# ------------------------------------------------------------
# Import argparse so we can pass in file paths for watchlist and customer CSVs.
# ------------------------------------------------------------
import argparse
# ------------------------------------------------------------
# Import pandas for data loading and manipulation.
# ------------------------------------------------------------
import pandas as pd
# ------------------------------------------------------------
# Import numpy for numeric operations and boolean math.
# ------------------------------------------------------------
import numpy as np
# ------------------------------------------------------------
# Import os for file path handling.
# ------------------------------------------------------------
import os
# ------------------------------------------------------------
# Import yaml to read config (e.g., the April backtest window).
# ------------------------------------------------------------
import yaml

# ------------------------------------------------------------
# Helper to load YAML configuration from the repository config.
# ------------------------------------------------------------
def load_config(path: str) -> dict:
    # Open the YAML file specified by path.
    with open(path, "r") as f:
        # Parse YAML and return a Python dictionary.
        return yaml.safe_load(f)

# ------------------------------------------------------------
# Helper to normalize the identifier for joining:
# prefer email if present, else return None (unmatched).
# ------------------------------------------------------------
def normalize_email_like(s: str) -> str | None:
    # Convert to string and lower-case for case-insensitive matching.
    x = str(s).strip().lower()
    # If the string contains an '@', treat it as an email and return.
    return x if ("@" in x and "." in x) else None

# ------------------------------------------------------------
# Core backtest logic: compare watchlist to actual cancellations in April 2025.
# ------------------------------------------------------------
def run_backtest(watchlist_csv: str, customer_csv: str, cfg: dict, out_results: str, out_metrics: str) -> None:
    # Read the historical watchlist CSV with hazard scores for March->April 2025.
    wl = pd.read_csv(watchlist_csv)
    # Read the raw customer CSV containing activation and cancellation dates to April 29, 2025.
    cust = pd.read_csv(customer_csv)
    # Standardize column names from the customer file for easier use.
    cust = cust.rename(columns={
        "Email": "Email",
        "Activation Date": "Activation_Date",
        "Cancellation Date": "Cancellation_Date"
    })
    # Parse date columns into proper datetime types.
    cust["Activation_Date"] = pd.to_datetime(cust["Activation_Date"], errors="coerce")
    cust["Cancellation_Date"] = pd.to_datetime(cust["Cancellation_Date"], errors="coerce")

    # Extract the backtest window (April 1..30, 2025) from config for clarity.
    start = pd.to_datetime(cfg["params"]["backtest_month_start"])
    end = pd.to_datetime(cfg["params"]["backtest_month_end"])

    # Create a join key from the watchlist where Name_or_Email might be an email or a name.
    wl["join_email"] = wl["Name_or_Email"].apply(normalize_email_like)
    # Create a join key from the customer file using the Email column directly.
    cust["join_email"] = cust["Email"].str.strip().str.lower()
    # Keep only rows with valid emails on the watchlist to ensure reliable matching.
    wl_matchable = wl[~wl["join_email"].isna()].copy()

    # Determine which customers were active at 2025-03-31 (the as_of date for this watchlist).
    as_of = pd.to_datetime(cfg["params"]["as_of"])
    # Active-as-of logic: activation <= as_of and (no cancellation or cancellation > as_of).
    active_as_of = (cust["Activation_Date"] <= as_of) & (
        cust["Cancellation_Date"].isna() | (cust["Cancellation_Date"] > as_of)
    )
    # Subset the customer table to those active on the as_of date.
    cust_active = cust.loc[active_as_of, ["join_email", "Activation_Date", "Cancellation_Date"]].copy()

    # Compute actual April churn flag: canceled between start and end inclusive.
    cust_active["actual_april_churn"] = cust_active["Cancellation_Date"].between(start, end, inclusive="both")

    # Left join watchlist entries (matchable by email) to their customer records.
    wl_j = wl_matchable.merge(cust_active, on="join_email", how="left")

    # Determine which watchlist entries had an observable outcome (i.e., we found a matching customer).
    wl_j["matched_customer"] = ~wl_j["Activation_Date"].isna()

    # Compute simple precision among matched watchlist entries: fraction that actually churned in April.
    precision_numer = wl_j["actual_april_churn"].fillna(False).sum()
    precision_denom = (wl_j["matched_customer"]).sum()
    precision = float(precision_numer) / float(precision_denom) if precision_denom > 0 else np.nan

    # For recall, we need all April churners among the active-as-of base, then see how many were flagged.
    april_churners = cust_active[cust_active["actual_april_churn"] == True][["join_email"]].drop_duplicates()
    # Mark watchlist emails as flagged.
    flagged = wl_matchable[["join_email"]].drop_duplicates()
    # True positives are those in both flagged and april churners.
    tp = april_churners.merge(flagged, on="join_email", how="inner")
    # Recall denominator is the number of April churners total.
    recall_denom = len(april_churners)
    recall = len(tp) / recall_denom if recall_denom > 0 else np.nan

    # Optional: calibration by hazard bins within the watchlist (only among matched records).
    wl_cal = wl_j[wl_j["matched_customer"]].copy()
    # If Avg_Hazard is present, create quantile bins to inspect calibration.
    if "Avg_Hazard" in wl_cal.columns and wl_cal["Avg_Hazard"].notna().any():
        # Create 5 quantile bins (quintiles) for a coarse calibration view.
        try:
            wl_cal["hz_bin"] = pd.qcut(wl_cal["Avg_Hazard"], q=5, duplicates="drop")
        except Exception:
            wl_cal["hz_bin"] = pd.cut(wl_cal["Avg_Hazard"], bins=5, include_lowest=True)
        # Compute observed April churn rate per hazard bin.
        calib = wl_cal.groupby("hz_bin", dropna=False)["actual_april_churn"].mean().reset_index(name="observed_rate")
    else:
        calib = pd.DataFrame(columns=["hz_bin", "observed_rate"])

    # Prepare detailed per-subscriber results for inspection and CSV export.
    results = wl_j.copy()
    # Add a simple classification label for readability.
    results["predicted_high_risk"] = True
    # Ensure output directory exists.
    os.makedirs(os.path.dirname(out_results), exist_ok=True)
    # Write the per-subscriber results CSV.
    results.to_csv(out_results, index=False)

    # Build a small summary metrics table including counts and percentages.
    metrics_rows = []
    metrics_rows.append({"metric": "watchlist_rows_total", "value": int(len(wl))})
    metrics_rows.append({"metric": "watchlist_rows_matchable_by_email", "value": int(len(wl_matchable))})
    metrics_rows.append({"metric": "watchlist_rows_matched_to_customer", "value": int(precision_denom)})
    metrics_rows.append({"metric": "precision_among_matched", "value": float(precision)})
    metrics_rows.append({"metric": "april_churners_total", "value": int(recall_denom)})
    metrics_rows.append({"metric": "true_positives_flagged", "value": int(len(tp))})
    metrics_rows.append({"metric": "recall_flagged_over_all_april_churners", "value": float(recall)})
    # If we computed calibration, append the rows, too.
    for _, r in calib.iterrows():
        metrics_rows.append({"metric": f"calibration_bin_{str(r['hz_bin'])}", "value": float(r["observed_rate"])})

    # Convert the metrics list to a DataFrame for CSV export.
    metrics = pd.DataFrame(metrics_rows)
    # Write the metrics CSV.
    metrics.to_csv(out_metrics, index=False)

    # Print a concise summary for the operator.
    print("Backtest summary")
    print(" - precision (matched watchlist):", precision)
    print(" - recall (of all April churners):", recall)
    print(f"Wrote results -> {out_results}")
    print(f"Wrote metrics -> {out_metrics}")

# ------------------------------------------------------------
# CLI entry to run the backtest with repo defaults.
# ------------------------------------------------------------
if __name__ == "__main__":
    # Construct an argument parser for inputs and overrides.
    ap = argparse.ArgumentParser(description="Backtest a March->April 2025 watchlist vs actual April churn outcomes.")
    # Add a required argument for the watchlist CSV path.
    ap.add_argument("--watchlist", required=True, help="Path to watchlist CSV (e.g., data/watchlists/watchlist_cancellation_risk_from_March_to_April_2025.csv)")
    # Add a required argument for the customer CSV with activation/cancellation.
    ap.add_argument("--customer", required=True, help="Path to raw customer CSV (e.g., data/customer_data_to_April_29_2025.csv)")
    # Optionally allow a custom config path; default to repo config.
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"), help="Path to config.yaml")
    # Optionally allow custom output paths for results and metrics.
    ap.add_argument("--out-results", default=os.path.join(os.path.dirname(__file__), "..", "data", "processed", "backtest_apr2025_results.csv"), help="Where to write per-subscriber results CSV")
    ap.add_argument("--out-metrics", default=os.path.join(os.path.dirname(__file__), "..", "data", "processed", "backtest_apr2025_metrics.csv"), help="Where to write summary metrics CSV")
    # Parse the provided CLI arguments.
    args = ap.parse_args()
    # Load the configuration from the given path.
    cfg = load_config(args.config)
    # Invoke the backtest process with the supplied file paths.
    run_backtest(args.watchlist, args.customer, cfg, args.out_results, args.out_metrics)
