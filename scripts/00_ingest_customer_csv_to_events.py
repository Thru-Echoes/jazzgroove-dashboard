# ------------------------------------------------------------
# Import argparse to parse command-line options for file paths.
# ------------------------------------------------------------
import argparse
# ------------------------------------------------------------
# Import pandas for reading and transforming CSV files.
# ------------------------------------------------------------
import pandas as pd
# ------------------------------------------------------------
# Import numpy for small numerical utilities (optional here).
# ------------------------------------------------------------
import numpy as np
# ------------------------------------------------------------
# Import os for path manipulations if needed.
# ------------------------------------------------------------
import os

# ------------------------------------------------------------
# Define a helper function to map raw "Current Plan Handle"
# into a simplified brand "Plan" label used across the app.
# ------------------------------------------------------------
def simplify_plan_handle(handle: str) -> str:
    # Convert the input to a string to safely handle NaN values.
    h = str(handle).lower().strip()
    # Use simple substring rules to tag "premium-plus-monthly".
    if "monthly" in h and "premium-plus" in h:
        return "premium-plus-monthly"
    # Tag an annual-like "premium-plus" (we treat other non-monthly premium-plus as annual).
    if "premium-plus" in h:
        return "premium-plus"
    # Tag a generic "unlimited-listening" as "premium-plus" unless you keep it separate.
    if "unlimited" in h or "listening" in h:
        return "premium-plus"
    # Fallback to the original handle for transparency.
    return h

# ------------------------------------------------------------
# Define the main conversion function: raw customer CSV -> events CSV.
# ------------------------------------------------------------
def convert_customer_csv_to_events(in_customer_csv: str, out_events_csv: str) -> None:
    # Read the raw customer CSV into a DataFrame.
    df = pd.read_csv(in_customer_csv)
    # Rename the columns we need into a consistent schema for processing.
    df = df.rename(columns={
        "Email": "Name_or_Email",
        "Current Plan Handle": "Plan",
        "Activation Date": "Start",
        "Cancellation Date": "End"
    })
    # Apply the plan simplification to normalize plan names across records.
    df["Plan"] = df["Plan"].apply(simplify_plan_handle)
    # Convert Start/End columns to pandas datetime for reliable filtering.
    df["Start"] = pd.to_datetime(df["Start"], errors="coerce")
    df["End"] = pd.to_datetime(df["End"], errors="coerce")
    # For still-active members (missing End), set a far-future date to indicate open interval.
    df["End"] = df["End"].fillna(pd.Timestamp("2099-12-31"))
    # Keep only the essential columns for the normalized events table.
    events = df[["Name_or_Email", "Plan", "Start", "End"]].copy()
    # Sort for readability (optional).
    events = events.sort_values(["Name_or_Email", "Start"])
    # Ensure output directory exists.
    os.makedirs(os.path.dirname(out_events_csv), exist_ok=True)
    # Write the normalized events as CSV for the rest of the pipeline.
    events.to_csv(out_events_csv, index=False)
    # Print a small confirmation for the operator.
    print(f"Wrote normalized events CSV -> {out_events_csv} (rows: {len(events):,})")

# ------------------------------------------------------------
# Entry point to parse arguments and run the conversion.
# ------------------------------------------------------------
if __name__ == "__main__":
    # Create an argument parser to accept input and output file paths.
    ap = argparse.ArgumentParser(description="Convert raw customer CSV to normalized events CSV for the dashboard.")
    # Add an argument for the input raw customer CSV path.
    ap.add_argument("--in-customer-csv", required=True, help="Path to raw customer CSV (e.g., data/customer_data_to_April_29_2025.csv)")
    # Add an argument for the output normalized events CSV path.
    ap.add_argument("--out-events-csv", required=True, help="Output path for normalized events CSV (e.g., data/raw/subscriptions.csv)")
    # Parse the command-line arguments.
    args = ap.parse_args()
    # Call the conversion function with the provided paths.
    convert_customer_csv_to_events(args.in_customer_csv, args.out_events_csv)
