#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------------------
# 00_ingest_customer_csv_to_events.py
# Raw Chargebee CSV -> normalized events.csv (all plans, exclusions applied)
# ------------------------------------------------------------

import argparse
import os
import re
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple

# ------------------------------------------------------------
# Plan normalization: keep all plans, but normalize premium-plus variants
# ------------------------------------------------------------
def simplify_plan_handle(handle: str) -> str:
    # Convert to lowercase string and strip whitespace; keep non-PP plans as-is (lowercased)
    h = str(handle).strip().lower()
    if "premium-plus" in h and "monthly" in h:
        return "premium-plus-monthly"
    if "premium-plus" in h:
        return "premium-plus"
    if "unlimited" in h or "listening" in h:
        # If your business wants 'unlimited-listening' treated as premium-plus, keep this
        return "premium-plus"
    return h  # preserve other plan slugs (lowercased) for brand windows

# ------------------------------------------------------------
# Build Email_or_Name: prefer Email, else <first-initial>_<last-name>
# ------------------------------------------------------------

def make_email_or_name(df: pd.DataFrame) -> pd.Series:
    # Prefer Email; else build <first-initial>_<last-name> (lowercase).
    first = df.get("First Name", pd.Series(index=df.index, dtype="object")).astype("string")
    last  = df.get("Last Name",  pd.Series(index=df.index, dtype="object")).astype("string")
    email = df.get("Email",      pd.Series(index=df.index, dtype="object")).astype("string")

    first = first.fillna("").str.strip()
    last  = last.fillna("").str.strip()
    email = email.fillna("").str.strip()

    first_initial = first.str[:1].str.lower()
    last_clean    = last.str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)

    fallback = (first_initial + "_" + last_clean).str.strip("_")
    fallback = fallback.mask(fallback.eq(""), "unknown")  # <-- if you want to drop 'unknown', see note below

    email_clean = email.str.lower()
    return np.where(email_clean.ne(""), email_clean, fallback)


def make_email_or_name_OLD(df: pd.DataFrame) -> pd.Series:
    first = df.get("First Name", pd.Series(index=df.index, dtype=object)).astype("string").fill_space().fillna("")
    last  = df.get("Last Name",  pd.Series(index=df.index, dtype=object)).astype("string").fill_space().fillna("")
    email = df.get("Email",      pd.Series(index=df.index, dtype=object)).astype("string").fillna("").str.strip()

    # first initial, lowercased
    first_initial = first.str.strip().str[:1].str.lower()
    last_clean    = last.str.strip().str.replace(r"[^A-Za-z0-9]+", "", regex=True).str.lower()
    fallback      = (first_initial + "_" + last_clean).str.strip("_")
    fallback      = fallback.mask(fallback.eq(""), "unknown")

    email_clean = email.str.strip()
    return np.where(email_clean.ne(""), email_clean.str.lower(), fallback)

# ------------------------------------------------------------
# Exclusions at ingest (raw table) based on First/Last/Email
# ------------------------------------------------------------
def exclude_people_and_domains(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop rows that match:
      - First Name starts with 'p' (case-insensitive) AND Last Name equals 'g' OR 'goldstein' (case-insensitive)
      - Email contains '@jazzgroove.com' or '@jazzgroove.org' (case-insensitive)
    Returns (kept_df, audit_df)
    """
    d0 = df.copy()

    first = d0.get("First Name", pd.Series(index=d0.index, dtype=object)).astype("string").fillna("").str.strip()
    last  = d0.get("Last Name",  pd.Series(index=d0.index, dtype=object)).astype("string").fillna("").str.strip()
    email = d0.get("Email",      pd.Series(index=d0.index, dtype=object)).astype("string").fillna("").str.strip()

    mask_pg = first.str.match(r"(?i)^\s*p") & (
        last.str.match(r"(?i)^\s*g{1,3}\b") | last.str.match(r"(?i)^\s*goldstein\b")
    )

    mask_email_paul_goldstein = email.str.contains(r"(?i)paul") & email.str.contains(r"(?i)goldstein")


    #mask_pg_variants = first.str.match(r"(?i)^\s*p") & (last.str.match(r"(?i)^\s*g$") | last.str.match(r"(?i)^\s*gg$") | last.str.match(r"(?i)^\s*ggg$"))

    mask_domain = email.str.contains(r"@jazzgroove\.com", case=False, regex=True) | \
                  email.str.contains(r"@jazzgroove\.org", case=False, regex=True)

    drop_mask = (mask_pg | mask_domain | mask_email_paul_goldstein).fillna(False)

    kept   = d0.loc[~drop_mask].copy()
    dropped = d0.loc[drop_mask].copy()

    # Build a small audit for visibility
    def nunique_safe(s: pd.Series) -> int:
        return int(s.nunique()) if s is not None else 0

    # Audit report
    audit = pd.DataFrame({
        "stage": ["EXCLUDE: ingest (paul/goldstein|@jazzgroove.com|@jazzgroove.org)"],
        "rows_before": [len(d0)],
        "rows_after":  [len(kept)],
        "delta_rows":  [len(kept) - len(d0)],
        "accounts_before": [d0["Email"].nunique() if "Email" in d0.columns else np.nan],
        "accounts_after":  [kept["Email"].nunique() if "Email" in kept.columns else np.nan],
        "delta_accounts":  [(kept["Email"].nunique() - d0["Email"].nunique()) if "Email" in d0.columns else np.nan],
        "dropped_rows_total": [len(dropped)],
        "drop_name_p_g":      [int(mask_pg.sum())],
        "drop_email_paul_goldstein": [int(mask_email_paul_goldstein.sum())],
        "drop_domain_jazzgroove":    [int(mask_domain.sum())],
        "dropped_accounts_total":    [nunique_safe(dropped.get("Email"))],
    })

    return kept, audit

# ------------------------------------------------------------
# Main: raw customer CSV -> normalized events.csv (all plans)
# ------------------------------------------------------------
def convert_customer_csv_to_events(in_customer_csv: str, out_events_csv: str) -> None:
    # Read raw customer CSV exactly as exported
    df_raw = pd.read_csv(in_customer_csv, dtype=str, keep_default_na=False).replace({"": np.nan})

    # 1) Apply ingest-time exclusions (Paul G variants + @jazzgroove.com / @jazzgroove.org)
    df_excluded, audit = exclude_people_and_domains(df_raw)

    # 2) Build Name_or_Email (prefer Email, else first-initial_lastname)
    df_excluded["Name_or_Email"] = make_email_or_name(df_excluded)

    # 3) Normalize plan + Start/End; do NOT drop non-PP plans, we need all plans for brand windows
    df_excluded["Plan"]  = df_excluded["Current Plan Handle"].astype("string") if "Current Plan Handle" in df_excluded else df_excluded["Current Plan Handle"]
    #df_excluded["Plan"]  = df_excluded["Plan"].astype("string").fillna("").map(simplify_plan_handle)

    # Coerce dates
    df_excluded["Start"] = pd.to_datetime(df_excluded["Activation Date"], errors="coerce", utc=True)
    df_excluded["End"]   = pd.to_datetime(df_excluded["Cancellation Date"], errors="coerce", utc=True)

    # 4) For still-active subscriptions, set a far-future sentinel End
    #    (Retention lib will right-censor to CAP_DATE later; Streamlit "active as-of" filters need End>=as_of)
    open_mask = df_excluded["End"].isna()
    df_excluded.loc[open_mask, "End"] = pd.Timestamp("2099-12-31", tz="UTC")

    # Drop rows with completely blank/NaN Name_or_Email before writing
    # (If you also want to drop the 'unknown' placeholder, uncomment the second mask line.)
    name_blank = df_excluded["Name_or_Email"].isna() | (df_excluded["Name_or_Email"].astype(str).str.strip() == "")
    # name_blank = name_blank | (df_filt["Name_or_Email"].str.lower() == "unknown")
    #df_out = df_excluded.loc[~name_blank, ["Name_or_Email", "Plan", "Start", "End"]].copy()
    df_out = df_excluded.copy()

    # Sort for readability
    df_out = df_out.sort_values(["Name_or_Email", "Start"])

    # FIXME: remove debugger
    import pdb 
    pdb.set_trace()


    # 5) Keep only essential columns for events.csv
    events_OLD = (
        df_out[["Name_or_Email", "Plan", "Start", "End"]]
        .dropna(subset=["Name_or_Email", "Start"])
        .sort_values(["Name_or_Email", "Start"])
        .copy()
    )

    # NEW Nov 11: KEEP ALL COLUMNS TO USE LATER 
    keep_cols = ["Name_or_Email","Plan","Start","End","First Name","Last Name","Email"]
    events = df_out[keep_cols].copy()

    # 6) Ensure output dir exists and write artifacts
    os.makedirs(os.path.dirname(out_events_csv), exist_ok=True)
    events.to_csv(out_events_csv, index=False)

    # Write audit file for operator
    audit_path = os.path.join(os.path.dirname(out_events_csv), "ingest_audit.csv")
    audit.to_csv(audit_path, index=False)

    print(f"[ingest] wrote: {out_events_csv}  rows={len(events):,}, accounts={events['Name_or_Email'].nunique():,}")
    print(audit.to_string(index=False))

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert raw customer CSV to normalized events.csv")
    ap.add_argument("--in-customer-csv", required=True, help="Path to raw customer CSV (e.g., data/raw/customer_data_to_April_29_2025.csv)")
    ap.add_argument("--out-events-csv",  required=True, help="Output events CSV (e.g., data/events.csv)")
    args = ap.parse_args()
    convert_customer_csv_to_events(args.in_customer_csv, args.out_events_csv)
