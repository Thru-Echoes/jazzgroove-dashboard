# ------------------------------------------------------------
# 01_prep_survival_and_risk.py
# Produce survival.csv, hazard.csv, and an initial watchlist.csv
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import yaml

# NEW: import the new orchestrator + constants from retention_lib
from retention_lib import (
    build_survival_and_hazard,  # brand-aware survival + hazard in one call
    ALLOWED_PLANS,              # {"premium-plus", "premium-plus-monthly"}
    CAP_DATE,                   # default: 2025-04-29
    find_early_annual_churn, 
)

# ------------------------------------------------------------
# Config loader
# ------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ------------------------------------------------------------
# Build a watchlist (same logic as the Streamlit app)
# ------------------------------------------------------------
def build_watchlist_for_as_of(events_df: pd.DataFrame,
                              hazard_df: pd.DataFrame,
                              as_of_date: pd.Timestamp,
                              allowed_plans=ALLOWED_PLANS) -> pd.DataFrame:
    """
    Create a ranked 'who looks riskiest next' list as of a month-end date.
    - Filters current activity to allowed_plans
    - Uses Entry_Plan for hazard lookups (plan segment is by first plan)
    - Computes 1-mo hazard and 3-mo combined risk (t, t+1, t+2)
    Output columns include:
      Name_or_Email, Entry_Plan, Months_Since_First, Hazard_Month, Avg_Hazard, Pred_3mo_Risk, h0, h1, h2
    """
    if events_df.empty or hazard_df.empty:
        return pd.DataFrame()

    df = events_df.copy()
    # normalize
    df["Start"] = pd.to_datetime(df["Start"], errors="coerce")
    df["End"]   = pd.to_datetime(df["End"],   errors="coerce")
    df["Plan"]  = df["Plan"].astype(str).str.lower()

    # Keep only allowed plans for the *current active* roster
    active_mask = (df["Start"] <= as_of_date) & (df["End"] >= as_of_date)
    active = df.loc[active_mask & df["Plan"].isin(set(p.lower() for p in allowed_plans)),
                    ["Name_or_Email", "Plan", "Start"]].copy()

    if active.empty:
        return pd.DataFrame()

    # Entry plan (first plan ever among allowed plans) + First_Start
    first_rows = (
        df.sort_values(["Name_or_Email", "Start"])
          .groupby("Name_or_Email", as_index=False)
          .first()
          .rename(columns={"Plan": "Entry_Plan", "Start": "First_Start"})
    )
    active = active.merge(first_rows[["Name_or_Email", "Entry_Plan", "First_Start"]],
                          on="Name_or_Email", how="left")
    active = active.dropna(subset=["First_Start"])
    active["First_Start"] = pd.to_datetime(active["First_Start"], errors="coerce")

    # Robust 1-indexed month difference (avoid Period subtraction pitfalls)
    asof_y, asof_m = as_of_date.year, as_of_date.month
    msf = (
        (asof_y - active["First_Start"].dt.year) * 12
        + (asof_m - active["First_Start"].dt.month)
        + 1
    )
    active["Months_Since_First"] = (
        pd.to_numeric(msf, errors="coerce").astype("Int64").clip(lower=1)
    )
    # Next month to score
    active["Hazard_Month"] = (active["Months_Since_First"] + 1).astype("Int64")

    # Join 1-month hazard by (Entry_Plan, Hazard_Month)
    hz = hazard_df.rename(columns={"Month": "Hazard_Month"}).copy()
    hz = hz.rename(columns={"Plan": "Hazard_Plan"})
    out = active.merge(
        hz[["Hazard_Plan", "Hazard_Month", "Avg_Hazard"]],
        left_on=["Entry_Plan", "Hazard_Month"],
        right_on=["Hazard_Plan", "Hazard_Month"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["Hazard_Plan"])

    # Build h0/h1/h2 from hazard table for 3-month combined risk
    out["Hazard_Month_0"] = out["Hazard_Month"]
    out["Hazard_Month_1"] = out["Hazard_Month_0"] + 1
    out["Hazard_Month_2"] = out["Hazard_Month_0"] + 2

    hz0 = hazard_df.rename(columns={"Month": "Hazard_Month_0", "Avg_Hazard": "h0", "Plan": "Hazard_Plan"}).copy()
    hz1 = hazard_df.rename(columns={"Month": "Hazard_Month_1", "Avg_Hazard": "h1", "Plan": "Hazard_Plan"}).copy()
    hz2 = hazard_df.rename(columns={"Month": "Hazard_Month_2", "Avg_Hazard": "h2", "Plan": "Hazard_Plan"}).copy()

    # Ensure right keys exist; if hazard had no plan segmentation, label as "All Plans" (defensive)
    for hzX in (hz0, hz1, hz2):
        if "Hazard_Plan" not in hzX.columns:
            hzX["Hazard_Plan"] = "All Plans"
    if "Entry_Plan" not in out.columns:
        out["Entry_Plan"] = "All Plans"

    # h0
    out = (out.merge(hz0[["Hazard_Plan", "Hazard_Month_0", "h0"]],
                     left_on=["Entry_Plan", "Hazard_Month_0"],
                     right_on=["Hazard_Plan", "Hazard_Month_0"],
                     how="left", validate="many_to_one")
             .drop(columns=["Hazard_Plan"]))
    # h1
    out = (out.merge(hz1[["Hazard_Plan", "Hazard_Month_1", "h1"]],
                     left_on=["Entry_Plan", "Hazard_Month_1"],
                     right_on=["Hazard_Plan", "Hazard_Month_1"],
                     how="left", validate="many_to_one")
             .drop(columns=["Hazard_Plan"]))
    # h2
    out = (out.merge(hz2[["Hazard_Plan", "Hazard_Month_2", "h2"]],
                     left_on=["Entry_Plan", "Hazard_Month_2"],
                     right_on=["Hazard_Plan", "Hazard_Month_2"],
                     how="left", validate="many_to_one")
             .drop(columns=["Hazard_Plan"]))

    out[["Avg_Hazard"]] = out[["Avg_Hazard"]].fillna(0.0)
    out[["h0", "h1", "h2"]] = out[["h0", "h1", "h2"]].fillna(0.0)
    out["Pred_3mo_Risk"] = 1.0 - (1.0 - out["h0"]) * (1.0 - out["h1"]) * (1.0 - out["h2"])

    return out.sort_values(["Avg_Hazard", "Pred_3mo_Risk"], ascending=False)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    # Resolve config path relative to scripts/
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    cfg = load_config(cfg_path)

    # Inputs/outputs from config
    events_csv = os.path.join(os.path.dirname(__file__), "..", cfg["paths"]["raw_events"])
    out_surv   = os.path.join(os.path.dirname(__file__), "..", cfg["paths"]["survival_csv"])
    out_hz     = os.path.join(os.path.dirname(__file__), "..", cfg["paths"]["hazard_csv"])
    out_wl     = os.path.join(os.path.dirname(__file__), "..", cfg["paths"]["watchlist_csv"])

    # Params
    grace      = int(cfg["params"].get("grace_months", 0))
    max_months = int(cfg["params"].get("max_months", 48))
    cap_date   = pd.to_datetime(cfg["params"].get("cap_date", str(CAP_DATE)))
    as_of      = pd.to_datetime(cfg["params"].get("as_of", cap_date))

    # Load events (already normalized by 00_ingest step)
    df_events = pd.read_csv(events_csv, parse_dates=["Start", "End"])

    # Safety: if Name_or_Email wasn't created upstream, create it now
    if "Name_or_Email" not in df_events.columns:
        em = df_events.get("Email", pd.Series(index=df_events.index, dtype="object")).astype(str).str.strip().str.lower()
        fn = df_events.get("First Name", pd.Series(index=df_events.index, dtype="object")).astype(str).str.strip().str.lower()
        ln = df_events.get("Last Name",  pd.Series(index=df_events.index, dtype="object")).astype(str).str.strip().str.lower()
        fallback = (fn.str.slice(0, 1).replace("", "x") + "_" + ln.replace("", "unknown").str.replace(r"\s+", "", regex=True))
        df_events["Name_or_Email"] = np.where(
            em.isna() | (em == "") | (em == "nan") | (em == "none"),
            fallback,
            em,
        )

    # Compute survival + hazard in one call (brand-aware, right-censored, entry-plan segmented)
    surv, haz, audit_df = build_survival_and_hazard(
        df_events,
        allowed_plans=ALLOWED_PLANS,
        cap_date=cap_date,
        max_months=max_months,
        grace_days=grace,
        already_events=True,   # we loaded events.csv (not a raw customer table)
        apply_exclusions=False, # exclusions should be done in 00_ingest; set True only if df has First/Last/Email
        return_audit = True
    )

    # Initial watchlist for the configured as_of month (helps CI and quick QA)
    watchlist = build_watchlist_for_as_of(df_events, haz, as_of)

    # Ensure output folders exist
    for p in (out_surv, out_hz, out_wl):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    # Save
    surv.to_csv(out_surv, index=False)
    haz.to_csv(out_hz, index=False)
    watchlist.to_csv(out_wl, index=False)

    # Save audit trail for inspection
    audit_csv = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "prep_audit.csv")
    os.makedirs(os.path.dirname(audit_csv), exist_ok=True)
    audit_df.to_csv(audit_csv, index=False)

    # Optional: early annual churn forensic list (brand-aware)
    # Rebuild brand windows + entry plans to feed the helper
    # (You can also refactor build_survival_and_hazard to return them.)
    from retention_lib import _normalize_event_columns, merge_brand_intervals, derive_entry_plan
    events_norm = _normalize_event_columns(df_events, cap_date=cap_date)
    brand_windows = merge_brand_intervals(events_norm, cap_date=cap_date, grace_days=grace)
    entry_plans   = derive_entry_plan(events_norm, allowed_plans=ALLOWED_PLANs)

    early_pp_2022 = find_early_churn = find_early_annual_churn(events_norm, brand_windows, entry_plans, months=12)
    early_pp_2022 = early_pp_2022[early_pp_2022["First_Start"].dt.year == 2022].copy()
    early_pp_2023 = find_early_annual_churn(events_norm, brand_windows, entry_plans, months=12)
    early_pp_2023 = early_pp_2023[early_pp_2023["First_Start"].dt.year == 2023].copy()

    # Export for review with billing logs
    early2022_csv = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "early_annual_cancels_2022.csv")
    early2023_csv = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "early_annual_cancels_2023.csv")
    early_pp_2022.to_csv(early2022_csv, index=False)
    early_pp_2023.to_csv(early2023_csv, index=False)

    print("Saved:")
    print(" -", out_surv)
    print(" -", out_hz)
    print(" -", out_wl)
    print(" -", audit_csv)
    print("Early annual cancels (2022):", len(early_pp_2022))
    print("Early annual cancels (2023):", len(early_pp_2023))

