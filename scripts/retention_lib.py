# ============================================================
# retention_lib.py 
# ============================================================
# Implementations for:
# - build_gap_adjusted_intervals(df, grace_months, cap_date, merge_across_plans, plan_col)
# - expand_intervals_to_records(intervals_df, max_months)
# - map_entry_plan(df, plan_col)
# - survival_from_records(rec_df)
# - compute_average_monthly_hazard(surv_df)
#
# The prep script imports these. Stubs below raise clear errors if left in place.

from __future__ import annotations
import pandas as pd 
import numpy as np 
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter
import calendar
import re
from typing import Dict, Optional, Iterable, Tuple
from pandas.api.types import is_datetime64tz_dtype

# -----------------------
# Project defaults
# -----------------------
ALLOWED_PLANS = {"premium-plus", "premium-plus-monthly"}
CAP_DATE = pd.Timestamp("2025-12-31 23:59:59")  # right-censoring cutoff, if April 29, 2025 = "2024-04-29"


# ------------------------------------------------------------
# Audit helpers: collect row/account counts at each pipeline stage
# ------------------------------------------------------------
from dataclasses import dataclass, field

@dataclass
class AuditEvent:
    # Human-friendly stage name, e.g. "EXCLUDE: paul/org"
    stage: str
    # Integer counts before and after the step
    rows_before: int
    rows_after: int
    # Optional counts for accounts and a free-form note
    accounts_before: Optional[int] = None
    accounts_after: Optional[int] = None
    note: str = ""

@dataclass
class AuditTracker:
    # In-memory list of AuditEvent entries
    events: list[AuditEvent] = field(default_factory=list)

    def add(self, stage: str, df_before: pd.DataFrame, df_after: pd.DataFrame, note: str = ""):
        # Compute row counts and distinct account counts (by Name_or_Email if present)
        rows_before = len(df_before)
        rows_after  = len(df_after)
        acc_before  = df_before["Name_or_Email"].nunique() if "Name_or_Email" in df_before.columns else None
        acc_after   = df_after["Name_or_Email"].nunique()  if "Name_or_Email" in df_after.columns  else None
        self.events.append(
            AuditEvent(stage=stage, rows_before=rows_before, rows_after=rows_after,
                       accounts_before=acc_before, accounts_after=acc_after, note=note)
        )

    def to_frame(self) -> pd.DataFrame:
        # Convert the audit trail to a DataFrame for easy printing/export
        if not self.events:
            return pd.DataFrame(columns=["stage","rows_before","rows_after","delta_rows","accounts_before","accounts_after","delta_accounts","note"])
        df = pd.DataFrame([e.__dict__ for e in self.events])
        df["delta_rows"]     = df["rows_after"] - df["rows_before"]
        df["delta_accounts"] = df["accounts_after"] - df["accounts_before"] if df["accounts_before"].notna().any() else np.nan
        return df[["stage","rows_before","rows_after","delta_rows","accounts_before","accounts_after","delta_accounts","note"]]


# -----------------------
# Utilities
# -----------------------

def _normalize_event_columns(df: pd.DataFrame, 
                             cap_date: Optional[pd.Timestamp] = None,
                             audit: Optional[AuditTracker] = None,
                             stage_label: str = "NORMALIZE: raw->events") -> pd.DataFrame:
    """
    Accepts a raw customer/subscription dataframe or an events dataframe.
    Returns a normalized events table with columns:
      ['Name_or_Email', 'Plan', 'Start', 'End', ('Subscription ID' optional)]
    - Maps 'Activation Date' -> 'Start', 'Cancellation Date' -> 'End' if needed
    - Lowercases Plan
    - Coerces dates
    - Optionally right-censors at cap_date
    """
    before = df.copy()
    d = df.copy()

    # Canonicalize identifier
    if "Name_or_Email" not in d.columns:
        # Try to build from Email; else <first_initial>_<last_name>
        em = d.get("Email", pd.Series(index=d.index, dtype="object")).astype(str).str.strip().str.lower()
        fn = d.get("First Name", pd.Series(index=d.index, dtype="object")).astype(str).str.strip().str.lower()
        ln = d.get("Last Name", pd.Series(index=d.index, dtype="object")).astype(str).str.strip().str.lower()
        fallback = (fn.str.slice(0, 1).replace("", "x") + "_" + ln.replace("", "unknown").str.replace(r"\s+", "", regex=True))
        d["Name_or_Email"] = np.where(em.isna() | (em == "") | (em == "nan") | (em == "none"), fallback, em)

    # Map date columns to Start/End if needed
    if "Start" not in d.columns and "Activation Date" in d.columns:
        d = d.rename(columns={"Activation Date": "Start"})
    if "End" not in d.columns and "Cancellation Date" in d.columns:
        d = d.rename(columns={"Cancellation Date": "End"})

    # Defensive: if still missing Start/End, raise a clear error
    for col in ("Start", "End"):
        if col not in d.columns:
            raise KeyError(f"[retention_lib] Expected a '{col}' column or a mappable source column (Activation/Cancellation Date).")

    # Plan normalization
    plan_col = None
    if "Plan" in d.columns:
        plan_col = "Plan"
    elif "Current Plan Handle" in d.columns:
        d = d.rename(columns={"Current Plan Handle": "Plan"})
        plan_col = "Plan"
    else:
        # If no plan column exists, create a generic one so downstream code is not blocked.
        d["Plan"] = "unknown"
        plan_col = "Plan"

    d[plan_col] = d[plan_col].astype(str).str.strip().str.lower()

    # **force tz-naive** (drop timezone if present)
    if is_datetime64tz_dtype(d["Start"]):
        d["Start"] = d["Start"].dt.tz_convert("UTC").dt.tz_localize(None)
    if is_datetime64tz_dtype(d["End"]):
        d["End"] = d["End"].dt.tz_convert("UTC").dt.tz_localize(None)

    # right-censor with a tz-naive cap
    if cap_date is not None:
        cap = pd.to_datetime(cap_date)
        if getattr(cap, "tzinfo", None) is not None:
            # drop tz on incoming cap if it is tz-aware
            cap = cap.tz_convert(None) if hasattr(cap, "tz_convert") else cap.tz_localize(None)
        d["End"] = d["End"].fillna(cap)
        d.loc[d["End"] > cap, "End"] = cap

    # Coerce dates
    #d["Start"] = pd.to_datetime(d["Start"], errors="coerce")
    #d["End"] = pd.to_datetime(d["End"], errors="coerce")

    # Right-censoring
    #if cap_date is not None:
    #    d["End"] = d["End"].fillna(cap_date)
    #    d.loc[d["End"] > cap_date, "End"] = cap_date

    # Keep a clean set of columns
    keep = ["Name_or_Email", "Plan", "Start", "End"]
    if "Subscription ID" in d.columns:
        keep.append("Subscription ID")
    d = d[keep].copy()

    # Drop nonsense rows
    cleaned = d.dropna(subset=["Name_or_Email", "Start"]).copy()
    #cleaned = cleaned[cleaned["Start"].notna()]


    # AUDIT 
    if audit is not None:
        audit.add(stage_label, before, cleaned, note="Normalize Start/End; cap future End; drop rows missing Start or ID.")

    return d


def _apply_global_exclusions(
        df: pd.DataFrame,
        audit: Optional[AuditTracker] = None,
        stage_label: str = "EXCLUDE: paul/org") -> pd.DataFrame:
    """
    Drop Paul Goldstein variants and internal org emails.
    """
    before = df.copy()
    d = df.copy()
    # Safe accessors
    fn = d.get("First Name", pd.Series(index=d.index, dtype="object")).astype(str).str.strip().str.lower()
    ln = d.get("Last Name",  pd.Series(index=d.index, dtype="object")).astype(str).str.strip().str.lower()
    em = d.get("Email",      pd.Series(index=d.index, dtype="object")).astype(str).str.strip().str.lower()

    mask_paul = fn.str.startswith("p", na=False) & (ln.str.startswith("g", na=False) | (ln == "goldstein"))
    mask_org  = em.str.contains(r"@jazzgroove\.org", case=False, na=False)

    drop_mask = mask_paul | mask_org
    kept = d.loc[~drop_mask].copy()

    if audit is not None:
        rows_dropped = int(drop_mask.sum())
        accounts_dropped = int(d.loc[drop_mask, "Name_or_Email"].nunique()) if "Name_or_Email" in d.columns else None
        note = f"dropped_rows={rows_dropped}, dropped_accounts={accounts_dropped}, rules=[first p* & last g*/goldstein; email contains @jazzgroove.com? no; @jazzgroove.org yes]"
        audit.add(stage_label, before, kept, note=note)

    return kept

def _apply_event_level_exclusions(
    events: pd.DataSource, 
    audit: Optional['AuditTracker']=None, 
    stage_label: str="EXCLUDE: events-level (domain/name)"
) -> pd.DataFrame:
    """
    Exclude rows using only what's available in an events table.
    Rules:
      - drop Name_or_Email containing '@jazzgroove.org'
      - drop rows where the *name-like* token looks like 'p*_*g' or 'p*_*goldstein'
        (we approximate first-initial 'p' and last name 'g' or 'goldstein')
    """
    before = events.copy()
    e = events.copy()
    key = e.get("Name_or_Email", pd.Series(index=e.index, dtype="object")).astype(str)

    # domain filter
    mask_org = key.str.contains(r"@jazzgroove\.org", case=False, na=False)

    # extract a name-ish token for non-email values: left of '@' or entire string if no '@'
    local = key.str.split("@").str[0].str.lower().str.replace(r"[^a-z0-9_\. -]", "", regex=True)

    # simple heuristic for first-initial p + last starting with g or 'goldstein'
    # e.g. 'p_goldstein', 'P.Goldstein', 'p g', 'paul_g', 'paul.goldstein'
    mask_pg = (
        local.str.contains(r"^\s*p[.\-_ ]*[a-z]*[.\-_ ]*(g(oldstein)?)(\b|$)", regex=True) |
        local.str.contains(r"^\s*p\b.*\b(goldstein)\b", regex=True)
    )

    drop_mask = mask_org | mask_pg
    kept = e.loc[~drop_mask].copy()

    if audit is not None:
        rows_before = len(before)
        rows_after  = len(kept)
        acc_before  = before["Name_or_Email"].nunique() if "Name_or_Email" in before.columns else None
        acc_after   = kept["Name_or_Email"].nunique()  if "Name_or_Email" in kept.columns  else None
        note = f"drop_rows={int(drop_mask.sum())}, drop_accounts={(before.loc[drop_mask,'Name_or_Email'].nunique() if 'Name_or_Email' in before.columns else 'n/a')}"
        audit.events.append(AuditEvent(stage=stage_label, rows_before=rows_before, rows_after=rows_after, accounts_before=acc_before, accounts_after=acc_after, note=note))

    return kept



def merge_brand_intervals(
        events: pd.DataFrame, 
        cap_date: pd.Timestamp = CAP_DATE, 
        grace_days: int = 0,
        audit: Optional[AuditTracker] = None,
        stage_label: str = "MERGE: brand windows") -> pd.DataFrame:
    """
    Collapse a member's subscription rows across ALL plans into brand-level intervals.
    - Adjacent/overlapping rows are merged; a gap > grace_days starts a new interval.
    - End dates are right-censored at cap_date.
    Returns: ['Name_or_Email', 'BrandStart', 'BrandEnd']
    """
    before = events.copy()

    e = events.copy()
    e["Start"] = pd.to_datetime(e["Start"], errors="coerce")
    e["End"]   = pd.to_datetime(e["End"],   errors="coerce")
    e = e.dropna(subset=["Name_or_Email", "Start"])
    e["End"] = e["End"].fillna(cap_date)
    e.loc[e["End"] > cap_date, "End"] = cap_date

    out = []
    for name, g in e.sort_values(["Name_or_Email", "Start"]).groupby("Name_or_Email", as_index=False):
        cur_s, cur_e = None, None
        for _, row in g.iterrows():
            s, en = row["Start"], row["End"]
            if cur_s is None:
                cur_s, cur_e = s, en
                continue
            if s <= (cur_e + pd.Timedelta(days=grace_days)):
                # extend
                if en > cur_e:
                    cur_e = en
            else:
                out.append((name, cur_s, cur_e))
                cur_s, cur_e = s, en
        if cur_s is not None:
            out.append((name, cur_s, cur_e))
    out = pd.DataFrame(out, columns=["Name_or_Email", "BrandStart", "BrandEnd"])

    if audit is not None:
        note = f"members={e['Name_or_Email'].nunique()} -> windows={len(out)}"
        audit.add(stage_label, before, out, note=note)

    return out


def derive_entry_plan(
        events: pd.DataFrame, 
        allowed_plans: Iterable[str] = ALLOWED_PLANS,
        audit: Optional[AuditTracker] = None,
        stage_label: str = "DERIVE: entry plan") -> pd.DataFrame:
    """
    Determine each member's Entry_Plan as the earliest of the allowed plans.
    Returns: ['Name_or_Email', 'Entry_Plan', 'First_Start']
    """
    before = events.copy()
    e = events.copy()
    e["Start"] = pd.to_datetime(e["Start"], errors="coerce")
    e["Plan"] = e["Plan"].astype(str).str.lower()

    first = (
        e[e["Plan"].isin(set(p.lower() for p in allowed_plans))]
        .sort_values(["Name_or_Email", "Start"])
        .groupby("Name_or_Email", as_index=False)
        .first()[["Name_or_Email", "Plan", "Start"]]
        .rename(columns={"Plan": "Entry_Plan", "Start": "First_Start"})
    )
    return first

# -----------------------
# Survival & Hazard
# -----------------------

def build_survival_from_brand_windows(
    brand_windows: pd.DataFrame,
    entry_plans: pd.DataFrame,
    max_months: int = 48,
) -> pd.DataFrame:
    """
    One lifetime per member:
      start = First_Start (first allowed plan),
      end   = first BrandEnd *at or after* First_Start (respects censoring already applied).
    """
    bw = brand_windows.copy()
    ep = entry_plans.copy()
    x = bw.merge(ep, on="Name_or_Email", how="inner")
    x["First_Start"] = pd.to_datetime(x["First_Start"], errors="coerce")
    x["BrandStart"]  = pd.to_datetime(x["BrandStart"], errors="coerce")
    x["BrandEnd"]    = pd.to_datetime(x["BrandEnd"],   errors="coerce")
    x = x.dropna(subset=["First_Start","BrandStart","BrandEnd"])

    # For each member, pick the first window whose [BrandStart, BrandEnd] overlaps First_Start or starts after it.
    chosen = []
    for name, g in x.sort_values(["BrandStart","BrandEnd"]).groupby("Name_or_Email"):
        fs = g["First_Start"].iloc[0]
        g = g.sort_values("BrandStart")
        # window that contains fs
        c = g[(g["BrandStart"] <= fs) & (g["BrandEnd"] >= fs)]
        if not c.empty:
            row = c.iloc[0]
        else:
            # otherwise the first window that starts after fs
            c = g[g["BrandStart"] > fs]
            if c.empty:
                continue
            row = c.iloc[0]
        chosen.append({
            "Name_or_Email": name,
            "Plan": row["Entry_Plan"],
            "Cohort Year": fs.year,
            "First_Start": fs,
            "End": row["BrandEnd"]
        })
    lifetimes = pd.DataFrame(chosen)
    if lifetimes.empty:
        return lifetimes.reindex(columns=["Plan","Cohort Year","Month","At_Risk","Retention"])

    # Emit one row per month from First_Start to End (inclusive), capped at max_months
    out_rows = []
    for _, r in lifetimes.iterrows():
        months = (r["End"].year - r["First_Start"].year) * 12 + (r["End"].month - r["First_Start"].month) + 1
        months = int(min(months, max_months))
        for m in range(1, months + 1):
            out_rows.append({
                "Plan": r["Plan"],
                "Cohort Year": int(r["Cohort Year"]),
                "Month": m,
                "At_Risk": 1
            })
    surv_long = pd.DataFrame(out_rows)
    curves = (
        surv_long.groupby(["Plan","Cohort Year","Month"], as_index=False)["At_Risk"]
                 .sum()
                 .rename(columns={"At_Risk":"Alive"})
    )
    den = curves[curves["Month"]==1][["Plan","Cohort Year","Alive"]].rename(columns={"Alive":"N0"})
    curves = curves.merge(den, on=["Plan","Cohort Year"], how="left")
    curves["At_Risk"] = curves["Alive"].astype(int)
    curves["Retention"] = (curves["Alive"] / curves["N0"]).astype(float)
    return curves[["Plan","Cohort Year","Month","At_Risk","Retention"]]


def build_survival_from_brand_windows_OLD(
    brand_windows: pd.DataFrame,
    entry_plans: pd.DataFrame,
    max_months: int = 48,
) -> pd.DataFrame:
    """
    Construct a monthly survival table (brand-level retention) segmented by Entry_Plan.
    Output columns:
      ['Plan', 'Cohort Year', 'Month', 'At_Risk', 'Retention']
    Notes:
    - Month is 1-indexed: first month after First_Start is Month=1.
    - Members are 'alive' while they are on ANY plan (brand windows).
    - Right-censoring already handled in brand_windows.
    """
    bw = brand_windows.copy()
    ep = entry_plans.copy()

    # Merge entry plan onto brand windows; drop members who never had an allowed Entry_Plan
    x = bw.merge(ep, on="Name_or_Email", how="inner")
    x["First_Start"] = pd.to_datetime(x["First_Start"], errors="coerce")
    x = x.dropna(subset=["First_Start", "BrandStart", "BrandEnd"])

    rows = []
    for _, r in x.iterrows():
        # Cohort Year is the year of the member's FIRST allowed plan
        cohort_year = r["First_Start"].year
        # We define tenure relative to First_Start so members who joined later in the year align
        start = r["First_Start"]
        end = r["BrandEnd"]
        if pd.isna(start) or pd.isna(end) or end < start:
            continue
        months = (end.year - start.year) * 12 + (end.month - start.month) + 1
        months = int(min(months, max_months))
        for m in range(1, months + 1):
            rows.append({
                "Plan": r["Entry_Plan"],       # <- name it 'Plan' for the app
                "Cohort Year": cohort_year,
                "Month": m,
                "At_Risk": 1
            })

    surv_long = pd.DataFrame(rows)
    if surv_long.empty:
        return surv_long

    curves = (
        surv_long.groupby(["Plan", "Cohort Year", "Month"], as_index=False)["At_Risk"]
        .sum()
        .rename(columns={"At_Risk": "Alive"})
    )

    # Denominator: Alive at Month 1 per (Plan, Cohort Year)
    den = curves[curves["Month"] == 1][["Plan", "Cohort Year", "Alive"]]
    den = den.rename(columns={"Alive": "N0"})
    curves = curves.merge(den, on=["Plan", "Cohort Year"], how="left")
    curves["At_Risk"] = curves["Alive"].astype(int)
    curves["Retention"] = (curves["Alive"] / curves["N0"]).astype(float)

    return curves[["Plan", "Cohort Year", "Month", "At_Risk", "Retention"]]

def build_pooled_hazard_from_survival_and_windows(
    survival: pd.DataFrame,
    brand_windows: pd.DataFrame,
    entry_plans: pd.DataFrame,
    cap_date: pd.Timestamp = CAP_DATE,
) -> pd.DataFrame:
    """
    Pooled hazard: exits are the *first* brand exit after First_Start for each member, not one per window.
    """
    surv = survival.copy()
    bw = brand_windows.copy()
    ep = entry_plans.copy()

    # Pool At_Risk across cohorts/months
    risk = surv.groupby(["Plan","Month"], as_index=False)["At_Risk"].sum()

    x = bw.merge(ep, on="Name_or_Email", how="inner")
    x["First_Start"] = pd.to_datetime(x["First_Start"], errors="coerce")
    x["BrandStart"]  = pd.to_datetime(x["BrandStart"], errors="coerce")
    x["BrandEnd"]    = pd.to_datetime(x["BrandEnd"],   errors="coerce")
    x = x.dropna(subset=["First_Start","BrandStart","BrandEnd"])

    # Find first exit (BrandEnd) at or after First_Start per member
    first_exit_rows = []
    for name, g in x.sort_values(["BrandStart","BrandEnd"]).groupby("Name_or_Email"):
        fs = g["First_Start"].iloc[0]
        g = g.sort_rows = g.sort_values("BrandStart")
        containing = g[(g["BrandStart"] <= fs) & (g["BrandEnd"] >= fs)]
        if not containing.empty:
            end = containing["BrandEnd"].iloc[0]
            ep_label = containing["Entry_Plan"].iloc[0]
        else:
            after = g[g["BrandStart"] > fs]
            if after.empty:
                continue
            row = after.iloc[0]
            end, ep_label = row["BrandEnd"], row["Entry_Plan"]
        # Exit month is the NEXT month after last active month; ignore if censored
        censored = end >= cap_date
        if censored:
            continue
        last_month = (end.year - fs.year) * 12 + (end.month - fs.month) + 1
        first_exit_rows.append({"Plan": ep_label, "Month": int(last_month + 1)})

    ex = pd.DataFrame(first_exit_rows)
    if ex.empty:
        haz = risk.copy()
        haz["Exits"] = 0
        haz["Avg_Hazard"] = 0.0
        return haz[["Plan","Month","At_Risk","Exits","Avg_Hazard"]]

    ex_count = ex.groupby(["Plan","Month"], as_index=False).size().rename(columns={"size":"Exits"})
    haz = risk.merge(ex_count, on=["Plan","Month"], how="left")
    haz["Exits"] = haz["Exits"].fillna(0).astype(int)
    haz["Avg_Hazard"] = (haz["Exits"] / haz["At_Risk"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return haz[["Plan","Month","At_Risk","Exits","Avg_Hazard"]]



def build_pooled_hazard_from_survival_and_windows_OLD(
    survival: pd.DataFrame,
    brand_windows: pd.DataFrame,
    entry_plans: pd.DataFrame,
    cap_date: pd.Timestamp = CAP_DATE,
) -> pd.DataFrame:
    """
    Compute pooled monthly hazard by tenure month and entry plan.
    Output columns:
      ['Plan', 'Month', 'At_Risk', 'Exits', 'Avg_Hazard']
    Semantics:
      - 'At_Risk' at Month m is pooled across cohorts from survival table.
      - 'Exits' at Month m count members whose BRAND window ended in Month m,
        and who are NOT censored at the cap date.
    """
    surv = survival.copy()
    bw = brand_windows.copy()
    ep = entry_plans.copy()

    # Pool At_Risk across cohorts to stabilize month estimates
    risk_pool = (
        surv.groupby(["Plan", "Month"], as_index=False)["At_Risk"]
        .sum()
    )

    # Determine each member's last observed tenure month and censoring
    x = bw.merge(ep, on="Name_or_Email", how="inner")
    x["First_Start"] = pd.to_datetime(x["First_Start"], errors="coerce")
    x = x.dropna(subset=["First_Start", "BrandEnd"])
    x["censored"] = (x["BrandEnd"] >= cap_date)

    # Last active month index relative to First_Start (1-indexed)
    last_month = ((x["BrandEnd"].dt.year - x["First_Start"].dt.year) * 12
                  + (x["BrandEnd"].dt.month - x["First_Start"].dt.month) + 1).astype("Int64")
    x["LastMonth"] = last_month

    # Exits occur in LastMonth + 1 ONLY for uncensored rows
    exits = x.loc[~x["censored"], ["Entry_Plan", "LastMonth"]].copy()
    exits["Month"] = exits["LastMonth"] + 1

    ex_count = (
        exits.groupby(["Entry_Plan", "Month"], as_index=False)
        .size()
        .rename(columns={"Entry_Plan": "Plan", "size": "Exits"})
    )

    haz = risk_pool.merge(ex_count, on=["Plan", "Month"], how="left")
    haz["Exits"] = haz["Exits"].fillna(0).astype(int)
    haz["Avg_Hazard"] = (haz["Exits"] / haz["At_Risk"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return haz[["Plan", "Month", "At_Risk", "Exits", "Avg_Hazard"]]

# ------------------------------------------------------------
# Early-annual-churn forensics (who appears to exit before 12 months?)
# ------------------------------------------------------------
def find_early_annual_churn(
    events: pd.DataFrame,
    brand_windows: pd.DataFrame,
    entry_plans: pd.DataFrame,
    months: int = 12,
) -> pd.DataFrame:
    """
    Return a table of premium-plus *entry* members whose brand window ends before `months`
    after First_Start (i.e., potential early annual cancellations). Includes flags indicating
    whether they had *other plan* coverage overlapping the first-year window.
    """
    e = events.copy()
    e["Start"] = pd.to_datetime(e["Start"], errors="coerce")
    e["End"]   = pd.to_datetime(e["End"],   errors="coerce")
    e["Plan"]  = e["Plan"].astype(str).str.lower()

    bw = brand_windows.copy()
    ep = entry_plans.copy()

    # Join entry info to brand windows
    x = bw.merge(ep, on="Name_or_Email", how="inner")
    x["First_Start"] = pd.to_datetime(x["First_Start"], errors="coerce")
    x["BrandEnd"]    = pd.to_datetime(x["BrandEnd"],    errors="coerce")

    # Focus on premium-plus entry only (annual)
    x = x[x["Entry_Plan"] == "premium-plus"].copy()

    # Compute 1-indexed tenure months at brand end
    x["LastMonth"] = ((x["BrandEnd"].dt.year - x["First_Start"].dt.year) * 12
                      + (x["BrandEnd"].dt.month - x["First_Start"].dt.month) + 1).astype("Int64")

    # Candidate early exits: brand ended before month threshold AND not censored
    x["censored"] = (x["BrandEnd"] >= CAP_DATE)
    early = x[(x["LastMonth"].notna()) & (x["LastMonth"] < months) & (~x["censored"])].copy()
    if early.empty:
        return early.assign(had_other_plan_overlap=False, other_plans="")

    # For each early-exit member, check if any other plan overlapped first-year window
    def _has_other_plan_overlap(row):
        # First-year window [First_Start, First_Start + months)
        start = row["First_Start"]
        end   = start + pd.DateOffset(months=months)
        mask  = (
            (e["Name_or_Email"] == row["Name_or_Email"]) &
            (e["Plan"] != "premium-plus") &  # other plans (e.g., premium-plus-monthly or others)
            (e["Start"] < end) &
            (e["End"]   > start)
        )
        others = e.loc[mask, "Plan"].unique().tolist()
        return pd.Series({"had_other_plan_overlap": len(others) > 0,
                          "other_plans": ", ".join(sorted(others))})

    overlaps = early.apply(_has_other_plan_overlap, axis=1)
    early = pd.concat([early.reset_index(drop=True), overlaps], axis=1)

    # Keep readable columns
    out = early[[
        "Name_or_Email", "Entry_Plan", "First_Start", "BrandEnd", "LastMonth",
        "had_other_plan_overlap", "other_plans"
    ]].sort_columns() if hasattr(early, "sort_columns") else early.sort_values(["First_Start", "LastMonth"])

    return out


# -----------------------
# Orchestrators
# -----------------------

def build_survival_and_hazard(
    df: pd.DataFrame,
    allowed_plans: Iterable[str] = ALLOWED_PLANS,
    cap_date: pd.Timestamp = CAP_DATE,
    max_months: int = 48,
    grace_days: int = 0,
    already_events: bool = False,
    apply_exclusions: bool = True,
    return_audit: bool = False,
):
    audit = AuditTracker() if return_audit else None
    d = df.copy()

    # If caller passed raw customers with First/Last/Email and wants exclusions
    if apply_exclusions and {"First Name","Last Name","Email"}.issubset(d.columns) and not already_events:
        d = _apply_global_exclusions(d, audit=audit, stage_label="EXCLUDE: paul/org (raw)")
    elif apply_exclusions and already_events:
        # events-level fallback (Name_or_Email + domain)
        d = _apply_event_level_exclusions(d, audit=audit, stage_label="EXCLUDE: events-level (domain/name)")
    else:
        if audit is not None:
            audit.add("EXCLUDE: skipped", d, d, note="apply_exclusions=False")

    events = _normalize_event_columns(d, cap_date=cap_date, audit=audit, stage_label="NORMALIZE: raw->events" if not already_events else "VERIFY: events (normalize)")
    brand_windows = merge_brand_intervals(events, cap_date=cap_date, grace_days=grace_days, audit=audit, stage_label="MERGE: brand windows")
    entry_plans   = derive_entry_plan(events, allowed_plans=allowed_plans, audit=audit, stage_label="DERIVE: entry plan")

    if entry_plans.empty or brand_windows.empty:
        surv = pd.DataFrame(columns=["Plan","Cohort Year","Month","At_Risk","Retention"])
        haz  = np.DataFrame(columns=["Plan","Month","At_Risk","Exits","Avg_Hazard"])
        return (surv, haz, audit.to_frame()) if return_audit else (surv, haz)

    surv = build_survival_from_brand_windows(brand_windows, entry_plans, max_months=max_months)
    haz  = build_pooled_hazard_from_survival_and_windows(surv, brand_windows, entry_plans, cap_date=cap_date)

    if return_audit:
        return surv, haz, audit.to_frame()
    return surv, haz


def build_survival_and_hazard_MAYBE(
    df: pd.DataFrame,
    allowed_plans: Iterable[str] = ALLOWED_PLANS,
    cap_date: pd.Timestamp = CAP_DATE,
    max_months: int = 48,
    grace_days: int = 0,
    already_events: bool = False,
    apply_exclusions: bool = True,
    return_audit: bool = False,
):
    """
    End-to-end builder for survival & hazard; optionally returns an audit DataFrame.
    """
    audit = AuditTracker() if return_audit else None

    d = df.copy()

    #  filter raw customer table
    if apply_exclusions and {"First Name","Last Name","Email"}.issubset(d.columns):
        d2 = _apply_global_exclusions(d, audit=audit, stage_label="EXCLUDE: paul/org")
    else:
        d2 = d.copy()
        if audit is not None:
            audit.add("EXCLUDE: skipped", d, d2, note="No exclusion columns or disabled")

    # Normalize to events (across ALL plans), with right-censoring
    events = _normalize_event_columns(d2, cap_date=cap_date, audit=audit, stage_label="NORMALIZE: raw->events")

    # Build brand windows (all plans merged)
    brand_windows = merge_brand_intervals(events, cap_date=cap_date, grace_days=grace_days, audit=audit, stage_label="MERGE: brand windows")

    # Entry plan (first allowed plan)
    entry_plans = derive_entry_plan(events, allowed_plans=allowed_plans, audit=audit, stage_label="DERIVE: entry plan")

    # Short-circuit if no cohorts
    if entry_plans.empty or brand_windows.empty:
        surv = pd.DataFrame(columns=["Plan","Cohort Year","Month","At_Risk","Retention"])
        haz  = pd.DataFrame(columns=["Plan","Month","At_Risk","Exits","Avg_Hazard"])
        if return_audit:
            return surv, haz, audit.to_frame()
        return surv, haz

    # Survival + hazard
    surv = build_survival_from_brand_windows(brand_windows, entry_plans, max_months=max_months)
    haz  = build_pooled_hazard_from_survival_and_windows(surv, brand_windows, entry_plans, cap_date=cap_date)

    if return_audit:
        return surv, haz, audit.to_frame()
    return surv, haz


def build_survival_and_hazard_OLD(
    df: pd.DataFrame,
    allowed_plans: Iterable[str] = ALLOWED_PLANS,
    cap_date: pd.Timestamp = CAP_DATE,
    max_months: int = 48,
    grace_days: int = 0,
    already_events: bool = False,
    apply_exclusions: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end builder for survival & hazard from either a raw customer table or an events table.
    Returns: (survival_df, hazard_df)
    - Survival has columns: ['Plan', 'Cohort Year', 'Month', 'At_Risk', 'Retention']
    - Hazard   has columns: ['Plan', 'Month', 'At_Risk', 'Exits', 'Avg_Hazard']
    """
    d = df.copy()

    # Optional safety exclusion (Paul/test) if caller passed raw customer table
    if apply_exclusions:
        # Only run if the columns exist; otherwise no-ops
        if {"First Name", "Last Name", "Email"}.issubset(set(d.columns)):
            d = _apply_global_exclusions(d)

    # Normalize to events across ALL plans
    if already_events:
        events = _normalize_event_columns(d, cap_date=cap_date)
    else:
        events = _normalize_event_columns(d, cap_date=cap_date)

    # Build brand windows and entry-plan segmentation
    brand_windows = merge_brand_intervals(events, cap_date=cap_date, grace_days=grace_days)
    entry_plans = derive_entry_plan(events, allowed_plans=allowed_plans)

    # People who never had an allowed entry plan are excluded from these cohorts
    # (but still contributed to brand windows if needed elsewhere)
    if entry_plans.empty or brand_windows.empty:
        return pd.DataFrame(columns=["Plan", "Cohort Year", "Month", "At_Risk", "Retention"]), \
               pd.DataFrame(columns=["Plan", "Month", "At_Risk", "Exits", "Avg_Hazard"])

    survival = build_survival_from_brand_windows(brand_windows, entry_plans, max_months=max_months)
    hazard   = build_pooled_hazard_from_survival_and_windows(survival, brand_windows, entry_plans, cap_date=cap_date)

    return survival, hazard


# -----------------------
# Backward-compat stubs
# -----------------------

def build_gap_adjusted_intervals(d: pd.DataFrame, grace_months: int = 0, cap_date: pd.Timestamp = CAP_DATE) -> pd.DataFrame:
    """
    Backward-compatible wrapper for older code paths that expected:
      - a dataframe with 'Activation Date'/'Cancellation Date'
      - plan-specific intervals
    We now:
      - accept either raw or normalized columns,
      - convert to Start/End,
      - right-censor,
      - and return brand-level intervals (across all plans) with an optional grace gap.
    Output columns: ['Name_or_Email', 'BrandStart', 'BrandEnd']
    """
    # Map old column names if present (prevents KeyError: 'Activation Date')
    df = d.copy()
    if "Activation Date" in df.columns and "Start" not in df.columns:
        df = df.rename(columns={"Activation Date": "Start"})
    if "Cancellation Date" in df.columns and "End" not in df.columns:
        df = df.rename(columns={"Cancellation Date": "End"})

    events = _normalize_event_columns(df, cap_date=cap_date)
    # Convert month-gap to days (approximate 30.44 days/month) for merging logic
    grace_days = int(round(grace_months * 30.44))
    return merge_brand_intervals(events, cap_date=cap_date, grace_days=grace_days)


# -----------------------
# OLD: maybe need to delete? 
# -----------------------

def build_gap_adjusted_intervals_OLD(df,
                                 grace_months=0,
                                 cap_date='2025-12-31',
                                 merge_across_plans=False,
                                 plan_col='Plan'):
    """
    Merge subscription intervals with a grace window.
    If merge_across_plans=True, merge across all plans per user (brand-level).
    Else, merge within each plan per user (plan-level).
    """
    cap_dt = pd.to_datetime(cap_date)
    d = df.copy()
    # Ensure columns exist & are datetimes
    if plan_col not in d.columns:
        d[plan_col] = d['Current Plan Handle']
    d['Activation Date'] = pd.to_datetime(d['Activation Date'])
    d['End Date'] = pd.to_datetime(d['End Date']).fillna(cap_dt).clip(upper=cap_dt)

    # Grouping key: across plans (brand) or within plan
    keys = ['Name_or_Email'] if merge_across_plans else ['Name_or_Email', plan_col]

    rows = []
    for key, g in d.sort_values(keys + ['Activation Date']).groupby(keys):
        g = g.reset_index(drop=True)
        cur_s = g.loc[0, 'Activation Date']
        cur_e = g.loc[0, 'End Date']
        for i in range(1, len(g)):
            s = g.loc[i, 'Activation Date']
            e = g.loc[i, 'End Date']
            # Merge if within grace (or overlapping)
            if (grace_months is None) or (s <= cur_e + relativedelta(months=grace_months)):
                cur_e = max(cur_e, e)
            else:
                out = {'Name_or_Email': g.loc[0, 'Name_or_Email'],
                       'Start': cur_s, 'End': cur_e}
                if not merge_across_plans: out[plan_col] = g.loc[0, plan_col]
                rows.append(out)
                cur_s = s 
                cur_e = e
        out = {'Name_or_Email': g.loc[0, 'Name_or_Email'],
               'Start': cur_s, 'End': cur_e}
        if not merge_across_plans: out[plan_col] = g.loc[0, plan_col]
        rows.append(out)

    intervals = pd.DataFrame(rows)
    # For brand-level, assign a single plan label so downstream code works unchanged
    if merge_across_plans:
        intervals[plan_col] = 'All'  # a single brand label
    return intervals

def expand_intervals_to_records_OLD(intervals, max_months=24):
    """
    Expand ['Name_or_Email','Plan','Start','End'] to user-month records with Month index from first start.
    Returns: ['Name_or_Email','Plan','Cohort Year','Month']
    """
    rows = []

    first = (
        intervals
        .groupby(['Name_or_Email','Plan'])['Start']
        .min()
        .reset_index(name='First_Start')
    )

    fdict = {(r.Name_or_Email, r.Plan): r.First_Start for r in first.itertuples()}

    for r in intervals.itertuples(index=False):
        user = r.Name_or_Email
        plan = r.Plan
        s = pd.to_datetime(r.Start)
        e = pd.to_datetime(r.End)
        start0 = fdict[(user, plan)]
        cohort_year = start0.year
        m = ((s.year - start0.year) * 12 + (s.month - start0.month) + 1)
        cur = s

        while cur <= e and m <= max_months:
            rows.append({
                'Name_or_Email': user,
                'Plan': plan,
                'Cohort Year': cohort_year,
                'Month': m
            })
            cur = cur + relativedelta(months=1)
            m = m + 1

    return pd.DataFrame(rows)

def map_entry_plan_OLD(df, plan_col='Plan'):
    """
    Map each user to their *plan of entry* (first plan ever taken).
    Returns: DataFrame ['Name_or_Email','Entry_Plan','First_Start']
    """
    tmp = df.copy()

    if plan_col not in tmp.columns:
        tmp[plan_col] = tmp['Current Plan Handle']

    tmp['Activation Date'] = pd.to_datetime(tmp['Activation Date'])

    first = (
        tmp.sort_values(['Name_or_Email','Activation Date'])
           .groupby('Name_or_Email')
           .first()
           .reset_index()
    )

    out = first[['Name_or_Email', plan_col, 'Activation Date']].copy()
    out = out.rename(columns={plan_col: 'Entry_Plan', 'Activation Date': 'First_Start'})

    return out

def survival_from_records_OLD(rec_df):
    """
    Compute survival by (Cohort Year, Plan, Month) from user-month records.
    Returns: ['Cohort Year','Plan','Month','Active','Cohort Size','Retention','Churn Rate']
    """
    cohort_sizes = (
        rec_df[rec_df['Month'] == 1]
        .groupby(['Cohort Year','Plan'])['Name_or_Email']
        .nunique()
        .reset_index(name='Cohort Size')
    )

    active = (
        rec_df
        .groupby(['Cohort Year','Plan','Month'])['Name_or_Email']
        .nunique()
        .reset_index(name='Active')
    )

    surv = active.merge(cohort_sizes, on=['Cohort Year','Plan'], how='left')
    surv['Retention'] = surv['Active'] / surv['Cohort Size']
    surv['Churn Rate'] = 1 - surv['Retention']

    return surv


def compute_average_monthly_hazard_OLD(surv_df, weight_by_cohort=True):
    """
    Input: survival DF with ['Plan','Cohort Year','Month','Active','Cohort Size', 'Retention Rate' or 'Retention']
    Output: average hazard per month for each Plan (and overall if Plan='All').

    hazard_t = (R_{t-1} - R_t) / R_{t-1}, averaged across cohorts.
    If weight_by_cohort=True, cohorts are weighted by their size (Month==1).
    """
    df = surv_df.copy()
    # Normalize retention column name
    if 'Retention' in df.columns:
        df['R'] = df['Retention']
    else:
        df['R'] = df['Retention Rate']

    # cohort weights
    w = (df[df['Month']==1][['Plan','Cohort Year','Cohort Size']]
         .drop_duplicates()
         .rename(columns={'Cohort Size':'Weight'}))

    # Compute hazard per (Plan, Cohort Year, Month)
    df = df.sort_values(['Plan','Cohort Year','Month'])
    df['R_prev'] = df.groupby(['Plan','Cohort Year'])['R'].shift(1)
    df['Hazard'] = (df['R_prev'] - df['R']) / df['R_prev']
    df.loc[df['Month']==1, 'Hazard'] = np.nan  # undefined at Month 1

    # Merge weights
    df = df.merge(w, on=['Plan','Cohort Year'], how='left')

    # Average by month & plan
    if weight_by_cohort:
        out = (df.dropna(subset=['Hazard'])
                 .groupby(['Plan','Month'])
                 .apply(lambda g: np.average(g['Hazard'], weights=g['Weight']))
                 .reset_index(name='Avg_Hazard'))
    else:
        out = (df.dropna(subset=['Hazard'])
                 .groupby(['Plan','Month'])['Hazard']
                 .mean()
                 .reset_index(name='Avg_Hazard'))
    return out