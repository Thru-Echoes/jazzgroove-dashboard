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
#CAP_DATE = pd.Timestamp("2025-12-31 23:59:59")  # right-censoring cutoff, if April 29, 2025 = "2024-04-29"
CAP_DATE = pd.Timestamp("2025-04-29 23:59:59")


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
    audit: Optional["AuditTracker"] = None,
    stage_label: str = "MERGE: brand windows",
) -> pd.DataFrame:
    """
    Collapse a member's subscription rows across ALL plans into brand-level intervals.
    Intervals are merged if the gap between a window's end and the next window's start
    is <= `grace_days`. All End dates are right-censored at `cap_date`.

    Output per row: ['Name_or_Email','BrandStart','BrandEnd']
    """
    before = events.copy()

    e = events.copy()
    e["Start"] = pd.to_datetime(e["Start"], errors="coerce")
    e["End"]   = pd.to_datetime(e["End"],   errors="coerce")

    # Drop tz to avoid tz-aware vs tz-naive comparisons
    for col in ("Start", "End"):
        if is_datetime64tz_dtype(e[col]):
            e[col] = e[col].dt.tz_convert("UTC").dt.tz_localize(None)

    e = e.dropna(subset=["Name_or_Email", "Start"])
    cap = pd.to_datetime(cap_date)
    e["End"] = e["End"].fillna(cap)
    e.loc[e["End"] > cap, "End"] = cap

    e = e.sort_values(["Name_or_Email", "Start"]).reset_index(drop=True)

    rows = []
    for name, g in e.groupby("Name_or_Email", sort=False):
        cur_s = None
        cur_e = None
        for _, r in g.iterrows():
            s, en = r["Start"], r["End"]
            if cur_s is None:
                cur_s, cur_e = s, en
                continue
            # Merge if the gap is within grace_days (or negative/zero)
            gap_days = (s - cur_e).days if pd.notna(s) and pd.notna(cur_e) else None
            if gap_days is None or gap_days <= max(grace_days, 0):
                if en > cur_e:
                    cur_e = en
            else:
                rows.append((name, cur_s, cur_e))
                cur_s, cur_e = s, en
        if cur_s is not None:
            rows.append((name, cur_s, cur_e))

    out = pd.DataFrame(rows, columns=["Name_or_Email", "BrandStart", "BrandEnd"])

    if audit is not None:
        note = f"members={before['Name_or_Email'].nunique()} -> windows={len(out)} ; grace_days={grace_days}"
        audit.add(stage_label, before, out, note=note)
    return out

def merge_brand_intervals_OLD(
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
    allowed_plans: list[str] | set[str] | None = None,
    plan_col: str = "Plan",
) -> pd.DataFrame:
    """
    Return first-ever plan & start per member with NO plan-name remapping.
    Columns: ['Name_or_Email','Entry_Plan','First_Start']
    """
    d = events.copy()
    d = d.dropna(subset=["Name_or_Email", plan_col, "Start"])
    d["Start"] = pd.to_datetime(d["Start"], errors="coerce")

    # Optional filter (default is None: no filtering)
    if allowed_plans is not None:
        d = d[d[plan_col].isin(allowed_plans)].copy()

    first_rows = (
        d.sort_values(["Name_or_Email", "Start"])
         .groupby("Name_or_Email", as_index=False)
         .first()   # the earliest row after sorting
    )
    out = first_rows.rename(columns={plan_col: "Entry_Plan", "Start": "First_Start"})
    return out[["Name_or_Email", "Entry_Plan", "First_Start"]]


# ------------------------------------------------------------
# Cohort / plan slicing helpers for account_windows
# ------------------------------------------------------------

def ensure_cohort_year_column(
    account_windows: pd.DataFrame,
    first_start_col: str = "First_Start",
    out_col: str = "Cohort Year",
) -> pd.DataFrame:
    """
    Ensure an integer 'Cohort Year' column exists on account_windows.
    We derive it from the year of the first activation (First_Start).
    """
    # Make a defensive copy so we don't mutate the caller
    d = account_windows.copy()
    # If the expected column isn't present yet, compute it from the 'First_Start' timestamp
    if out_col not in d.columns:
        d[out_col] = pd.to_datetime(d[first_start_col], errors="coerce").dt.year.astype("Int64")
    # Return the augmented DataFrame
    return d


def split_by_cohort_year(
    account_windows: pd.DataSheet,  # type: ignore (alias for readability)
    cohort_year: int,
    first_start_col: str = "First_Start",
) -> pd.DataFrame:
    """
    Return only the rows from account_windows that belong to the requested cohort year,
    defined by the year component of First_Start (or an existing 'Cohort Year' column if present).
    """
    # Ensure we have a 'Cohort Year' column to filter by
    d = ensure_cohort_year_column(account_windows, first_start_col=first_start_col, out_col="Cohort Year")
    # Filter rows where the derived cohort year equals the requested year
    out = d[d["Cohort Year"] == cohort_year].copy()
    # Return the filtered DataFrame
    return out


def split_by_entry_plans(
    account_windows: pd.DataFrame,
    plans: list[str],
    plan_col: str = "Entry_Plan",
) -> pd.DataFrame:
    """
    Return only rows whose entry plan value is in the user-specified list of plan labels.
    Matching is done case-insensitively.
    """
    # Create a case-insensitive set of desired plan labels
    target = {p.lower() for p in plans}
    # Create a defensive copy
    d = account_windows.copy()
    # Create a temporary lowercase view of the plan column for matching
    d["_plan_lc"] = d[plan_col].astype("string").str.lower()
    # Filter rows whose normalized plan is in the target set
    out = d[d["_plan_lc"].isin(target)].copy()
    # Drop the helper column before returning
    out = out.drop(columns=["_plan_lc"])
    # Return the filtered DataFrame
    return out


# -----------------------
# BUILD ONE ROW PER ACCOUNT (contiguous brand block) 
# -----------------------

def _extract_identities_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return unique identities ['Name_or_Email','First Name','Last Name','Email'] found in df,
    or an empty DataFrame with those columns if not present.
    """
    wants = ["Name_or_Email", "First Name", "Last Name", "Email"]
    have = [c for c in wants if c in df.columns]
    if not have or "Name_or_Email" not in have:
        # ensure we return these columns even if empty
        return pd.DataFrame(columns=wants)
    ids = df[have].copy()
    # Lightweight normalization (do NOT lowercase the key; we rely on exact match)
    ids["Name_or_Email"] = ids["Name_or_Email"].astype("string").str.strip()
    return ids.drop_duplicates(subset=["Name_or_Email"])


# --- NEW: Build one row per (account × contiguous brand block) without changing plan names ---
def build_account_brand_blocks(
    events: pd.DataFrame,
    identities: pd.DataFrame | None = None,
    grace_days: int = 0,
    cap_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Inputs
    ------
    events      : DataFrame with at least ['Name_or_Email','Plan','Start','End'] (ALL plans; no renaming)
    identities  : Optional DataFrame with ['Name_or_Email','First Name','Last Name','Email'] to enrich rows
    grace_days  : Merge consecutive subscriptions into the same brand block if gap <= grace_days
    cap_date    : Right-censoring date (e.g., '2025-04-29')

    Output
    ------
    One row per Name_or_Email per contiguous brand block with columns:
      Name_or_Email, First Name, Last Name, Email,
      Entry_Plan, First_Start,
      BrandStart, BrandEnd, LastMonth, Number_of_Subscriptions,
      Cancel_ts, Cancel_Month_Index, censored,
      had_other_plan_overlap, other_plans

    Notes
    -----
    • No plan-name simplification or filtering is applied.
    • LastMonth counts months from BrandStart→BrandEnd (1-indexed), so overlapping subs don’t double-count.
    • Cancel_ts is BrandEnd iff not censored; censored means BrandEnd >= cap_date.
    • other_plans are distinct plan slugs observed inside the block EXCLUDING the Entry_Plan.
    """
    d = events.copy()

    d["Name_or_Email"] = d["Name_or_Email"].astype("string").str.strip()


    # Datetime hygiene (tz-naive + cap)
    d["Start"] = pd.to_datetime(d["Start"], errors="coerce")
    d["End"]   = pd.to_datetime(d["End"],   errors="coerce")

    if is_datetime64tz_dtype(d["Start"]):
        d["Start"] = d["Start"].dt.tz_convert("UTC").dt.tz_localize(None)
    if is_datetime64tz_dtype(d["End"]):
        d["End"] = d["End"].dt.tz_convert("UTC").dt.tz_localize(None)

    cap = None
    if cap_date is not None:
        cap = pd.to_datetime(cap_date)

        # FIXME: remove debuggers 
        #import pdb 
        #pdb.set_trace()

        # Fill open intervals with the cap, then clip any future End to cap
        d["End"] = d["End"].fillna(cap)
        d.loc[d["End"] > cap, "End"] = cap

    # Make sure tz-naive (avoid tz-aware vs naive comparisons later)
    for col in ("Start", "End"):
        if pd.api.types.is_datetime64_any_dtype(d[col]):
            try:
                d[col] = d[col].dt.tz_localize(None)
            except Exception:
                pass

    # Build contiguous brand windows across ALL plans (no renaming)
    brand_windows = merge_brand_intervals(d, cap_date=cap_date, grace_days=grace_days)

    # Derive Entry_Plan & First_Start from raw data (NO filtering/simplification)
    ep = derive_entry_plan(d, allowed_plans=None, plan_col="Plan")  # ensure plan-preserving
    ep["First_Start"] = pd.to_datetime(ep["First_Start"], errors="coerce")
    try:
        ep["First_Start"] = ep["First_Start"].dt.tz_localize(None)
    except Exception:
        pass

    # Identity enrichment: prefer explicit identities; otherwise fall back to events
    ident_keys = ["Name_or_Email", "First Name", "Last Name", "Email"]
    if identities is not None and not identities.empty:
        id_ok = [c for c in ident_keys if c in identities.columns]
        ids = identities[id_ok].copy()
    else:
        ids = _extract_identities_from_df(d)  # <— pull identities from events if present

    if not ids.empty:
        ids["Name_or_Email"] = ids["Name_or_Email"].astype("string").str.strip()
        ids = ids.drop_duplicates(subset=["Name_or_Email"])


    out_rows = []


    # Process per member
    for name, g_win in brand_windows.sort_values(["Name_or_Email","BrandStart"]).groupby("Name_or_Email"):
        # entry info
        ep_row = ep.loc[ep["Name_or_Email"] == name]
        if ep_row.empty:
            entry_plan  = pd.NA
            first_start = pd.NaT
        else:
            entry_plan  = ep_row["Entry_Plan"].iloc[0]
            first_start = ep_row["First_Start"].iloc[0]

        # identity info
        if not ids.empty:
            id_row = ids.loc[ids["Name_or_Email"] == name].head(1)
            first_name = id_row["First Name"].iloc[0] if ("First Name" in id_row.columns and not id_row.empty) else pd.NA
            last_name  = id_row["Last Name"].iloc[0] if ("Last Name"  in id_row.columns and not id_row.empty) else pd.NA
            email      = id_row["Email"].iloc[0]      if ("Email"      in id_row.columns and not id_row.empty) else pd.NA
        else:
            first_name = last_name = email = pd.NA

        # All events for this member (to count subs & detect other plans inside each block)
        ev_m = d[d["Name_or_Email"] == name].copy()

        for _, w in g_win.sort_values("BrandStart").iterrows():
            bs = pd.to_datetime(w["BrandStart"], errors="coerce")
            be = pd.to_datetime(w["BrandEnd"],   errors="coerce")
            try:
                bs = bs.tz_localize(None)
                be = be.tz_localize(None)
            except Exception:
                pass

            # Which raw subscriptions overlap this block?
            in_block = ev_m[(ev_m["Start"] <= be) & (ev_m["End"] >= bs)]
            nsubs    = int(len(in_block))

            # Plans observed inside the block (distinct, preserve order)
            plans_in_block = in_block["Plan"].astype(str).tolist()
            # Other plans (distinct) compared to Entry_Plan
            #other_list = sorted(pd.unique([p for p in plans_in_block if p != entry_plan]))
            
            other_list = sorted(set(p for p in plans_in_block if p != entry_plan))
            had_other  = len(other_list) > 0

            # Tenure in months (1-indexed) from BrandStart→BrandEnd
            last_month = (
                (be.year - bs.year) * 12 + (be.month - bs.month) + 1
            ) if (pd.notna(bs) and pd.notna(be)) else pd.NA

            # Censoring at cap
            cens = False
            if cap is not None and pd.notna(be):
                cens = bool(be >= cap)

            cancel_ts = pd.NaT if cens else be
            cancel_mi = (
                (be.year - first_start.year) * 12 + (be.month - first_start.month) + 1
            ) if (pd.notna(be) and pd.notna(first_start)) else pd.NA

            out_rows.append({
                "Name_or_Email": name,
                "First Name": first_name,
                "Last Name": last_name,
                "Email": email,

                "Entry_Plan": entry_plan,     # first-ever plan (raw, unmodified)
                "First_Start": first_start,   # first-ever start

                "BrandStart": bs,
                "BrandEnd": be,
                "LastMonth": int(last_month) if last_month is not pd.NA else pd.NA,
                "Number_of_Subscriptions": nsubs,

                "Cancel_ts": cancel_ts,
                "Cancel_Month_Index": int(cancel_mi) if cancel_mi is not pd.NA else pd.NA,
                "censored": cens,

                "had_other_plan_overlap": had_other,
                "other_plans": ", ".join(other_list),
            })

    res = pd.DataFrame(out_rows)
    # Stable ordering
    if not res.empty:
        res = res.sort_values(["Name_or_Email","BrandStart"], kind="stable").reset_index(drop=True)

    return res



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

    # drop tz if present to avoid tz-aware vs tz-naive comparison errors
    for col in ("First_Start", "BrandStart", "BrandEnd"):
        if is_datetime64tz_dtype(x[col]):
            x[col] = x[col].dt.tz_convert("UTC").dt.tz_localize(None)

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
    cap_date: pd.Timestamp = CAP_DATE,
) -> pd.DataFrame:
    """
    Identify premium-plus *entry* members whose *contiguous brand* tenure from the start
    of the brand block that contains their First_Start is < `months` (default: 12) and
    who are NOT censored by the cap date.

    Assumptions & semantics:
      • `events` is the ALL-plans normalized events table: ['Name_or_Email','Plan','Start','End'].
      • `brand_windows` is built from ALL plans via merge_brand_intervals(..., grace_days=...),
        so any PP→PPM (or other plan) switches with gaps <= grace_days are *already merged*
        into a single contiguous brand window.
      • `entry_plans` is from derive_entry_plan(events), includes:
            ['Name_or_Email','Entry_Plan','First_Start'] where Entry_Plan ∈ {'premium-plus','premium-plus-monthly'}.
      • We restrict to Entry_Plan == 'premium-plus' (annual cohort).
      • For each annual member, we find the brand window that *contains* their First_Start.
        That window’s BrandStart/BrandEnd define the contiguous brand block used for tenure.
      • LastMonth is computed as 1-indexed months from BrandStart to BrandEnd (contiguous brand tenure).
        This counts time across *any plans* within the brand block (PP→PPM transitions inside `grace_days` included).
      • A member is counted as "early" if LastMonth < `months` and NOT censored (BrandEnd < cap_date).

    Returns one row per Name_or_Email with stable column names:
      ['Name_or_Email','Entry_Plan','First_Start','BrandStart','BrandEnd',
       'LastMonth','Cancel_ts','Cancel_Month_Index','censored',
       'had_other_plan_overlap','other_plans']

      - First_Start: first start on the ENTRY plan (unchanged)
      - BrandStart/BrandEnd: start/end of the contiguous brand block that contains First_Start
      - LastMonth: months from BrandStart to BrandEnd (1-indexed, contiguous brand tenure)
      - Cancel_ts: BrandEnd if not censored, else NaT
      - Cancel_Month_Index: months from First_Start to BrandEnd (1-indexed, for backwards compatibility)
      - censored: True if BrandEnd >= cap_date (brand still active at cap)
      - had_other_plan_overlap: True if any non-PP plan overlaps [BrandStart, BrandEnd]
      - other_plans: comma-separated list of distinct non-PP plan slugs overlapping the block
    """
    cap = pd.to_datetime(cap_date)

    # Normalize inputs defensively
    e = events.copy()
    e["Plan"]  = e["Plan"].astype(str).str.lower()
    e["Start"] = pd.to_datetime(e["Start"], errors="coerce")
    e["End"]   = pd.to_datetime(e["End"],   errors="coerce")

    bw = brand_windows.copy()
    bw["BrandStart"] = pd.to_datetime(bw["BrandStart"], errors="coerce")
    bw["BrandEnd"]   = pd.to_datetime(bw["BrandEnd"],   errors="coerce")

    ep = entry_plans.copy()
    ep["Entry_Plan"]   = ep["Entry_Plan"].astype(str).str.lower()
    ep["First_Start"]  = pd.to_datetime(ep["First_Start"], errors="coerce")

    # Focus on annual (premium-plus) entry only
    ep_pp = ep[ep["Entry_Plan"] == "premium-plus"].copy()
    if ep_pp.empty:
        return pd.DataFrame(
            columns=[
                "Name_or_Email","Entry_Plan","First_Start","BrandStart","BrandEnd",
                "LastMonth","Cancel_ts","Cancel_Month_Index","censored",
                "had_other_plan_overlap","other_plans"
            ]
        )

    # Join brand windows to entry info
    x = bw.merge(ep_pp[["Name_or_Email","Entry_Plan","First_Start"]],
                 on="Name_or_Email", how="inner")
    x = x.dropna(subset=["First_Start","BrandStart","BrandEnd"]).copy()

    out_rows = []

    for name, g in x.groupby("Name_or_Email"):
        g = g.sort_values("BrandStart")
        fs = g["First_Start"].iloc[0]

        # find the brand window(s) that contain First_Start
        cov = g[(g["BrandStart"] <= fs) & (g["BrandEnd"] >= fs)]

        if not cov.empty:
            # because brand_windows is already merged by grace_days, there is typically one row here
            brand_start = cov["BrandStart"].min()
            brand_end   = cov["BrandEnd"].max()
        else:
            # Fallback: if no window contains First_Start (unexpected), pick the first window starting after fs
            after = g[g["BrandStart"] >= fs].sort_values("BrandStart")
            if after.empty:
                # No window at/after entry → treat as anomaly; keep the latest end before fs
                sel = g.iloc[[-1]]
            else:
                sel = after.iloc[[0]]
            brand_start = sel["BrandStart"].iloc[0]
            brand_end   = sel["BrandEnd"].iloc[0]

        # Contiguous brand tenure from BrandStart to BrandEnd (1-indexed months)
        # This counts PP and any other plans merged into this block by grace_days.
        last_month = (
            (brand_end.year - brand_start.year) * 12
            + (brand_end.month - brand_start.month)
            + 1
        )

        # Censoring at cap date
        cens = bool(brand_end >= cap)

        # For backwards compatibility:
        cancel_ts = pd.NaT if cens else brand_end
        cancel_month_from_first = (
            (brand_end.year - fs.year) * 12 + (brand_end.month - fs.month) + 1
        )

        # Context: did they have any non-PP plan overlapping this contiguous block?
        any_other_mask = (
            (e["Name_or_Email"] == name)
            & (e["Plan"] != "premium-plus")
            & (e["Start"] < brand_end)
            & (e["End"]   > brand_start)
        )
        other_plans = sorted(e.loc[any_other_mask, "Plan"].dropna().unique().tolist())
        had_other_overlap = len(other_plans) > 0

        out_rows.append({
            "Name_or_Email": name,
            "Entry_Plan": g["Entry_Plan"].iloc[0],
            "First_Start": fs,
            "BrandStart": brand_start,
            "BrandEnd": brand_end,
            "LastMonth": int(last_month) if pd.notna(brand_end) and pd.notna(brand_start) else pd.NA,
            "Cancel_ts": cancel_ts,
            "Cancel_Month_Index": int(cancel_month_from_first) if pd.notna(brand_end) and pd.notna(fs) else pd.NA,
            "censored": cens,
            "had_other_plan_overlap": had_other_overlap,
            "other_plans": ", ".join(other_plans),
        })

    early = pd.DataFrame(out_rows)

    # One row per account, no duplicates expected; if any, collapse by taking max BrandEnd/min BrandStart
    if not early.empty and early.duplicated("Name_or_Email").any():
        # merge duplicates conservatively: earliest BrandStart, latest BrandEnd, recompute durations/flags
        early = (
            early.sort_values(["Name_or_Email","BrandStart","BrandEnd"])
                 .groupby("Name_or_Email", as_index=False)
                 .agg({
                     "Entry_Plan":"first",
                     "First_Start":"min",
                     "BrandStart":"min",
                     "BrandEnd":"max",
                     "censored":"max",
                     "had_other_plan_overlap":"max",
                     "other_plans": lambda s: ", ".join(sorted(set(", ".join(s.dropna()).split(", ")))) if s.notna().any() else "",
                 })
        )
        # recompute tenure fields
        early["LastMonth"] = (
            (early["BrandEnd"].dt.year - early["BrandStart"].dt.year) * 12
            + (early["BrandEnd"].dt.month - early["BrandStart"].dt.month)
            + 1
        ).astype("Int64")
        early["Cancel_ts"] = early.apply(lambda r: pd.NaT if r["BrandEnd"] >= cap else r["BrandEnd"], axis=1)
        early["Cancel_Month_Index"] = (
            (early["BrandEnd"].dt.year - early["First_Start"].dt.year) * 12
            + (early["BrandEnd"].dt.month - early["First_Start"].dt.month)
            + 1
        ).astype("Int64")

    # Order columns consistently
    cols = [
        "Name_or_Email","Entry_Plan","First_Start","BrandStart","BrandEnd",
        "LastMonth","Cancel_ts","Cancel_Month_Index","censored",
        "had_other_plan_overlap","other_plans"
    ]
    for c in cols:
        if c not in early.columns:
            early[c] = pd.NA
    early = early[cols].sort_values(["First_Start","Name_or_Email"]).reset_index(drop=True)

    return early



def find_early_annual_churn_OLD(
    events: pd.DataFrame,
    brand_windows: pd.DataFrame,
    entry_plans: pd.DataFrame,
    months: int = 12,
    cap_date: pd.Timestamp = CAP_DATE,
) -> pd.DataFrame:
    """
    Identify *premium-plus (annual entry)* members whose *continuous* brand coverage
    from First_Start ends before `months` (e.g., 12) months, after merging plan windows
    with `merge_brand_intervals(..., grace_days=...)`.

    One row per Name_or_Email:
      - 'brand_coverage_start' = First_Start (from entry_plans).
      - 'brand_coverage_end'   = BrandEnd of the merged brand window that *covers* First_Start.
      - 'tenure_months_at_end' = 1-indexed month at coverage end (BrandEnd vs First_Start), may be negative
      - 'early_churn_ltN'      = True if not censored and tenure_months_at_end < months
      - 'brand_censored_at_cap' flagged if brand_coverage_end >= cap_date (cannot know real end)
      - 'anomaly_no_window_at_entry' True if no brand window covers First_Start (data issue)
      - 'anomaly_negative_months' True if tenure_months_at_end < 1 (indicates data problem)
      - 'had_nonpp_any_after_start' and 'nonpp_plans_first_year' are *informational* (not used to suppress churn)

    IMPORTANT:
      This function assumes brand_windows were built with the desired `grace_days` so that
      plan switches within grace are already merged into one window. If PP → PPM occurs
      within grace_days, it is part of the same 'continuous' window and counts toward
      tenure; if the gap exceeds grace, it does not.
    """
    cap = pd.to_datetime(cap_date)

    # Normalize and lower-case plan
    e = events.copy()
    e["Plan"]  = e["Plan"].astype(str).str.lower()
    e["Start"] = pd.to_datetime(e["Start"], errors="coerce")
    e["End"]   = pd.to_datetime(e["End"],   errors="coerce")

    bw = brand_windows.copy()
    bw["BrandStart"] = pd.to_datetime(bw["BrandStart"], errors="coerce")
    bw["BrandEnd"]   = pd.to_datetime(bw["BrandEnd"],   errors="coerce")

    ep = entry_plans.copy()
    ep["Entry_Plan"]  = ep["Entry_Plan"].astype(str).str.lower()
    ep["First_Start"] = pd.to_datetime(ep["First_Start"], errors="coerce")

    # Premium-plus entry cohort
    ep_pp = ep[ep["Entry_Plan"] == "premium-plus"].copy()
    if ep_pp.empty:
        return pd.DataFrame(columns=[
            "Name_or_Email","Entry_Plan",
            #"brand_first_start_any","brand_windows_count",
            "BrandStart", "brand_windows_count",
            #"brand_coverage_start","brand_coverage_end",
            "First_Start","LastMonth",
            "brand_censored_at_cap",
            "tenure_months_at_end","anomaly_no_window_at_entry","anomaly_negative_months",
            "early_churn_ltN","had_nonpp_any_after_start","nonpp_plans_first_year"
        ])

    # Diagnostic: earliest brand start across all plans
    brand_first = (
        bw.groupby("Name_or_Email", as_index=False)["BrandStart"]
          .min()
          .rename(columns={"BrandStart": "brand_first_start_any"})
    )

    # Join brand windows to entry to limit to members with PP entry
    df = bw.merge(ep_pp[["Name_or_Email", "Entry_Plan", "First_Start"]], on="Name_or_Email", how="right")

    # Select the contiguous brand coverage window that COVERS First_Start
    def pick_window_covering_entry(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("BrandStart")
        fs = g["First_Start"].iloc[0]
        cover = g[(g["BrandStart"] <= fs) & (g["BrandEnd"] >= fs)]
        anomaly_no_window = False
        if not cover.empty:
            end = cover["BrandEnd"].max()  # windows already merged by grace_days; take max to be safe
            cov_start = fs                 # contiguous coverage from entry starts at First_Start
        else:
            # Data anomaly: no brand window covers entry
            anomaly_no_window = True
            after = g[g["BrandStart"] >= fs].sort_values("BrandStart")
            # If no window after entry, end=last seen end; else end of the first window after entry
            end = (after["BrandEnd"].iloc[0] if not after.empty else g["BrandEnd"].max())
            cov_start = fs
        return pd.Series({"brand_coverage_start": cov_start, "brand_coverage_end": end, "anomaly_no_window_at_entry": anomaly_no_window})

    picked = (
        df.groupby("Name_or_Email", as_index=False)
          .apply(pick_window_covering_entry)
          .reset_index(drop=True)
    )

    # Merge diagnostics
    win_counts = df.groupby("Name_or_Email").size().reset_index(name="brand_windows_count")
    base = (
        ep_pp.merge(brand_first, on="Name_or_Email", how="left")
             .merge(picked, on="Name_or_Email", how="left")
             .merge(win_counts, on="Name_or_Email", how="left")
    )

    # Censor & tenure from First_Start to contiguous brand end
    base["brand_censored_at_cap"] = base["brand_coverage_end"] >= cap
    base["tenure_months_at_end"] = (
        (base["brand_coverage_end"].dt.year  - base["First_Start"].dt.year)  * 12 +
        (base["brand_coverage_end"].dt.month - base["First_Start"].dt.month) + 1
    ).astype("Int64")

    base["anomaly_negative_months"] = base["tenure_months_at_end"].notna() & (base["tenure_months_at_end"] < 1)

    base["early_churn_ltN"] = (
        base["brand_censored_at_cap"].eq(False) &
        base["tenure_months_at_end"].notna() &
        (base["tenure_months_at_end"] < np.int64(months))
    )

    # Informational context on non-PP after start and overlapping first year (not used to suppress churn)
    ectx = e[e["Name_or_Email"].isin(base["Name_or_Email"])]
    ectx["Plan"]  = ectx["Plan"].str.lower()
    ectx["Start"] = pd.to_datetime(ectx["Start"], errors="coerce")
    ectx["End"]   = pd.to_datetime(ectx["End"],   errors="coerce")

    def nonpp_ctx(row):
        sid = row["Name_or_Email"]
        fs  = row["First_Activation"] if "First_Activation" in row else row["brand_coverage_start"]
        fy  = row["brand_coverage_start"] + pd.DateOffset(months=months)
        sub = ectx[(ectx["Name_or_Email"] == sid) & (ectx["Plan"] != "premium-plus")]
        if sub.empty:
            return pd.Series({"had_nonpp_any_after_start": False, "nonpp_plans_first_year": ""})
        any_after = bool((sub["End"] > fs).any())
        ov = (sub["Start"] < fy) & (sub["End"] > fs)
        plans = ", ".join(sorted(sub.loc[ov, "Plan"].dropna().unique().tolist()))
        return pd.Series({"had_nonpp_any_after_start": any_after, "nonpp_plans_first_year": plans})

    ctx = base.apply(nonpp_ctx, axis=1)
    base = pd.concat([base, ctx], axis=1)

    # Keep only early-churn rows
    early = base[base["early_churn_ltN"] == True].copy()

    # Final columns (one row per member)
    keep_cols = [
        "Name_or_Email", "Entry_Plan",
        "brand_first_start_any", "brand_windows_count",
        "brand_coverage_start", "brand_coverage_end", "brand_censored_at_cap",
        "tenure_months_at_end", "anomaly_no_window_at_entry", "anomaly_negative_months",
        "early_churn_ltN",
        "had_nonpp_any_after_start", "nonpp_plans_first_year",
    ]
    return early.loc[:, keep_cols].sort_values(["brand_coverage_start", "tenure_months_at_end", "Name_or_Email"])



def find_early_annual_churn_OLD2(
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
    entry_plans   = derive_entry_plan(events, allowed_plans=allowed_plans)

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


# ------------------------------------------------------------
# Brand-tenure summary from account_windows
# ------------------------------------------------------------

def compute_avg_brand_tenure(
    account_windows: pd.DataFrame,
    # Optional: restrict to these entry-plan labels (case-insensitive)
    include_plans: list[str] | None = None,
    # When True, also compute an aggregate row 'both' over these plans
    include_both: bool = True,
    both_plans: tuple[str, str] = ("premium-plus", "premium-plus-monthly"),
    # When True, also compute an aggregate 'all' across all entry plans
    include_all: bool = True,
    # Name of the entry plan column
    plan_col: str = "Entry_Plan",
    # Name of the first activation timestamp column
    first_start_col: str = "First_Start",
    # Name of the monthly tenure column (if missing, we compute it)
    tenure_col: str = "LastMonth",
    # Name of the right-censor flag column (if missing, we compute it from BrandEnd and cap_date)
    censored_col: str = "censored",
    # Right-censoring cutoff if 'censored' is unavailable
    cap_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Compute brand-tenure summary per entry plan and cohort year.

    Returns a DataFrame with columns:
      ['Entry_Plan','Cohort Year','members','avg_tenure_months',
       'median_tenure_months','std_tenure_months','censored_share'].

    - 'members' counts distinct Name_or_Email within each (plan,year) group.
    - 'avg/median/std' summarize the 'tenure_col' values (treated as months).
    - 'censored_share' is the fraction of rows with censored==True in the group.
    - You can limit to specific entry plans via include_plans.
    - Set include_browse 'both' to also compute an aggregate for both_plans (e.g., ('premium-plus','premium-plus-monthly')).
    - Set include_all to also compute a single aggregate across all entry plans present.
    """
    # Start from a defensive copy
    d = account_windows.copy()

    # Ensure we have a 'Cohort Year'
    d = ensure_cohort_year_column(d, first_start_col=first_start_col, out_col="Cohort Year")

    # Normalize types for plan and optionally filter by plans
    d[plan_col] = d[plan_col].astype("string")
    if include_plans is not None and len(include_plans) > 0:
        # Build a case-insensitive filter
        keep = {p.lower() for p in include_plans}
        d = d[d[plan_col].str.lower().isin(keep)].copy()

    # Ensure we have a numeric tenure column; if missing, compute from BrandStart/BrandEnd
    if tenure_col not in d.columns:
        # Compute tenure in months from BrandStart->BrandEnd (Month 1 is the start month)
        d[tenure_col] = (
            (pd.to_datetime(d["BrandEnd"]).dt.year  - pd.to_datetime(d["BrandStart"]).dt.year) * 12 +
            (pd.to_datetime(d["BrandEnd"]).dt.month - pd.to_datetime(d["BrandStart"]).dt.month) + 1
        ).astype("Int64")

    # Ensure we have a boolean 'censored' column, or derive it from cap_date
    if censored_col not in d.columns:
        # If caller didn't pass a cap_date, we can't infer; set to False rather than erroring
        if cap_date is None:
            d[censored_col] = False
        else:
            d[censored_col] = (pd.to_datetime(d["BrandEnd"]) >= pd.to_datetime(cap_date))

    # Compute the base summary by (Entry_Plan, Cohort Year)
    base = (
        d.groupby([plan_col, "Cohort Year"])
         .agg(
             members=("Name_or_Email", lambda s: s.nunique()),
             avg_tenure_months=(tenure_col, "mean"),
             median_tenure_months=(tenure_col, "median"),
             std_tenure_months=(tenure_col, "std"),
             censored_share=(censored_col, "mean"),
         )
         .reset_index()
         .rename(columns={plan_col: "Entry_Plan"})
    )

    # Prepare a list of frames to concatenate
    frames = [base]

    # Optionally add a 'both' aggregate across the two specified plans
    if include_both:
        # Lowercase set for matching
        both_set = {p.lower() for p in both_plans}
        # Filter only those two plans
        d_both = d[d[plan_col].str.lower().isin(both_set)].copy()
        if not d_both.empty:
            both_summary = (
                d_both.groupby("Cohort Year")
                      .agg(
                          members=("Name_or_Email", lambda s: s.nunique()),
                          avg_tenure_months=(tenure_col, "mean"),
                          median_tenure_months=(tenure_col, "median"),
                          std_tenure_months=(tenure_col, "std"),
                          censored_share=(censored_col, "mean"),
                      )
                      .reset_index()
            )
            # Tag the aggregate as 'both'
            both_summary.insert(0, "Entry_Plan", "both")
            frames.append(both_summary)

    # Optionally add an 'all' aggregate across all entry plans present
    if include_all:
        all_summary = (
            d.groupby("Cohort Year")
             .agg(
                 members=("Name_or_Email", lambda s: s.nunique()),
                 avg_tenure_months=(tenure_col, "mean"),
                 median_tenure_months=(tenure_col, "median"),
                 std_tenure_months=(tenure_col, "std"),
                 censored_share=(censored_col, "mean"),
             )
             .reset_index()
        )
        all_summary.insert(0, "Entry_Plan", "all")
        frames.append(all_summary)

    # Concatenate all pieces into the final summary table
    out = pd.concat(frames, ignore_index=True)

    # FIXME: remove debugger 
    #import pdb 
    #pdb.set_trace()

    # Order the columns exactly as requested
    out = out[["Entry_Plan","Cohort Year","members","avg_tenure_months","median_tenure_months"]]
    return out


# ---------------------
# Conversions 
# ---------------------

def compute_first_plan_changes(
    events: pd.DataFrame,
    id_col: str   = "Name_or_Email",
    plan_col: str = "Plan",
    start_col: str = "Start",
    # first activation column to determine the clock start for the horizon window
    first_start_col: str = "First_Start",
) -> pd.DataFrame:
    """
    For each member on the event timeline:
      - Identify their first (entry) plan and its first_start timestamp.
      - Find the first subsequent event where plan != entry plan.
      - Return one row per member with first change info (or NaN if none).
    Output columns:
      ['Name_or_Email','Entry_Plan','First_Start','First_Change_Plan','First_Change_Start','Months_to_Change'].
    """
    # Defensive copy
    e = events.copy()
    # Normalize key columns
    e[id_col]    = e[id_col].astype("string").str.strip()
    e[plan_col]  = e[plan_col].astype("string").str.strip()
    e[start_col] = pd.to_datetime(e[start_col], errors="coerce")
    # Build each member's entry record (first activation)
    firsts = (
        e.sort_values([id_col, start_col])
         .groupby(id_col, as_index=False)
         .first()
         [[id_col, plan_col, start_col]]
         .rename(columns={plan_col: "Entry_Plan", start_col: "First_Start"})
    )
    # Join back to all rows to label them with each member's entry plan + first start
    e2 = e.merge(firsts, on=id_col, how="left")
    # Keep only rows strictly AFTER the first start
    e2 = e2[e2[start_col] > e2["First_Start"]].copy()
    # Remove rows where the plan hasn't changed (same as entry)
    e2 = e2[e2[plan_col] != e2["Entry_Plan"]].copy()
    # Get the earliest change per member
    first_change = (
        e2.sort_values([id_col, start_col])
          .groupby(id_col, as_index=False)
          .first()
    )
    # Compute months from entry to first change (Month 1 = month of entry)
    first_change["Months_to_Change"] = (
        (first_change[start_col].dt.year  - first_change["First_Start"].dt.year) * 12
        + (first_change[start_col].dt.month - first_change["First_Start"].dt.month)
        + 1
    ).astype("Int64")

    # Keep only columns of interest
    out = first_change[[id_col, "Entry_Plan", "First_Start", plan_col, start_col, "Months_to_Change"]].copy()
    # Rename new columns clearly
    out = out.rename(columns={plan_col: "First_Change_Plan", start_col: "First_Change_Start"})
    return out

def summarize_conversions(
    first_changes: pd.DataFrame,
    account_windows: pd.DataFrame,
    horizon_months: int = 12,
    # When grouping by cohort too, pass by=["Entry_Cohort","Entry_Plan","First_Change_Plan"]
    by: list[str] = ["Entry_Plan", "First_Change_Plan"],
) -> pd.DataFrame:
    """
    Aggregate first-change events into conversion rates.
    - 'first_changes' is the 1-row-per-member output of compute_first_plan_changes()
    - 'account_windows' gives you denominators (members per entry plan/cohort)

    Returns a DataFrame with counts AND rates:
      group columns (per 'by') + ['conversions_within','entry_members','conversion_rate'].
    """
    # Defensive copies
    fc = first_changes.copy()
    aw = account_windows.copy()

    # Attach Entry Cohort for optional grouping
    aw["Cohort Year"] = pd.to_datetime(aw["First_Start"], errors="coerce").dt.year
    

    # FIXME: remove debugger 
    #import pdb 
    #pdb.set_trace()

    # One row per member for denominators
    entrants = (
        aw.sort_values(["Name_or_Email","First_Start"])
          .drop_duplicates(subset=["Name_or_Email"])
          [["Name_or_Email","Cohort Year","Entry_Plan"]]
          .copy()
    )

    # ---- Filter first changes to those within the horizon ----
    if "Months_to_Change" not in fc.columns:
        raise KeyError("first_changes is missing 'Months_to_Change'")

    fc["Within_Horizon"] = fc["Months_to_Change"].le(horizon_months)

    # Avoid Entry_Plan column collision: drop any pre-existing Entry_Plan / Entry_Cohort on fc
    # (we'll take the canonical values from 'entrants')
    for col in ("Entry_Plan","Entry_Cohort"):
        if col in fc.columns:
            fc = fc.drop(columns=[col])

    # Merge in denominators' keys (canonical Entry_Plan / Entry_Cohort)
    fc = fc.merge(
        entrants,
        on="Name_or_Email",
        how="left",
        validate="one_to_one"  # first_changes should be 1 row per member
    )

    # ---- Build numerator: conversions within horizon grouped by requested keys ----
    # Only rows that actually changed within the horizon count in the numerator
    numer = (
        fc[fc["Within_Horizon"]]
        .groupby([k for k in by if k in fc.columns], as_index=False)
        .agg(conversions_within=("Name_or_Email","nunique"))
    )

    # FIXME: remove debugger 
    #import pdb 
    #pdb.set_trace()

    # ---- Build denominator counts: unique entrants per relevant entry keys ----
    # Denominator grouping keys are the subset of 'by' that belong to entrants (Entry_Plan / Entry_Cohort)
    entry_keys = [k for k in by if k in ("Entry_Plan","Cohort Year")]
    if entry_keys:
        denom = (
            entrants.groupby(entry_keys, as_index=False)
                    .agg(entry_members=("Name_or_Email","nunique"))
        )
    else:
        # If user didn't group by any entry key, denominator is global entrants count
        denom = pd.DataFrame({
            "entry_members": [entrants["Name_or_Email"].nunique()]
        })

    # ---- Join numerators to denominators and compute rates ----
    # Determine join keys between numer and denom
    join_keys = entry_keys.copy()
    # If 'First_Change_Plan' is in by, it doesn't belong to the denominator;
    # left-merge numerator onto denom with the entry keys, then fill NaNs
    out = denom.merge(numer, on=join_keys, how="left")
    out["conversions_within"] = out["conversions_within"].fillna(0).astype(int)

    # Compute conversion rate safely
    out["conversion_rate"] = (out["conversions_within"] / out["entry_members"]).replace([np.inf, -np.inf], 0.0)

    # Reorder: put all group columns first, then metrics
    ordered_cols = [k for k in by if k in out.columns] + ["entry_members","conversions_within","conversion_rate"]
    # Include 'Entry_Cohort' if requested but not in 'by' (rare)
    for k in ("Entry_Plan","Cohort Year","First_Change_Plan"):
        if (k in out.columns) and (k not in ordered_cols):
            ordered_cols.insert(0, k)

    out = out[ordered_cols].sort_values(by=[c for c in by if c in out.columns])

    return out