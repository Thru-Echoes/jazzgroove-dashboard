#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_prep_survival_and_risk.py (verification-only)

For now this script ONLY:
  • Loads ALL-plans events (same file your notebook uses)
  • Builds account_windows with grace_days=90 and your CAP date
  • Reproduces:
      - Entry counts by Entry_Plan (premium-plus / premium-plus-monthly) for 2021–2023
      - Early annual churners (<12 months) for 2021–2023
  • Writes account_windows.csv so you can inspect it
  • Prints OK/MISMATCH checks
  • Does NOT build survival/hazard/watchlist yet

Run:
  python scripts/01_prep_survival_and_risk.py \
      --events-path data/events.csv \
      --cap 2025-12-31 \
      --grace-days 90
"""

# ───────────────────────────────────────────────────────────
# Section 0 — Imports & paths
# ───────────────────────────────────────────────────────────
import os
import sys
import argparse
import pandas as pd
from dateutil.relativedelta import relativedelta

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from retention_lib import (
    _normalize_event_columns,
    merge_brand_intervals,
    build_account_brand_blocks,
    derive_entry_plan,
    ensure_cohort_year_column,
    find_early_annual_churn,
    ALLOWED_PLANS,
    build_pooled_hazard_from_survival_and_windows
)

# ───────────────────────────────────────────────────────────
# NEW Section — Survival builder from account_windows
# ───────────────────────────────────────────────────────────
def build_survival_curves_from_account_windows(
    account_windows: pd.DataFrame,
    max_months: int = 48,
) -> pd.DataFrame:
    """
    Build survival curves directly from account_windows.

    Semantics:
      - Each row in account_windows represents one contiguous brand block for an account,
        with columns including: Entry_Plan, BrandStart, LastMonth.
      - We define the cohort year as BrandStart.year (matches how you sliced accounts_*_YYYY).
      - For each account-window row, we treat months 1..LastMonth as 'alive'.
      - We aggregate across accounts to get Alive/At_Risk per (Entry_Plan, Cohort Year, Month).
      - Retention = Alive / Alive_at_month_1 within each (Entry_Plan, Cohort Year).

    Output columns:
      ['Plan','Cohort Year','Month','At_Risk','Retention']
    """
    d = account_windows.copy()

    # Keep only rows with a valid Entry_Plan, BrandStart, LastMonth
    d = d.dropna(subset=["Entry_Plan", "BrandStart", "LastMonth"]).copy()
    d["LastMonth"] = d["LastMonth"].astype("Int64")
    d = d[d["LastMonth"] > 0].copy()

    # Cohort Year defined by BrandStart.year (this matches your accounts_premium_plus_YYYY logic)
    d["Cohort Year"] = pd.to_datetime(d["BrandStart"], errors="coerce").dt.year.astype("Int64")

    # Restrict to the plans the app cares about (PP & PPM) so the curves match
    allowed = {p.lower() for p in ALLOWED_PLANS}
    d["_plan_lc"] = d["Entry_Plan"].astype(str).str.lower()
    d = d[d["_plan_lc"].isin(allowed)].copy()

    rows = []
    for _, r in d.iterrows():
        plan = r["Entry_Plan"]
        cohort_year = int(r["Cohort Year"])
        # continuous tenure in months for this window
        months = int(min(r["LastMonth"], max_months))
        for m in range(1, months + 1):
            rows.append({
                "Plan": plan,
                "Cohort Year": cohort_year,
                "Month": m,
                "At_Risk": 1,
            })

    if not rows:
        return pd.DataFrame(columns=["Plan","Cohort Year","Month","At_Risk","Retention"])

    surv_long = pd.DataFrame(rows)
    curves = (
        surv_long.groupby(["Plan","Cohort Year","Month"], as_index=False)["At_Risk"]
                 .sum()
                 .rename(columns={"At_Risk":"Alive"})
    )

    # Denominator = Alive at Month 1 for each (Plan, Cohort Year)
    den = curves[curves["Month"] == 1][["Plan","Cohort Year","Alive"]].rename(columns={"Alive":"N0"})
    curves = curves.merge(den, on=["Plan","Cohort Year"], how="left")

    curves["At_Risk"]   = curves["Alive"].astype(int)
    curves["Retention"] = (curves["Alive"] / curves["N0"]).astype(float)

    return curves[["Plan","Cohort Year","Month","At_Risk","Retention"]]

# ───────────────────────────────────────────────────────────
# NEW Section — Backtest builder (brand windows + hazard)
# ───────────────────────────────────────────────────────────
def build_backtest_rows(
    events: pd.DataFrame,
    brand_windows: pd.DataFrame,
    hazard: pd.DataFrame,
    entry_plans_allowed: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon_months: int,
    cap: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build a simple backtest table:
      - one row per member active at `as_of`
      - uses hazard (Avg_Hazard) for their next-month cancellation risk
      - labels whether they actually cancel within `horizon_months` after `as_of`

    Output columns:
      ['Name_or_Email','Entry_Plan','First_Start',
       'Months_Since_First','Hazard_Month','Avg_Hazard_next',
       'BrandEnd','churn_within_horizon']
    """
    # Normalize basics
    e = events.copy()
    e["Start"] = pd.to_datetime(e["Start"], errors="coerce")
    e["End"]   = pd.to_datetime(e["End"],   errors="coerce")

    bw = brand_windows.copy()
    bw["BrandStart"] = pd.to_datetime(bw["BrandStart"], errors="coerce")
    bw["BrandEnd"]   = pd.to_datetime(bw["BrandEnd"],   errors="coerce")

    ep = entry_plans_allowed.copy()
    ep["First_Start"] = pd.to_datetime(ep["First_Start"], errors="coerce")

    # 1) find brand-level actives at as_of (BrandStart <= as_of <= BrandEnd)
    active_bw = bw[(bw["BrandStart"] <= as_of) & (bw["BrandEnd"] >= as_of)].copy()
    active_bw = active_bw.merge(ep, on="Name_or_Email", how="inner")  # bring Entry_Plan + First_Start
    if active_bw.empty:
        return pd.DataFrame(columns=[
            "Name_or_Email","Entry_Plan","First_Start",
            "Months_Since_First","Hazard_Month","Avg_Hazard_next",
            "BrandEnd","churn_within_horizon"
        ])

    # 2) compute tenure at as_of and hazard lookup month
    asof_y, asof_m = as_of.year, as_of.month
    msf = (
        (asof_y - active_bw["First_Start"].dt.year) * 12
        + (asof_m - active_bw["First_Start"].dt.month)
        + 1
    )
    active_bw["Months_Since_First"] = pd.to_numeric(msf, errors="coerce").astype("Int64").clip(lower=1)
    active_bw["Hazard_Month"] = (active_bw["Months_Since_First"] + 1).astype("Int64")

    # 3) pull next-month hazard per (Entry_Plan, Hazard_Month)
    hz = hazard.copy()
    hz = hz.rename(columns={"Plan": "Entry_Plan"})  # so we can join on Entry_Plan
    hz = hz[["Entry_Plan","Month","Avg_Hazard"]].rename(columns={"Month":"Hazard_Month","Avg_Hazard":"Avg_Hazard_next"})

    out = active_bw.merge(
        hz,
        on=["Entry_Plan","Hazard_Month"],
        how="left"
    )

    # 4) label churn-within-horizon by BrandEnd
    horizon_end = as_of + relativedelta(months=horizon_months)
    out["BrandEnd"] = pd.to_datetime(out["BrandEnd"], errors="coerce")
    out["churn_within_horizon"] = (
        (out["BrandEnd"] > as_of) &
        (out["BrandEnd"] <= horizon_end) &
        (out["BrandEnd"] < cap)  # ignore censored at cap
    )

    # Final columns
    keep = [
        "Name_or_Email","Entry_Plan","First_Start",
        "Months_Since_First","Hazard_Month","Avg_Hazard_next",
        "BrandEnd","churn_within_horizon"
    ]
    return out[keep].sort_values(["Entry_Plan","Months_Since_First","Name_or_Email"]).reset_index(drop=True)


# ───────────────────────────────────────────────────────────
# Section 1 — CLI parsing (so we can exactly match notebook)
# ───────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Verify account_windows + counts match notebook and build survival/risk/backtest.")
    ap.add_argument("--events-path",
                    default=os.path.join(REPO_ROOT, "data", "events.csv"),
                    help="Path to ALL-plans events.csv (same one used in the notebook).")
    ap.add_argument("--cap",
                    default="2025-12-31",
                    help="CAP (right-censor) date, e.g., 2025-12-31 (use notebook value).")
    ap.add_argument("--grace-days",
                    type=int, default=90,
                    help="Grace window in DAYS for merging brand windows (use notebook value).")
    # Optional “expected” numbers; defaults are your notebook targets
    ap.add_argument("--exp-pp",  default="2021:853,2022:645,2023:742",
                    help="Expected PP entry counts as 'YYYY:N,...'.")
    ap.add_argument("--exp-ppm", default="2021:337,2022:237,2023:310",
                    help="Expected PPM entry counts as 'YYYY:N,...'.")
    ap.add_argument("--exp-early-pp", default="2021:12,2022:8,2023:23",
                    help="Expected early PP churners as 'YYYY:N,...'.")
    # NEW: backtest args (optional)
    ap.add_argument("--backtest-as-of",
                    default="", help="Optional as-of date for backtest (YYYY-MM-DD). If blank, backtest is skipped.")
    ap.add_argument("--backtest-horizon-months",
                    type=int, default=3, help="Backtest horizon in months (e.g. 3).")
    return ap.parse_args()

def parse_expectation_map(s: str) -> dict[int, int]:
    out = {}
    s = (s or "").strip()
    if not s:
        return out
    for part in s.split(","):
        k,v = part.split(":")
        out[int(k.strip())] = int(v.strip())
    return out

# ───────────────────────────────────────────────────────────
# Section 2 — Helpers for counts/printing
# ───────────────────────────────────────────────────────────
def count_entry_by_year(account_windows: pd.DataFrame, plan_label: str, years=(2021, 2022, 2023)) -> dict[int,int]:
    """Count unique entrants by (Entry_Plan==plan_label, Cohort Year). Entrant = first row per member."""
    aw = account_windows.copy()
    # entrants = first window per member by First_Start
    entrants = (
        aw.sort_values(["Name_or_Email", "First_Start"])
          .drop_duplicates(subset=["Name_or_Email"])
          [["Name_or_Email", "Entry_Plan", "First_Start"]]
          .copy()
    )
    entrants["Cohort Year"] = pd.to_datetime(entrants["First_Start"], errors="coerce").dt.year
    m = entrants["Entry_Plan"].astype(str).str.lower() == plan_label.lower()
    res = {}
    for y in years:
        res[y] = int(entrants.loc[m & (entrants["Cohort Year"] == y), "Name_or_Email"].nunique())
    return res

def count_early_pp_by_year(events: pd.DataFrame,
                           brand_windows: pd.DataFrame,
                           entry_plans_all: pd.DataFrame,
                           cap: pd.Timestamp,
                           years=(2021, 2022, 2023)) -> dict[int,int]:
    """
    Use the SAME early-churn function as the notebook, but pass entry_plans derived
    from ALL plans (allowed_plans=None). Cohort year = First_Start year.
    """
    early = find_early_annual_churn(events, brand_windows, entry_plans_all, months=12, cap_date=cap)
    if early.empty:
        return {y: 0 for y in years}
    early["Cohort Year"] = pd.to_datetime(early["First_Start"], errors="coerce").dt.year
    res = {}
    for y in years:
        res[y] = int(early.loc[early["Cohort Year"] == y, "Name_or_Email"].nunique())
    return res

def print_check(label: str, got_map: dict[int,int], exp_map: dict[int,int]) -> None:
    ok = got_map == exp_map
    status = "OK" if ok else "MISMATCH"
    print(f"[{status}] {label}")
    print(f"  expected: {exp_map}")
    print(f"  got     : {got_map}")

# ───────────────────────────────────────────────────────────
# Section 3 — Main (replicate notebook steps exactly)
# ───────────────────────────────────────────────────────────
def main():
    args = parse_args()
    events_path = args.events_path
    cap = pd.to_datetime(args.cap)
    grace_days = int(args.grace_days)

    exp_pp   = parse_expectation_map(args.exp_pp)
    exp_ppm  = parse_expectation_map(args.exp_ppm)
    exp_early= parse_expectation_map(args.exp_early_pp)

    # 1) Load ALL‑plans events (same file as notebook)
    print(f"[i] Loading events: {events_path}")
    events = pd.read_csv(events_path, parse_dates=["Start", "End"])
    print(f"[i] rows={len(events):,} accounts={events['Name_or_Email'].nunique():,}")

    # 2) (Notebook builds account_windows from *events*, not events_norm)
    #    We still normalize once for brand_windows (sanity), but keep parity with notebook call.
    events_norm   = _normalize_event_columns(events, cap_date=cap)
    brand_windows = merge_brand_intervals(events, cap_date=cap, grace_days=grace_days)

    # 3) Build account_windows exactly like notebook
    print(f"[i] Building account_windows (grace_days={grace_days}, cap={cap.date()})")
    account_windows = build_account_brand_blocks(
        events,
        identities=None,         # notebook passes None
        grace_days=grace_days,
        cap_date=cap,
    )

    account_windows = ensure_cohort_year_column(account_windows, first_start_col="First_Start")
    out_windows = os.path.join(REPO_ROOT, "data", "processed", "account_windows.csv")
    os.makedirs(os.path.dirname(out_windows), exist_ok=True)
    account_windows.to_csv(out_windows, index=False)
    print(f"[save] account_windows -> {out_windows} (rows={len(account_windows):,})")

    # 4) Find number of subscribers per cohort 
    accounts_premium_plus = account_windows[account_windows['Entry_Plan'] == 'premium-plus'].copy()
    accounts_premium_plus_monthly = account_windows[account_windows['Entry_Plan'] == 'premium-plus-monthly'].copy()

    accounts_premium_plus_2021 = accounts_premium_plus[accounts_premium_plus['BrandStart'].dt.year == 2021].copy()
    accounts_premium_plus_2022 = accounts_premium_plus[accounts_premium_plus['BrandStart'].dt.year == 2022].copy()
    accounts_premium_plus_2023 = accounts_premium_plus[accounts_premium_plus['BrandStart'].dt.year == 2023].copy()
    accounts_premium_plus_2024 = accounts_premium_plus[accounts_premium_plus['BrandStart'].dt.year == 2024].copy()

    accounts_premium_plus_monthly_2021 = accounts_premium_plus_monthly[accounts_premium_plus_monthly['BrandStart'].dt.year == 2021].copy()
    accounts_premium_plus_monthly_2022 = accounts_premium_plus_monthly[accounts_premium_plus_monthly['BrandStart'].dt.year == 2022].copy()
    accounts_premium_plus_monthly_2023 = accounts_premium_plus_monthly[accounts_premium_plus_monthly['BrandStart'].dt.year == 2023].copy()
    accounts_premium_plus_monthly_2024 = accounts_premium_plus_monthly[accounts_premium_plus_monthly['BrandStart'].dt.year == 2024].copy()

    print(f"Number of premium-plus rows in 2021: {accounts_premium_plus_2021.shape[0]}")
    print(f"number of unique Name_or_Email: {accounts_premium_plus_2021['Name_or_Email'].nunique()}") 

    print(f"Number of premium-plus-monthly rows in 2021: {accounts_premium_plus_monthly_2021.shape[0]}")
    print(f"number of unique Name_or_Email: {accounts_premium_plus_monthly_2021['Name_or_Email'].nunique()}") 

    print(f"Total new (premium-plus + premium-plus-monthly) in 2021: {accounts_premium_plus_2021.shape[0] + accounts_premium_plus_monthly_2021.shape[0]}")

    print("\n----\n")

    print(f"Number of premium-plus rows in 2022: {accounts_premium_plus_2022.shape[0]}")
    print(f"number of unique Name_or_Email: {accounts_premium_plus_2022['Name_or_Email'].nunique()}") 

    print(f"Number of premium-plus-monthly rows in 2022: {accounts_premium_plus_monthly_2022.shape[0]}")
    print(f"number of unique Name_or_Email: {accounts_premium_plus_monthly_2022['Name_or_Email'].nunique()}") 

    print(f"Total new (premium-plus + premium-plus-monthly) in 2022: {accounts_premium_plus_2022.shape[0] + accounts_premium_plus_monthly_2022.shape[0]}")

    print("\n----\n")

    print(f"Number of premium-plus rows in 2023: {accounts_premium_plus_2023.shape[0]}")
    print(f"number of unique Name_or_Email: {accounts_premium_plus_2023['Name_or_Email'].nunique()}") 

    print(f"Number of premium-plus-monthly rows in 2023: {accounts_premium_plus_monthly_2023.shape[0]}")
    print(f"number of unique Name_or_Email: {accounts_premium_plus_monthly_2023['Name_or_Email'].nunique()}") 

    print(f"Total new (premium-plus + premium-plus-monthly) in 2023: {accounts_premium_plus_2023.shape[0] + accounts_premium_plus_monthly_2023.shape[0]}")

    print("\n----\n")

    print(f"Number of premium-plus rows in 2024: {accounts_premium_plus_2024.shape[0]}")
    print(f"number of unique Name_or_Email: {accounts_premium_plus_2024['Name_or_Email'].nunique()}") 

    print(f"Number of premium-plus-monthly rows in 2024: {accounts_premium_plus_monthly_2024.shape[0]}")
    print(f"number of unique Name_or_Email: {accounts_premium_plus_monthly_2024['Name_or_Email'].nunique()}") 

    print(f"Total new (premium-plus + premium-plus-monthly) in 2024: {accounts_premium_plus_2024.shape[0] + accounts_premium_plus_monthly_2024.shape[0]}")

    # 5) Determine if number of early premium-plus churners matches expected 
    early_2021 = accounts_premium_plus_2021[accounts_premium_plus_2021['LastMonth'] < 13].copy()
    num_early_2021 = len(early_2021)
    print(f"Ratio early exit in 2021: {(num_early_2021 / len(accounts_premium_plus_2021)):.3f}")

    early_2022 = accounts_premium_plus_2022[accounts_premium_plus_2022['LastMonth'] < 13].copy()
    num_early_2022 = len(early_2022)
    print(f"Ratio early exit in 2022: {(num_early_2022 / len(accounts_premium_plus_2022)):.3f}")

    early_2023 = accounts_premium_plus_2023[accounts_premium_plus_2023['LastMonth'] < 13].copy()
    num_early_2023 = len(early_2023)
    print(f"Ratio early exit in 2023: {(num_early_2023 / len(accounts_premium_plus_2023)):.3f}")

    assert (num_early_2021 == 12), "Early 2021 churners does not match expected!"
    assert (num_early_2022 == 8), "Early 2021 churners does not match expected!"
    assert (num_early_2023 == 23), "Early 2021 churners does not match expected!"

    # 6) Build survival_curves.csv from account_windows using the same logic
    print("\n[i] Building survival_curves.csv from account_windows ...")
    surv = build_survival_curves_from_account_windows(account_windows, max_months=48)
    out_surv = os.path.join(REPO_ROOT, "data", "processed", "survival_curves.csv")
    os.makedirs(os.path.dirname(out_surv), exist_ok=True)
    surv.to_csv(out_surv, index=False)
    print(f"[save] survival_curves -> {out_surv} (rows={len(surv):,})")

    # FIXME: remove debugger 
    #import pdb 
    #pdb.set_trace() 

    # 7) Build pooled_hazard.csv from survival_curves + brand_windows
    print("\n[i] Building pooled_hazard.csv (cancellation risk by tenure month) ...")
    entry_plans_allowed = derive_entry_plan(events, allowed_plans=ALLOWED_PLANS)
    haz = build_pooled_hazard_from_survival_and_windows(surv, brand_windows, entry_plans_allowed, cap_date=cap)
    out_hz = os.path.join(REPO_ROOT, "data", "processed", "pooled_hazard.csv")
    os.makedirs(os.path.dirname(out_hz), exist_ok=True)
    haz.to_csv(out_hz, index=False)
    print(f"[save] pooled_hazard -> {out_hz} (rows={len(haz):,})")

    # 8) Build backtest_rows.csv if backtest-as-of is provided
    if args.backtest_as_of:
        as_of = pd.to_datetime(args.backtest_as_of)
        horizon = int(args.backtest_horizon_months)
        print(f"\n[i] Building backtest_rows.csv (as_of={as_of.date()}, horizon={horizon} months) ...")
        backtest_rows = build_backtest_rows(
            events=events,
            brand_windows=brand_windows,
            hazard=haz,
            entry_plans_allowed=entry_plans_allowed,
            as_of=as_of,
            horizon_months=horizon,
            cap=cap,
        )
        out_bt = os.path.join(REPO_ROOT, "data", "processed", "backtest_rows.csv")
        os.makedirs(os.path.dirname(out_bt), exist_ok=True)
        backtest_rows.to_csv(out_bt, index=False)
        print(f"[save] backtest_rows -> {out_bt} (rows={len(backtest_rows):,})")
    else:
        print("\n[i] No --backtest-as-of provided; skipping backtest_rows.csv.\n")

    print("\n[i] Verification + survival/risk/backtest build complete.\n")


if __name__ == "__main__":
    main()
