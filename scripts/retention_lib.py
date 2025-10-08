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

import pandas as pd 
import numpy as np 
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter
import calendar

def build_gap_adjusted_intervals(df,
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

def expand_intervals_to_records(intervals, max_months=24):
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

def map_entry_plan(df, plan_col='Plan'):
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

def survival_from_records(rec_df):
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

def compute_average_monthly_hazard_OLD(surv, weight_by_cohort=False):
    """
    NEW pooled hazard calculation:
    - Input: `surv` from compute_gap_adjusted_survival with columns
      ['Cohort Year','Plan','Month','Active','Cohort Size','Retention Rate','Churn Rate'].
    - Output: DataFrame with ['Plan','Month','Avg_Hazard'] where
      Avg_Hazard = sum(Events) / sum(At_Risk) pooled across cohorts.

    Notes:
    * We ignore the old weight_by_cohort flag; pooling replaces it.
    * Hazards are defined for Month >= 2 (Month 1 has no prior interval).
    """
    s = surv.copy()
    s = s.sort_values(['Plan', 'Cohort Year', 'Month'])

    # Active in the previous month within each (Plan, Cohort Year)
    s['Active_prev'] = s.groupby(['Plan', 'Cohort Year'])['Active'].shift(1)

    # For Month==1, set Active_prev to the cohort size
    is_m1 = s['Month'] == 1
    s.loc[is_m1, 'Active_prev'] = s.loc[is_m1, 'Cohort Size']

    # At-risk at month t is the active at t-1 (Active_prev)
    s['At_Risk'] = s['Active_prev']

    # Events are those who were active at t-1 but not at t
    s['Events'] = (s['Active_prev'] - s['Active']).clip(lower=0)

    # Hazards start at Month 2 (there is no interval before Month 1)
    s = s[s['Month'] >= 2]

    pooled = (
        s.groupby(['Plan', 'Month'], as_index=False)[['Events', 'At_Risk']]
         .sum()
    )

    # Pooled hazard; clip to [0,1] for safety
    pooled['Avg_Hazard'] = (pooled['Events'] / pooled['At_Risk']).clip(lower=0, upper=1)

    return pooled[['Plan', 'Month', 'Avg_Hazard']]

def compute_average_monthly_hazard(surv_df, weight_by_cohort=True):
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