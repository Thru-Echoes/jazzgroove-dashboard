# Churn Analysis and Streamlit Dashboard 

This project handles churn / retention analysis of customer data **from 2019 up to April 29, 2025.** 

## Overview of Steps 

There are several steps of data processing that are required before being able to run the Streamlit dashboard. 

### Step 1: Raw CSV -> normalized events.csv 

Process the raw .csv and create the normalized events.csv data. 

Columns expected in *customer_data_to_April_29_2025.csv:* 

```bash

```

Ingest data with code: 

```bash
python scripts/00_ingest_customer_csv_to_events.py \
  --in-customer-csv data/raw/customer_data_to_April_29_2025.csv \
  --out-events-csv  data/events.csv
```

### Step 2: Build survival, hazard, and watchlist data 

What this script does under the hood

#### Uses retention_lib.py to:

- Collapse all plan rows into brand windows (member-level “alive”)

- Determine the Entry_Plan (first plan among premium-plus, premium-plus-monthly)

- Compute brand-level survival segmented by Entry_Plan

- Compute pooled hazards per tenure month


#### Builds a watchlist using the same logic as the app:

- Avg_Hazard for next month, and

- Pred_3mo_Risk = 1 - (1-h0)(1-h1)(1-h2)

```bash
python scripts/01_prep_survival_and_risk.py --events-path scripts/events.csv --cap 2025-12-31 --grace-days 90
```

Or, run with backtests: 

```bash
python scripts/01_prep_survival_and_risk.py \
  --events-path scripts/events.csv \
  --cap 2025-12-31 \
  --grace-days 90 \
  --backtest-as-of 2025-01-31 \
  --backtest-horizon-months 3
```

### Step 3: Run Streamlit app 

```bash
streamlit run app/streamlit_app.py
```


### Step 4: Backtest 

To test if a January 2025 list catches Feb - Apr cancels: 

```bash
python scripts/02b_backtest_multimonth.py \
    --hazard-as-of 2025-01-31 \
    --eval-end 2025-04-30 \
    --top-n 5000 \
    --min-hazard 0.00
```