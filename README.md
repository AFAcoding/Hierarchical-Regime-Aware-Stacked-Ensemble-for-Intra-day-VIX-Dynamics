# Stacked Multi-Model Ensemble for Intraday VIX Prediction with Regime Awareness

**Author:** Aleix Francia Albert  
**Project:** MSc Thesis – Open Source Release  

---

# Overview

This repository implements a **regime-aware stacked ensemble system** designed to **predict intraday movements of the CBOE VIX** using machine learning models and market-derived features.

The objective is to capture **non-linear relationships, volatility clustering, and market regime dynamics** through a **multi-model architecture** trained on historical financial data.

The system is built for:

- Quantitative research
- Volatility modelling
- Risk monitoring
- Trading experimentation

The project also includes a **fully automated cloud pipeline** that performs daily **data extraction, feature engineering, database storage, and monitoring**.

---

# Key Capabilities

## Intraday VIX Direction Classification

The system predicts categorical VIX movements:

| Class | Meaning |
|------|------|
| `0` | Neutral |
| `1` | VIX Up |
| `2` | VIX Down |

This converts a continuous volatility signal into a **classification problem suitable for ensemble models**.

---

## Regime-Aware Feature Engineering

Features are designed to capture different **market regimes**, including:

- Realized volatility
- Implied volatility behaviour
- Cross-market correlations
- Momentum
- Credit spreads
- Macro proxies

---

## Stacked Multi-Model Architecture

The modelling framework uses multiple supervised learning models combined into a **stacked ensemble**, improving:

- predictive stability
- robustness to regime changes
- generalization performance

Typical base models may include:

- Random Forest
- Gradient Boosting
- LightGBM
- Logistic models

---

## Automated Cloud Pipeline

A **serverless Azure Function** performs the entire ETL pipeline daily:

1. Download market data
2. Compute financial features
3. Generate prediction targets
4. Store results in MongoDB
5. Send monitoring reports

---

## Cloud Database Storage

All processed features and labels are stored in **MongoDB**, enabling:

- historical dataset reconstruction
- model training
- prediction monitoring

---

## Real-Time Monitoring

The system automatically sends **daily market diagnostics to Discord**, including:

- feature values
- 5-day changes
- market signals
- diagnostic charts

---

# System Architecture

The system follows a simple automated pipeline:

```
Market Data (Yahoo Finance)
        │
        ▼
Azure Function ETL Pipeline
        │
        ▼
Feature Engineering
        │
        ▼
MongoDB Storage
        │
        ▼
Model Training / Predictions
        │
        ▼
Discord Monitoring
```

The **Azure Function** acts as the central component responsible for **daily ETL operations**.

---

# Azure Function Pipeline

The Azure Function executes a **daily automated pipeline** consisting of:

1. Data Extraction  
2. Feature Engineering  
3. Target Construction  
4. MongoDB Storage  
5. Discord Reporting  

The pipeline runs automatically through a **timer trigger**.

---

# Pipeline Step 1 — Data Extraction

Market data is retrieved using **Yahoo Finance (`yfinance`)**.

Assets include several market segments.

## Equity Market

- `^GSPC` → S&P 500

## Volatility Market

- `^VIX` → CBOE Volatility Index
- `^VIX3M` → 3-Month VIX

## Fixed Income Volatility

- `^MOVE` → MOVE Index

## Macro Assets

- `DX-Y.NYB` → US Dollar Index
- `GC=F` → Gold
- `CL=F` → Crude Oil

## Credit Market

- `HYG` → High Yield ETF
- `LQD` → Investment Grade Credit ETF

All datasets are cleaned and merged into a **single time-indexed dataset**.

---

# Pipeline Step 2 — Feature Engineering

A wide range of **financial features** is generated to capture market dynamics.

---

## Returns & Realized Volatility

```python
dataset["Return_SPX"] = dataset["Close_SP500"].pct_change()

dataset["RV_5d"] = dataset["Return_SPX"].rolling(5).std() * np.sqrt(252)
dataset["RV_10d"] = dataset["Return_SPX"].rolling(10).std() * np.sqrt(252)
dataset["RV_21d"] = dataset["Return_SPX"].rolling(21).std() * np.sqrt(252)
```

These features measure **short- and medium-term realized volatility of the equity market**.

---

## VIX Behaviour

Several indicators describe the **structure and dynamics of implied volatility**.

Examples include:

- rolling volatility
- moving averages
- lagged values
- percentile ranks
- mean-reversion signals

Example implementation:

```python
dataset["VIX_MA_10"] = dataset["Close_VIX"].rolling(10).mean()
dataset["VIX_STD_10"] = dataset["Close_VIX"].rolling(10).std()

dataset["VIX_Zscore"] = (
    dataset["Close_VIX"] - dataset["VIX_MA_20"]
) / dataset["VIX_STD_10"]
```

These metrics help detect **extreme volatility regimes**.

---

## Cross-Market Signals

Interactions between markets often provide useful predictive signals.

Example:

```python
dataset["SPX_VIX_Corr_21d"] = (
    dataset["Return_SPX"]
    .rolling(21)
    .corr(dataset["Return_VIX"])
)
```

These signals capture **risk-on vs risk-off behaviour**.

---

# Target Construction

The prediction target is the **intraday return of the VIX**.

```python
dataset["Intraday_VIX_Return"] = (
    dataset["Close_VIX"] - dataset["Open_VIX"]
) / dataset["Open_VIX"]
```

The continuous return is converted into **three categories** using quantiles.

```python
q_up = dataset["Intraday_VIX_Return"].quantile(0.66)
q_down = dataset["Intraday_VIX_Return"].quantile(0.33)

dataset["Intraday_VIX_Move"] = np.where(
    dataset["Intraday_VIX_Return"] >= q_up, 1,
    np.where(dataset["Intraday_VIX_Return"] <= q_down, 2, 0)
)
```

This produces a **balanced classification problem**.

---

# MongoDB Storage

All processed features and targets are stored in **MongoDB**.

Each document represents a **daily market snapshot**.

Example:

```json
{
  "_id": "2008-01-02",
  "Intraday_VIX_Move": 1,
  "Open_SP500": 1467.969970703125,
  "Open_VIX": 22.579999923706055,
  "Open_MOVE": 148.6999969482422,
  "Close_DXY": 75.97000122070312,
  "Close_GOLD": 857,
  "Close_OIL": 99.62000274658203,
  "Close_HYG": 32.46661376953125,
  "Close_LQD": 52.85487365722656,
  "Drawdown": -0.07538573835983953,
  "Momentum_1M": -0.022941774668626125,
  "Momentum_3M": -0.06431400554049238,
  "Momentum_6M": -0.05096169587331145,
  "SPX_Gap": -0.0002656124195213074,
  "SPX_Volume_Norm": 1.0694843950416666,
  "RV_5d": 0.12323419006941308,
  "RV_10d": 0.15612578987057835,
  "RV_21d": 0.17864943140866799,
  "RV_21d_Sq": 0.03191561934264037,
  "VIX_Lag1": 22.5,
  "VIX_Lag2": 20.739999771118164,
  "VIX_Lag5": 18.600000381469727,
  "VIX_MA_5": 21.06599998474121,
  "VIX_MA_10": 20.729999923706053,
  "VIX_MA_20": 21.628999996185303,
  "VIX_STD_5": 1.8047105609744962,
  "VIX_STD_10": 1.7578396000033067,
  "VIX_Zscore": 0.87664430822115,
  "VIX_Zscore_Sq": 0.7685052431365387,
  "VIX_MeanRev": 2.440000152587892,
  "VIX_Percentile": 0.7744435612082671,
  "VIX_Trend": -0.3697488734814982,
  "VIX_Vol_5d": 0.03767876849142836,
  "VIX_Vol_10d": 0.06383982494675126,
  "VIX_Vol_21d": 0.059399284005228176,
  "VIX3M_Spread": -0.43000030517578125,
  "VIX_Contango": 0.018558493904181184,
  "VIX_RV_Spread": 22.991350644885276,
  "IV_RV_Ratio": 129.69534743881502,
  "SPX_VIX_Corr_21d": -0.8668932270886801,
  "VIX_MOVE_Ratio": 0.1558170850827838,
  "VIX_Gap": 0.003555552164713542
}
```

Database writes use **MongoDB upserts**, preventing duplicate entries.

---

# Automated Discord Monitoring

A reporting module sends **daily diagnostics to Discord**. -> https://discord.gg/j2KshhYmxp 

Reports include:

- feature values
- 5-day changes
- market indicators
- volatility metrics
- charts

Example output:

<img width="692" height="664" alt="image" src="https://github.com/user-attachments/assets/2654a848-b61f-4e76-85ff-f0fa75bbe649" />

The system also generates **diagnostic charts comparing VIX levels and SPX intraday behaviour**.

For security reasons, the **Discord webhook URL is not included in the public repository**.

---

# Execution Schedule

The Azure Function runs automatically using a **timer trigger**.

```
Monday – Friday
09:35 AM (ET)
```

This ensures the dataset is **updated daily before the main trading session evolves**.

---

# Azure Function Code

The following section contains the **full open-source implementation of the ETL pipeline**.

```python
import logging
import os
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from pymongo import MongoClient, UpdateOne
import certifi
import azure.functions as func
import requests
from io import BytesIO
import matplotlib.pyplot as plt

app = func.FunctionApp()

@app.timer_trigger(
    schedule="0 35 9 * * 1-5",  # Monday to Friday 9:35 AM ET
    arg_name="myTimer",
    run_on_startup=False,
    use_monitor=True
)
def timer_trigger_dbvix(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.warning("The timer is past due!")

    logging.info("Timer started.")

    # --- ETL ---
```
### Full source code:  [`function_app.py`](function_app.py)
---
