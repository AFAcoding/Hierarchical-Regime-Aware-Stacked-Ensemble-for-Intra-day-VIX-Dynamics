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
    sp500 = yf.Ticker("^GSPC").history(period="3y")
    vix = yf.Ticker("^VIX").history(period="3y")
    move = yf.Ticker("^MOVE").history(period="3y")
    vix3m = yf.Ticker("^VIX3M").history(period="3y")
    dxy = yf.Ticker("DX-Y.NYB").history(period="3y")
    gold = yf.Ticker("GC=F").history(period="3y")
    oil = yf.Ticker("CL=F").history(period="3y")
    hvyg = yf.Ticker("HYG").history(period="3y")
    ivig = yf.Ticker("LQD").history(period="3y")

    sp500 = sp500.drop(columns=["Dividends","Stock Splits"])
    vix = vix.drop(columns=["Dividends","Stock Splits","Volume"])
    move = move.drop(columns=["Dividends","Stock Splits","Volume"])
    vix3m = vix3m.drop(columns=["Dividends","Stock Splits","Volume"])
    dxy = dxy.drop(columns=["Dividends","Stock Splits","Volume"])
    gold = gold.drop(columns=["Dividends","Stock Splits","Volume"])
    oil = oil.drop(columns=["Dividends","Stock Splits","Volume"])
    hvyg = hvyg.drop(columns=["Dividends","Stock Splits","Volume"])
    ivig = ivig.drop(columns=["Dividends","Stock Splits","Volume"])

    def rename_asset(df,suffix):
        return df.rename(columns={
            "Open":f"Open_{suffix}",
            "High":f"High_{suffix}",
            "Low":f"Low_{suffix}",
            "Close":f"Close_{suffix}",
            "Volume":f"Volume_{suffix}"
        })

    sp500 = rename_asset(sp500,"SP500")
    vix = rename_asset(vix,"VIX")
    move = rename_asset(move,"MOVE")
    vix3m = rename_asset(vix3m,"VIX3M")
    dxy = rename_asset(dxy,"DXY")
    gold = rename_asset(gold,"GOLD")
    oil = rename_asset(oil,"OIL")
    hvyg = rename_asset(hvyg,"HYG")
    ivig = rename_asset(ivig,"LQD")

    for df in [sp500,vix,move,vix3m,dxy,gold,oil,hvyg,ivig]:
        df.index = df.index.tz_localize(None)

    dataset = pd.concat([sp500,vix,move,vix3m,dxy,gold,oil,hvyg,ivig],axis=1).sort_index()

    dataset["Return_SPX"]  = dataset["Close_SP500"].pct_change(fill_method=None)
    dataset["Return_VIX"]  = dataset["Close_VIX"].pct_change(fill_method=None)
    dataset["Return_MOVE"] = dataset["Close_MOVE"].pct_change(fill_method=None)
    dataset["Return_VIX3M"]= dataset["Close_VIX3M"].pct_change(fill_method=None)
    dataset["RV_5d"] = dataset["Return_SPX"].rolling(5).std()*np.sqrt(252)
    dataset["RV_10d"] = dataset["Return_SPX"].rolling(10).std()*np.sqrt(252)
    dataset["RV_21d"] = dataset["Return_SPX"].rolling(21).std()*np.sqrt(252)
    dataset["VIX_Vol_5d"] = dataset["Return_VIX"].rolling(5).std()
    dataset["VIX_Vol_10d"] = dataset["Return_VIX"].rolling(10).std()
    dataset["VIX_Vol_21d"] = dataset["Return_VIX"].rolling(21).std()
    dataset["VIX_Lag1"] = dataset["Close_VIX"].shift(1)
    dataset["VIX_Lag2"] = dataset["Close_VIX"].shift(2)
    dataset["VIX_Lag5"] = dataset["Close_VIX"].shift(5)
    dataset["VIX_MA_5"] = dataset["Close_VIX"].rolling(5).mean()
    dataset["VIX_MA_10"] = dataset["Close_VIX"].rolling(10).mean()
    dataset["VIX_MA_20"] = dataset["Close_VIX"].rolling(20).mean()
    dataset["VIX_STD_5"] = dataset["Close_VIX"].rolling(5).std()
    dataset["VIX_STD_10"] = dataset["Close_VIX"].rolling(10).std()
    dataset["VIX_Percentile"] = dataset["Close_VIX"].rank(pct=True)
    dataset["SPX_Volume_Norm"] = dataset["Volume_SP500"]/dataset["Volume_SP500"].rolling(252).mean()
    dataset["VIX3M_Spread"] = dataset["Close_VIX"] - dataset["Close_VIX3M"]
    dataset["VIX_Contango"] = dataset["Close_VIX3M"]/dataset["Close_VIX"] - 1
    dataset["SPX_Gap"] = (dataset["Open_SP500"] - dataset["Close_SP500"].shift(1))/dataset["Close_SP500"].shift(1)
    dataset["VIX_Gap"] = (dataset["Open_VIX"] - dataset["Close_VIX"].shift(1))/dataset["Close_VIX"].shift(1)
    dataset["Drawdown"] = dataset["Close_SP500"]/dataset["Close_SP500"].cummax() -1
    dataset["Momentum_1M"] = dataset["Close_SP500"]/dataset["Close_SP500"].shift(21)-1
    dataset["Momentum_3M"] = dataset["Close_SP500"]/dataset["Close_SP500"].shift(63)-1
    dataset["Momentum_6M"] = dataset["Close_SP500"]/dataset["Close_SP500"].shift(126)-1
    dataset["VIX_Zscore"] = (dataset["Close_VIX"]-dataset["VIX_MA_20"])/dataset["VIX_STD_10"]
    dataset["VIX_MeanRev"] = dataset["Close_VIX"] - dataset["VIX_MA_10"]
    dataset["IV_RV_Ratio"] = dataset["Close_VIX"]/dataset["RV_21d"]
    dataset["VIX_RV_Spread"] = dataset["Close_VIX"] - dataset["RV_21d"]
    dataset["VIX_Trend"] = dataset["Close_VIX"].ewm(span=21,adjust=False).mean() - dataset["Close_VIX"].ewm(span=63,adjust=False).mean()
    dataset["VIX_MOVE_Ratio"] = dataset["Close_VIX"]/dataset["Close_MOVE"]
    dataset["SPX_VIX_Corr_21d"] = dataset["Return_SPX"].rolling(21).corr(dataset["Return_VIX"])
    dataset["RV_21d_Sq"] = dataset["RV_21d"]**2
    dataset["VIX_Zscore_Sq"] = dataset["VIX_Zscore"]**2
    dataset["Intraday_VIX_Return"] = (dataset["Close_VIX"]-dataset["Open_VIX"])/dataset["Open_VIX"]
    q_up = dataset["Intraday_VIX_Return"].quantile(0.66)
    q_down = dataset["Intraday_VIX_Return"].quantile(0.33)
    dataset["Intraday_VIX_Move"] = np.where(dataset["Intraday_VIX_Return"]>=q_up,1,np.where(dataset["Intraday_VIX_Return"]<=q_down,2,0))

    feature_cols = [
        "Open_SP500","Open_VIX","Open_MOVE",
        "Drawdown",
        "Momentum_1M","Momentum_3M","Momentum_6M",
        "RV_5d","RV_10d","RV_21d",
        "VIX_Vol_5d","VIX_Vol_10d","VIX_Vol_21d",
        "VIX_Lag1","VIX_Lag2","VIX_Lag5",
        "VIX_MA_5","VIX_MA_10","VIX_MA_20",
        "VIX_STD_5","VIX_STD_10","VIX_Percentile",
        "SPX_Volume_Norm",
        "VIX3M_Spread","VIX_Contango",
        "SPX_Gap","VIX_Gap",
        "VIX_Zscore","VIX_Zscore_Sq","VIX_MeanRev",
        "IV_RV_Ratio","VIX_RV_Spread","VIX_Trend",
        "VIX_MOVE_Ratio","SPX_VIX_Corr_21d","RV_21d_Sq",
        "Close_DXY","Close_GOLD","Close_OIL",
        "Close_HYG","Close_LQD"
    ]

    data_final = dataset[feature_cols + ["Intraday_VIX_Move"]]

    # --- MongoDB ---
    mongo_uri = os.environ.get("mongo_uri")
    client = MongoClient(mongo_uri, tls=True, tlsCAFile=certifi.where())
    db = client["DB_VIX"]
    collection = db["vix_data"]

    data_mongo = data_final.copy()
    data_mongo["_id"] = data_mongo.index.astype(str)
    records = data_mongo.to_dict("records")

    operations = [
        UpdateOne({"_id": r["_id"]}, {"$setOnInsert": r}, upsert=True)
        for r in records
    ]
    if operations:
        result = collection.bulk_write(operations)
        logging.info(f"New records inserted: {result.upserted_count}")
    else:
        logging.info("No new data to insert.")

    logging.info("Timer finished.")
    
    # Send last info to Discord
    last_info_dict = data_final.to_dict(orient="records")[0]

    # Data 5 days ago
    prev_info_dict = data_final.tail(6).head(1).to_dict(orient="records")[0]

    # Metrics we want to show as percentage points
    pp_metrics = ["Drawdown", "Momentum_1M", "Momentum_3M", "Momentum_6M",
                "RV_5d", "RV_10d", "RV_21d", "VIX_Vol_5d", "VIX_Vol_21d"]

    # Construct technical column
    technical_dict = {}
    for k in last_info_dict.keys():
        if isinstance(last_info_dict[k], (int, float)) and isinstance(prev_info_dict[k], (int, float)):
            diff = last_info_dict[k] - prev_info_dict[k]
            if k in pp_metrics:
                diff *= 100  # convertimos a puntos porcentuales
                technical_dict[k] = f"{diff:+.2f} p.p."
            else:
                technical_dict[k] = f"{diff:+.2f}"
        else:
            technical_dict[k] = "-"

    # Values 
    value_str_dict = {}
    for k, v in last_info_dict.items():
        if k in pp_metrics:
            value_str_dict[k] = f"{v*100:.2f}%"  # convertimos a porcentaje
        else:
            value_str_dict[k] = f"{v:.2f}"
            
    col_width = max(len(k) for k in last_info_dict.keys())
    val_width = max(len(v) for v in value_str_dict.values())
    tech_width = max(len(v) for v in technical_dict.values())
    tech_width = max(tech_width, len("Change(5d)"))

    table_header = f"| {'Feature'.ljust(col_width)} | {'Value'.rjust(val_width)} | {'Change(5d)'.rjust(tech_width)} |"
    table_divider = f"|{'-'*(col_width+2)}|{'-'*(val_width+2)}|{'-'*(tech_width+2)}|"

    table_rows = "\n".join([
        f"| {k.ljust(col_width)} | {value_str_dict[k].rjust(val_width)} | {technical_dict[k].rjust(tech_width)} |"
        for k in last_info_dict.keys()
    ])

    last_info_str = f"```\n{table_header}\n{table_divider}\n{table_rows}\n```"

    # Send to Discord
    webhook = os.environ.get("webhook")

    selected_features = [
    "Open_SP500",
    "Open_VIX",
    "Drawdown",
    "Momentum_1M",
    "Momentum_3M",
    "RV_5d",
    "RV_21d",
    "VIX_Vol_5d",
    "VIX_Vol_21d",
    "VIX_Lag1",
    "VIX_MA_20",
    "VIX_STD_10",
    "VIX_Percentile",
    "VIX3M_Spread",
    "VIX_Contango",
    "SPX_VIX_Corr_21d",
    "Close_DXY",
    "Close_GOLD",
    "Close_OIL"]
    
    last, prev = data_final.iloc[-1], data_final.iloc[-6]

    last_date = data_final.index[-1].strftime("%Y-%m-%d")
    prev_date = data_final.index[-6].strftime("%Y-%m-%d")

    pp = {"Drawdown","Momentum_1M","Momentum_3M","RV_5d","RV_21d","VIX_Vol_5d","VIX_Vol_21d"}

    rows = []
    for k in selected_features:
        if k in last.index and isinstance(last[k], (int,float)):
            name = k
            val = f"{last[k]*100:.2f}%" if k in pp else f"{last[k]:.2f}"
            diff = last[k] - prev[k]
            chg = f"{diff*100:+.2f}pp" if k in pp else f"{diff:+.2f}"
            emo = "🟢" if diff>0 else "🔴" if diff<0 else "⚪"
            rows.append((name,val,emo,chg))

    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(r[1]) for r in rows)
    w4 = max(len(r[3]) for r in rows)

    table = "\n".join(f"{k:<{w1}} {v:>{w2}} {e}{c:>{w4}}" for k,v,e,c in rows)
    title = f"Market Snapshot {last_date}  (Δ vs {prev_date})"

    # -------- CHART --------

    last_150 = data_final.tail(150).copy()

    last_150["VIX_Smooth"] = (last_150["Open_VIX"].rolling(30,1).mean().rolling(7,1).mean())

    last_150["SPX_Smooth"] = ((last_150["Open_SP500"] / last_150["Open_SP500"].shift(-1) - 1)*100).rolling(7,1).mean()

    fig, ax1 = plt.subplots(figsize=(16,5))

    ax1.plot(last_150.index,last_150["VIX_Smooth"],lw=2,color="blue",alpha=0.7,label="VIX")

    ax2 = ax1.twinx()
    ax2.plot(last_150.index,last_150["SPX_Smooth"],lw=2,color="red",alpha=0.5,label="SPX %")
    ax2.fill_between(last_150.index,last_150["SPX_Smooth"],0,alpha=0.1,color="red")
    ax2.axhline(0,color="gray",ls="--",lw=1)

    plt.title("VIX vs SPX Intra-day % (150d)")
    plt.xticks(rotation=45)
    plt.grid(axis="y",ls="--",alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf,format="png")
    buf.seek(0)

    msg = f"**{title}**\n```\n{table}\n```"

    requests.post(webhook,data={"content": msg},files={"file": ("chart.png", buf, "image/png")})

    buf.close()
    plt.close(fig)
```

---
