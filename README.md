# 📊 Stacked Multi-Model Ensemble for Intraday VIX Prediction with Regime Awareness

**Author:** Aleix Francia Albert  
**Project:** MSc Thesis – Open Source Release  

---

## Overview

This repository implements a **regime-aware, stacked multi-model ensemble** for **categorical intraday prediction of the CBOE VIX**, designed for **research, risk management, and quantitative trading**.  

The system captures **non-linear dependencies** and **regime-based market dynamics** using multiple supervised learning models in a **stacked ensemble framework**.

**Key Capabilities:**

- Predict intraday VIX movements across Low/Medium/High volatility regimes  
- Incorporate historical VIX metrics, S&P 500 indicators, and derived features  
- Automated daily pipeline for cloud-based feature updates and predictions  
- Real-time reporting via Discord alerts  

---

## Azure Function Pipeline

The core of the project is an **automated Azure Function pipeline**, executing market ETL, feature computation, and MongoDB storage **every trading day**.  

### Pipeline Highlights

1. **Data Extraction**
   - Pulls historical market data for:
     - S&P 500 (`^GSPC`)
     - VIX (`^VIX`)
     - MOVE index (`^MOVE`)
   - Uses `yfinance` for historical OHLC prices  
   - Cleans and aligns timestamps  

2. **Feature Engineering**
   - **Returns & Realized Volatility**
     ```python
     dataset["Return_SPX"]  = dataset["Close_SP500"].pct_change()
     dataset["RV_5d"] = dataset["Return_SPX"].rolling(5).std() * np.sqrt(252)
     ```
   - **VIX-specific metrics**
     - VIX volatility (`VIX_Vol_5d`, `VIX_Vol_21d`)  
     - Mean reversion: rolling averages, Z-scores (`VIX_MA_10`, `VIX_MA_21`, `VIX_Zscore`)  
     - ΔVIX term structure proxy (`VIX_Trend`)  
   - **Cross-market & momentum**
     - S&P/VIX correlations (`SPX_VIX_Corr_21d`)  
     - Momentum 1M, 3M, 6M  
   - **Target Construction**
     ```python
     dataset["Intraday_VIX_Return"] = (dataset["Close_VIX"] - dataset["Open_VIX"]) / dataset["Open_VIX"]
     dataset["Intraday_VIX_Move"] = np.where(
         dataset["Intraday_VIX_Return"] >= q_up, 1,
         np.where(dataset["Intraday_VIX_Return"] <= q_down, 2, 0)
     )
     ```

3. **MongoDB Storage**
   - Saves processed features and target labels in MongoDB for historical tracking  
   - Performs **upserts** to prevent duplicate entries  
   - Example document:
     ```json
     {
       "_id": "2026-03-08T09:30:00Z",
       "Open_VIX": 22.34,
       "Close_VIX": 23.01,
       "Open_SP500": 4087.12,
       "Drawdown": -0.014,
       "Momentum_1M": 0.05,
       "VIX_Zscore": 1.23,
       "VIX_Trend": "MeanReversion",
       "VIX_Regime": "High",
       "Features": {
         "RV_5d": 0.018,
         "RV_10d": 0.022,
         "VIX_Vol_21d": 0.035,
         "SPX_Gap": -0.002
       },
       "Stacked_Model_Prediction": "Up",
       "Model_Probabilities": {
         "RandomForest": 0.65,
         "LightGBM": 0.72,
         "GradientBoosting": 0.68
       },
       "Updated_At": "2026-03-08T09:31:00Z"
     }
     ```

4. **Automated Discord Reporting**
   - Sends daily updates of latest metrics and 5-day changes  
   - Generates formatted tables of features and technical metrics  
   - Webhook URL is **blurred for security** in the public repository  

5. **Execution Schedule**
   - Triggered **Monday to Friday at 15:35 ET** via Azure Function timer

---

## Example Azure Function Code (Open Source Template)

```python
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from pymongo import MongoClient, UpdateOne
import certifi
import azure.functions as func
import requests

app = func.FunctionApp()

@app.timer_trigger(schedule="0 35 15 * * 1-5", arg_name="myTimer")
def timer_trigger_dbvix(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.warning("The timer is past due!")

    logging.info("Timer started.")

    # --- ETL: download market data ---
    sp500 = yf.Ticker("^GSPC").history(period="20y").drop(columns=["Dividends","Stock Splits"])
    vix   = yf.Ticker("^VIX").history(period="20y").drop(columns=["Dividends","Stock Splits","Volume"])
    move  = yf.Ticker("^MOVE").history(period="20y").drop(columns=["Dividends","Stock Splits","Volume"])

    # ...feature engineering (returns, RV, momentum, mean-reversion, VIX_Zscore)...

    # --- MongoDB upsert ---
    mongo_uri = "mongodb+srv://<USERNAME>:<PASSWORD>@<CLUSTER>/DB_VIX"
    client = MongoClient(mongo_uri, tls=True, tlsCAFile=certifi.where())
    db = client["DB_VIX"]
    collection = db["vix_data"]
    # Convert dataframe to dict and perform bulk upsert
    data_mongo["_id"] = data_mongo.index.astype(str)
    operations = [UpdateOne({"_id": r["_id"]}, {"$setOnInsert": r}, upsert=True) for r in data_mongo.to_dict("records")]
    collection.bulk_write(operations)

    # --- Discord Notification ---
    # webhook_url = "<BLURRED_FOR_SECURITY>"
    # requests.post(webhook_url, json={"content": "<formatted metrics table>"})

    logging.info("Timer finished.")
