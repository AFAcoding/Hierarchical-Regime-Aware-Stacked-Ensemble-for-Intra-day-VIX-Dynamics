# Stacked Multi-Model Ensemble for Intraday VIX Prediction with Regime Awareness

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

## Azure Function Code (Open Source Template)
```python
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from pymongo import MongoClient, UpdateOne
import certifi
import azure.functions as func
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_score
from hmmlearn.hmm import VariationalGaussianHMM

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

    # --- Feature engineering ---
    for df, suffix in zip([sp500, vix, move], ["SP500","VIX","MOVE"]):
        df.index = df.index.tz_localize(None)
        df.rename(columns={
            "Open": f"Open_{suffix}", "High": f"High_{suffix}",
            "Low": f"Low_{suffix}", "Close": f"Close_{suffix}", 
            "Volume": f"Volume_{suffix}"
        }, inplace=True)

    dataset = pd.concat([sp500, vix, move], axis=1).sort_index()
    dataset["Return_SPX"]  = dataset["Close_SP500"].pct_change()
    dataset["Return_VIX"]  = dataset["Close_VIX"].pct_change()
    dataset["Return_MOVE"] = dataset["Close_MOVE"].pct_change()

    # Realized volatility
    dataset["RV_5d"]  = dataset["Return_SPX"].rolling(5).std() * np.sqrt(252)
    dataset["RV_10d"] = dataset["Return_SPX"].rolling(10).std() * np.sqrt(252)
    dataset["RV_21d"] = dataset["Return_SPX"].rolling(21).std() * np.sqrt(252)

    # Momentum
    dataset["Momentum_1M"] = dataset["Close_SP500"].pct_change(21)
    dataset["Momentum_3M"] = dataset["Close_SP500"].pct_change(63)
    dataset["Momentum_6M"] = dataset["Close_SP500"].pct_change(126)

    # Mean reversion / Z-score
    dataset["VIX_MA_10"] = dataset["Open_VIX"].rolling(10).mean()
    dataset["VIX_MA_21"] = dataset["Open_VIX"].rolling(21).mean()
    dataset["VIX_STD_21"] = dataset["Open_VIX"].rolling(21).std()
    dataset["VIX_Zscore"] = (dataset["Open_VIX"] - dataset["VIX_MA_21"]) / dataset["VIX_STD_21"]
    dataset["VIX_MeanRev"] = dataset["Open_VIX"] - dataset["VIX_MA_10"]

    # Target
    dataset["Intraday_VIX_Return"] = (dataset["Close_VIX"] - dataset["Open_VIX"]) / dataset["Open_VIX"]
    q_up, q_down = dataset["Intraday_VIX_Return"].quantile([0.66, 0.33])
    dataset["Intraday_VIX_Move"] = np.where(
        dataset["Intraday_VIX_Return"] >= q_up, 1,
        np.where(dataset["Intraday_VIX_Return"] <= q_down, 2, 0)
    )

    # Drop NaNs
    feature_cols = [
        "Open_SP500","Open_VIX","Open_MOVE",
        "Drawdown","Momentum_1M","Momentum_3M","Momentum_6M",
        "RV_5d","RV_10d","RV_21d","VIX_Zscore","VIX_MeanRev"
    ]
    data_final = dataset[feature_cols + ["Intraday_VIX_Move"]].dropna()

    # --- ML Pipeline: Stacked Ensemble with HMM regime ---
    # Base models
    model_r_forest_simple = RandomForestClassifier(
        n_estimators=1000, max_depth=6, min_samples_split=20,
        min_samples_leaf=10, max_features='sqrt', bootstrap=True,
        oob_score=True, n_jobs=-1, random_state=42
    )
    model_r_forest_complex = RandomForestClassifier(
        n_estimators=1000, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt', bootstrap=True,
        n_jobs=-1, random_state=42
    )
    stack_classify = VotingClassifier(
        estimators=[('r_forest_simple', model_r_forest_simple),
                    ('r_forest_complex', model_r_forest_complex)],
        voting='soft', n_jobs=-1
    )

    # Example splits (to be replaced with your train/test splits)
    splits = [(data_final[feature_cols].iloc[:-50], data_final[feature_cols].iloc[-50:],
               data_final["Intraday_VIX_Move"].iloc[:-50], data_final["Intraday_VIX_Move"].iloc[-50:])]

    hmm_cols = ["RV_21d", "VIX_Zscore"]
    scaler = StandardScaler()
    pred_real = pd.DataFrame()
    scores = []

    for i, (X_train, X_test, Y_train, Y_test) in enumerate(splits):
        if X_test is None or Y_test is None:
            continue

        # HMM Regime Detection
        X_train_hmm_scaled = scaler.fit_transform(X_train[hmm_cols])
        hmm = VariationalGaussianHMM(n_components=3, covariance_type="full", n_iter=1400,
                                     tol=1e-3, random_state=42, init_params="stmc", params="stmc")
        hmm.fit(X_train_hmm_scaled)
        train_probs = hmm.predict_proba(X_train_hmm_scaled)
        X_test_hmm_scaled = scaler.transform(X_test[hmm_cols])
        test_probs = hmm.predict_proba(X_test_hmm_scaled)
        for s in range(train_probs.shape[1]):
            X_train[f"hmm_state_{s}"] = train_probs[:, s]
            X_test[f"hmm_state_{s}"] = test_probs[:, s]

        # Train stacked model
        stack_classify.fit(X_train, Y_train)
        preds_stack = stack_classify.predict(X_test)

        pred_real = pd.concat([
            pred_real,
            pd.DataFrame({
                'pred_class': preds_stack,
                'real_class': Y_test.values.ravel(),
                'residual_class': Y_test.values.ravel() - preds_stack,
                'test_regime': hmm.predict(X_test_hmm_scaled)
            }, index=Y_test.index)
        ])

        precision = precision_score(Y_test, preds_stack, zero_division=0, average="macro")
        scores.append(precision)
        logging.info(f"Split {i+1}/{len(splits)} completed")

    logging.info(f"Precision mean: {np.mean(scores)}")

    # --- MongoDB upsert ---
    mongo_uri = "mongodb+srv://<USERNAME>:<PASSWORD>@<CLUSTER>/DB_VIX"
    client = MongoClient(mongo_uri, tls=True, tlsCAFile=certifi.where())
    db = client["DB_VIX"]
    collection = db["vix_data"]

    data_mongo = data_final.copy()
    data_mongo["_id"] = data_mongo.index.astype(str)
    operations = [UpdateOne({"_id": r["_id"]}, {"$setOnInsert": r}, upsert=True)
                  for r in data_mongo.to_dict("records")]
    if operations:
        result = collection.bulk_write(operations)
        logging.info(f"Inserted {result.upserted_count} new records")
    else:
        logging.info("No new records to insert")

    # --- Discord Notification ---
    # webhook_url = "<BLURRED_FOR_SECURITY>"
    # requests.post(webhook_url, json={"content": "<formatted metrics table>"})

    logging.info("Timer finished.")
```
