# 📊 Stacked Multi-Model Ensemble for Intraday VIX Prediction with Regime Awareness

**Author:** Aleix Francia Albert  
**Project:** MSc Thesis  

---

## Overview

This project implements a **regime-aware, stacked multi-model ensemble** for **categorical intraday prediction of the CBOE VIX**.  
The system combines multiple supervised machine learning models in a **stacked ensemble framework**, capturing **non-linear relationships** and **regime-dependent market dynamics**, delivering insights for **risk management**, **hedging**, and **quantitative trading strategies**.

**Key Objectives:**

- Predict intraday VIX movements under different market regimes  
- Incorporate historical VIX metrics, S&P 500 indicators, and derived volatility features  
- Ensure robustness during periods of high volatility and structural regime shifts

---

## Methodology

### 1. Data Collection & Feature Engineering
- Historical intraday VIX and S&P 500 prices  
- Derived volatility metrics: realized volatility (`RV_5d`, `RV_10d`, `RV_21d`), VIX implied volatility (`VIX_Vol_5d`, `VIX_Vol_21d`)  
- Market regime signals computed using **Markov Chains**  
- Feature engineering to capture:
  - Momentum (`Momentum_1M`, `Momentum_3M`, `Momentum_6M`)  
  - Mean-reversion (`VIX_MeanRev`)  
  - Intraday dynamics (`Intraday_VIX_Move`)  

### 2. Exploratory Data Analysis (EDA)
- Correlation analysis across VIX horizons (`VIX_5d`, `VIX_10d`, `VIX_15d`, etc.)  
- ΔVIX analysis by regime and quantile bins  
- Detection of anomalies and extreme events

### 3. Machine Learning Pipeline
- Individual models: `RandomForestClassifier`, `GradientBoostingClassifier`, `LightGBM`  
- **Stacked Ensemble:** Combines individual model outputs with weighted stacking  
- **Regime-aware inference:** Adjusts predictions based on detected market regimes (Low / Medium / High volatility)

### 4. Deployment & Monitoring
- Automated cloud pipeline implemented with **Azure Functions**  
- Daily computation of features, regime signals, and model predictions  
- Storage in **MongoDB** for structured access  
- Automated alerts and reporting through a **Discord bot**

---

## MongoDB Data Schema

Each intraday observation is stored as a **JSON document**:

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
    ....
  },
  "Stacked_Model_Prediction": "Increase",
  "Model_Probabilities": {
    "RandomForest": 0.65,
    "LightGBM": 0.72,
    "GradientBoosting": 0.68
  },
  "Updated_At": "2026-03-08T09:31:00Z"
}
