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
    # --- DOWNLOAD ---
    tickers = {
        "SP500": "^GSPC",
        "VIX": "^VIX",
        "MOVE": "^MOVE",
        "VIX3M": "^VIX3M",
        "DXY": "DX-Y.NYB",
        "GOLD": "GC=F",
        "OIL": "CL=F",
        "HYG": "HYG",
        "LQD": "LQD"
    }

    data = {}
    for name, ticker in tickers.items():
        df = yf.Ticker(ticker).history(period="20y")

        df = df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")

        if name != "SP500":
            df = df.drop(columns=["Volume"], errors="ignore")

        df = df.rename(columns={
            "Open": f"Open_{name}",
            "High": f"High_{name}",
            "Low": f"Low_{name}",
            "Close": f"Close_{name}",
            "Volume": f"Volume_{name}"
        })

        df.index = df.index.tz_localize(None)
        data[name] = df

    # --- MERGE + ALIGNMENT ---
    dataset = pd.concat(data.values(), axis=1).sort_index()
    dataset = dataset.asfreq("B")
    dataset = dataset.ffill()

    # --- SHIFTED SERIES (ANTI-LEAKAGE CORE) ---
    close_spx = dataset["Close_SP500"].shift(1)
    close_vix = dataset["Close_VIX"].shift(1)
    close_move = dataset["Close_MOVE"].shift(1)
    close_vix3m = dataset["Close_VIX3M"].shift(1)

    return_spx = close_spx.pct_change()
    return_vix = close_vix.pct_change()
    return_move = close_move.pct_change()
    return_vix3m = close_vix3m.pct_change()

    # --- RETURNS ---
    dataset["Return_SPX"]  = return_spx
    dataset["Return_VIX"]  = return_vix
    dataset["Return_MOVE"] = return_move
    dataset["Return_VIX3M"]= return_vix3m

    # --- VOL ---
    dataset["RV_5d"]  = return_spx.rolling(5).std() * np.sqrt(252)
    dataset["RV_10d"] = return_spx.rolling(10).std() * np.sqrt(252)
    dataset["RV_21d"] = return_spx.rolling(21).std() * np.sqrt(252)

    dataset["VIX_Vol_5d"]  = return_vix.rolling(5).std()
    dataset["VIX_Vol_10d"] = return_vix.rolling(10).std()
    dataset["VIX_Vol_21d"] = return_vix.rolling(21).std()

    # --- LAGS ---
    dataset["VIX_Lag1"] = dataset["Close_VIX"].shift(1)
    dataset["VIX_Lag2"] = dataset["Close_VIX"].shift(2)
    dataset["VIX_Lag5"] = dataset["Close_VIX"].shift(5)

    # --- MOVING STATS (NO LEAKAGE focused) ---
    dataset["VIX_MA_5"]  = close_vix.rolling(5).mean()
    dataset["VIX_MA_10"] = close_vix.rolling(10).mean()
    dataset["VIX_MA_20"] = close_vix.rolling(20).mean()

    dataset["VIX_STD_5"]  = close_vix.rolling(5).std()
    dataset["VIX_STD_10"] = close_vix.rolling(10).std()
    dataset["VIX_STD_20"] = close_vix.rolling(20).std()

    dataset["VIX_Percentile"] = close_vix.rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )

    # --- VOLUME ---
    dataset["SPX_Volume_Norm"] = dataset["Volume_SP500"] / (
        dataset["Volume_SP500"].rolling(252).mean() + 1e-8
    )

    # --- STRUCTURE ---
    dataset["VIX3M_Spread"] = close_vix - close_vix3m
    dataset["VIX_Contango"] = close_vix3m / (close_vix + 1e-8) - 1

    # --- GAPS ---
    dataset["SPX_Gap"] = (dataset["Open_SP500"] - close_spx) / (close_spx + 1e-8)
    dataset["VIX_Gap"] = (dataset["Open_VIX"] - close_vix) / (close_vix + 1e-8)

    # --- TREND / MOMENTUM ---
    dataset["Drawdown"] = close_spx / close_spx.cummax() - 1

    dataset["Momentum_1M"] = close_spx / close_spx.shift(21) - 1
    dataset["Momentum_3M"] = close_spx / close_spx.shift(63) - 1
    dataset["Momentum_6M"] = close_spx / close_spx.shift(126) - 1

    dataset["VIX_Zscore"] = (
        close_vix - dataset["VIX_MA_20"]
    ) / (dataset["VIX_STD_20"] + 1e-8)

    dataset["VIX_MeanRev"] = close_vix - dataset["VIX_MA_10"]

    dataset["IV_RV_Ratio"] = close_vix / (dataset["RV_21d"] + 1e-8)
    dataset["VIX_RV_Spread"] = close_vix - dataset["RV_21d"]

    dataset["VIX_Trend"] = (
        close_vix.ewm(span=21, adjust=False).mean()
        - close_vix.ewm(span=63, adjust=False).mean()
    )

    dataset["VIX_MOVE_Ratio"] = close_vix / (close_move + 1e-8)

    dataset["SPX_VIX_Corr_21d"] = return_spx.rolling(21).corr(return_vix)

    dataset["RV_21d_Sq"] = dataset["RV_21d"] ** 2
    dataset["VIX_Zscore_Sq"] = dataset["VIX_Zscore"] ** 2

    # --- MACRO FEATURES (NO LEAKAGE focus) ---
    dataset["DXY_overnight"]  = dataset["Open_DXY"]  / dataset["Open_DXY"].shift(1)  - 1
    dataset["GOLD_overnight"] = dataset["Open_GOLD"] / dataset["Open_GOLD"].shift(1) - 1
    dataset["OIL_overnight"]  = dataset["Open_OIL"]  / dataset["Open_OIL"].shift(1)  - 1

    # --- TARGET (categorical and balanced q1,q2,q3) ---
    dataset["Intraday_VIX_Return"] = (
        dataset["Close_VIX"] - dataset["Open_VIX"]
    ) / (dataset["Open_VIX"] + 1e-8)

    dataset["q_up"] = dataset["Intraday_VIX_Return"].shift(1).rolling(252).quantile(0.66)
    dataset["q_down"] = dataset["Intraday_VIX_Return"].shift(1).rolling(252).quantile(0.33)

    dataset["Intraday_VIX_Move"] = np.where(
        dataset["Intraday_VIX_Return"] >= dataset["q_up"], 1,
        np.where(dataset["Intraday_VIX_Return"] <= dataset["q_down"], 2, 0)
    )

    # --- FEATURES ---
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
        "Open_DXY","Open_GOLD","Open_OIL",
        "Open_HYG","Open_LQD",
        "DXY_overnight","GOLD_overnight","OIL_overnight"
    ]

    data_final = dataset[feature_cols + ["Intraday_VIX_Move"]].dropna()

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
                diff *= 100  # transform to percentage points
                technical_dict[k] = f"{diff:+.2f} p.p."
            else:
                technical_dict[k] = f"{diff:+.2f}"
        else:
            technical_dict[k] = "-"

    # Values 
    value_str_dict = {}
    for k, v in last_info_dict.items():
        if k in pp_metrics:
            value_str_dict[k] = f"{v*100:.2f}%"  # transform to percentage points
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
    
    last, prev = data_final.iloc[-1], data_final.iloc[-6]

    last_date = data_final.index[-1].strftime("%Y-%m-%d")
    prev_date = data_final.index[-6].strftime("%Y-%m-%d")

    pp = {"Drawdown","Momentum_1M","Momentum_3M","RV_5d","RV_21d","VIX_Vol_5d","VIX_Vol_21d"}

    rows = []
    for k in feature_cols:
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

    last_150["VIX_Smooth"] = last_150["Open_VIX"].ewm(span=10, adjust=False).mean()

    last_150["SPX_Return"] = (last_150["Open_SP500"] / last_150["Open_SP500"].shift(1) - 1) * 100

    fig, ax1 = plt.subplots(figsize=(16,5))

    ax1.plot(last_150.index,last_150["VIX_Smooth"],lw=2,color="blue",alpha=0.7,label="VIX")

    ax2 = ax1.twinx()
    ax2.plot(last_150.index,last_150["SPX_Return"],lw=2,color="red",alpha=0.5,label="SPX %")
    ax2.fill_between(last_150.index,last_150["SPX_Return"],0,alpha=0.1,color="red")
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
