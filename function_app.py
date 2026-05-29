import logging
import os
import pandas as pd
import numpy as np
import yfinance as yf
import certifi
import azure.functions as func
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

app = func.FunctionApp()

@app.timer_trigger(
    schedule="0 35 9 * * 1-5",
    arg_name="myTimer",
    run_on_startup=False,
    use_monitor=True
)
def timer_trigger_dbvix(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.warning("The timer is past due!")

    logging.info("Timer started.")

    # -------------------------
    # DATA
    # -------------------------
    tickers = {
        "SP500": "^GSPC",
        "VIX": "^VIX"
    }

    data = {}
    for name, ticker in tickers.items():
        df = yf.Ticker(ticker).history(period="20y")
        df = df.rename(columns={"Close": f"Close_{name}", "Open": f"Open_{name}"})
        df.index = df.index.tz_localize(None)
        data[name] = df

    dataset = pd.concat(data.values(), axis=1).sort_index().ffill()

    # -------------------------
    # FEATURES
    # -------------------------
    close_spx = dataset["Close_SP500"].shift(1)
    close_vix = dataset["Close_VIX"].shift(1)

    dataset["RV_21d"] = close_spx.pct_change().rolling(21).std() * np.sqrt(252)

    dataset["VIX_MA_20"] = close_vix.rolling(20).mean()
    dataset["VIX_STD_20"] = close_vix.rolling(20).std()

    dataset["VIX_Zscore"] = (
        close_vix - dataset["VIX_MA_20"]
    ) / (dataset["VIX_STD_20"] + 1e-8)

    dataset["Intraday_VIX_Return"] = (
        dataset["Close_VIX"] - dataset["Open_VIX"]
    ) / (dataset["Open_VIX"] + 1e-8)

    dataset["q_up"] = dataset["Intraday_VIX_Return"].shift(1).rolling(252).quantile(0.66)
    dataset["q_down"] = dataset["Intraday_VIX_Return"].shift(1).rolling(252).quantile(0.33)

    dataset["Intraday_VIX_Move"] = np.where(
        dataset["Intraday_VIX_Return"] >= dataset["q_up"], 1,
        np.where(dataset["Intraday_VIX_Return"] <= dataset["q_down"], 2, 0)
    )

    feature_cols = ["Open_SP500", "Open_VIX", "VIX_Zscore", "RV_21d"]

    data_final = dataset[feature_cols + ["Intraday_VIX_Move", "q_up", "q_down"]].dropna()

    # -------------------------
    # MODEL
    # -------------------------
    X = data_final[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    y = data_final["Intraday_VIX_Move"]

    model = LogisticRegression(C=75, max_iter=2000)
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    prediction = pd.DataFrame({
        "pred_class": preds,
        "p0": probs[:, 0],
        "p1": probs[:, 1],
        "p2": probs[:, 2],
    }, index=data_final.index)

    # -------------------------
    # LAST ROW
    # -------------------------
    last = prediction.tail(1).iloc[0]
    prev = prediction.tail(6).head(1).iloc[0]

    last_date = data_final.index[-1].strftime("%Y-%m-%d")
    prev_date = data_final.index[-6].strftime("%Y-%m-%d")

    # -------------------------
    # QUANTILES
    # -------------------------
    q_down = data_final["q_down"].iloc[-1]
    q_up = data_final["q_up"].iloc[-1]

    if np.isnan(q_down) or np.isnan(q_up):
        note = "Quantiles not ready"
        q_down, q_up = 0.0, 0.0
    else:
        note = ""

    low, high = sorted([q_down, q_up])

    # -------------------------
    # TABLE COMPARISON
    # -------------------------
    rows = [
        ("Pred", last["pred_class"], prev["pred_class"]),
        ("Prob Down", f"{last['p0']:.2%}", f"{prev['p0']:.2%}"),
        ("Prob Neutral", f"{last['p1']:.2%}", f"{prev['p1']:.2%}"),
        ("Prob Up", f"{last['p2']:.2%}", f"{prev['p2']:.2%}")
    ]

    table = "\n".join(
        f"{k:<15} {v:>10}  |  Δ prev: {p}"
        for k, v, p in rows
    )

    # -------------------------
    # CHART
    # -------------------------
    last_150 = data_final.tail(150)

    fig, ax1 = plt.subplots(figsize=(12, 4))

    ax1.plot(last_150.index, dataset["Close_VIX"].loc[last_150.index], label="VIX")
    ax2 = ax1.twinx()
    ax2.plot(last_150.index, dataset["Close_SP500"].loc[last_150.index], color="red", label="SPX")

    plt.title("VIX vs SPX (150D)")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    # -------------------------
    # OUTPUT TEXT
    # -------------------------
    proba_text = f"""
*Model Output*
------------------------
Decrease VIX Prob: {last['p0']:.2%}
Neutral VIX Prob: {last['p1']:.2%}
Increase VIX Prob: {last['p2']:.2%}

Confidence: {max(last['p0'], last['p1'], last['p2']):.2%}

Note:
Downside Move → <{round(100*low,2)}%
Neutral Move  → {round(100*low,2)}% to {round(100*high,2)}%
Upside Move   → >{round(100*high,2)}%

{note}
"""

    msg = f"""
**Market Snapshot {last_date} (vs {prev_date})**

{table}
"""

    # -------------------------
    # DISCORD
    # -------------------------
    webhook = os.environ.get("webhook")

    requests.post(
        webhook,
        data={"content": msg + "\n\n" + proba_text},
        files={"file": ("chart.png", buf, "image/png")}
    )

    logging.info("Done.")
