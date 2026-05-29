import logging
import os
import pandas as pd
import numpy as np
import yfinance as yf
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

        df = df.rename(columns={
            "Open": f"Open_{name}",
            "Close": f"Close_{name}",
        })

        df.index = df.index.tz_localize(None)
        data[name] = df

    dataset = pd.concat(data.values(), axis=1).sort_index().ffill()

    # -------------------------
    # FEATURES
    # -------------------------
    close_spx = dataset["Close_SP500"].shift(1)
    close_vix = dataset["Close_VIX"].shift(1)

    dataset["VIX_Lag1"] = dataset["Close_VIX"].shift(1)

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

    feature_cols = [
        "Open_SP500",
        "Open_VIX",
        "VIX_Lag1",
        "RV_21d",
        "VIX_Zscore"
    ]

    data_final = dataset[feature_cols + ["Intraday_VIX_Move", "q_up", "q_down"]].dropna()

    # -------------------------
    # MODEL
    # -------------------------
    X = data_final[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    y = data_final["Intraday_VIX_Move"]

    model = LogisticRegression(C=75, max_iter=6000)
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    prediction = pd.DataFrame({
        "pred_class": preds,
        "proba_class_0": probs[:, 0],
        "proba_class_1": probs[:, 1],
        "proba_class_2": probs[:, 2],
    }, index=data_final.index)

    # -------------------------
    # DISCORD OUTPUT
    # -------------------------
    webhook = os.environ.get("webhook")

    last_row = prediction.iloc[-1]
    last = data_final.iloc[-1]
    prev = data_final.iloc[-6]

    last_date = data_final.index[-1].strftime("%Y-%m-%d")
    prev_date = data_final.index[-6].strftime("%Y-%m-%d")

    # -------------------------
    # TABLE COMPARISON
    # -------------------------
    pp = {"RV_21d"}

    rows = []
    for k in feature_cols:
        if k in last.index:
            val = f"{last[k]*100:.2f}%" if k in pp else f"{last[k]:.2f}"
            diff = last[k] - prev[k]
            chg = f"{diff*100:+.2f}pp" if k in pp else f"{diff:+.2f}"
            emo = "🟢" if diff > 0 else "🔴" if diff < 0 else "⚪"
            rows.append((k, val, emo, chg))

    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(r[1]) for r in rows)
    w4 = max(len(r[3]) for r in rows)

    table = "\n".join(
        f"{k:<{w1}} {v:>{w2}} {e}{c:>{w4}}"
        for k, v, e, c in rows
    )

    title = f"Market Snapshot {last_date} (Δ vs {prev_date})"

    # -------------------------
    # QUANTILES + PROB
    # -------------------------
    q_down = data_final["q_down"].iloc[-1]
    q_up = data_final["q_up"].iloc[-1]

    if np.isnan(q_down) or np.isnan(q_up):
        note = "Quantiles not ready"
        q_down, q_up = 0.0, 0.0
    else:
        note = ""

    low, high = sorted([q_down, q_up])

    proba_text = f"""
*Model Output*
------------------------
Decrease VIX Prob: {last_row['proba_class_0']:.2%}
Neutral VIX Prob: {last_row['proba_class_1']:.2%}
Increase VIX Prob: {last_row['proba_class_2']:.2%}

Quantiles:
Downside Move → <{round(100*low,2)}%
Neutral Move  → {round(100*low,2)}% to {round(100*high,2)}%
Upside Move   → >{round(100*high,2)}%

{note}
"""

    # -------------------------
    # CHART
    # -------------------------
    last_150 = data_final.tail(150).copy()

    last_150["VIX_Smooth"] = last_150["Open_VIX"].ewm(span=10).mean()
    last_150["SPX_Return"] = last_150["Open_SP500"].pct_change() * 100

    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(last_150.index, last_150["VIX_Smooth"], label="VIX", color="blue")
    ax2 = ax1.twinx()

    ax2.plot(last_150.index, last_150["SPX_Return"], label="SPX %", color="red", alpha=0.6)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)

    plt.title("VIX vs SPX (150D)")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    msg = f"**{title}**\n```\n{table}\n```"

    requests.post(
        webhook,
        data={"content": msg + "\n\n" + proba_text},
        files={"file": ("chart.png", buf, "image/png")}
    )

    buf.close()
    plt.close(fig)

    logging.info("Done.")
