import logging
import resend
import os
import pandas as pd
import numpy as np
import yfinance as yf
import azure.functions as func
import requests
from io import BytesIO
import base64
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

    #---------------OUTPUT---------------
    webhook = os.environ.get("webhook")

    last_row = prediction.iloc[-1]
    last = data_final.iloc[-1]
    prev = data_final.iloc[-6]

    last_date = data_final.index[-1].strftime("%Y-%m-%d")
    prev_date = data_final.index[-6].strftime("%Y-%m-%d")

    # TABLE COMPARISON

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

    # QUANTILES + PROB

    q_down = data_final["q_down"].iloc[-1]
    q_up = data_final["q_up"].iloc[-1]

    if np.isnan(q_down) or np.isnan(q_up):
        note = "Quantiles not ready"
        q_down, q_up = 0.0, 0.0
    else:
        note = ""

    low, high = sorted([q_down, q_up])

    proba_text = f"""
    Model Output
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

    # CHART
    last_150 = data_final.tail(150).copy()

    # Smoothing + returns
    last_150["VIX_Smooth"] = last_150["Open_VIX"].ewm(span=10).mean()
    last_150["SPX_Return"] = last_150["Open_SP500"].pct_change() * 100

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # --- VIX (left axis)
    line1 = ax1.plot(
        last_150.index,
        last_150["VIX_Smooth"],
        color="blue",
        linewidth=2,
        label="VIX (EWMA 10)"
    )
    ax1.set_ylabel("VIX Level", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, alpha=0.25)

    # --- SPX (right axis)
    ax2 = ax1.twinx()

    line2 = ax2.plot(
        last_150.index,
        last_150["SPX_Return"],
        color="red",
        alpha=0.7,
        linewidth=1.5,
        label="SPX Daily Return (%)"
    )

    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("SPX Return (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # --- Title
    plt.title("Market Regime: VIX vs SPX (Last 150 Days)", fontsize=14, fontweight="bold")

    # --- Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.tight_layout()

    # --- Save buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    # DISCORD MESSAGE

    msg = f"**{title}**\n```\n{table}\n```"

    requests.post(
        webhook,
        data={"content": msg + "\n\n" + proba_text},
        files={"file": ("chart.png", buf, "image/png")}
    )

    # EMAIL (RESEND)

    resend.api_key = os.environ["RESEND_API_KEY"]
    
    html_table = """
    <table style="border-collapse: collapse; font-family: monospace; font-size: 12px;">
    """

    for k, v, e, c in rows:
        html_table += f"""
        <tr>
            <td style="padding:4px 10px; text-align:left;">{k}</td>
            <td style="padding:4px 10px; text-align:right;">{v}</td>
            <td style="padding:4px 10px; text-align:left;">{e}{c}</td>
        </tr>
    """

    html_table += "</table>"

    html_body = f"""
    <h2>{title}</h2>

    {html_table}

    <h3>Model Output</h3>

    <ul>
        <li><b>Decrease VIX Prob:</b> {last_row['proba_class_0']:.2%}</li>
        <li><b>Neutral VIX Prob:</b> {last_row['proba_class_1']:.2%}</li>
        <li><b>Increase VIX Prob:</b> {last_row['proba_class_2']:.2%}</li>
    </ul>

    <h3>Quantiles</h3>

    <ul>
        <li>Downside Move → &lt;{round(100*low,2)}%</li>
        <li>Neutral Move → {round(100*low,2)}% to {round(100*high,2)}%</li>
        <li>Upside Move → &gt;{round(100*high,2)}%</li>
    </ul>

    <p>{note}</p>
    """

    chart_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    email_response = resend.Emails.send({
        "from": "Market Bot <onboarding@resend.dev>",
        "to": ["afranciaa2501@gmail.com"],
        "subject": title,
        "html": html_body,
        "attachments": [
            {
                "filename": "vix_spx_chart.png",
                "content": chart_base64
            }
        ]
    })

    print(email_response)

    buf.close()
    plt.close(fig)
