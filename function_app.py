import logging, os, pandas as pd, numpy as np, yfinance as yf, certifi, requests, matplotlib.pyplot as plt, azure.functions as func
from io import BytesIO
from pymongo import MongoClient, UpdateOne
from sklearn.decomposition import PCA
from hmmlearn.hmm import VariationalGaussianHMM
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = func.FunctionApp()

@app.timer_trigger(schedule="0 40 9 * * 1-5", arg_name="myTimer", run_on_startup=False, use_monitor=True)
def timer_trigger_dbvix(myTimer: func.TimerRequest):

    if myTimer.past_due:
        logging.warning("The timer is past due!")
    logging.info("Timer started.")

    # ---------------- DATA ----------------
    tickers = {
        "SP500":"^GSPC","VIX":"^VIX","MOVE":"^MOVE","VIX3M":"^VIX3M",
        "DXY":"DX-Y.NYB","GOLD":"GC=F","OIL":"CL=F","HYG":"HYG","LQD":"LQD"
    }

    data = {}
    for n,t in tickers.items():
        df = yf.Ticker(t).history("20y").drop(columns=["Dividends","Stock Splits"], errors="ignore")
        if n != "SP500":
            df = df.drop(columns=["Volume"], errors="ignore")

        df = df.rename(columns=lambda c: f"{c}_{n}")
        df.index = df.index.tz_localize(None)
        data[n] = df

    d = pd.concat(data.values(), axis=1).asfreq("B").ffill()

    cspx, cvix, cmove, cvix3m = (
        d["Close_SP500"].shift(1),
        d["Close_VIX"].shift(1),
        d["Close_MOVE"].shift(1),
        d["Close_VIX3M"].shift(1)
    )

    r = lambda x: x.pct_change()

    # ---------------- FEATURES ----------------
    d["Return_SPX"], d["Return_VIX"], d["Return_MOVE"], d["Return_VIX3M"] = r(cspx), r(cvix), r(cmove), r(cvix3m)

    d["RV_5d"], d["RV_10d"], d["RV_21d"] = [r(cspx).rolling(x).std()*np.sqrt(252) for x in (5,10,21)]
    d["VIX_Vol_5d"], d["VIX_Vol_10d"], d["VIX_Vol_21d"] = [r(cvix).rolling(x).std() for x in (5,10,21)]

    d["Drawdown"] = cspx/cspx.cummax()-1
    d["Mom_1M"], d["Mom_3M"], d["Mom_6M"] = cspx/cspx.shift(21)-1, cspx/cspx.shift(63)-1, cspx/cspx.shift(126)-1

    d["VIX_Z"] = (cvix-cvix.rolling(20).mean())/(cvix.rolling(20).std()+1e-8)
    d["IV_RV"] = cvix/(d["RV_21d"]+1e-8)

    # ---------------- TARGET ----------------
    d["q_up"] = d["Return_VIX"].shift(1).rolling(252).quantile(.66)
    d["q_down"] = d["Return_VIX"].shift(1).rolling(252).quantile(.33)

    d["y"] = np.select(
        [d["Return_VIX"] >= d["q_up"], d["Return_VIX"] <= d["q_down"]],
        [1,2], 0
    )

    data_final = d.dropna()

    # ---------------- MONGO ----------------
    client = MongoClient(os.getenv("mongo_uri"), tls=True, tlsCAFile=certifi.where())
    col = client.DB_VIX.vix_data

    col.bulk_write([
        UpdateOne({"_id":i},{"$setOnInsert":r},upsert=True)
        for i,r in data_final.assign(_id=data_final.index.astype(str)).to_dict("records")
    ])

    # ---------------- MODEL ----------------
    def clean(X):
        return X.replace([np.inf,-np.inf],np.nan).ffill().fillna(0)

    X, y = clean(data_final.drop(columns=["y"])), data_final["y"]

    Xtr, Xte, ytr, yte = X.iloc[:-1], X.iloc[-1:], y.iloc[:-1], y.iloc[-1:]

    pca = Pipeline([
        ("c", FunctionTransformer(clean)),
        ("s", StandardScaler()),
        ("p", PCA(14))
    ])

    Xtr_p = pd.DataFrame(pca.fit_transform(Xtr))
    Xte_p = pd.DataFrame(pca.transform(Xte))

    hmm = VariationalGaussianHMM(3, covariance_type="full", n_iter=1000).fit(Xtr_p.iloc[:, :3])

    tr_p = hmm.predict_proba(Xtr_p.iloc[:, :3])
    te_p = hmm.predict_proba(Xte_p.iloc[:, :3])

    for i in range(3):
        Xtr_p[f"h{i}"] = tr_p[:, i]
        Xte_p[f"h{i}"] = te_p[:, i]

    model = LogisticRegression(C=75, max_iter=6000).fit(Xtr_p, ytr)

    pred = model.predict(Xte_p)[0]
    prob = model.predict_proba(Xte_p)[0]

    # ---------------- OUTPUT (UNCHANGED LOGIC) ----------------
    last_info_dict = data_final.to_dict(orient="records")[0]
    prev_info_dict = data_final.tail(6).head(1).to_dict(orient="records")[0]

    pp_metrics = ["Drawdown","Momentum_1M","Momentum_3M","Momentum_6M","RV_5d","RV_10d","RV_21d","VIX_Vol_5d","VIX_Vol_21d"]

    technical_dict = {}
    for k in last_info_dict:
        if isinstance(last_info_dict[k], (int,float)) and isinstance(prev_info_dict[k], (int,float)):
            diff = last_info_dict[k]-prev_info_dict[k]
            technical_dict[k] = f"{diff*100:+.2f} p.p." if k in pp_metrics else f"{diff:+.2f}"
        else:
            technical_dict[k] = "-"

    value_str_dict = {
        k: f"{v*100:.2f}%" if k in pp_metrics else f"{v:.2f}"
        for k,v in last_info_dict.items()
    }

    col_width = max(len(k) for k in last_info_dict)
    val_width = max(len(v) for v in value_str_dict.values())
    tech_width = max(len(v) for v in technical_dict.values())

    table_header = f"| {'Feature'.ljust(col_width)} | {'Value'.rjust(val_width)} | {'Change(5d)'.rjust(tech_width)} |"
    table_divider = f"|{'-'*(col_width+2)}|{'-'*(val_width+2)}|{'-'*(tech_width+2)}|"

    table_rows = "\n".join(
        f"| {k.ljust(col_width)} | {value_str_dict[k].rjust(val_width)} | {technical_dict[k].rjust(tech_width)} |"
        for k in last_info_dict
    )

    last_info_str = f"```\n{table_header}\n{table_divider}\n{table_rows}\n```"

    proba_text = f"""
    *Model Output*
    ------------------------
    Prediction: {pred}

    Decrease VIX Prob: {prob[0]:.1%}
    Neutral VIX Prob: {prob[1]:.1%}
    Increase VIX Prob: {prob[2]:.1%}

    Note:
        Downside Move → < {round(100*d['q_down'].iloc[-1],1)}%
        Neutral Move  → {round(100*d['q_down'].iloc[-1],1)}% to {round(100*d['q_up'].iloc[-1],1)}%
        Upside Move   → > {round(100*d['q_up'].iloc[-1],1)}%
    """

    # webhook untouched
    webhook = os.getenv("webhook")

    requests.post(webhook, data={"content": proba_text})
