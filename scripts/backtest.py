from __future__ import annotations
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load, dump

from scripts.utils import load_params, load_processed
from scripts.train import build_pipeline
from scripts.metrics import summarize

def main():
    params = load_params()
    bt = params["backtest"]
    tr = params["training"]

    df = load_processed(train=True).sort_values(tr["date_col"]).reset_index(drop=True)
    date_col, target = tr["date_col"], tr["target"]
    id_cols = tr["id_cols"]
    drop_cols = set([target, date_col] + tr.get("features_to_drop", []))

    all_cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = [c for c in all_cols if str(df[c].dtype) in ("object", "category") or c in id_cols]
    num_cols = [c for c in all_cols if c not in cat_cols]

    horizon = int(bt["horizon_days"])
    step    = int(bt["step_days"])
    start_date = pd.to_datetime(bt["start_date"]) if bt.get("start_date") else df[date_col].min() + timedelta(days=180)

    # janelas deslizantes
    cut = start_date
    records = []
    while True:
        train_df = df[df[date_col] <= cut].copy()
        test_df  = df[(df[date_col] > cut) & (df[date_col] <= cut + timedelta(days=horizon))].copy()
        if test_df.empty:
            break

        pipe = build_pipeline(tr["model"], tr, num_cols, cat_cols)
        pipe.fit(train_df[all_cols], train_df[target])

        preds = pipe.predict(test_df[all_cols])
        mets  = summarize(test_df[target].values, preds)
        rec = {"cutoff": str(cut.date()), "n_train": len(train_df), "n_test": len(test_df), **mets}
        records.append(rec)

        cut = cut + timedelta(days=step)

    Path("artifacts/backtests").mkdir(parents=True, exist_ok=True)
    df_bt = pd.DataFrame(records)
    df_bt.to_csv("artifacts/backtests/metrics_by_split.csv", index=False)
    summary = {k: float(np.mean(df_bt[k])) for k in ["mae","rmse","mape","smape"]} if not df_bt.empty else {}
    with open("artifacts/metrics/backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Backtest por janelas salvo em artifacts/backtests/metrics_by_split.csv")
    print("📄 Resumo em artifacts/metrics/backtest_summary.json")

if __name__ == "__main__":
    main()
