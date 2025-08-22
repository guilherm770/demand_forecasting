# scripts/evaluate.py
from __future__ import annotations
import argparse, os, json
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd

from scripts.utils import load_processed, load_params
from scripts.metrics import summarize

MODEL_INFO_FILE = Path("artifacts/model_info/model_uri.txt")

def resolve_model_uri(cli_uri: str | None, params: dict) -> str:
    if cli_uri:
        return cli_uri
    if MODEL_INFO_FILE.exists():
        for line in MODEL_INFO_FILE.read_text().splitlines():
            line = line.strip()
            if line:
                return line
    # fallback: último run do experimento do params.yaml
    exp_name = params.get("mlflow", {}).get("experiment", "default")
    exp = mlflow.get_experiment_by_name(exp_name)
    if not exp:
        raise SystemExit(f"Experimento '{exp_name}' não encontrado e nenhum --model-uri/model_uri.txt informado.")
    runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        raise SystemExit(f"Nenhum run encontrado no experimento '{exp_name}'.")
    run_id = runs.iloc[0]["run_id"]
    return f"runs:/{run_id}/model"

def main(model_uri_cli: str | None):
    # Tracking URI (usa env ou localhost)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    params = load_params("params.yaml")
    tconf = params["training"]

    model_uri = resolve_model_uri(model_uri_cli, params)
    print(f"🔗 Carregando modelo do MLflow: {model_uri}")

    # carrega o pipeline sklearn logado no MLflow
    pipe = mlflow.sklearn.load_model(model_uri)

    # dados de teste
    df_test = load_processed(train=False)

    # colunas de entrada
    feat_cols = list(getattr(pipe, "feature_names_in_", []))
    if not feat_cols:
        drop_cols = set([tconf["target"], tconf["date_col"]] + tconf.get("features_to_drop", []))
        feat_cols = [c for c in df_test.columns if c not in drop_cols]

    X = df_test[feat_cols]
    y = df_test[tconf["target"]] if tconf["target"] in df_test.columns else None

    preds = pipe.predict(X)

    Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
    out = {"note": "Sem y_true no test"} if y is None else summarize(y.values, preds)

    with open("artifacts/metrics/test_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print("✅ Avaliação salva em artifacts/metrics/test_metrics.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Avaliação usando modelo do MLflow")
    ap.add_argument("--model-uri", help="URI do modelo (runs:/... ou models:/NAME/Stage)", default=None)
    args = ap.parse_args()
    main(args.model_uri)
