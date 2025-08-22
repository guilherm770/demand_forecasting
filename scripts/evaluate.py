from __future__ import annotations
import json
from pathlib import Path
from joblib import load
from scripts.utils import load_processed, latest_model_path
from scripts.metrics import summarize

def main():
    model_path = latest_model_path()
    if not model_path:
        raise SystemExit("Nenhum modelo encontrado em artifacts/models/")
    pipe = load(model_path)

    df_test = load_processed(train=False)
    # inferir colunas a partir do pipeline:
    pre = pipe.named_steps["pre"]
    # recupera features do dataframe original (todas exceto alvo e data)
    # assumindo as mesmas colunas da fase de treino:
    # Para simplicidade, vamos usar as colunas do dataframe presente no pipeline
    # via atributo feature_names_in_ (desde sklearn 1.0)
    feat_cols = list(getattr(pipe, "feature_names_in_", []))
    if not feat_cols:
        # fallback: tentar todas colunas que não são 'sales' e 'date'
        feat_cols = [c for c in df_test.columns if c not in ("sales", "date")]

    X = df_test[feat_cols]
    y = df_test["sales"] if "sales" in df_test.columns else None

    preds = pipe.predict(X)

    Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
    out = {"note": "Sem y_true no test" }
    if y is not None:
        out = summarize(y.values, preds)

    with open("artifacts/metrics/test_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print("✅ Avaliação salva em artifacts/metrics/test_metrics.json")

if __name__ == "__main__":
    main()
