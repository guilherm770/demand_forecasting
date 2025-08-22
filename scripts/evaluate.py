# scripts/evaluate.py
from __future__ import annotations
import argparse, os, json
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import tempfile
import joblib

from scripts.utils import load_processed, load_params
from scripts.metrics import summarize

MODEL_INFO_FILE = Path("artifacts/model_info/model_uri.txt")

def setup_mlflow():
    """Configure MLflow with MinIO settings"""
    # Set MinIO environment variables
    os.environ.setdefault('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9100')
    os.environ.setdefault('AWS_ACCESS_KEY_ID', 'minioadmin')
    os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'minioadmin')
    os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')
    
    # Set tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")

def resolve_model_uri(cli_uri: str | None, params: dict) -> str:
    if cli_uri:
        return cli_uri
    if MODEL_INFO_FILE.exists():
        for line in MODEL_INFO_FILE.read_text().splitlines():
            line = line.strip()
            if line and line.startswith('runs:/'):
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

def load_model_from_uri(model_uri):
    """Load model from URI, handling both MLflow format and joblib format"""
    try:
        # First try to load as MLflow model
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"❌ Erro ao carregar modelo MLflow: {e}")
        print("📝 Tentando carregar como joblib...")
        
        # Download artifacts and try to find a joblib file
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download all artifacts from the model directory
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri, 
                dst_path=tmp_dir
            )
            
            # Look for joblib files
            joblib_files = list(Path(local_path).glob("*.joblib"))
            if not joblib_files:
                # Look for any file that might be a model
                all_files = list(Path(local_path).rglob("*"))
                model_files = [f for f in all_files if f.is_file() and not f.name.startswith('.')]
                
                if not model_files:
                    raise FileNotFoundError("Nenhum arquivo de modelo encontrado nos artefatos")
                
                # Try to load the first file as joblib
                try:
                    return joblib.load(model_files[0])
                except Exception as joblib_error:
                    raise Exception(f"Não foi possível carregar nenhum arquivo como modelo: {joblib_error}")
            
            # Load the first joblib file found
            return joblib.load(joblib_files[0])

def main(model_uri_cli: str | None):
    # Configure MLflow
    setup_mlflow()

    params = load_params("params.yaml")
    tconf = params["training"]

    model_uri = resolve_model_uri(model_uri_cli, params)
    print(f"🔗 Carregando modelo do MLflow: {model_uri}")

    try:
        # Try to load the model
        pipe = load_model_from_uri(model_uri)
        print("✅ Modelo carregado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        print("📝 Tentando carregar modelo localmente...")
        
        # Fallback: try to load from local path if available
        local_model_path = "artifacts/model"
        if Path(local_model_path).exists():
            try:
                # Look for joblib files in local directory
                joblib_files = list(Path(local_model_path).glob("*.joblib"))
                if joblib_files:
                    pipe = joblib.load(joblib_files[0])
                    print("✅ Modelo carregado localmente")
                else:
                    raise FileNotFoundError("Nenhum arquivo .joblib encontrado no diretório local")
            except Exception as local_error:
                print(f"❌ Falha ao carregar modelo local: {local_error}")
                raise SystemExit("Não foi possível carregar o modelo")
        else:
            raise SystemExit("Nenhum modelo local encontrado")

    # Rest of your evaluation code...
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