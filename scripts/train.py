from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from datetime import datetime
import psutil
import gc
import warnings
import tempfile
import joblib

import numpy as np
import pandas as pd
# from joblib import dump  # não precisamos mais gravar .pkl local
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# ---- garantir imports absolutos "scripts.*" mesmo rodando 'python scripts/train.py'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------------------------------

from scripts.utils import load_params, load_processed, split_train_test_by_days
from scripts.metrics import summarize

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Suprimir warnings de sklearn que estão aparecendo
warnings.filterwarnings('ignore', message='invalid value encountered in divide')

def check_memory_usage():
    """Monitora uso de memória e limpa garbage collection se necessário"""
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        gc.collect()
        print(f"⚠️  Alto uso de memória ({memory_percent:.1f}%) - executando garbage collection")
    return memory_percent


def sample_data_if_needed(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Aplica sampling nos dados se configurado para modo desenvolvimento"""
    training_config = params.get("training", {})
    if not training_config.get("development_mode", False):
        return df

    original_size = len(df)

    # Aplicar sample_fraction
    sample_fraction = training_config.get("sample_fraction", 1.0)
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)
        print(f"📊 Dados reduzidos por fração: {original_size:,} → {len(df):,} ({sample_fraction:.2%})")

    # Aplicar max_samples
    max_samples = training_config.get("max_samples")
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"📊 Dados limitados: {len(df):,} amostras (máximo: {max_samples:,})")

    if len(df) != original_size:
        print(f"✅ Dataset final: {len(df):,} amostras ({len(df)/original_size:.2%} do original)")

    return df


def _ohe_compat():
    """OneHotEncoder compatível com sklearn < 1.2 (sem sparse_output)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(model_name: str, params: dict, numeric_cols, cat_cols):
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    training_config = params.get("training", {})
    # Usar StandardScaler mais simples para desenvolvimento
    if training_config.get("development_mode"):
        scaler = StandardScaler(with_mean=False)
    else:
        scaler = StandardScaler()

    pre = ColumnTransformer(
        transformers=[
            ("num", scaler, numeric_cols),
            ("cat", _ohe_compat(), cat_cols),
        ],
        remainder="drop"
    )

    if model_name == "rf":
        rf_params = (training_config.get("rf") or {}).copy()
        # Ajustar n_jobs para não sobrecarregar
        max_jobs = psutil.cpu_count() - 1 if psutil.cpu_count() > 1 else 1
        if rf_params.get("n_jobs") in (-1, None) or rf_params.get("n_jobs", 1) > max_jobs:
            rf_params["n_jobs"] = max_jobs
            print(f"🔧 Ajustado n_jobs para {max_jobs}")

        model = RandomForestRegressor(
            n_estimators=rf_params.get("n_estimators", 200),
            max_depth=rf_params.get("max_depth"),
            min_samples_split=rf_params.get("min_samples_split", 2),
            min_samples_leaf=rf_params.get("min_samples_leaf", 1),
            max_features=rf_params.get("max_features", "auto") if rf_params.get("max_features") else None,
            n_jobs=rf_params.get("n_jobs"),
            random_state=training_config.get("random_state", 42),
        )
    elif model_name == "xgb":
        xgb_params = (training_config.get("xgb") or {}).copy()
        max_jobs = psutil.cpu_count() - 1 if psutil.cpu_count() > 1 else 1
        if xgb_params.get("n_jobs", -1) in (-1, None) or xgb_params.get("n_jobs", 1) > max_jobs:
            xgb_params["n_jobs"] = max_jobs
            print(f"🔧 Ajustado n_jobs para {max_jobs}")

        model = XGBRegressor(
            n_estimators=xgb_params.get("n_estimators", 400),
            learning_rate=xgb_params.get("learning_rate", 0.05),
            max_depth=xgb_params.get("max_depth", 8),
            subsample=xgb_params.get("subsample", 0.8),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
            reg_lambda=xgb_params.get("reg_lambda", 1.0),
            reg_alpha=xgb_params.get("reg_alpha", 0.0),
            n_jobs=xgb_params.get("n_jobs"),
            random_state=training_config.get("random_state", 42),
            tree_method="hist",
        )
    else:
        raise ValueError(f"Modelo não suportado: {model_name}")

    return Pipeline([("pre", pre), ("model", model)])


def setup_mlflow(experiment_name: str, tracking_uri: str | None) -> bool:
    """
    Configura MLflow para servidor remoto se disponível, com fallback para local.
    - tracking_uri: usa o que veio por argumento, depois env MLFLOW_TRACKING_URI, depois http://localhost:5000
    """
    try:
        # 1) escolher tracking uri
        tracking_uri = (
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI")
            or "http://localhost:5000"
        )
        mlflow.set_tracking_uri(tracking_uri)

        # 2) testar conectividade
        mlflow.search_experiments(max_results=1)

        # 3) setar experimento
        mlflow.set_experiment(experiment_name)

        print(f"✅ MLflow: tracking_uri={mlflow.get_tracking_uri()} | experiment={experiment_name}")

        # 4) dica sobre artifacts (apenas mensagem de ajuda)
        if not os.getenv("MLFLOW_S3_ENDPOINT_URL"):
            print("ℹ️  Se o servidor MLflow NÃO estiver com --serve-artifacts, "
                  "defina MLFLOW_S3_ENDPOINT_URL/AWS_* no cliente para subir artefatos ao MinIO.")

        return True

    except Exception as e:
        # fallback local
        local_uri = "file:./mlruns"
        mlflow.set_tracking_uri(local_uri)
        try:
            mlflow.search_experiments(max_results=1)
            mlflow.set_experiment(experiment_name)
            print(f"⚠️  Não conectou no servidor ({e}). Usando MLflow local: {local_uri}")
            return True
        except Exception as e2:
            print(f"⚠️  MLflow desabilitado: {e2}")
            return False


def main(
    config_path: str = "params.yaml",
    model_override: str | None = None,
    experiment: str | None = None,
    mlflow_uri: str | None = None,
):
    print(f"🚀 Iniciando treinamento com configuração: {config_path}")

    # Monitorar recursos iniciais
    initial_memory = psutil.virtual_memory().percent
    available_gb = psutil.virtual_memory().available / (1024**3)
    print(f"💾 Memória disponível: {available_gb:.1f}GB ({initial_memory:.1f}% em uso)")

    # Carregar parâmetros
    params = load_params(config_path)

    # Configurar MLflow (prioriza argumento/ENV para servidor remoto)
    experiment_name = experiment or params.get("mlflow", {}).get("experiment", "default")
    mlflow_enabled = setup_mlflow(experiment_name, tracking_uri=mlflow_uri)

    # Carregar dados
    print("📁 Carregando dados processados...")
    df = load_processed(train=True)
    print(f"📊 Dados originais: {len(df):,} linhas, {len(df.columns)} colunas")

    # Aplicar sampling se necessário
    df = sample_data_if_needed(df, params)
    check_memory_usage()

    # Configuração de treinamento
    tconf = params["training"]
    date_col = tconf["date_col"]
    target   = tconf["target"]
    id_cols  = tconf["id_cols"]
    drop_cols = set([target, date_col] + tconf.get("features_to_drop", []))

    # Preparar features
    all_cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = [c for c in all_cols if str(df[c].dtype) in ("object", "category") or c in id_cols]
    num_cols = [c for c in all_cols if c not in cat_cols]

    print(f"🔧 Features: {len(all_cols)} total ({len(num_cols)} numéricas, {len(cat_cols)} categóricas)")

    # Split temporal
    print(f"✂️  Split temporal: holdout de {tconf['test_size_days']} dias")
    train_df, test_df = split_train_test_by_days(df, date_col, tconf["test_size_days"])
    print(f"📊 Train: {len(train_df):,} | Holdout: {len(test_df):,}")

    X_train = train_df[all_cols]
    y_train = train_df[target]
    X_test  = test_df[all_cols] if len(test_df) > 0 else pd.DataFrame()
    y_test  = test_df[target] if target in test_df.columns and len(test_df) > 0 else None

    # Liberar memória
    del df, train_df, test_df
    gc.collect()
    check_memory_usage()

    # Construir pipeline
    model_name = model_override or tconf["model"]
    print(f"🤖 Modelo: {model_name}")
    pipe = build_pipeline(model_name, params.copy(), num_cols, cat_cols)

    # Cross-validation temporal
    cv_config = params["cv"]
    n_splits = cv_config["n_splits"]
    print(f"🔄 Cross-validation temporal: {n_splits} folds")

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=cv_config.get("gap", 0))

    cv_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train), 1):
        print(f"  📁 Fold {fold}/{n_splits}")

        Xt, yt = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        Xv, yv = X_train.iloc[va_idx], y_train.iloc[va_idx]

        pipe_fold = build_pipeline(model_name, params, num_cols, cat_cols)
        pipe_fold.fit(Xt, yt)
        preds = pipe_fold.predict(Xv)
        fold_metrics = summarize(yv.values, preds)
        cv_metrics.append(fold_metrics)

        print(f"    MAE: {fold_metrics['mae']:.4f}, RMSE: {fold_metrics['rmse']:.4f}")

        # Limpeza de memória
        del pipe_fold, Xt, yt, Xv, yv, preds
        gc.collect()

    # Resumo do CV
    cv_summary = {k: float(np.mean([m[k] for m in cv_metrics])) for k in cv_metrics[0].keys()}
    print(f"✅ CV médio - MAE: {cv_summary['mae']:.4f}, RMSE: {cv_summary['rmse']:.4f}")

    # Treino final
    print("🏋️  Treinamento final no dataset completo...")
    pipe.fit(X_train, y_train)

    # Avaliação no holdout
    holdout_metrics = {}
    if y_test is not None and len(X_test) > 0:
        print("🎯 Avaliação no holdout...")
        preds = pipe.predict(X_test)
        holdout_metrics = summarize(y_test.values, preds)
        print(f"🎯 Holdout - MAE: {holdout_metrics['mae']:.4f}, RMSE: {holdout_metrics['rmse']:.4f}")

    # ---- MLflow logging (modelo e métricas)
    run_uri_print = None
    if mlflow_enabled:
        try:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            config_name = Path(config_path).stem
            run_name = f"{model_name}-{config_name}-{ts}"
            print(f"📊 Logando no MLflow: {run_name}")

            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                run_uri_print = f"runs:/{run_id}/model"

                # Parâmetros básicos
                log_params = {
                    "model": model_name,
                    "config_file": config_name,
                    "features_total": len(all_cols),
                    "num_cols": len(num_cols),
                    "cat_cols": len(cat_cols),
                    "test_size_days": tconf["test_size_days"],
                    "cv_folds": n_splits,
                    "train_samples": len(X_train),
                    "development_mode": tconf.get("development_mode", False),
                    "available_memory_gb": available_gb,
                    "cpu_count": psutil.cpu_count(),
                    "sample_fraction": tconf.get("sample_fraction", 1.0),
                }
                if model_name == "rf" and "rf" in tconf:
                    log_params.update({f"rf_{k}": v for k, v in tconf["rf"].items()})
                elif model_name == "xgb" and "xgb" in tconf:
                    log_params.update({f"xgb_{k}": v for k, v in tconf["xgb"].items()})

                mlflow.log_params(log_params)

                # Métricas
                mlflow.log_metrics({f"cv_{k}": v for k, v in cv_summary.items()})
                if holdout_metrics:
                    mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout_metrics.items()})

                # Modelo (sem gravar .pkl local)
                # usa um pequeno exemplo/assinatura p/ melhor rastreabilidade
                try:
                    sample_X = X_train.iloc[:100].copy()
                except Exception:
                    sample_X = X_train.copy()

                signature = None
                try:
                    preds_sample = pipe.predict(sample_X)
                    signature = infer_signature(sample_X, preds_sample)
                except Exception:
                    pass
                
                print("🔎 artifact_uri =", mlflow.get_artifact_uri())
        
                try:
                    # Log model with a try-catch for compatibility
                    mlflow.sklearn.log_model(
                        sk_model=pipe,
                        name="model",
                        signature=signature,
                        input_example=sample_X.iloc[:5] if len(sample_X) > 5 else sample_X,
                    )
                    print("✅ Model logged successfully")
                except Exception as e:
                    print(f"⚠️  Error logging model: {e}")
                    # Fallback: save model as artifact
                    try:
                        import joblib
                        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
                            joblib.dump(pipe, tmp.name)
                            mlflow.log_artifact(tmp.name, artifact_path="model")
                            os.unlink(tmp.name)
                        print("✅ Model saved as artifact (fallback)")
                    except Exception as fallback_error:
                        print(f"❌ Failed to save model as artifact: {fallback_error}")

                # Também subir o params.yaml como artefato leve
                try:
                    mlflow.log_artifact(config_path)
                except Exception as e:
                    print(f"⚠️  Não foi possível logar params.yaml: {e}")

        except Exception as e:
            print(f"⚠️  Erro no MLflow (modelo/artefatos): {e}")
            print("📝 Continuando sem salvar o modelo localmente, apenas métricas locais.")

    # Salvar métricas locais (útil para DVC)
    Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
    with open("artifacts/metrics/cv_metrics.json", "w") as f:
        json.dump(cv_summary, f, indent=2)
    if holdout_metrics:
        with open("artifacts/metrics/holdout_metrics.json", "w") as f:
            json.dump(holdout_metrics, f, indent=2)
    
    models_info_dir = Path("artifacts/model_info")
    models_info_dir.mkdir(parents=True, exist_ok=True)
    info_path = models_info_dir / "model_uri.txt"
    with open(info_path, "w") as f:
        if mlflow_enabled and run_uri_print:
            f.write(f"{run_uri_print}\n")
            f.write(f"tracking_uri={mlflow.get_artifact_uri()}\n")
        else:
            f.write("MODEL_NOT_LOGGED\n")
    print(f"📝 Info do modelo escrita em: {info_path}")

    final_memory = psutil.virtual_memory().percent
    print(f"✅ Treinamento concluído!")
    print(f"💾 Uso de memória: {initial_memory:.1f}% → {final_memory:.1f}%")
    if run_uri_print:
        print(f"🏆 Modelo logado no MLflow em: {run_uri_print}")
    else:
        print("🏆 Modelo não foi salvo localmente e não há URI de run disponível.")

    # Resumo final
    print(f"\n📊 RESUMO FINAL:")
    print(f"   🎯 CV - MAE: {cv_summary['mae']:.4f}, RMSE: {cv_summary['rmse']:.4f}")
    if holdout_metrics:
        print(f"   🎯 Holdout - MAE: {holdout_metrics['mae']:.4f}, RMSE: {holdout_metrics['rmse']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Treinamento local/servidor com MLflow")
    ap.add_argument("--config", help="Arquivo de configuração", default="params.yaml")
    ap.add_argument("--model", help="rf|xgb (override do config)", default=None)
    ap.add_argument("--experiment", help="Nome do experimento MLflow", default=None)
    ap.add_argument("--mlflow-uri", help="Tracking URI (ex: http://localhost:5000)", default=None)
    args = ap.parse_args()

    main(
        config_path=args.config,
        model_override=args.model,
        experiment=args.experiment,
        mlflow_uri=args.mlflow_uri,
    )
