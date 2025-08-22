from __future__ import annotations
import argparse, json, os
from pathlib import Path
from datetime import datetime
import psutil
import gc

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from scripts.utils import load_params, load_processed, split_train_test_by_days
from scripts.metrics import summarize

import mlflow

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

def build_pipeline(model_name: str, params: dict, numeric_cols, cat_cols):
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    # Preprocessamento mais eficiente para recursos limitados
    training_config = params.get("training", {})
    
    # Para desenvolvimento, usar StandardScaler mais simples
    if training_config.get("development_mode", False):
        scaler = StandardScaler(with_mean=False)  # Mais eficiente
    else:
        scaler = StandardScaler()

    pre = ColumnTransformer(
        transformers=[
            ("num", scaler, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )

    if model_name == "rf":
        rf_params = params["training"]["rf"].copy()
        # Verificar e ajustar n_jobs para não sobrecarregar sistema
        max_jobs = psutil.cpu_count() - 1 if psutil.cpu_count() > 1 else 1
        if rf_params.get("n_jobs", -1) == -1 or rf_params.get("n_jobs", 1) > max_jobs:
            rf_params["n_jobs"] = max_jobs
            print(f"🔧 Ajustado n_jobs para {max_jobs} (deixando 1 CPU livre)")
        
        model = RandomForestRegressor(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_split=rf_params["min_samples_split"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            max_features=rf_params.get("max_features", "auto"),
            n_jobs=rf_params["n_jobs"],
            random_state=params["training"]["random_state"],
        )
    elif model_name == "xgb":
        xgb_params = params["training"]["xgb"].copy()
        max_jobs = psutil.cpu_count() - 1 if psutil.cpu_count() > 1 else 1
        if xgb_params.get("n_jobs", -1) == -1 or xgb_params.get("n_jobs", 1) > max_jobs:
            xgb_params["n_jobs"] = max_jobs
            print(f"🔧 Ajustado n_jobs para {max_jobs}")
        
        model = XGBRegressor(
            n_estimators=xgb_params["n_estimators"],
            learning_rate=xgb_params["learning_rate"],
            max_depth=xgb_params["max_depth"],
            subsample=xgb_params["subsample"],
            colsample_bytree=xgb_params["colsample_bytree"],
            reg_lambda=xgb_params["reg_lambda"],
            reg_alpha=xgb_params.get("reg_alpha", 0.0),
            n_jobs=xgb_params["n_jobs"],
            random_state=params["training"]["random_state"],
            tree_method="hist",  # Mais eficiente
        )
    else:
        raise ValueError(f"Modelo não suportado: {model_name}")

    return Pipeline([("pre", pre), ("model", model)])

def main(config_path: str = "params.yaml", model_override: str | None = None, experiment: str | None = None):
    print(f"🚀 Iniciando treinamento com configuração: {config_path}")
    
    # Monitorar recursos iniciais
    initial_memory = psutil.virtual_memory().percent
    available_gb = psutil.virtual_memory().available / (1024**3)
    print(f"💾 Memória disponível: {available_gb:.1f}GB ({initial_memory:.1f}% em uso)")
    
    # Carregar parâmetros
    params = load_params(config_path)

    # MLflow
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    experiment_name = experiment or params["mlflow"]["experiment"]
    mlflow.set_experiment(experiment_name)
    print(f"📊 MLflow experiment: {experiment_name}")

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
    drop_cols= set([target, date_col] + tconf.get("features_to_drop", []))

    # Preparar features
    all_cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = [c for c in all_cols if str(df[c].dtype) in ("object", "category") or c in id_cols]
    num_cols = [c for c in all_cols if c not in cat_cols]
    
    print(f"🔧 Features: {len(all_cols)} total ({len(num_cols)} numéricas, {len(cat_cols)} categóricas)")

    # Split temporal: train/holdout
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
    pipe = build_pipeline(model_name, params, num_cols, cat_cols)

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
        
        # Limpeza de memória após cada fold
        del pipe_fold, Xt, yt, Xv, yv, preds
        gc.collect()
        check_memory_usage()

    # Resumo do CV
    cv_summary = {k: float(np.mean([m[k] for m in cv_metrics])) for k in cv_metrics[0].keys()}
    print(f"✅ CV médio - MAE: {cv_summary['mae']:.4f}, RMSE: {cv_summary['rmse']:.4f}")

    # Treino final no conjunto completo
    print("🏋️  Treinamento final no dataset completo...")
    pipe.fit(X_train, y_train)

    # Avaliação no holdout (se disponível)
    holdout_metrics = {}
    if y_test is not None and len(X_test) > 0:
        print("🎯 Avaliação no holdout...")
        preds = pipe.predict(X_test)
        holdout_metrics = summarize(y_test.values, preds)
        print(f"🎯 Holdout - MAE: {holdout_metrics['mae']:.4f}, RMSE: {holdout_metrics['rmse']:.4f}")

    # Persistência
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    
    # Nome do modelo inclui configuração usada
    config_name = Path(config_path).stem
    model_path = Path(f"artifacts/models/model-{model_name}-{config_name}-{ts}.pkl")
    
    print(f"💾 Salvando modelo: {model_path}")
    dump(pipe, model_path)

    # Salvar métricas
    Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
    
    with open("artifacts/metrics/cv_metrics.json", "w") as f:
        json.dump(cv_summary, f, indent=2)
    
    if holdout_metrics:
        with open("artifacts/metrics/holdout_metrics.json", "w") as f:
            json.dump(holdout_metrics, f, indent=2)

    # MLflow logging
    run_name = f"{model_name}-{config_name}-{ts}"
    print(f"📊 Logando no MLflow: {run_name}")
    
    with mlflow.start_run(run_name=run_name):
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
        }
        
        # Parâmetros do modelo
        if model_name == "rf" and "rf" in tconf:
            log_params.update({f"rf_{k}": v for k, v in tconf["rf"].items()})
        elif model_name == "xgb" and "xgb" in tconf:
            log_params.update({f"xgb_{k}": v for k, v in tconf["xgb"].items()})

        # Parâmetros de recursos
        log_params.update({
            "available_memory_gb": available_gb,
            "cpu_count": psutil.cpu_count(),
            "sample_fraction": tconf.get("sample_fraction", 1.0),
        })

        mlflow.log_params(log_params)

        # Métricas
        mlflow.log_metrics({f"cv_{k}": v for k, v in cv_summary.items()})
        if holdout_metrics:
            mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout_metrics.items()})

        # Artefatos
        mlflow.log_artifact(model_path.as_posix())
        mlflow.log_artifact(config_path)

    final_memory = psutil.virtual_memory().percent
    print(f"✅ Treinamento concluído!")
    print(f"💾 Uso de memória: {initial_memory:.1f}% → {final_memory:.1f}%")
    print(f"🏆 Melhor modelo salvo: {model_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Treinamento otimizado para recursos limitados")
    ap.add_argument("--config", help="Arquivo de configuração", default="params.yaml")
    ap.add_argument("--model", help="rf|xgb (override do config)", default=None)
    ap.add_argument("--experiment", help="Nome do experimento MLflow", default=None)
    args = ap.parse_args()
    
    main(config_path=args.config, model_override=args.model, experiment=args.experiment)