from __future__ import annotations
from pathlib import Path
import yaml
import pandas as pd
from datetime import timedelta
import psutil
import os
import mlflow

def load_params(path: str = "params.yaml") -> dict:
    """Carrega parâmetros com fallback para params.yaml se o arquivo não existir"""
    if not Path(path).exists():
        fallback = "params.yaml"
        if Path(fallback).exists():
            print(f"⚠️  {path} não encontrado, usando {fallback}")
            path = fallback
        else:
            raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {path}")
    
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_processed(train=True) -> pd.DataFrame:
    """Carrega dados processados com verificação de existência"""
    base = Path("data/processed")
    
    if train:
        file_path = base / "train_processed.parquet"
    else:
        file_path = base / "test_processed.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    # Monitorar carregamento
    print(f"📁 Carregando: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"📊 Carregado: {len(df):,} linhas, {len(df.columns)} colunas")
    
    return df

def split_train_test_by_days(df: pd.DataFrame, date_col: str, test_size_days: int):
    """Split temporal com informações detalhadas"""
    df = df.sort_values(date_col).reset_index(drop=True)
    
    max_date = df[date_col].max()
    min_date = df[date_col].min()
    split_date = max_date - timedelta(days=int(test_size_days))
    
    print(f"📅 Período total: {min_date} até {max_date}")
    print(f"✂️  Data de corte: {split_date}")
    
    train_df = df[df[date_col] <= split_date].copy()
    test_df = df[df[date_col] > split_date].copy()
    
    print(f"📊 Train: {len(train_df):,} linhas ({min_date} - {split_date})")
    print(f"📊 Test: {len(test_df):,} linhas ({split_date} - {max_date})")
    
    return train_df, test_df

def latest_model_path(model_dir: str = "artifacts/models") -> Path | None:
    """Encontra o modelo mais recente com informações detalhadas"""
    models_dir = Path(model_dir)
    if not models_dir.exists():
        print(f"⚠️  Diretório de modelos não existe: {models_dir}")
        return None
    
    candidates = sorted(models_dir.glob("model-*.pkl"))
    if not candidates:
        print(f"⚠️  Nenhum modelo encontrado em: {models_dir}")
        return None
    
    latest = candidates[-1]
    print(f"🎯 Modelo mais recente: {latest}")
    return latest

def check_system_resources():
    """Verifica recursos do sistema disponíveis"""
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    resources = {
        "memory_total_gb": memory.total / (1024**3),
        "memory_available_gb": memory.available / (1024**3),
        "memory_percent": memory.percent,
        "cpu_count": cpu_count,
        "cpu_count_logical": psutil.cpu_count(logical=True),
    }
    
    print("🔍 Recursos do Sistema:")
    print(f"  💾 RAM Total: {resources['memory_total_gb']:.1f}GB")
    print(f"  💾 RAM Disponível: {resources['memory_available_gb']:.1f}GB ({100-resources['memory_percent']:.1f}%)")
    print(f"  🖥️  CPUs: {resources['cpu_count']} físicos, {resources['cpu_count_logical']} lógicos")
    
    return resources

def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = None):
    """Configura experimento MLflow com verificação de conectividade"""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    try:
        # Verificar se o servidor MLflow está acessível
        mlflow.search_experiments()
        print(f"✅ MLflow conectado: {mlflow.get_tracking_uri()}")
    except Exception as e:
        print(f"⚠️  Problema com MLflow: {e}")
        print("📝 Continuando sem logging remoto...")
        return False
    
    try:
        mlflow.set_experiment(experiment_name)
        exp = mlflow.get_experiment_by_name(experiment_name)
        print(f"📊 Experimento: {experiment_name} (ID: {exp.experiment_id})")
        return True
    except Exception as e:
        print(f"⚠️  Erro ao configurar experimento: {e}")
        return False

def estimate_memory_usage(df: pd.DataFrame) -> dict:
    """Estima uso de memória do DataFrame"""
    memory_usage = df.memory_usage(deep=True).sum()
    memory_mb = memory_usage / (1024**2)
    memory_gb = memory_mb / 1024
    
    # Estimativa para diferentes operações
    estimates = {
        "current_mb": memory_mb,
        "current_gb": memory_gb,
        "train_test_split_gb": memory_gb * 2,  # 2 cópias
        "preprocessing_gb": memory_gb * 3,     # Pipeline cria cópias
        "cross_validation_gb": memory_gb * 4,  # Múltiplos folds
    }
    
    return estimates

def recommend_config_for_data(df: pd.DataFrame) -> str:
    """Recomenda configuração baseada no tamanho dos dados"""
    memory_est = estimate_memory_usage(df)
    available_gb = psutil.virtual_memory().available / (1024**3)
    
    print(f"📊 Estimativa de memória para processamento: {memory_est['cross_validation_gb']:.1f}GB")
    print(f"💾 Memória disponível: {available_gb:.1f}GB")
    
    if memory_est['cross_validation_gb'] > available_gb * 0.8:
        return "params.yaml"
    elif memory_est['cross_validation_gb'] > available_gb * 0.5:
        return "params.yaml"
    else:
        return "params_cloud.yaml"

def create_backup_config(source_config: str = "params.yaml", 
                        backup_config: str = "params_backup.yaml",
                        sample_fraction: float = 0.1):
    """Cria configuração backup com sampling para emergências"""
    if not Path(source_config).exists():
        print(f"⚠️  Arquivo fonte não existe: {source_config}")
        return None
    
    with open(source_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modificar para modo desenvolvimento
    if 'training' not in config:
        config['training'] = {}
    
    config['training']['development_mode'] = True
    config['training']['sample_fraction'] = sample_fraction
    config['training']['max_samples'] = 10000
    
    # Reduzir parâmetros de modelo
    if 'rf' in config['training']:
        config['training']['rf']['n_estimators'] = min(50, config['training']['rf'].get('n_estimators', 100))
        config['training']['rf']['n_jobs'] = 2
    
    if 'xgb' in config['training']:
        config['training']['xgb']['n_estimators'] = min(100, config['training']['xgb'].get('n_estimators', 200))
        config['training']['xgb']['n_jobs'] = 2
    
    # Reduzir CV
    if 'cv' in config:
        config['cv']['n_splits'] = min(3, config['cv'].get('n_splits', 5))
    
    with open(backup_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"💾 Configuração backup criada: {backup_config}")
    return backup_config

# Manter compatibilidade com código existente
def get_system_info():
    """Wrapper para check_system_resources para compatibilidade"""
    return check_system_resources()