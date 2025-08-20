from pathlib import Path
from typing import Dict, List
import yaml
import re

CATALOG_PATH = Path("features/feature_store.yaml")

def load_catalog() -> Dict:
    if not CATALOG_PATH.exists():
        return {}
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def list_declared_features(catalog: Dict) -> List[str]:
    feats = []
    # base.temporal / base.categorical / base.interaction
    base = catalog.get("base", {})
    for group in ("temporal", "categorical", "interaction"):
        for it in base.get(group, []) or []:
            feats.append(it.get("name", ""))
    # outras seções
    for section in ("temporal", "lags", "statistical"):
        for it in catalog.get(section, []) or []:
            feats.append(it.get("name", ""))
    # expande curingas comuns para fins de documentação (mantém curinga no metadata)
    return [f for f in feats if f]

def validate_catalog(catalog: Dict):
    """Validações leves para falhar cedo caso haja erros óbvios."""
    # checagem de lags
    for it in catalog.get("lags", []) or []:
        n = it.get("name", "")
        if not (n.startswith("sales_lag_") or n.startswith("sales_roll_mean_")):
            # permite std se você tiver cadastrado
            if not n.startswith("sales_roll_std_"):
                raise ValueError(f"Lag inválido no catálogo (esperado prefixo sales_lag_ ou sales_roll_mean_): {n}")
    # nomes vazios
    for section, items in catalog.items():
        if isinstance(items, list):
            for it in items:
                if not it.get("name"):
                    raise ValueError(f"Item sem 'name' em '{section}' no feature_store.yaml")
        elif isinstance(items, dict):
            for group, lst in items.items():
                for it in lst or []:
                    if not it.get("name"):
                        raise ValueError(f"Item sem 'name' em 'base.{group}' no feature_store.yaml")
