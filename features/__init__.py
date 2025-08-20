"""
Pacote de engenharia de features.

Este módulo organiza diferentes tipos de features utilizadas no pipeline.
Cada arquivo implementa um conjunto específico de transformações.
"""

from .base_features import BaseFeatureEngine
from .temporal_features import add_temporal_components
from .lag_features import add_lag_features, add_rolling_features
from .statistical_features import add_statistical_features

__all__ = [
    "BaseFeatureEngine",
    "add_temporal_components",
    "add_lag_features",
    "add_rolling_features",
    "add_statistical_features",
]
