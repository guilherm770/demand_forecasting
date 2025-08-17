import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class BaseFeatureEngine:
    """
    Engine para criação de features fundamentais.
    
    Esta classe interage com o sistema de dados para extrair
    características básicas que servem de base para features
    mais complexas.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features temporais básicas.
        
        Interage com colunas de data para extrair componentes
        temporais que capturam sazonalidade e tendências.
        """
        logger.info("Criando features temporais básicas...")
        
        df = df.copy()
        
        # Components temporais básicos
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month  
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Features cíclicas (preservam relação circular do tempo)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)  
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Features de calendario
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        temporal_features = [
            'year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year',
            'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            'is_weekend', 'is_month_start', 'is_month_end'
        ]
        
        self.feature_names.extend(temporal_features)
        logger.info(f"Criadas {len(temporal_features)} features temporais")
        
        return df
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa features categóricas com encoding apropriado.
        
        Interage com o sistema para aplicar transformações que
        mantêm poder preditivo das variáveis categóricas.
        """
        logger.info("Processando features categóricas...")
        
        df = df.copy()
        
        # One-hot encoding para lojas (baixa cardinalidade)
        if df['store'].nunique() <= 20:
            store_dummies = pd.get_dummies(df['store'], prefix='store')
            df = pd.concat([df, store_dummies], axis=1)
            self.feature_names.extend(store_dummies.columns.tolist())
            
        # One-hot encoding para itens (se cardinalidade permitir)
        if df['item'].nunique() <= 50:
            item_dummies = pd.get_dummies(df['item'], prefix='item')
            df = pd.concat([df, item_dummies], axis=1) 
            self.feature_names.extend(item_dummies.columns.tolist())
            
        logger.info("Features categóricas processadas")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de interação entre variáveis.
        
        Captura relacionamentos não-lineares que podem
        ser importantes para o modelo preditivo.
        """
        logger.info("Criando features de interação...")
        
        df = df.copy()
        
        # Interação store-item (combinação única)
        df['store_item'] = df['store'].astype(str) + '_' + df['item'].astype(str)
        
        # Interações temporais
        df['store_month'] = df['store'].astype(str) + '_' + df['month'].astype(str)
        df['item_month'] = df['item'].astype(str) + '_' + df['month'].astype(str)
        df['store_dow'] = df['store'].astype(str) + '_' + df['day_of_week'].astype(str)
        
        interaction_features = ['store_item', 'store_month', 'item_month', 'store_dow']
        self.feature_names.extend(interaction_features)
        
        logger.info(f"Criadas {len(interaction_features)} features de interação")
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline completo de transformação de features.
        
        Orquestra todas as transformações em sequência,
        mantendo estado para aplicação futura em novos dados.
        """
        logger.info("Iniciando pipeline de feature engineering...")
        
        df = self.create_temporal_features(df)
        df = self.create_categorical_features(df) 
        df = self.create_interaction_features(df)
        
        logger.info(f"Pipeline concluído. Total de features: {len(self.feature_names)}")
        return df
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista de nomes das features criadas."""
        return self.feature_names.copy()