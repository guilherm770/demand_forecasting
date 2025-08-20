import pandas as pd
import logging

logger = logging.getLogger(__name__)

def add_temporal_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona componentes temporais adicionais:
    - Trimestre
    - Flag de feriado
    """
    logger.info("Criando componentes temporais adicionais...")
    df = df.copy()
    
    df['quarter'] = df['date'].dt.quarter
    df['is_holiday'] = df['date'].isin(get_holidays()).astype(int)
    
    logger.info("Features temporais adicionais criadas: ['quarter', 'is_holiday']")
    return df

def get_holidays():
    """
    Retorna lista simples de feriados fixos.
    (Pode ser expandido com calendários oficiais)
    """
    return pd.to_datetime([
        "2025-01-01",  # Ano Novo
        "2025-12-25",  # Natal
    ])
