import pandas as pd
import logging

logger = logging.getLogger(__name__)

def add_lag_features(df: pd.DataFrame, lags=[1, 7, 30]) -> pd.DataFrame:
    """
    Cria features de defasagem (lags) de vendas.
    """
    logger.info(f"Criando lag features: {lags}")
    df = df.copy()
    
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby('store_item')['sales'].shift(lag)
    
    logger.info(f"Criadas {len(lags)} lag features")
    return df

def add_rolling_features(df: pd.DataFrame, windows=[7, 14, 30]) -> pd.DataFrame:
    """
    Cria features de janelas móveis (rolling stats).
    """
    logger.info(f"Criando rolling features: {windows}")
    df = df.copy()
    
    for w in windows:
        df[f'sales_roll_mean_{w}'] = (
            df.groupby('store_item')['sales']
              .shift(1)
              .rolling(w)
              .mean()
        )
    
    logger.info(f"Criadas {len(windows)} rolling features")
    return df
