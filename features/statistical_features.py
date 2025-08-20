import pandas as pd
import logging

logger = logging.getLogger(__name__)

def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features estatísticas baseadas em agregações.
    """
    logger.info("Criando statistical features...")
    df = df.copy()
    
    df['sales_mean_store'] = df.groupby('store')['sales'].transform('mean')
    df['sales_std_item'] = df.groupby('item')['sales'].transform('std')
    
    logger.info("Features estatísticas criadas: ['sales_mean_store', 'sales_std_item']")
    return df
