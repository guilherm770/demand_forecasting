import kaggle
import os
from pathlib import Path

def download_dataset():
    """
    Realiza download automático do dataset do Kaggle.
    
    O script verifica se os dados já existem para evitar downloads
    desnecessários e mantém a integridade dos dados raw.
    """
    raw_path = Path('data/raw')
    raw_path.mkdir(parents=True, exist_ok=True)
    
    # Verifica se os arquivos já existem
    if not (raw_path/'train.csv').exists():
        print("Baixando dataset do Kaggle...")
        kaggle.api.competition_download_files(
            'demand-forecasting-kernels-only', 
            path='data/raw',
            unzip=True
        )
        print("Download concluído!")
    else:
        print("Dataset já existe. Pulando download.")

if __name__ == "__main__":
    download_dataset()