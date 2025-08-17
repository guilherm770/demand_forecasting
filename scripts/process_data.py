import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from datetime import datetime
import sys
import os

# Adicionar diretório de features ao path
sys.path.append(str(Path(__file__).parent.parent))

from features.base_features import BaseFeatureEngine
from scripts.data_validation import main as validate_data

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processador principal de dados com pipeline reproduzível.
    
    Integra validação, transformação e armazenamento em um
    fluxo controlado e monitorado.
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.feature_engine = BaseFeatureEngine(self.config.get('features', {}))
        self.processing_metadata = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'errors': [],
            'pipeline_status': 'IN_PROGRESS'  # Initialize status
        }
        
    def _load_config(self, config_path: str) -> dict:
        """Carrega configuração do pipeline."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Retorna configuração padrão do pipeline."""
        return {
            'input': {
                'train_path': 'data/raw/train.csv',
                'test_path': 'data/raw/test.csv'
            },
            'output': {
                'processed_dir': 'data/processed',
                'train_output': 'train_processed.parquet',
                'test_output': 'test_processed.parquet', 
                'metadata_output': 'processing_metadata.yaml'
            },
            'features': {
                'create_lags': True,
                'lag_periods': [1, 7, 30],
                'create_rolling_stats': True,
                'rolling_windows': [7, 14, 30]
            },
            'validation': {
                'run_validation': True,
                'fail_on_validation_error': True
            }
        }
    
    def validate_input_data(self) -> bool:
        """
        Executa validação de dados de entrada.
        
        Interage com o sistema de validação Great Expectations
        para garantir qualidade dos dados antes do processamento.
        """
        logger.info("🔍 Iniciando validação de dados...")
        
        try:
            if self.config['validation']['run_validation']:
                validate_data()
                self.processing_metadata['steps_completed'].append('validation')
                logger.info("✅ Validação concluída com sucesso")
                return True
            else:
                logger.info("⏭️ Validação pulada conforme configuração")
                return True
                
        except Exception as e:
            error_msg = f"Erro na validação: {str(e)}"
            logger.error(error_msg)
            self.processing_metadata['errors'].append(error_msg)
            
            if self.config['validation']['fail_on_validation_error']:
                raise
            return False
    
    def load_data(self) -> tuple:
        """
        Carrega dados de treino e teste.
        
        Interage com sistema de arquivos para carregar dados
        com otimizações de performance e parsing inteligente.
        """
        logger.info("📥 Carregando dados...")
        
        try:
            # Carregar dados de treino
            train_path = Path(self.config['input']['train_path'])
            if not train_path.exists():
                raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_path}")
                
            df_train = pd.read_csv(
                train_path,
                parse_dates=['date'],
                dtype={'store': 'int16', 'item': 'int16', 'sales': 'int32'}
            )
            
            # Carregar dados de teste se existirem
            test_path = Path(self.config['input']['test_path'])
            df_test = None
            if test_path.exists():
                df_test = pd.read_csv(
                    test_path,
                    parse_dates=['date'],
                    dtype={'store': 'int16', 'item': 'int16'}
                )
            
            logger.info(f"✅ Dados carregados - Treino: {df_train.shape}, Teste: {df_test.shape if df_test is not None else 'N/A'}")
            self.processing_metadata['steps_completed'].append('data_loading')
            
            return df_train, df_test
            
        except Exception as e:
            error_msg = f"Erro no carregamento: {str(e)}"
            logger.error(error_msg)
            self.processing_metadata['errors'].append(error_msg)
            raise
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features avançadas incluindo lags e estatísticas móveis.
        
        Interage com dados históricos para criar features que
        capturam padrões temporais complexos.
        """
        logger.info("🔧 Criando features avançadas...")
        
        df = df.copy().sort_values(['store', 'item', 'date'])
        
        # Features básicas
        df = self.feature_engine.fit_transform(df)
        
        # Features de lag se configuradas
        if self.config['features']['create_lags']:
            lag_periods = self.config['features']['lag_periods']
            for lag in lag_periods:
                df[f'sales_lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)
                
        # Estatísticas móveis se configuradas
        if self.config['features']['create_rolling_stats']:
            windows = self.config['features']['rolling_windows']
            for window in windows:
                # Médias móveis
                df[f'sales_rolling_mean_{window}'] = (
                    df.groupby(['store', 'item'])['sales']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(drop=True)
                )
                
                # Desvio padrão móvel
                df[f'sales_rolling_std_{window}'] = (
                    df.groupby(['store', 'item'])['sales']
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(drop=True)
                )
        
        # Features de tendência
        df['sales_trend'] = (
            df.groupby(['store', 'item'])['sales']
            .pct_change(periods=7)
            .fillna(0)
        )
        
        logger.info("✅ Features avançadas criadas")
        self.processing_metadata['steps_completed'].append('feature_engineering')
        
        return df
    
    def save_processed_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame = None):
        """
        Salva dados processados em formato otimizado.
        
        Interage com sistema de arquivos para persistir dados
        transformados com compressão e particionamento eficientes.
        """
        logger.info("💾 Salvando dados processados...")
        
        try:
            output_dir = Path(self.config['output']['processed_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Salvar dados de treino
            train_output_path = output_dir / self.config['output']['train_output']
            df_train.to_parquet(
                train_output_path,
                index=False,
                compression='snappy',
                engine='pyarrow'
            )
            
            logger.info(f"✅ Dados de treino salvos: {train_output_path}")
            
            # Salvar dados de teste se existirem
            if df_test is not None:
                test_output_path = output_dir / self.config['output']['test_output']
                df_test.to_parquet(
                    test_output_path,
                    index=False,
                    compression='snappy',
                    engine='pyarrow'
                )
                logger.info(f"✅ Dados de teste salvos: {test_output_path}")
            
            # Salvar metadata do processamento
            self.processing_metadata['end_time'] = datetime.now().isoformat()
            self.processing_metadata['total_features'] = len(df_train.columns)
            self.processing_metadata['train_shape'] = df_train.shape
            self.processing_metadata['test_shape'] = df_test.shape if df_test is not None else None
            
            metadata_path = output_dir / self.config['output']['metadata_output']
            with open(metadata_path, 'w') as f:
                yaml.dump(self.processing_metadata, f, default_flow_style=False)
            
            logger.info(f"📋 Metadata salva: {metadata_path}")
            self.processing_metadata['steps_completed'].append('data_saving')
            
        except Exception as e:
            error_msg = f"Erro ao salvar dados: {str(e)}"
            logger.error(error_msg)
            self.processing_metadata['errors'].append(error_msg)
            raise
    
    def run_full_pipeline(self):
        """
        Executa pipeline completo de processamento.
        
        Orquestra todas as etapas em sequência com tratamento
        de erros e logging detalhado.
        """
        logger.info("🚀 Iniciando pipeline completo de processamento...")
        
        try:
            # 1. Validação
            self.validate_input_data()
            
            # 2. Carregamento
            df_train, df_test = self.load_data()
            
            # 3. Feature Engineering - Treino
            df_train_processed = self.create_advanced_features(df_train)
            
            # 4. Feature Engineering - Teste (se existir)
            df_test_processed = None
            if df_test is not None:
                # Aplicar mesmas transformações nos dados de teste
                # mas sem usar informações futuras
                logger.info("🔧 Processando dados de teste...")
                df_test_processed = self.feature_engine.fit_transform(df_test)
            
            # 5. Salvar dados processados
            self.save_processed_data(df_train_processed, df_test_processed)
            
            # 6. Set success status before generating report
            self.processing_metadata['pipeline_status'] = 'SUCCESS'
            
            # 7. Relatório final
            self._generate_processing_report()
            
            logger.info("🎉 Pipeline concluído com sucesso!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline falhou: {str(e)}")
            self.processing_metadata['pipeline_status'] = 'FAILED'
            self.processing_metadata['failure_reason'] = str(e)
            # Generate report even on failure
            self._generate_processing_report()
            raise
    
    def _generate_processing_report(self):
        """Gera relatório detalhado do processamento."""
        logger.info("📊 Gerando relatório de processamento...")
        
        # Ensure end_time is set if not already
        if 'end_time' not in self.processing_metadata:
            self.processing_metadata['end_time'] = datetime.now().isoformat()
        
        report = {
            'Pipeline Status': self.processing_metadata.get('pipeline_status', 'UNKNOWN'),
            'Processing Time': f"{self.processing_metadata['start_time']} - {self.processing_metadata['end_time']}",
            'Steps Completed': self.processing_metadata['steps_completed'],
            'Total Features': self.processing_metadata.get('total_features', 'N/A'),
            'Train Data Shape': self.processing_metadata.get('train_shape', 'N/A'),
            'Test Data Shape': self.processing_metadata.get('test_shape', 'N/A'),
            'Errors': self.processing_metadata['errors']
        }
        
        logger.info("=" * 50)
        logger.info("📋 RELATÓRIO DE PROCESSAMENTO")
        logger.info("=" * 50)
        
        for key, value in report.items():
            logger.info(f"{key}: {value}")
        
        logger.info("=" * 50)

def main():
    """Função principal que executa o pipeline de processamento."""
    try:
        processor = DataProcessor(config_path="params.yaml")
        processor.run_full_pipeline()
    except Exception as e:
        logger.error(f"Erro crítico no processamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()