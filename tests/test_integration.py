import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from datetime import datetime
import sys

# Adicionar diretório do projeto ao PYTHONPATH
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from features.base_features import BaseFeatureEngine
from features.temporal_features import add_temporal_components
from features.lag_features import add_lag_features  # (não usado direto, mantido p/ compat.)
from features.statistical_features import add_statistical_features
from scripts.data_validation import main as validate_data
from scripts.registry_loader import load_catalog, list_declared_features, validate_catalog

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processador principal de dados com pipeline reproduzível.
    Integra validação, transformação e armazenamento.
    """

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.feature_engine = BaseFeatureEngine(self.config.get('features', {}))
        self.processing_metadata = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'errors': [],
            'pipeline_status': 'IN_PROGRESS'
        }
        # Catálogo de features
        self.catalog = load_catalog()
        validate_catalog(self.catalog)
        self.declared_features = list_declared_features(self.catalog)

    def _load_config(self, config_path: str) -> dict:
        """Carrega configuração do pipeline."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
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
                'enable_temporal_components': True,
                'create_lags': True,
                'lag_periods': [1, 7, 30],
                'create_rolling_stats': True,
                'rolling_windows': [7, 14, 30],
                'create_statistical': True
            },
            'validation': {
                'run_validation': True,
                'fail_on_validation_error': True
            }
        }

    def validate_input_data(self) -> bool:
        """Executa validação de dados de entrada via Great Expectations."""
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

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Carrega dados de treino e teste."""
        logger.info("📥 Carregando dados...")
        try:
            train_path = Path(self.config['input']['train_path'])
            if not train_path.exists():
                raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_path}")

            df_train = pd.read_csv(
                train_path,
                parse_dates=['date'],
                dtype={'store': 'int16', 'item': 'int16', 'sales': 'int32'}
            )

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

    def create_advanced_features(self, df_train: pd.DataFrame, df_test: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Gera features consistentes para treino e teste (sem vazamento).
        - Concatena (train + test) para garantir mesmas dummies/interações.
        - Lags/Rolling usam apenas passado (shift(1)) e o histórico do treino para o teste.
        - Estatísticas aprendidas só no treino e aplicadas ao teste via map/join.
        """
        logger.info("🔧 Criando features consistentes para treino e teste...")

        # Cópias e flags
        df_train = df_train.copy().sort_values(['store', 'item', 'date'])
        df_train['_is_test'] = 0

        df_test_in = None
        test_had_sales = False
        if df_test is not None:
            df_test_in = df_test.copy().sort_values(['store', 'item', 'date'])
            test_had_sales = 'sales' in df_test_in.columns
            df_test_in['_is_test'] = 1
            if not test_had_sales:
                # cria coluna 'sales' vazia só para permitir lags/rolling consistentes
                df_test_in['sales'] = np.nan

        # Concatena
        df_all = pd.concat([df_train, df_test_in] if df_test_in is not None else [df_train], ignore_index=True)

        # Base (temporais/categóricas/interações) em conjunto p/ mesmas colunas
        df_all = self.feature_engine.fit_transform(df_all)

        cfg = self.config.get('features', {})

        # Temporais adicionais (não dependem de sales)
        if cfg.get('enable_temporal_components', True):
            df_all = add_temporal_components(df_all)

        # Ordena antes de lags/rolling
        df_all = df_all.sort_values(['store', 'item', 'date'])

        # Lags
        if cfg.get('create_lags', True):
            for lag in cfg.get('lag_periods', [1, 7, 30]):
                df_all[f'sales_lag_{lag}'] = df_all.groupby(['store', 'item'])['sales'].shift(lag)

        # Rolling (sem ponto atual)
        if cfg.get('create_rolling_stats', True):
            windows = cfg.get('rolling_windows', [7, 14, 30])

            def _apply_rolling(g):
                # garante dtype float e evita usar o ponto atual
                s = g['sales'].astype(float)
                for w in windows:
                    g[f'sales_rolling_mean_{w}'] = s.shift(1).rolling(w, min_periods=1).mean()
                    g[f'sales_rolling_std_{w}']  = s.shift(1).rolling(w, min_periods=1).std()
                return g

            df_all = (
                df_all
                .groupby(['store', 'item'], group_keys=False, sort=False)
                .apply(_apply_rolling)
            )

        # Estatísticas aprendidas no treino
        if cfg.get('create_statistical', True):
            mask_train = (df_all['_is_test'] == 0)
            train_view = df_all[mask_train]
            store_mean = train_view.groupby('store')['sales'].mean()
            item_std = train_view.groupby('item')['sales'].std()

            df_all['sales_mean_store'] = df_all['store'].map(store_mean)
            df_all['sales_std_item'] = df_all['item'].map(item_std)

        # Separa de volta
        df_train_out = df_all[df_all['_is_test'] == 0].drop(columns=['_is_test'])
        df_test_out = None
        if df_test_in is not None:
            df_test_out = df_all[df_all['_is_test'] == 1].drop(columns=['_is_test'])
            if not test_had_sales:
                # remove a coluna 'sales' do teste se ela não existia originalmente
                df_test_out = df_test_out.drop(columns=['sales'])

        logger.info("✅ Features criadas para treino e teste com consistência")
        self.processing_metadata['steps_completed'].append('feature_engineering')
        return df_train_out, df_test_out

    def _save_metadata(self, df_train: pd.DataFrame, df_test: pd.DataFrame | None, metadata_path: Path):
        """Salva metadata detalhado do processamento."""
        base_cols = {'date', 'store', 'item', 'sales'}
        generated_cols = sorted([c for c in df_train.columns if c not in base_cols])

        meta = {
            'declared_in_catalog': sorted(self.declared_features),
            'actually_generated': generated_cols,
            'engine_feature_names': self.feature_engine.get_feature_names(),
            'params_used': self.config.get('features', {}),
            'catalog_file': str(Path('features/feature_store.yaml').resolve()),
        }
        self.processing_metadata.update(meta)

        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.processing_metadata, f, sort_keys=False, allow_unicode=True)

    def save_processed_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame = None):
        """Salva dados processados e metadata."""
        logger.info("💾 Salvando dados processados...")
        try:
            output_dir = Path(self.config['output']['processed_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            train_output_path = output_dir / self.config['output']['train_output']
            df_train.to_parquet(
                train_output_path,
                index=False,
                compression='snappy',
                engine='pyarrow'
            )
            logger.info(f"✅ Dados de treino salvos: {train_output_path}")

            if df_test is not None:
                test_output_path = output_dir / self.config['output']['test_output']
                df_test.to_parquet(
                    test_output_path,
                    index=False,
                    compression='snappy',
                    engine='pyarrow'
                )
                logger.info(f"✅ Dados de teste salvos: {test_output_path}")

            # atualizar e salvar metadata
            self.processing_metadata['end_time'] = datetime.now().isoformat()
            self.processing_metadata['total_features'] = len(df_train.columns)
            self.processing_metadata['train_shape'] = tuple(df_train.shape)
            self.processing_metadata['test_shape'] = tuple(df_test.shape) if df_test is not None else None

            metadata_path = output_dir / self.config['output']['metadata_output']
            self._save_metadata(df_train, df_test, metadata_path)

            logger.info(f"📋 Metadata salva: {metadata_path}")
            self.processing_metadata['steps_completed'].append('data_saving')

        except Exception as e:
            error_msg = f"Erro ao salvar dados: {str(e)}"
            logger.error(error_msg)
            self.processing_metadata['errors'].append(error_msg)
            raise

    def run_full_pipeline(self):
        """Executa pipeline completo de processamento."""
        logger.info("🚀 Iniciando pipeline completo de processamento...")
        try:
            # 1) Validação
            self.validate_input_data()

            # 2) Carregamento
            df_train, df_test = self.load_data()

            # 3) Feature Engineering (treino + teste consistentes)
            df_train_processed, df_test_processed = self.create_advanced_features(df_train, df_test)

            # 4) Salvar
            self.save_processed_data(df_train_processed, df_test_processed)

            # 5) Status e relatório
            self.processing_metadata['pipeline_status'] = 'SUCCESS'
            self._generate_processing_report()
            logger.info("🎉 Pipeline concluído com sucesso!")

        except Exception as e:
            logger.error(f"❌ Pipeline falhou: {str(e)}")
            self.processing_metadata['pipeline_status'] = 'FAILED'
            self.processing_metadata['failure_reason'] = str(e)
            self._generate_processing_report()
            raise

    def _generate_processing_report(self):
        """Gera relatório detalhado do processamento."""
        logger.info("📊 Gerando relatório de processamento...")

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
