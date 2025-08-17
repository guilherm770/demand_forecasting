import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).parent.parent))
from scripts.process_data import DataProcessor

class TestPipelineIntegration(unittest.TestCase):
    """
    Testes de integração que verificam funcionamento
    do pipeline completo em ambiente controlado.
    """
    
    def setUp(self):
        """Criar ambiente temporário para testes."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Criar estrutura de diretórios
        (self.test_path / 'data' / 'raw').mkdir(parents=True)
        (self.test_path / 'data' / 'processed').mkdir(parents=True)
        
        # Criar dados sintéticos
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'store': [1, 2] * 50,
            'item': [1] * 100,
            'sales': range(100, 200)
        })
        
        test_data.to_csv(self.test_path / 'data' / 'raw' / 'train.csv', index=False)
    
    def tearDown(self):
        """Limpar ambiente de teste."""
        shutil.rmtree(self.test_dir)
    
    def test_end_to_end_pipeline(self):
        """Testa pipeline completo end-to-end."""
        # Configurar processador para usar diretório de teste
        config = {
            'input': {
                'train_path': str(self.test_path / 'data' / 'raw' / 'train.csv'),
                'test_path': str(self.test_path / 'data' / 'raw' / 'test.csv')
            },
            'output': {
                'processed_dir': str(self.test_path / 'data' / 'processed'),
                'train_output': 'train_processed.parquet',
                'test_output': 'test_processed.parquet',
                'metadata_output': 'processing_metadata.yaml'
            },
            'features': {
                'create_lags': False,  # Simplificar para teste
                'create_rolling_stats': False
            },
            'validation': {
                'run_validation': False  # Pular validação em teste
            }
        }
        
        # Executar pipeline
        processor = DataProcessor()
        processor.config = config
        
        # Testar carregamento
        df_train, df_test = processor.load_data()
        self.assertEqual(len(df_train), 100)
        
        # Testar feature engineering
        df_processed = processor.create_advanced_features(df_train)
        self.assertGreater(df_processed.shape[1], df_train.shape[1])
        
        # Testar salvamento
        processor.save_processed_data(df_processed)
        
        # Verificar se arquivos foram criados
        output_file = self.test_path / 'data' / 'processed' / 'train_processed.parquet'
        self.assertTrue(output_file.exists())
        
        # Verificar se dados podem ser carregados novamente
        df_reloaded = pd.read_parquet(output_file)
        self.assertEqual(len(df_reloaded), len(df_processed))

if __name__ == '__main__':
    unittest.main()
    