import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Adicionar path das features
sys.path.append(str(Path(__file__).parent.parent))
from features.base_features import BaseFeatureEngine

class TestFeatureEngine(unittest.TestCase):
    """
    Testes unitários para o sistema de feature engineering.
    
    Cada teste interage com componentes específicos do sistema
    para validar comportamento esperado e detectar regressões.
    """
    
    def setUp(self):
        """Configuração de dados de teste para cada caso."""
        self.engine = BaseFeatureEngine()
        
        # Criar dataset sintético para testes
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        stores = [1, 2, 3]
        items = [1, 2]
        
        self.test_data = []
        for date in dates:
            for store in stores:
                for item in items:
                    sales = np.random.randint(10, 100)
                    self.test_data.append({
                        'date': date,
                        'store': store,
                        'item': item,
                        'sales': sales
                    })
        
        self.df_test = pd.DataFrame(self.test_data)
    
    def test_temporal_features_creation(self):
        """Testa criação de features temporais."""
        result = self.engine.create_temporal_features(self.df_test)
        
        # Verificar se features foram criadas
        expected_features = [
            'year', 'month', 'day', 'day_of_week', 'day_of_year',
            'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            'is_weekend', 'is_month_start', 'is_month_end'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, result.columns, f"Feature {feature} não encontrada")
        
        # Verificar ranges das features cíclicas
        self.assertTrue(result['month_sin'].between(-1, 1).all())
        self.assertTrue(result['month_cos'].between(-1, 1).all())
        
        # Verificar lógica de weekend
        weekend_days = result[result['day_of_week'].isin([5, 6])]
        self.assertTrue((weekend_days['is_weekend'] == 1).all())
    
    def test_categorical_features_encoding(self):
        """Testa encoding de features categóricas."""
        result = self.engine.create_categorical_features(self.df_test)
        
        # Verificar se dummies foram criadas
        store_dummies = [col for col in result.columns if col.startswith('store_')]
        item_dummies = [col for col in result.columns if col.startswith('item_')]
        
        self.assertTrue(len(store_dummies) > 0, "Dummies de store não criadas")
        self.assertTrue(len(item_dummies) > 0, "Dummies de item não criadas")
        
        # Verificar se encoding é mutuamente exclusivo
        for store in self.df_test['store'].unique():
            store_col = f'store_{store}'
            if store_col in result.columns:
                store_rows = result[result['store'] == store]
                self.assertTrue((store_rows[store_col] == 1).all())
    
    def test_interaction_features(self):
        """Testa criação de features de interação."""
        # Primeiro criar features temporais (dependência)
        df_with_temporal = self.engine.create_temporal_features(self.df_test)
        result = self.engine.create_interaction_features(df_with_temporal)
        
        # Verificar features de interação
        interaction_features = ['store_item', 'store_month', 'item_month', 'store_dow']
        
        for feature in interaction_features:
            self.assertIn(feature, result.columns, f"Feature de interação {feature} não encontrada")
        
        # Verificar formato das interações
        sample_store_item = result['store_item'].iloc[0]
        self.assertIsInstance(sample_store_item, str)
        self.assertIn('_', sample_store_item)
    
    def test_pipeline_integration(self):
        """Testa integração completa do pipeline."""
        result = self.engine.fit_transform(self.df_test)
        
        # Verificar se shape aumentou (features adicionadas)
        self.assertGreater(result.shape[1], self.df_test.shape[1])
        
        # Verificar se dados originais foram preservados
        original_cols = ['date', 'store', 'item', 'sales']
        for col in original_cols:
            self.assertIn(col, result.columns)
            pd.testing.assert_series_equal(result[col], self.df_test[col])
        
        # Verificar se não há features vazias
        feature_names = self.engine.get_feature_names()
        self.assertTrue(len(feature_names) > 0)
        
        for feature in feature_names:
            if feature in result.columns:
                null_count = result[feature].isnull().sum()
                self.assertLessEqual(null_count / len(result), 0.1, 
                                   f"Feature {feature} tem muitos valores nulos")

class TestDataValidation(unittest.TestCase):
    """Testes para sistema de validação de dados."""
    
    def test_data_quality_checks(self):
        """Testa verificações básicas de qualidade."""
        # Dados válidos
        valid_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'store': [1] * 10,
            'item': [1] * 10, 
            'sales': [50] * 10
        })
        
        # Verificações básicas
        self.assertFalse(valid_data.isnull().any().any())
        self.assertTrue((valid_data['sales'] >= 0).all())
        self.assertTrue(valid_data['store'].dtype in ['int64', 'int32', 'int16'])
    
    def test_data_schema_validation(self):
        """Testa validação de schema."""
        schema = {
            'required_columns': ['date', 'store', 'item', 'sales'],
            'date_columns': ['date'],
            'numeric_columns': ['store', 'item', 'sales']
        }
        
        test_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'store': [1, 2],
            'item': [1, 1],
            'sales': [50, 60]
        })
        
        # Verificar colunas obrigatórias
        for col in schema['required_columns']:
            self.assertIn(col, test_data.columns)

if __name__ == '__main__':
    unittest.main()
    