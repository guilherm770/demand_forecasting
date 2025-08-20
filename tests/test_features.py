import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
import tempfile
from typing import Tuple

# Adicionar path do projeto (raiz) para imports absolutos
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from features.base_features import BaseFeatureEngine
from scripts.registry_loader import load_catalog, list_declared_features
from scripts.process_data import DataProcessor


class TestFeatureEngine(unittest.TestCase):
    """
    Testes unitários para o sistema de feature engineering.
    """

    def setUp(self):
        """Dataset sintético para cada caso."""
        self.engine = BaseFeatureEngine()

        dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')  # 31 dias
        stores = [1, 2]
        items = [1, 2]

        rows = []
        rng = np.random.default_rng(42)
        for d in dates:
            for s in stores:
                for it in items:
                    rows.append({
                        'date': d,
                        'store': s,
                        'item': it,
                        'sales': int(rng.integers(10, 100))
                    })
        self.df = pd.DataFrame(rows)

    def test_temporal_features_creation(self):
        """Cria features temporais básicas."""
        result = self.engine.create_temporal_features(self.df)

        expected = [
            'year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year',
            'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            'is_weekend', 'is_month_start', 'is_month_end'
        ]
        for feat in expected:
            self.assertIn(feat, result.columns, f"Feature {feat} não encontrada")

        self.assertTrue(result['month_sin'].between(-1, 1).all())
        self.assertTrue(result['month_cos'].between(-1, 1).all())

        weekend_rows = result[result['day_of_week'].isin([5, 6])]
        self.assertTrue((weekend_rows['is_weekend'] == 1).all())

    def test_categorical_features_encoding(self):
        """Testa one-hot de store/item (baixa cardinalidade)."""
        result = self.engine.create_categorical_features(self.df)

        store_dummy_cols = [c for c in result.columns if c.startswith('store_')]
        item_dummy_cols = [c for c in result.columns if c.startswith('item_')]

        self.assertGreater(len(store_dummy_cols), 0, "Dummies de store não criadas")
        self.assertGreater(len(item_dummy_cols), 0, "Dummies de item não criadas")

        # Para cada linha, a dummy correspondente deve ser 1
        for s in result['store'].unique():
            col = f'store_{s}'
            if col in result.columns:
                self.assertTrue((result.loc[result['store'] == s, col] == 1).all())

        for it in result['item'].unique():
            col = f'item_{it}'
            if col in result.columns:
                self.assertTrue((result.loc[result['item'] == it, col] == 1).all())

    def test_interaction_features(self):
        """Testa criação de features de interação."""
        df_tmp = self.engine.create_temporal_features(self.df)
        result = self.engine.create_interaction_features(df_tmp)

        expected = ['store_item', 'store_month', 'item_month', 'store_dow']
        for feat in expected:
            self.assertIn(feat, result.columns, f"Feature {feat} não encontrada")

        sample = result['store_item'].iloc[0]
        self.assertIsInstance(sample, str)
        self.assertIn('_', sample)

    def test_pipeline_integration(self):
        """Pipeline completo do BaseFeatureEngine aumenta o número de colunas e preserva as originais."""
        result = self.engine.fit_transform(self.df)

        self.assertGreater(result.shape[1], self.df.shape[1])

        original_cols = ['date', 'store', 'item', 'sales']
        for col in original_cols:
            self.assertIn(col, result.columns)
            pd.testing.assert_series_equal(result[col].reset_index(drop=True),
                                           self.df[col].reset_index(drop=True))

        feature_names = self.engine.get_feature_names()
        self.assertGreater(len(feature_names), 0)
        for feat in feature_names:
            self.assertIn(feat, result.columns)

    def test_catalog_loading_and_alignment(self):
        """Catálogo deve declarar features base e elas devem ser geradas pelo engine."""
        catalog = load_catalog()
        declared = set(list_declared_features(catalog))

        self.assertIn('year', declared, "Catálogo não declara 'year' em base.temporal")

        out = self.engine.fit_transform(self.df)
        self.assertIn('year', out.columns, "Engine não gerou 'year'")

    def test_params_presence(self):
        """params.yaml deve conter as chaves usadas pelo pipeline/dvc."""
        params_path = ROOT / 'params.yaml'
        self.assertTrue(params_path.exists(), "params.yaml não encontrado")

        with open(params_path, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f) or {}

        self.assertIn('features', params, "Seção 'features' ausente em params.yaml")
        feats = params['features']
        required_keys = [
            'enable_temporal_components',
            'create_lags', 'lag_periods',
            'create_rolling_stats', 'rolling_windows',
            'create_statistical'
        ]
        for k in required_keys:
            self.assertIn(k, feats, f"Chave '{k}' ausente em params.yaml->features")

    def test_train_test_same_feature_space(self):
        """Treino e teste devem ter as MESMAS colunas de features (exceto 'sales' se o teste não tiver target)."""
        # simula um teste sem coluna 'sales'
        df_test = self.df.drop(columns=['sales'])

        proc = DataProcessor(config_path=str(ROOT / 'params.yaml'))
        proc.config['validation']['run_validation'] = False  # não chamar GE em unit test

        train_out, test_out = proc.create_advanced_features(self.df, df_test)

        # 'sales' somente no treino se o teste não tinha target
        self.assertIn('sales', train_out.columns)
        self.assertNotIn('sales', test_out.columns)

        # mesmo espaço de features
        train_feats = set(train_out.columns) - {'sales'}
        test_feats = set(test_out.columns)
        self.assertSetEqual(train_feats, test_feats, "Train e test não possuem o mesmo espaço de features")

    def test_no_data_leakage_in_rolling_and_lags(self):
        """
        Verifica que lags/rolling não usam o valor atual e respeitam fronteiras de grupo.
        """
        # Constrói dataset com 2 grupos bem definidos
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        df_tr = pd.DataFrame({
            'date': list(dates)*2,
            'store': [1]*5 + [2]*5,
            'item':  [1]*10,
            'sales': list(range(1,6)) + list(range(101,106))  # grupos com escalas diferentes
        })
        # teste sem target
        df_te = df_tr[['date','store','item']].copy()

        proc = DataProcessor(config_path=str(ROOT / 'params.yaml'))
        proc.config['validation']['run_validation'] = False
        proc.config['features']['create_lags'] = True
        proc.config['features']['lag_periods'] = [1]
        proc.config['features']['create_rolling_stats'] = True
        proc.config['features']['rolling_windows'] = [3]
        proc.config['features']['enable_temporal_components'] = False
        proc.config['features']['create_statistical'] = False

        train_out, test_out = proc.create_advanced_features(df_tr, df_te)

        # Checa valores esperados de rolling mean no treino, por grupo
        def expected_roll_mean(series, w):
            arr = series.tolist()
            exp = []
            for i in range(len(arr)):
                past = arr[max(0, i-w):i]  # até i-1
                exp.append(np.nan if len(past) == 0 else float(np.mean(past)))
            return exp

        for group_val, grp_df in train_out.sort_values(['store','date']).groupby('store'):
            got = grp_df['sales_rolling_mean_3'].tolist()
            exp = expected_roll_mean(grp_df['sales'], 3)
            for i, (g, e) in enumerate(zip(got, exp)):
                if np.isnan(e):
                    self.assertTrue(np.isnan(g), f"group={group_val}, idx={i}: esperado NaN, obtido {g}")
                else:
                    self.assertAlmostEqual(g, e, places=6, msg=f"group={group_val}, idx={i}: got {g} != expected {e}")

            # lag_1 deve ser a venda anterior dentro do grupo
            self.assertIn('sales_lag_1', grp_df.columns)
            self.assertTrue(np.isnan(grp_df['sales_lag_1'].iloc[0]))  # primeira do grupo é NaN
            self.assertTrue((grp_df['sales_lag_1'].iloc[1:].values == grp_df['sales'].iloc[:-1].values).all())

    def test_feature_group_boundaries(self):
        """Mudança de grupo (store/item) deve resetar lags/rolling (primeira linha de cada grupo = NaN)."""
        df_tr = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01','2023-01-02','2023-01-01','2023-01-02']),
            'store': [1,1,2,2],
            'item':  [1,1,1,1],
            'sales': [10,20,30,40]
        })
        df_te = df_tr[['date','store','item']].copy()  # sem sales

        proc = DataProcessor(config_path=str(ROOT / 'params.yaml'))
        proc.config['validation']['run_validation'] = False
        proc.config['features']['create_lags'] = True
        proc.config['features']['lag_periods'] = [1]
        proc.config['features']['create_rolling_stats'] = True
        proc.config['features']['rolling_windows'] = [2]
        proc.config['features']['enable_temporal_components'] = False
        proc.config['features']['create_statistical'] = False

        train_out, _ = proc.create_advanced_features(df_tr, df_te)
        train_out = train_out.sort_values(['store','item','date'])

        # primeira linha de cada grupo: lag e rolling devem ser NaN
        first_idx_per_group = train_out.groupby(['store','item']).head(1).index
        self.assertTrue(train_out.loc[first_idx_per_group, 'sales_lag_1'].isna().all())
        self.assertTrue(train_out.loc[first_idx_per_group, 'sales_rolling_mean_2'].isna().all())

    def test_save_outputs_and_metadata_tempdir(self):
        """Verifica que save_processed_data escreve parquet e metadata em um diretório temporário."""
        proc = DataProcessor(config_path=str(ROOT / 'params.yaml'))
        proc.config['validation']['run_validation'] = False

        # simula teste sem target
        df_test = self.df.drop(columns=['sales'])
        train_out, test_out = proc.create_advanced_features(self.df, df_test)

        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "data/processed"
            proc.config['output']['processed_dir'] = str(outdir)
            proc.save_processed_data(train_out, test_out)

            train_pq = outdir / proc.config['output']['train_output']
            test_pq  = outdir / proc.config['output']['test_output']
            meta_yml = outdir / proc.config['output']['metadata_output']

            self.assertTrue(train_pq.exists(), "Parquet de treino não foi salvo")
            self.assertTrue(test_pq.exists(),  "Parquet de teste não foi salvo")
            self.assertTrue(meta_yml.exists(),  "Metadata não foi salvo")

            # metadata deve conter chaves principais
            meta = yaml.safe_load(open(meta_yml, 'r', encoding='utf-8'))
            for key in ['declared_in_catalog', 'actually_generated', 'engine_feature_names', 'params_used']:
                self.assertIn(key, meta, f"Chave '{key}' ausente no metadata")


if __name__ == '__main__':
    unittest.main()
