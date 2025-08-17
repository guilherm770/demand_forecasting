import os
import json
from pathlib import Path
import great_expectations as gx
from great_expectations.core import ExpectationSuite
import logging
import pandas as pd

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/validation_results")
OUT_FILE = OUT_DIR / "summary.json"

def setup_data_context():
    os.environ["GE_USAGE_STATS"] = "False"
    context = gx.get_context(mode="ephemeral")
    logger.info("Contexto Great Expectations criado com sucesso")
    return context

def create_expectation_suite_and_validate(context, data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")

    suite = ExpectationSuite(name="raw_data_quality_suite")

    try:
        datasource = context.data_sources.add_pandas(name="pandas_datasource")
    except Exception:
        datasource = context.data_sources.get("pandas_datasource")

    try:
        csv_asset = datasource.add_csv_asset(
            name="raw_data_csv",
            filepath_or_buffer=data_path,
        )
    except Exception:
        csv_asset = datasource.get_asset("raw_data_csv")

    batch_request = csv_asset.build_batch_request()
    validator = context.get_validator(batch_request=batch_request, expectation_suite=suite)

    logger.info("Aplicando expectativas...")
    expected_columns = ['date', 'store', 'item', 'sales']
    validator.expect_table_columns_to_match_ordered_list(expected_columns)

    for column in expected_columns:
        if column in df.columns:
            validator.expect_column_values_to_not_be_null(column)

    if 'sales' in df.columns:
        sales_min = max(0, df['sales'].min())
        sales_max = min(1_000_000, df['sales'].max() * 1.1)
        validator.expect_column_values_to_be_between('sales', min_value=sales_min, max_value=sales_max)
        validator.expect_column_values_to_be_between('sales', min_value=0)

    if 'store' in df.columns:
        validator.expect_column_values_to_be_between('store', min_value=1, max_value=50)
    if 'item' in df.columns:
        validator.expect_column_values_to_be_between('item', min_value=1, max_value=100)

    for column in ['store', 'item', 'sales']:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            validator.expect_column_values_to_be_in_type_list(column, ['int', 'int64', 'float', 'float64'])

    if 'date' in df.columns:
        validator.expect_column_values_to_match_strftime_format('date', '%Y-%m-%d')

    validator.expect_compound_columns_to_be_unique(['date', 'store', 'item'])

    logger.info("Expectativas aplicadas. Executando validação...")
    validator.save_expectation_suite(discard_failed_expectations=False)
    result = validator.validate()

    return result, validator, df

def build_summary(result, df):
    """Monta um dicionário com o resumo da validação."""
    stats = getattr(result, "statistics", {}) or {}
    failed = []
    for r in getattr(result, "results", []) or []:
        if not getattr(r, "success", True):
            exp = getattr(r, "expectation_config", None)
            exp_name = getattr(exp, "expectation_type", "unknown") if exp else "unknown"
            failed.append(exp_name)

    summary = {
        "success": bool(getattr(result, "success", False)),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "statistics": {
            "evaluated_expectations": int(stats.get("evaluated_expectations", 0)),
            "successful_expectations": int(stats.get("successful_expectations", 0)),
            "unsuccessful_expectations": int(stats.get("unsuccessful_expectations", 0)),
            "success_percent": float(stats.get("success_percent", 0.0)),
        },
        "failed_expectations": failed[:20],  # limite para não inflar o json
    }
    return summary

def write_summary_file(summary: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Resumo salvo em: {OUT_FILE}")

def show_data_debug_info(df):
    logger.info("\n" + "="*50)
    logger.info("📊 INFORMAÇÕES DOS DADOS PARA DEBUG")
    logger.info("="*50)
    logger.info(f"📏 Shape: {df.shape} (linhas x colunas)")
    logger.info(f"📋 Colunas: {list(df.columns)}")
    logger.info(f"\n🔍 TIPOS DE DADOS:")
    for col, dtype in df.dtypes.items():
        logger.info(f"   {col}: {dtype}")
    logger.info(f"\n📄 PRIMEIRAS 5 LINHAS:")
    logger.info(df.head(5).to_string(index=False))
    logger.info("="*50)

def main():
    context = setup_data_context()
    data_path = 'data/raw/train.csv'

    # Garantir geração do summary.json mesmo em falha
    result = None
    df = None
    try:
        result, validator, df = create_expectation_suite_and_validate(context, data_path)
        success = bool(getattr(result, "success", False))
        if not success:
            logger.error("❌ VALIDAÇÃO FALHOU!")
            show_data_debug_info(df)
        else:
            logger.info("✅ VALIDAÇÃO PASSOU COM SUCESSO!")
    except Exception as e:
        logger.error(f"💥 Erro fatal: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        # Em caso de exceção antes da validação, ainda tentamos escrever um summary mínimo
        if df is None:
            df = pd.DataFrame()
        class _EmptyResult:  # objeto mínimo para build_summary
            success = False
            statistics = {"evaluated_expectations": 0, "successful_expectations": 0, "unsuccessful_expectations": 0, "success_percent": 0.0}
            results = []
        result = _EmptyResult()
    finally:
        # Sempre gera o summary.json para satisfazer o DVC
        try:
            summary = build_summary(result, df if df is not None else pd.DataFrame())
            write_summary_file(summary)
        except Exception as e2:
            logger.error(f"Não foi possível escrever {OUT_FILE}: {e2}")

if __name__ == "__main__":
    main()
