import os
import great_expectations as gx
from great_expectations.core import ExpectationSuite
import logging
import pandas as pd

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_data_context():
    """Cria contexto com configuração mínima e robusta"""
    try:
        # Desabilitar estatísticas de uso via variável de ambiente
        os.environ["GE_USAGE_STATS"] = "False"
        
        # Configuração mínima do contexto
        context = gx.get_context(mode="ephemeral")
        
        logger.info("Contexto Great Expectations criado com sucesso")
        return context
    except Exception as e:
        logger.error(f"Erro ao criar contexto: {e}")
        raise

def create_expectation_suite_and_validate(context, data_path):
    """Cria expectativas e valida dados usando a API moderna"""
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
        
        # Ler dados para análise prévia
        df = pd.read_csv(data_path)
        logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        # Criar suite de expectativas
        suite_name = "raw_data_quality_suite"
        suite = ExpectationSuite(name=suite_name)

        # Data source pandas (idempotente)
        try:
            datasource = context.data_sources.add_pandas(name="pandas_datasource")
        except Exception:
            datasource = context.data_sources.get("pandas_datasource")

        # Asset de CSV apontando para o arquivo
        try:
            csv_asset = datasource.add_csv_asset(
                name="raw_data_csv",
                filepath_or_buffer=data_path,
            )
        except Exception:
            csv_asset = datasource.get_asset("raw_data_csv")

        # BatchRequest sem parâmetros (whole table)
        batch_request = csv_asset.build_batch_request()

        # Validator via batch_request (não use 'batch=')
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite=suite,
        )
        
        logger.info("Aplicando expectativas...")
        
        # Definir expectativas
        expected_columns = ['date', 'store', 'item', 'sales']
        
        validator.expect_table_columns_to_match_ordered_list(expected_columns)
        
        for column in expected_columns:
            if column in df.columns:
                validator.expect_column_values_to_not_be_null(column)
        
        if 'sales' in df.columns:
            sales_min = max(0, df['sales'].min())
            sales_max = min(1000000, df['sales'].max() * 1.1)
            validator.expect_column_values_to_be_between('sales', min_value=sales_min, max_value=sales_max)
        
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
        validator.expect_column_values_to_be_between('sales', min_value=0)
        
        logger.info("Expectativas aplicadas. Executando validação...")
        
        # Salvar suite
        validator.save_expectation_suite(discard_failed_expectations=False)
        
        # Executar validação
        result = validator.validate()
        
        return result, validator, df
        
    except Exception as e:
        logger.error(f"Erro ao processar dados: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def analyze_validation_result(result, df):
    """Analisa e reporta os resultados da validação"""
    try:
        stats = result.statistics
        success_count = stats.get('successful_expectations', 0)
        total_count = stats.get('evaluated_expectations', 0)
        failed_count = stats.get('unsuccessful_expectations', 0)
        
        if result.success:
            logger.info("✅ VALIDAÇÃO PASSOU COM SUCESSO!")
            logger.info(f"📊 Expectativas bem-sucedidas: {success_count}/{total_count}")
            return True
        else:
            logger.error("❌ VALIDAÇÃO FALHOU!")
            logger.error(f"📊 Expectativas falhas: {failed_count}/{total_count}")
            
            # Mostrar detalhes das expectativas que falharam
            logger.error("\n🔍 DETALHES DAS FALHAS:")
            for i, result_item in enumerate(result.results):
                if not result_item.success:
                    expectation_config = result_item.expectation_config
                    expectation_type = expectation_config.expectation_type
                    logger.error(f"\n❌ Falha {i+1}: {expectation_type}")
                    
                    # Mostrar configuração da expectativa
                    if hasattr(expectation_config, 'kwargs'):
                        relevant_config = {k: v for k, v in expectation_config.kwargs.items() 
                                         if k not in ['batch_id', 'result_format']}
                        if relevant_config:
                            logger.error(f"   Configuração: {relevant_config}")
                    
                    # Mostrar detalhes do resultado
                    result_details = result_item.result
                    if hasattr(result_details, 'partial_unexpected_list'):
                        unexpected_list = getattr(result_details, 'partial_unexpected_list', [])
                        if unexpected_list:
                            logger.error(f"   Valores inesperados (primeiros 5): {unexpected_list[:5]}")
                    
                    if hasattr(result_details, 'unexpected_count'):
                        unexpected_count = getattr(result_details, 'unexpected_count', 0)
                        if unexpected_count > 0:
                            logger.error(f"   Total de valores inesperados: {unexpected_count}")
                    
                    if hasattr(result_details, 'element_count'):
                        element_count = getattr(result_details, 'element_count', 0)
                        logger.error(f"   Total de elementos: {element_count}")
            
            # Mostrar informações dos dados para debug
            show_data_debug_info(df)
            
            return False
            
    except Exception as e:
        logger.error(f"Erro na análise dos resultados: {e}")
        return False

def show_data_debug_info(df):
    """Mostra informações detalhadas dos dados para debug"""
    logger.info("\n" + "="*50)
    logger.info("📊 INFORMAÇÕES DOS DADOS PARA DEBUG")
    logger.info("="*50)
    
    # Informações básicas
    logger.info(f"📏 Shape: {df.shape} (linhas x colunas)")
    logger.info(f"📋 Colunas: {list(df.columns)}")
    
    # Tipos de dados
    logger.info(f"\n🔍 TIPOS DE DADOS:")
    for col, dtype in df.dtypes.items():
        logger.info(f"   {col}: {dtype}")
    
    # Amostra dos dados
    logger.info(f"\n📄 PRIMEIRAS 5 LINHAS:")
    for i in range(min(5, len(df))):
        row_data = {}
        for col in df.columns:
            val = df.iloc[i][col]
            if isinstance(val, str) and len(val) > 20:
                val = val[:20] + "..."
            row_data[col] = val
        logger.info(f"   {i+1}: {row_data}")
    
    # Estatísticas numéricas
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        logger.info(f"\n📈 ESTATÍSTICAS NUMÉRICAS:")
        for col in numeric_cols:
            stats = {
                'min': df[col].min(),
                'max': df[col].max(),
                'média': round(df[col].mean(), 2),
                'únicos': df[col].nunique()
            }
            logger.info(f"   {col}: {stats}")
    
    # Valores nulos
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.info(f"\n❌ VALORES NULOS:")
        for col, count in null_counts[null_counts > 0].items():
            pct = round((count / len(df)) * 100, 2)
            logger.info(f"   {col}: {count} ({pct}%)")
    else:
        logger.info(f"\n✅ NENHUM VALOR NULO ENCONTRADO")
    
    # Duplicatas
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info(f"\n⚠️  LINHAS DUPLICADAS: {duplicates}")
        if 'date' in df.columns and 'store' in df.columns and 'item' in df.columns:
            key_duplicates = df.duplicated(['date', 'store', 'item']).sum()
            logger.info(f"   Duplicatas por [date, store, item]: {key_duplicates}")
    
    logger.info("="*50)

def main():
    try:
        # Configurar contexto
        context = setup_data_context()
        
        # Processar dados
        data_path = 'data/raw/train.csv'
        result, validator, df = create_expectation_suite_and_validate(context, data_path)
        
        # Analisar resultados
        success = analyze_validation_result(result, df)
        
        if not success:
            logger.error("\n🚨 VALIDAÇÃO DE DADOS FALHOU!")
            logger.info("\n💡 PRÓXIMOS PASSOS:")
            logger.info("   1. Analise as informações dos dados mostradas acima")
            logger.info("   2. Ajuste as expectativas no código conforme necessário")
            logger.info("   3. Corrija problemas de qualidade nos dados se aplicável")
            logger.info("   4. Execute novamente a validação")
            exit(1)
        
        logger.info("\n🎉 VALIDAÇÃO CONCLUÍDA COM SUCESSO!")
        logger.info("✅ Todos os critérios de qualidade foram atendidos")
        
    except Exception as e:
        logger.error(f"💥 Erro fatal: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        exit(1)

if __name__ == "__main__":
    main()