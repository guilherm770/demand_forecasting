#!/usr/bin/env python3
"""
Script orquestrador que executa pipeline completo com tratamento
de erros robusto e logging detalhado.
"""

import sys
import logging
from pathlib import Path
import yaml
from datetime import datetime

# Importar módulos do sistema
from download_data import download_dataset
from data_validation import main as validate_data
from process_data import DataProcessor

def setup_logging():
    """Configura logging centralizado do sistema."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('FullPipeline')

def load_pipeline_config() -> dict:
    """Carrega configuração completa do pipeline."""
    config_path = Path('config/pipeline.yaml')
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Configuração padrão
    return {
        'pipeline': {
            'skip_download': False,
            'skip_validation': False,
            'run_monitoring': True,
            'fail_on_error': True
        },
        'notifications': {
            'on_success': True,
            'on_failure': True
        }
    }

def run_full_pipeline():
    """
    Executa pipeline completo end-to-end com orquestração
    inteligente e recuperação de erros.
    """
    logger = setup_logging()
    config = load_pipeline_config()
    
    pipeline_start = datetime.now()
    logger.info("🚀 Iniciando pipeline completo...")
    
    try:
        # Fase 1: Download de dados
        if not config['pipeline']['skip_download']:
            logger.info("📥 Fase 1: Download de dados")
            download_dataset()
            logger.info("✅ Download concluído")
        else:
            logger.info("⏭️ Download pulado conforme configuração")
        
        # Fase 2: Validação
        if not config['pipeline']['skip_validation']:
            logger.info("🔍 Fase 2: Validação de dados")
            validate_data()
            logger.info("✅ Validação concluída")
        else:
            logger.info("⏭️ Validação pulada conforme configuração")
        
        # Fase 3: Processamento
        logger.info("🔧 Fase 3: Processamento e feature engineering")
        processor = DataProcessor()
        processor.run_full_pipeline()
        logger.info("✅ Processamento concluído")
        
        # Sucesso
        pipeline_duration = datetime.now() - pipeline_start
        logger.info(f"🎉 Pipeline concluído com sucesso em {pipeline_duration}")
        
        if config['notifications']['on_success']:
            _send_success_notification(pipeline_duration)
        
        return True
        
    except Exception as e:
        pipeline_duration = datetime.now() - pipeline_start
        logger.error(f"❌ Pipeline falhou após {pipeline_duration}: {str(e)}")
        
        if config['notifications']['on_failure']:
            _send_failure_notification(str(e), pipeline_duration)
        
        if config['pipeline']['fail_on_error']:
            sys.exit(1)
        
        return False

def _send_success_notification(duration):
    """Envia notificação de sucesso (placeholder para integração)."""
    print(f"✅ SUCESSO: Pipeline concluído em {duration}")

def _send_failure_notification(error, duration):
    """Envia notificação de falha (placeholder para integração)."""  
    print(f"❌ FALHA: Pipeline falhou após {duration} - {error}")

if __name__ == "__main__":
    run_full_pipeline()