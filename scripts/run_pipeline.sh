#!/bin/bash
set -e

echo "🔄 Executando pipeline DVC completo..."

# Verificar status atual
echo "📊 Status do pipeline:"
dvc status

# Reproduzir pipeline completo
echo "🚀 Reproduzindo pipeline..."
dvc repro

# Mostrar métricas
echo "📈 Métricas dos experimentos:"
dvc metrics show

# Comparar com experimentos anteriores
echo "🔍 Comparação com versões anteriores:"
dvc metrics diff

# Push para remote storage
echo "☁️ Enviando artefatos para storage remoto..."
dvc push

# Commit das mudanças
#echo "💾 Versionando mudanças..."
#git add .
#git commit -m "Pipeline executado - $(date '+%Y-%m-%d %H:%M:%S')"

echo "✅ Pipeline concluído e versionado!"