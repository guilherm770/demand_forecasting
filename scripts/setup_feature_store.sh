#!/bin/bash

echo "🚀 Iniciando setup do Feature Store..."
dvc init

# Subir infraestrutura
docker compose up -d --build
echo "⏳ Aguardando MinIO inicializar..."
sleep 15

# Configurar DVC remote
echo "🔧 Configurando DVC..."
dvc remote add -d minio-storage s3://store-item-demand-forecasting
dvc remote modify minio-storage endpointurl http://localhost:9100
dvc remote modify minio-storage access_key_id minioadmin  
dvc remote modify minio-storage secret_access_key minioadmin
dvc remote modify minio-storage ssl_verify false

echo "✅ Feature Store configurado com sucesso!"
echo "🌐 Console MinIO: http://localhost:9101"
echo "🌐 Console MLflow: http://localhost:5000"
echo "📊 Usuário: minioadmin | Senha: minioadmin"