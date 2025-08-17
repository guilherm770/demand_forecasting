from pathlib import Path
import zipfile
import os

import kaggle

COMP = "demand-forecasting-kernels-only"

def download_dataset():
    """
    Faz o download do dataset do Kaggle e extrai para data/raw.
    Evita redownload se os arquivos já existirem.
    """
    raw_path = Path("data/raw")
    raw_path.mkdir(parents=True, exist_ok=True)

    expected_files = ["train.csv", "test.csv", "sample_submission.csv"]
    if all((raw_path / f).exists() for f in expected_files):
        print("Dataset já existe. Pulando download.")
        return

    print("Baixando dataset do Kaggle...")

    # Baixa todos os arquivos da competição como um .zip
    kaggle.api.competition_download_files(
        COMP,
        path=str(raw_path),
        force=False,   # mude para True se quiser sobrescrever
        quiet=False
    )

    # O Kaggle salva como data/raw/{COMP}.zip
    zip_path = raw_path / f"{COMP}.zip"
    if not zip_path.exists():
        # Algumas versões salvam como {COMP}.zip mesmo com path diferente
        # tenta fallback no diretório atual
        alt = Path(f"{COMP}.zip")
        if alt.exists():
            alt.replace(zip_path)

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_path)
        # opcional: remove o zip
        zip_path.unlink(missing_ok=True)
        print("Download concluído e arquivos extraídos.")
    else:
        raise FileNotFoundError(
            f"Arquivo ZIP não encontrado em {zip_path}. "
            "Verifique permissões da competição e autenticação do Kaggle."
        )

if __name__ == "__main__":
    download_dataset()
