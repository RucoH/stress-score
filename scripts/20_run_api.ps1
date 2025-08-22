# scripts/20_run_api.ps1  (uvicorn PATH'te olmasa da çalışır)

# Gerekli paketler (idempotent)
python -m pip install -U fastapi "uvicorn[standard]" pandas pyarrow

# API'yi başlat (PATH'e ihtiyaç yok)
cd $PSScriptRoot\..
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000