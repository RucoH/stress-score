# scripts/01_fetch_kaggle.py
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

DATASETS = [
    ("mohamedasem318/wesad-full-dataset", "data/raw/wesad"),
    ("orvile/biostress-dataset",          "data/raw/biostress"),
]

def ensure_dirs():
    for _, out in DATASETS:
        Path(out).mkdir(parents=True, exist_ok=True)

def main():
    ensure_dirs()
    api = KaggleApi()
    api.authenticate()
    for ds, out in DATASETS:
        print(f"[download] {ds} -> {out}")
        api.dataset_download_files(ds, path=out, unzip=True)
    print("[ok] İndirme tamam. Envantere geçebilirsin.")

if __name__ == "__main__":
    main()
