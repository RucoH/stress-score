# scripts/07_quick_report.py (v2)
import sys, argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEF_PATH = ROOT / "data/processed/scores_v01.parquet"
OUT_DIR  = ROOT / "data/processed"

CANDIDATES = [
    ROOT / "data/processed/scores_v02_calibrated.parquet",
    ROOT / "data/processed/scores_v01.parquet",
]
try:
    DEF_PATH = next(p for p in CANDIDATES if p.exists())
except StopIteration:
    DEF_PATH = ROOT / "data/processed/scores_v01.parquet"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(DEF_PATH), help="Parquet skor yolu")
    return ap.parse_args()

def pick_col(df, cal_name, raw_name):
    return cal_name if cal_name in df.columns else raw_name

def main():
    args = parse_args()
    p = Path(args.path)
    if not p.exists():
        print(f"[error] skor dosyası yok: {p}")
        sys.exit(1)

    df = pd.read_parquet(p)

    # kalibre varsa onları kullan
    score_col = pick_col(df, "score_cal", "score")
    band_col  = pick_col(df, "band_cal",  "band")

    print("[info] toplam pencere:", len(df))
    print("[info] tarih aralığı :", df['t_start'].min(), "→", df['t_start'].max())
    print("[info] kaynaklar     :", df['source'].value_counts().to_dict())

    print("\n[band dağılımı]")
    band_counts = df[band_col].value_counts().rename_axis('band').reset_index(name='count')
    print(band_counts)

    per_subj = (df
        .groupby(['source','subject_id'])
        .agg(n=(score_col,'size'),
             score_mean=(score_col,'mean'),
             score_std=(score_col,'std'),
             high_pct=(band_col, lambda s: (s.isin(['high','critical'])).mean()))
        .reset_index()
        .sort_values(['source','subject_id'])
    )

    top5 = df.sort_values(score_col, ascending=False).head(5)[
        ['source','subject_id','t_start','t_end',score_col,band_col,'confidence','reasons','hr_mean','temp_mean','qa_artifact_ratio']
    ]

    low_conf_pct = (df['confidence'].eq('low').mean()) if 'confidence' in df.columns else np.nan

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    band_counts.to_csv(OUT_DIR / "_band_counts.csv", index=False)
    per_subj.to_csv(OUT_DIR / "_scores_by_subject.csv", index=False)
    top5.to_csv(OUT_DIR / "_top5_windows.csv", index=False)

    print("\n[ok] kayıtlar:")
    print("  - data/processed/_band_counts.csv")
    print("  - data/processed/_scores_by_subject.csv")
    print("  - data/processed/_top5_windows.csv")
    print(f"[info] düşük güven oranı: {low_conf_pct:.1%}")

if __name__ == "__main__":
    main()
