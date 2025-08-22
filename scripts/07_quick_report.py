# scripts/07_quick_report.py  (v4 — reasons breakdown + safe columns)
import sys, argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEF_PATH = ROOT / "data/processed/scores_v01.parquet"
OUT_DIR  = ROOT / "data/processed"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(DEF_PATH), help="Parquet skor yolu (v01/v02/v03)")
    ap.add_argument("--topk", type=int, default=5, help="Top-K pencereleri yazdır")
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

    # subject bazında özet
    per_subj = (df
        .groupby(['source','subject_id'])
        .agg(n=(score_col,'size'),
             score_mean=(score_col,'mean'),
             score_std=(score_col,'std'),
             high_pct=(band_col, lambda s: (s.isin(['high','critical'])).mean()))
        .reset_index()
        .sort_values(['source','subject_id'])
    )

    # Top-K pencereler — opsiyonel kolonlar varsa ekle
    base_cols = ['source','subject_id','t_start','t_end',score_col,band_col]
    optional  = [c for c in ['confidence','reasons','hr_mean','temp_mean','qa_artifact_ratio'] if c in df.columns]
    top_cols  = base_cols + optional
    topk = df.sort_values(score_col, ascending=False).head(args.topk)[top_cols]

    # reasons breakdown (varsa)
    if "reasons" in df.columns:
        rs = (df["reasons"].fillna("")
              .str.split(";", expand=True)
              .stack()
              .str.strip()
              .rename("reason"))
        rs = rs[rs.ne("")]
        reasons_counts = (rs.value_counts()
                          .rename_axis("reason")
                          .reset_index(name="count")) if not rs.empty else pd.DataFrame(columns=["reason","count"])
    else:
        reasons_counts = pd.DataFrame(columns=["reason","count"])

    # düşük güven oranı (varsa)
    low_conf_pct = (df['confidence'].eq('low').mean()) if 'confidence' in df.columns else np.nan

    # --- çıktı dosyaları ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    band_counts.to_csv(OUT_DIR / "_band_counts.csv", index=False)
    per_subj.to_csv(OUT_DIR / "_scores_by_subject.csv", index=False)
    topk.to_csv(OUT_DIR / f"_top{args.topk}_windows.csv", index=False)
    reasons_counts.to_csv(OUT_DIR / "_reasons_counts.csv", index=False)

    print("\n[ok] kayıtlar:")
    print("  - data/processed/_band_counts.csv")
    print("  - data/processed/_scores_by_subject.csv")
    print(f"  - data/processed/_top{args.topk}_windows.csv")
    print("  - data/processed/_reasons_counts.csv")
    if 'confidence' in df.columns:
        print(f"[info] düşük güven oranı: {low_conf_pct:.1%}")
    else:
        print("[info] 'confidence' kolonu yok (inference dosyası için normal).")

if __name__ == "__main__":
    main()
