# scripts/18_export_excel.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT / "data" / "processed" / "cohort"
OUT_XLSX = C_DIR / "summary.xlsx"

SUBJ_CSV   = C_DIR / "_subject_summary.csv"
BANDS_CSV  = C_DIR / "_cohort_band_counts.csv"
PIVOT_CSV  = C_DIR / "_band_by_subject.csv"
ALL_PQ     = C_DIR / "all_scores.parquet"

def main():
    # varsa yoksa uyarı
    missing = [p.name for p in [SUBJ_CSV, BANDS_CSV, PIVOT_CSV] if not p.exists()]
    if missing:
        print(f"[error] eksik dosya: {', '.join(missing)}")
        print("Önce çalıştırın:  python scripts/17_cohort_summary.py")
        sys.exit(1)

    df_subj  = pd.read_csv(SUBJ_CSV)
    df_bands = pd.read_csv(BANDS_CSV)
    df_pivot = pd.read_csv(PIVOT_CSV)

    # ufak temizlik / sıralama
    if "risk_pct" in df_subj.columns:
        df_subj.sort_values(["risk_pct","score_mean"], ascending=[False,False], inplace=True)

    # meta
    meta_rows = []
    if ALL_PQ.exists():
        all_df = pd.read_parquet(ALL_PQ)
        n_rows = len(all_df)
        n_subj = all_df["subject_id"].nunique() if "subject_id" in all_df.columns else np.nan
        meta_rows = [
            ("toplam pencere", int(n_rows)),
            ("toplam subject", int(n_subj) if not pd.isna(n_subj) else "NA"),
        ]
    else:
        meta_rows = [("not", "all_scores.parquet bulunamadı")]

    # Excel yaz
    C_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as xw:
        # Sheet 1: Özet (metadata)
        meta_df = pd.DataFrame(meta_rows, columns=["anahtar","değer"])
        meta_df.to_excel(xw, sheet_name="Özet", index=False)
        ws_meta = xw.sheets["Özet"]
        ws_meta.set_column("A:A", 20)
        ws_meta.set_column("B:B", 25)

        # Sheet 2: Subject Summary
        df_subj.to_excel(xw, sheet_name="Subject_Summary", index=False)
        ws = xw.sheets["Subject_Summary"]
        ws.freeze_panes(1, 1)
        # sütun genişlikleri
        for i, col in enumerate(df_subj.columns, start=0):
            width = max(12, min(28, int(df_subj[col].astype(str).str.len().quantile(0.9)) + 2))
            ws.set_column(i, i, width)
        # yüzdeler
        pct_cols = [c for c in df_subj.columns if c.endswith("_pct")]
        percent_fmt = xw.book.add_format({"num_format": "0.0%"})
        for c in pct_cols:
            j = df_subj.columns.get_loc(c)
            ws.set_column(j, j, 12, percent_fmt)
        # sayı formatı
        if "n" in df_subj.columns:
            jn = df_subj.columns.get_loc("n")
            ws.set_column(jn, jn, 10, xw.book.add_format({"num_format": "0"}))
            # data bar
            ws.conditional_format(1, jn, len(df_subj), jn, {
                "type": "data_bar", "bar_color": "#B7DEE8"
            })
        # risk heatmap
        if "risk_pct" in df_subj.columns:
            jr = df_subj.columns.get_loc("risk_pct")
            ws.conditional_format(1, jr, len(df_subj), jr, {
                "type": "3_color_scale",
                "min_color": "#63BE7B",  # yeşil
                "mid_color": "#FFEB84",
                "max_color": "#F8696B"   # kırmızı
            })

        # Sheet 3: Band Dağılımı
        df_bands.to_excel(xw, sheet_name="Cohort_Bands", index=False)
        ws_b = xw.sheets["Cohort_Bands"]
        ws_b.freeze_panes(1, 0)
        ws_b.set_column("A:A", 14)
        ws_b.set_column("B:B", 12, xw.book.add_format({"num_format": "0"}))
        # data bar
        if "count" in df_bands.columns:
            j = df_bands.columns.get_loc("count")
            ws_b.conditional_format(1, j, len(df_bands), j, {
                "type": "data_bar", "bar_color": "#9CC3E6"
            })

        # Sheet 4: Band x Subject pivot
        df_pivot.to_excel(xw, sheet_name="Band_by_Subject", index=False)
        ws_p = xw.sheets["Band_by_Subject"]
        ws_p.freeze_panes(1, 1)
        # sayıları 0 formatında ve daha geniş sütun
        for i, col in enumerate(df_pivot.columns, start=0):
            fmt = xw.book.add_format({"num_format": "0"})
            w = 12 if i == 0 else 10
            ws_p.set_column(i, i, w, fmt)

    print(f"[ok] Excel rapor hazır → {OUT_XLSX.as_posix()}")

if __name__ == "__main__":
    main()
