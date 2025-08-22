# scripts/17_cohort_summary.py  (v2 — safe column picking & numeric casting)
import sys, argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INFER_DIR = ROOT / "data" / "processed" / "infer"
OUT_DIR   = ROOT / "data" / "processed" / "cohort"

def load_all():
    paths = sorted(INFER_DIR.glob("e4infer_*/scores_v02_calibrated.parquet"))
    if not paths:
        # yine de v01 deneyelim
        paths = sorted(INFER_DIR.glob("e4infer_*/scores_v01.parquet"))
    if not paths:
        raise SystemExit(f"[error] infer klasöründe skor bulunamadı: {INFER_DIR.as_posix()}")

    frames = []
    for p in paths:
        df = pd.read_parquet(p).copy()

        # subject yoksa klasör adından al (e4infer_S13 → S13)
        if "subject_id" not in df.columns:
            subj = p.parent.name.split("_", 1)[-1]
            df["subject_id"] = subj

        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # ---- etkin kolonları oluştur (score_eff / band_eff)
    # score_cal varsa onu kullan, yoksa score
    if "score_cal" in all_df.columns:
        all_df["score_eff"] = all_df["score_cal"]
    else:
        all_df["score_eff"] = all_df["score"]

    # band_cal varsa onu kullan, yoksa band
    if "band_cal" in all_df.columns:
        all_df["band_eff"] = all_df["band_cal"]
    else:
        all_df["band_eff"] = all_df["band"]

    # numerik güvence
    all_df["score_eff"] = pd.to_numeric(all_df["score_eff"], errors="coerce")

    return all_df

def main():
    ap = argparse.ArgumentParser(description="Cohort summary for all e4infer_* sessions")
    ap.add_argument("--min-windows", type=int, default=10, help="Özete dahil olmak için asgari pencere sayısı")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()

    score_col = "score_eff"
    band_col  = "band_eff"

    # ---- toplam band dağılımı
    cohort_counts = (df[band_col].value_counts(dropna=False)
                     .rename_axis("band")
                     .reset_index(name="count")
                     .sort_values("count", ascending=False))
    cohort_counts.to_csv(OUT_DIR / "_cohort_band_counts.csv", index=False)

    # ---- subject bazlı özet (yalnızca numerik skorlardan)
    grp = df.groupby("subject_id", as_index=False)
    subj = grp.agg(
        n=(score_col, "size"),
        score_mean=(score_col, "mean"),
        score_std=(score_col, "std"),
        high_crit_pct=(band_col, lambda s: (s.isin(["high","critical"])).mean()),
        elevated_pct=(band_col, lambda s: (s == "elevated").mean()),
        low_pct=(band_col, lambda s: (s == "low").mean()),
        calm_pct=(band_col, lambda s: (s == "calm").mean()),
        unknown_pct=(band_col, lambda s: (s == "unknown").mean()),
    )
    subj = subj.loc[subj["n"] >= args.min_windows].copy()
    subj["risk_pct"] = subj["high_crit_pct"] + subj["elevated_pct"]
    subj.sort_values(["risk_pct", "score_mean"], ascending=[False, False], inplace=True)
    subj.round(4).to_csv(OUT_DIR / "_subject_summary.csv", index=False)

    # ---- pivot: subject x band
    pivot = (df.pivot_table(index="subject_id", columns=band_col,
                            values=score_col, aggfunc="size", fill_value=0)
             .reset_index())
    pivot.to_csv(OUT_DIR / "_band_by_subject.csv", index=False)

    # ---- birleşik tüm skorları kaydet (yardımcı kolonları saklayıp saklamamak serbest)
    df.to_parquet(OUT_DIR / "all_scores.parquet", index=False)

    # ---- görseller
    # 1) cohort band bar
    plt.figure(figsize=(8,5))
    plt.bar(cohort_counts["band"].astype(str), cohort_counts["count"].astype(int))
    plt.title("Cohort band dağılımı")
    plt.xlabel("band"); plt.ylabel("adet")
    plt.tight_layout(); plt.savefig(OUT_DIR / "_cohort_band_counts.png", dpi=150); plt.close()

    # 2) subject risk bar (ilk 12)
    top = subj.head(12)
    if not top.empty:
        plt.figure(figsize=(10,5))
        plt.bar(top["subject_id"].astype(str), (top["risk_pct"]*100.0))
        plt.title("Subject risk yüzdesi (elevated + high + critical)")
        plt.xlabel("subject"); plt.ylabel("risk %")
        plt.tight_layout(); plt.savefig(OUT_DIR / "_subject_risk_top12.png", dpi=150); plt.close()

    print("[ok] Cohort özet →", OUT_DIR.as_posix())
    print(" - _cohort_band_counts.csv / .png")
    print(" - _subject_summary.csv  (leaderboard)")
    print(" - _band_by_subject.csv")
    print(" - all_scores.parquet")
    if not top.empty:
        print(" - _subject_risk_top12.png")

if __name__ == "__main__":
    main()
