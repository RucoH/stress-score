# scripts/10_session_report.py  (v1.1 — JSON fix)
import sys, argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

ROOT = Path(__file__).resolve().parents[1]

def pick_col(df, cal_name, raw_name):
    return cal_name if cal_name in df.columns else raw_name

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="scores_v02_calibrated.parquet (veya v01)")
    ap.add_argument("--outdir", default=None, help="Çıktı klasörü (varsayılan dosyanın yanına)")
    ap.add_argument("--smooth", type=int, default=3, help="Hareketli ortalama pencere boyu (adet)")
    ap.add_argument("--topk", type=int, default=5, help="En yüksek K segment")
    return ap.parse_args()

def contiguous_segments(df, score_col, band_col, thr=80.0, min_len=2):
    """score_col >= thr olan ardışık pencereleri segment olarak birleştirir."""
    m = df[score_col] >= thr
    segs = []
    if not m.any():
        return segs
    idx = np.where(m.values)[0]
    start = idx[0]; prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            if (prev - start + 1) >= min_len:
                sl = df.iloc[start:prev+1]
                segs.append((sl.iloc[0]["t_start"], sl.iloc[-1]["t_end"],
                             float(sl[score_col].mean()), sl[band_col].mode().iat[0]))
            start = prev = i
    if (prev - start + 1) >= min_len:
        sl = df.iloc[start:prev+1]
        segs.append((sl.iloc[0]["t_start"], sl.iloc[-1]["t_end"],
                     float(sl[score_col].mean()), sl[band_col].mode().iat[0]))
    segs.sort(key=lambda x: x[2], reverse=True)
    return segs

def main():
    args = parse_args()
    p = Path(args.path)
    if not p.exists():
        print(f"[error] dosya yok: {p}")
        sys.exit(1)
    outdir = Path(args.outdir) if args.outdir else p.parent

    df = pd.read_parquet(p).sort_values("t_start").reset_index(drop=True)
    score_col = pick_col(df, "score_cal", "score")
    band_col  = pick_col(df, "band_cal",  "band")

    # opsiyonel yumuşatma (grafik için)
    vis_col = score_col
    if args.smooth and args.smooth > 1:
        df["_score_smooth"] = df[score_col].rolling(args.smooth, min_periods=1).mean()
        vis_col = "_score_smooth"

    # Zaman serisi grafiği
    plt.figure(figsize=(11,7))
    plt.plot(df["t_start"], df[vis_col], marker='.', linewidth=1)
    plt.title(f"Stres skoru zaman serisi ({p.stem})")
    plt.xlabel("zaman"); plt.ylabel("score (0–100)")
    plt.tight_layout()
    out_png = outdir / "_session_timeline.png"
    plt.savefig(out_png, dpi=160)
    plt.close()

    # Band dağılımı ve top segmentler
    band_counts = df[band_col].value_counts().rename_axis("band").reset_index(name="count")
    segs = contiguous_segments(df, score_col=score_col, band_col=band_col, thr=80.0, min_len=2)
    seg_df = pd.DataFrame(segs, columns=["t_start","t_end","score_mean","major_band"]).head(args.topk)

    # JSON/CSV için zamanları stringe çevir (JSON serialization fix)
    if not seg_df.empty:
        for c in ["t_start", "t_end"]:
            if c in seg_df.columns:
                seg_df[c] = seg_df[c].astype(str)

    # Çıktılar
    outdir.mkdir(parents=True, exist_ok=True)
    band_counts.to_csv(outdir / "_band_counts.csv", index=False)
    seg_df.to_csv(outdir / "_top_segments.csv", index=False)

    meta = {
        "n_windows": int(len(df)),
        "time_range": [str(df['t_start'].min()), str(df['t_start'].max())],
        "used_score_col": score_col,
        "used_band_col": band_col,
        "top_segments": seg_df.to_dict(orient="records"),
    }
    # default=str => Timestamp vs. otomatik string'e çevrilir
    (outdir / "_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )

    print(f"[ok] Raporlar → {outdir.as_posix()}")
    print(" - _session_timeline.png")
    print(" - _band_counts.csv")
    print(" - _top_segments.csv")
    print(" - _summary.json")

if __name__ == "__main__":
    main()
