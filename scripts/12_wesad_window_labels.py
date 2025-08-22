# scripts/12_wesad_window_labels.py  (tz-fix)
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORES_V01 = ROOT / "data/processed/scores_v01.parquet"
INTERVALS  = ROOT / "configs/wesad_intervals.csv"
OUT_PARQ   = ROOT / "data/interim/wesad_window_labels.parquet"

def to_utc_naive(s: pd.Series) -> pd.Series:
    # her şeyi UTC yap, sonra tz bilgisini düşür (naive)
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s.dt.tz_convert("UTC").dt.tz_localize(None)

def main():
    if not SCORES_V01.exists():
        print(f"[error] yok: {SCORES_V01}"); sys.exit(1)
    if not INTERVALS.exists():
        print(f"[error] yok: {INTERVALS}"); sys.exit(1)

    df = pd.read_parquet(SCORES_V01)
    df = df[df["source"] == "wesad"].copy()
    if df.empty:
        print("[error] v01 içinde 'wesad' yok."); sys.exit(1)

    # Zamanları UTC-naive’a çevir
    df["t_start"] = to_utc_naive(df["t_start"])
    df["t_end"]   = to_utc_naive(df["t_end"])

    iv = pd.read_csv(INTERVALS, parse_dates=["start_time","end_time"])
    iv["start_time"] = to_utc_naive(iv["start_time"])
    iv["end_time"]   = to_utc_naive(iv["end_time"])

    out_rows = []
    for sid, g in df.groupby("subject_id"):
        g = g.copy()
        g["_mid"] = g["t_start"] + (g["t_end"] - g["t_start"]) / 2

        ivs = iv[iv["subject_id"].astype(str) == str(sid)].copy()
        if ivs.empty:
            continue

        intervals = pd.IntervalIndex.from_arrays(ivs["start_time"], ivs["end_time"], closed="both")
        labels = ivs["label"].tolist()

        lab = []
        for t in g["_mid"].values:
            found = "unknown"
            for inter, name in zip(intervals, labels):
                if (t >= inter.left) and (t <= inter.right):
                    found = name
                    break
            lab.append(found)

        g["wesad_label"] = lab
        out_rows.append(g[["source","subject_id","t_start","t_end","wesad_label"]])

    if not out_rows:
        print("[error] hiç label üretilmedi."); sys.exit(1)

    out = pd.concat(out_rows, ignore_index=True)
    OUT_PARQ.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQ, index=False)
    print(f"[ok] Pencere etiketleri → {OUT_PARQ.as_posix()} (n={len(out)})")
    print(out["wesad_label"].value_counts())

if __name__ == "__main__":
    main()
