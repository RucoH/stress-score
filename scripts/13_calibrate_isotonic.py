# scripts/13_calibrate_isotonic.py  (tz-fix)
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_SCORES  = ROOT / "data/processed/scores_v01.parquet"
IN_LABELS  = ROOT / "data/interim/wesad_window_labels.parquet"
SPEC_PATH  = ROOT / "configs/score_spec.yaml"
OUT_PATH   = ROOT / "data/processed/scores_v03_isotonic.parquet"
MODEL_DIR  = ROOT / "models"

def _load_yaml(p):
    import yaml
    with open(p,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def _to_band(s, bands_cfg):
    items = sorted(bands_cfg.items(), key=lambda kv: float(kv[1][0]))
    for name, (lo,hi) in items:
        if float(lo) <= s <= float(hi):
            return name
    return "unknown"

def _to_utc_naive(s):
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s.dt.tz_convert("UTC").dt.tz_localize(None)

def main():
    try:
        from sklearn.isotonic import IsotonicRegression
        import joblib
    except Exception:
        print("[error] scikit-learn ve joblib gerekli: pip install scikit-learn joblib")
        sys.exit(1)

    if not IN_SCORES.exists() or not IN_LABELS.exists():
        print(f"[error] eksik: {IN_SCORES} veya {IN_LABELS}")
        sys.exit(1)

    df = pd.read_parquet(IN_SCORES)
    lab = pd.read_parquet(IN_LABELS)

    # sadece WESAD
    df = df[df["source"]=="wesad"].copy()

    # tz standardizasyonu (iki tarafta da aynı)
    df["t_start"] = _to_utc_naive(df["t_start"])
    df["t_end"]   = _to_utc_naive(df["t_end"])
    lab["t_start"] = _to_utc_naive(lab["t_start"])
    lab["t_end"]   = _to_utc_naive(lab["t_end"])

    df = df.merge(lab, on=["source","subject_id","t_start","t_end"], how="left")

    ymap = {"baseline":0, "stress":1}
    df = df[df["wesad_label"].isin(ymap.keys())].copy()
    if df.empty:
        print("[error] baseline ve stress etiketli pencere bulunamadı."); sys.exit(1)

    x = df["score"].astype(float).values
    y = df["wesad_label"].map(ymap).astype(int).values

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)
    prob = iso.transform(x)
    df["score_iso"] = (prob * 100.0).clip(0,100)

    spec = _load_yaml(SPEC_PATH); bands = spec["bands"]
    df["band_iso"] = df["score_iso"].apply(lambda s: _to_band(float(s), bands))
    df["score_cal"] = df["score_iso"]; df["band_cal"] = df["band_iso"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[ok] İzotonik kalibre skor → {OUT_PATH.as_posix()} (n={len(df)})")
    print(df["band_cal"].value_counts())

    import joblib
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, MODEL_DIR / "isotonic_v1.joblib")
    print(f"[ok] Model kaydedildi → {(MODEL_DIR / 'isotonic_v1.joblib').as_posix()}")

if __name__ == "__main__":
    main()
