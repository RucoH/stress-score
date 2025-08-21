# scripts/08_calibrate_distributional.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_cfg():
    with open(ROOT/"configs/calibration.yaml","r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def pct_map(scores: pd.Series, percentiles, targets):
    # girdideki percentil noktalarını bul
    xp = np.percentile(scores.dropna().values, percentiles).astype(float)
    fp = np.array(targets, dtype=float)
    # monotonluk/kenar durumları
    xp[0]  = min(xp[0],  scores.min())
    xp[-1] = max(xp[-1], scores.max())
    # map
    mapped = np.interp(scores.values, xp, fp)
    return np.clip(mapped, 0.0, 100.0)

def main():
    cfg = load_cfg()
    inp  = ROOT / cfg["in_path"]
    outp = ROOT / cfg["out_path"]
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp)
    if "score" not in df.columns:
        print("[error] Beklenen kolon 'score' yok:", inp)
        sys.exit(1)

    # knots
    P, T = zip(*cfg["knots"])
    P = list(P); T = list(T)

    df = df.copy()
    df["score_cal"] = pct_map(df["score"].astype(float), P, T)

    # band'ı yeniden ata (spec bands aynı kalsın)
    # bands'ı configs/score_spec.yaml'dan okuyalım
    with open(ROOT/"configs/score_spec.yaml","r",encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    bands = spec["bands"]

    # küçük yardımcı
    def to_band(s, bands_cfg=bands):
        items = sorted(bands_cfg.items(), key=lambda kv: float(kv[1][0]))
        for name, (lo, hi) in items:
            if float(lo) <= s <= float(hi):
                return name
        return "unknown"

    df["band_cal"] = df["score_cal"].apply(to_band)

    # confidence aynen taşınır; istersen burada da kural koyabiliriz.
    df.to_parquet(outp, index=False)
    print(f"[ok] Kalibre skor → {outp.as_posix()} (n={len(df)})")

    # küçük özet
    print("[info] band_cal dağılımı:")
    print(df["band_cal"].value_counts())

if __name__ == "__main__":
    main()
