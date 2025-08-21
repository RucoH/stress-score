# scripts/05_build_baseline.py
import sys, yaml
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def load_cfg():
    with open(ROOT/"configs/baseline.yaml","r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def rank01(x: pd.Series) -> pd.Series:
    return x.rank(method="average", pct=True)

def main():
    cfg = load_cfg()
    feats_path = ROOT/cfg["io"]["features_parquet"]
    out_path   = ROOT/cfg["io"]["out_parquet"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(feats_path)
    rows = []
    for (source, subj), g in df.groupby(["source","subject_id"], dropna=False):
        g = g.copy()
        # proxy = 0.6*rank(hr_mean) + 0.4*rank(scr_count + scr_amp_sum)
        scr_combo = (g["scr_count"].fillna(0).astype(float) +
                     g["scr_amp_sum"].fillna(0).astype(float))
        proxy = (cfg["proxy"]["hr_weight"] * rank01(g["hr_mean"].fillna(g["hr_mean"].median())) +
                 cfg["proxy"]["scr_weight"] * rank01(scr_combo))
        q = float(cfg["proxy"]["quantile"])
        thr = proxy.quantile(q)

        base = g.loc[proxy <= thr]
        if base.empty:
            # güvenli fallback: tümünden hesapla
            base = g

        rec = {"source": source, "subject_id": str(subj) if pd.notna(subj) else ""}
        for col in cfg["features_keep"]:
            s = base[col].astype(float)
            mu = float(s.mean()) if s.notna().any() else np.nan
            sd = float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan
            rec[f"{col}__mu"] = mu
            rec[f"{col}__sd"] = sd if (sd and sd>0) else 1e-6
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"[ok] Baseline istatistikleri → {out_path.as_posix()} (n={len(out)})")

if __name__ == "__main__":
    main()
