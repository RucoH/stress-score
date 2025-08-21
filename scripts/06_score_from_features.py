# scripts/06_score_from_features.py
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd

# --- src/ path ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stresscore.scoring.logistic import clipped_logistic
from stresscore.scoring.combine import weighted_mean
from stresscore.scoring.overrides import apply_overrides
from stresscore.scoring.bands import to_band

SCORE_SPEC = "configs/score_spec.yaml"
FEATS_IN   = "data/processed/features.parquet"
BASE_IN    = "data/interim/baseline_stats.parquet"
OUT_PATH   = "data/processed/scores_v01.parquet"

def load_yaml(p: Path):
    import yaml
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def zscore(x, mu, sd):
    if sd is None or sd == 0 or (isinstance(sd, float) and (math.isnan(sd) or sd == 0.0)):
        sd = 1e-6
    return (x - mu) / sd

def comp_ppg_hr(row, base_row):
    vals = []
    # +hr_mean ; -(rmssd, sdnn, pnn50)
    for col, sign in [
        ("hr_mean", +1.0),
        ("rmssd",  -1.0),
        ("sdnn",   -1.0),
        ("pnn50",  -1.0),
    ]:
        x = row.get(col)
        mu = base_row.get(f"{col}__mu"); sd = base_row.get(f"{col}__sd")
        if x is not None and pd.notna(x) and (mu is not None):
            vals.append(sign * zscore(float(x), float(mu), float(sd)))
    if not vals:
        return np.nan
    return float(np.mean(vals))

def comp_eda(row, base_row):
    vals = []
    for col in ["scl_median","scl_slope","scr_count","scr_amp_sum"]:
        x = row.get(col)
        mu = base_row.get(f"{col}__mu"); sd = base_row.get(f"{col}__sd")
        if x is not None and pd.notna(x) and (mu is not None):
            vals.append(zscore(float(x), float(mu), float(sd)))
    if not vals:
        return np.nan
    return float(np.mean(vals))

def comp_temp(row, base_row):
    vals = []
    # -temp_mean (soğuma ↑ stres), +|temp_slope|
    x = row.get("temp_mean")
    mu = base_row.get("temp_mean__mu"); sd = base_row.get("temp_mean__sd")
    if x is not None and pd.notna(x) and (mu is not None):
        vals.append(-zscore(float(x), float(mu), float(sd)))
    xs = row.get("temp_slope")
    mu2 = base_row.get("temp_slope__mu"); sd2 = base_row.get("temp_slope__sd")
    if xs is not None and pd.notna(xs) and (mu2 is not None):
        vals.append(abs(zscore(float(xs), float(mu2), float(sd2))) * 0.5)
    if not vals:
        return np.nan
    return float(np.mean(vals))

def _to_float(x):
    try:
        return float(x)
    except Exception:
        try:
            return float(np.asarray(x, dtype="float64").ravel()[0])
        except Exception:
            return float("nan")

def main():
    spec = load_yaml(ROOT / SCORE_SPEC)

    # esnek okuma: logistic_transform | logistic
    log_cfg = spec.get("logistic_transform") or spec.get("logistic") or {}
    log_a = float(log_cfg.get("a", 0.9))
    log_b = float(log_cfg.get("b", 0.0))

    clip_z = float(spec.get("normalization", {}).get("clip_z", 3.0))
    feat_cfg = spec.get("features", {}) or {}
    def w(k, default): return float(feat_cfg.get(k, {}).get("weight", default))
    weights = {"ppg_hr": w("ppg_hr", 0.35), "eda": w("eda", 0.35), "temp": w("temp", 0.30)}

    feats = pd.read_parquet(ROOT / FEATS_IN)
    base  = pd.read_parquet(ROOT / BASE_IN)
    base_ix = {(r["source"], str(r["subject_id"])): r for _, r in base.iterrows()}

    rows = []
    for _, r in feats.iterrows():
        key = (r["source"], str(r["subject_id"]))
        base_row = base_ix.get(key)
        if base_row is None:
            continue

        # bileşen z'leri
        z_ppg  = comp_ppg_hr(r, base_row)
        z_eda  = comp_eda(r, base_row)
        z_temp = comp_temp(r, base_row)

        comp_scores = {}
        if pd.notna(z_ppg):
            comp_scores["ppg_hr"] = clipped_logistic(_to_float(z_ppg),  log_a, log_b, clip_z)
        if pd.notna(z_eda):
            comp_scores["eda"]    = clipped_logistic(_to_float(z_eda),  log_a, log_b, clip_z)
        if pd.notna(z_temp):
            comp_scores["temp"]   = clipped_logistic(_to_float(z_temp), log_a, log_b, clip_z)

        if not comp_scores:
            continue

        used_weights = {k: weights[k] for k in comp_scores.keys()}
        s_raw = weighted_mean(comp_scores, used_weights)

        # overrides
        temp_c = float(r["temp_mean"]) if "temp_mean" in r and pd.notna(r["temp_mean"]) else None
        hr_bpm = float(r["hr_mean"])   if "hr_mean"   in r and pd.notna(r["hr_mean"])   else None
        qa     = float(r["qa_artifact_ratio"]) if "qa_artifact_ratio" in r and pd.notna(r["qa_artifact_ratio"]) else None

        s_adj, conf, reasons = apply_overrides(s_raw, temp_c, hr_bpm, qa, spec)
        band = to_band(s_adj, spec["bands"])

        rows.append({
            "source": r["source"],
            "subject_id": r["subject_id"],
            "t_start": r["t_start"],
            "t_end":   r["t_end"],
            "score_raw": round(s_raw, 3),
            "score":     round(s_adj, 3),
            "band": band,
            "confidence": conf,
            "reasons": ";".join(reasons) if reasons else "",
            "ppg_hr_score": comp_scores.get("ppg_hr"),
            "eda_score":    comp_scores.get("eda"),
            "temp_score":   comp_scores.get("temp"),
            "qa_artifact_ratio": r.get("qa_artifact_ratio"),
            "hr_mean": r.get("hr_mean"),
            "temp_mean": r.get("temp_mean"),
        })

    if not rows:
        print("[warn] Skor üretilemedi.")
        return

    out_df = pd.DataFrame(rows)
    out_p = ROOT / OUT_PATH
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_p, index=False)
    print(f"[ok] Skorlar yazıldı → {out_p.as_posix()}  (n={len(out_df)})")

if __name__ == "__main__":
    main()
