# scripts/09_infer_e4_session.py  (v2.2 — overrides + isotonic/percentile auto-calibration)
import sys, argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

# --- src/ path ---
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# our modules
from stresscore.preprocess.utils import e4_key_from_filename, load_empatica_e4, to_4hz
from stresscore.preprocess.make_windows import make_window_index
from stresscore.features.assemble import compute_window_features
from stresscore.scoring.logistic import clipped_logistic
from stresscore.scoring.bands import to_band
from stresscore.scoring.overrides import apply_overrides   # <-- YENİ

# config paths
CFG_PREP   = ROOT / "configs" / "preprocess.yaml"
CFG_FEATS  = ROOT / "configs" / "features.yaml"
CFG_BASE   = ROOT / "configs" / "baseline.yaml"
CFG_SCORE  = ROOT / "configs" / "score_spec.yaml"
CFG_CALIB  = ROOT / "configs" / "calibration.yaml"
MODEL_ISO  = ROOT / "models" / "isotonic_v1.joblib"

def load_yaml(p: Path):
    import yaml
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_args():
    ap = argparse.ArgumentParser(description="Infer stress score from a new Empatica E4 folder")
    ap.add_argument("--input", required=True, help="E4 klasörü (BVP.csv, EDA.csv, TEMP.csv, HR.csv, ACC.csv)")
    ap.add_argument("--subject", default="NEW", help="subject_id etiketi (ör. S99)")
    ap.add_argument("--source",  default="e4infer", help="source etiketi")
    ap.add_argument("--outdir",  default="data/processed/infer", help="çıktı kök klasörü")
    return ap.parse_args()

# ---- helpers ----
def _rank01(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)

def _z(x, mu, sd):
    if sd is None or sd == 0 or (isinstance(sd,float) and (math.isnan(sd) or sd==0.0)):
        sd = 1e-6
    return (x - mu) / sd

def _compute_baseline_stats(df_feats: pd.DataFrame, base_cfg: dict) -> dict:
    hrw = float(base_cfg["proxy"]["hr_weight"])
    scw = float(base_cfg["proxy"]["scr_weight"])
    q   = float(base_cfg["proxy"]["quantile"])

    g = df_feats.copy()
    scr_combo = (g["scr_count"].fillna(0).astype(float) + g["scr_amp_sum"].fillna(0).astype(float))
    proxy = hrw * _rank01(g["hr_mean"].fillna(g["hr_mean"].median())) + scw * _rank01(scr_combo)
    thr = proxy.quantile(q)
    base = g.loc[proxy <= thr]
    if base.empty: base = g

    stats = {}
    for col in base_cfg["features_keep"]:
        s = base[col].astype(float)
        mu = float(s.mean()) if s.notna().any() else np.nan
        sd = float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan
        stats[f"{col}__mu"] = mu
        stats[f"{col}__sd"] = sd if (sd and sd>0) else 1e-6
    return stats

def _comp_ppg_hr(row, base_row):
    vals=[]
    for col,sgn in [("hr_mean",+1.0),("rmssd",-1.0),("sdnn",-1.0),("pnn50",-1.0)]:
        x=row.get(col); mu=base_row.get(f"{col}__mu"); sd=base_row.get(f"{col}__sd")
        if x is not None and pd.notna(x) and (mu is not None): vals.append(sgn*_z(float(x),float(mu),float(sd)))
    return float(np.mean(vals)) if vals else np.nan

def _comp_eda(row, base_row):
    vals=[]
    for col in ["scl_median","scl_slope","scr_count","scr_amp_sum"]:
        x=row.get(col); mu=base_row.get(f"{col}__mu"); sd=base_row.get(f"{col}__sd")
        if x is not None and pd.notna(x) and (mu is not None): vals.append(_z(float(x),float(mu),float(sd)))
    return float(np.mean(vals)) if vals else np.nan

def _comp_temp(row, base_row):
    vals=[]
    x=row.get("temp_mean"); mu=base_row.get("temp_mean__mu"); sd=base_row.get("temp_mean__sd")
    if x is not None and pd.notna(x) and (mu is not None): vals.append(-_z(float(x),float(mu),float(sd)))
    xs=row.get("temp_slope"); mu2=base_row.get("temp_slope__mu"); sd2=base_row.get("temp_slope__sd")
    if xs is not None and pd.notna(xs) and (mu2 is not None): vals.append(abs(_z(float(xs),float(mu2),float(sd2)))*0.5)
    return float(np.mean(vals)) if vals else np.nan

def _to_float(v):
    try: return float(v)
    except Exception:
        try: return float(np.asarray(v, dtype="float64").ravel()[0])
        except Exception: return float("nan")

def load_e4_folder(input_dir: Path) -> dict:
    """Return dict: {'bvp': df, 'eda': df, 'temp': df, 'hr': df, 'acc': df} (DatetimeIndex)"""
    input_dir = Path(input_dir)
    files = list(input_dir.glob("*.csv"))
    if not files:
        files = list(input_dir.rglob("*.csv"))  # alt klasörleri de tara
    if not files:
        raise FileNotFoundError(f"E4 klasöründe CSV bulunamadı: {input_dir}")

    sigs = {}
    for p in files:
        key = e4_key_from_filename(p.name)
        if key is None:
            continue
        df, fs = load_empatica_e4(p, key)
        sigs[key] = df
    if not sigs:
        raise FileNotFoundError(f"E4 CSV’leri bulunamadı (BVP/EDA/TEMP/HR/ACC yok): {input_dir}")
    return sigs

# ---- calibration helpers ----
def apply_calibration(df_s01: pd.DataFrame, score_cfg: dict, calib_cfg: dict) -> pd.DataFrame:
    """
    Auto calibration:
      - If isotonic model exists: use it.
      - If calibrated scores are (near) constant: fallback to percentile_map.
      - Else if no model: percentile_map.
    """
    method = str(calib_cfg.get("method","auto")).lower()

    def _percentile_map(df):
        knots = calib_cfg.get("knots", [[0,10],[10,25],[25,40],[50,55],[75,70],[90,85],[100,100]])
        P, T = zip(*knots); P=list(P); T=list(T)
        x = df["score"].values.astype(float)

        xp = np.percentile(x, P).astype(float)
        if np.allclose(xp, xp[0]):
            base = x[0] if len(x) else 0.0
            xp = np.array([base-1e-6, base, base+1e-6, base+2e-6, base+3e-6, base+4e-6, base+5e-6], dtype=float)
        fp = np.array(T, dtype=float)
        xp[0] = min(xp[0], x.min()); xp[-1] = max(xp[-1], x.max())
        df["score_cal"] = np.clip(np.interp(x, xp, fp), 0, 100)
        return df

    used = "none"
    iso_ok = False
    if method in ("auto","isotonic","isotonic_if_available") and MODEL_ISO.exists():
        try:
            import joblib
            iso = joblib.load(MODEL_ISO)
            prob = iso.predict(df_s01["score"].values.astype(float))  # [0,1]
            df_s01["score_cal"] = np.clip(prob * 100.0, 0, 100)
            used = "isotonic"
            iso_ok = True
        except Exception:
            iso_ok = False

    if not iso_ok:
        df_s01 = _percentile_map(df_s01); used = "percentile"

    if np.nanstd(df_s01["score_cal"].values) < 1e-3:
        df_s01 = _percentile_map(df_s01); used = "percentile_fallback"

    df_s01["band_cal"] = df_s01["score_cal"].apply(lambda s: to_band(float(s), score_cfg["bands"]))
    df_s01.attrs["calibration_used"] = used
    return df_s01

def main():
    args = parse_args()
    inp   = Path(args.input)
    subj  = args.subject
    source= args.source
    outroot = ROOT / args.outdir
    outdir  = outroot / f"{source}_{subj}"
    outdir.mkdir(parents=True, exist_ok=True)

    # load cfgs
    prep  = load_yaml(CFG_PREP)
    feats_cfg = load_yaml(CFG_FEATS)
    base_cfg  = load_yaml(CFG_BASE)
    score_cfg = load_yaml(CFG_SCORE)
    calib_cfg = load_yaml(CFG_CALIB)

    fs_target = float(prep["target_fs_hz"])
    wsec = int(prep["window_sec"]); hsec = int(prep["hop_sec"])

    # 1) Read E4 & resample
    sigs = load_e4_folder(inp)
    resampled = {k: to_4hz(df, fs_target) for k, df in sigs.items()}

    # 2) Window index
    spans = []
    need_ppg_or_hr = (("bvp" in resampled) or ("hr" in resampled))
    need_eda = "eda" in resampled
    need_temp= "temp" in resampled
    if not (need_ppg_or_hr and need_eda and need_temp):
        missing = [x for x,ok in {"bvp/hr":need_ppg_or_hr,"eda":need_eda,"temp":need_temp}.items() if not ok]
        raise RuntimeError(f"Gerekli sinyaller eksik: {missing}")

    for key in ["bvp","hr","eda","temp"]:
        if key in resampled:
            spans.append((resampled[key].index[0], resampled[key].index[-1]))
    start = max(s for s,_ in spans); end = min(e for _,e in spans)
    widx = make_window_index(start, end, wsec, hsec)
    if widx.empty:
        raise RuntimeError("Pencere üretilemedi (ortak aralık yok).")

    # 3) Features
    rows=[]
    for _,row in widx.iterrows():
        t0=row["t_start"]; t1=row["t_end"]
        win_data={}
        for k,df in resampled.items():
            sl = df.loc[(df.index>=t0)&(df.index<t1)]
            if not sl.empty:
                if isinstance(sl, pd.DataFrame) and sl.shape[1]==1:
                    win_data[k]=sl.iloc[:,0]
                else:
                    win_data[k]=sl
        if ("eda" not in win_data) or ("temp" not in win_data) or (("bvp" not in win_data) and ("hr" not in win_data)):
            continue
        feats = compute_window_features(win_data, fs_target, feats_cfg)
        feats_row = {"source":source, "subject_id":subj, "t_start":t0, "t_end":t1}
        feats_row.update(feats)
        rows.append(feats_row)

    if not rows:
        raise RuntimeError("Bu oturumda özellik çıkmadı (pencereler boş).")

    df_feats = pd.DataFrame(rows)
    df_feats.to_parquet(outdir / "features.parquet", index=False)

    # 4) Baseline (tek subject)
    base_stats = _compute_baseline_stats(df_feats, base_cfg)

    # 5) Scoring v0.1 (ham + overrides)
    log_cfg = score_cfg.get("logistic_transform") or score_cfg.get("logistic") or {}
    log_a = float(log_cfg.get("a", 0.9)); log_b = float(log_cfg.get("b", 0.0))
    clip_z = float(score_cfg.get("normalization",{}).get("clip_z",3.0))
    weights = {
        "ppg_hr": float(score_cfg["features"]["ppg_hr"]["weight"]),
        "eda":    float(score_cfg["features"]["eda"]["weight"]),
        "temp":   float(score_cfg["features"]["temp"]["weight"]),
    }

    score_rows=[]
    for _, r in df_feats.iterrows():
        z_ppg  = _comp_ppg_hr(r, base_stats)
        z_eda  = _comp_eda(r, base_stats)
        z_temp = _comp_temp(r, base_stats)

        comp_scores={}
        if pd.notna(z_ppg):  comp_scores["ppg_hr"] = clipped_logistic(_to_float(z_ppg),  log_a, log_b, clip_z)
        if pd.notna(z_eda):  comp_scores["eda"]    = clipped_logistic(_to_float(z_eda),  log_a, log_b, clip_z)
        if pd.notna(z_temp): comp_scores["temp"]   = clipped_logistic(_to_float(z_temp), log_a, log_b, clip_z)
        if not comp_scores:  continue

        used_w = {k: weights[k] for k in comp_scores.keys()}
        tw = sum(used_w.values()) or 1.0
        s_raw = sum(comp_scores[k]*used_w[k] for k in comp_scores.keys())/tw

        # --- overrides (fever/HR/kalite) ---
        temp_c = float(r["temp_mean"]) if pd.notna(r.get("temp_mean")) else None
        hr_bpm = float(r["hr_mean"])   if pd.notna(r.get("hr_mean")) else None
        qa     = float(r["qa_artifact_ratio"]) if pd.notna(r.get("qa_artifact_ratio")) else None
        s_adj, conf, reasons = apply_overrides(float(s_raw), temp_c, hr_bpm, qa, score_cfg)

        score_rows.append({
            "source": source, "subject_id": subj,
            "t_start": r["t_start"], "t_end": r["t_end"],
            "score_raw": round(float(s_raw),3),
            "score":     round(float(s_adj),3),                 # <-- overrides sonrası skor
            "band": to_band(float(s_adj), score_cfg["bands"]), # band raw değil, adjusted üzerinden
            "confidence": conf,
            "reasons": ";".join(reasons) if reasons else "",
            "ppg_hr_score": comp_scores.get("ppg_hr"),
            "eda_score": comp_scores.get("eda"),
            "temp_score": comp_scores.get("temp"),
            "qa_artifact_ratio": r.get("qa_artifact_ratio"),
            "hr_mean": r.get("hr_mean"),
            "temp_mean": r.get("temp_mean"),
        })

    if not score_rows:
        raise RuntimeError("Skor üretilemedi (bileşen skorları boş).")

    df_s01 = pd.DataFrame(score_rows)
    df_s01.to_parquet(outdir / "scores_v01.parquet", index=False)

    # 6) Calibration (auto + fallback)
    df_s01 = apply_calibration(df_s01, score_cfg, calib_cfg)
    out_cal = outdir / "scores_v02_calibrated.parquet"
    df_s01.to_parquet(out_cal, index=False)

    # 7) Tiny report CSVs
    bc = df_s01["band_cal"].value_counts().rename_axis("band").reset_index(name="count")
    bc.to_csv(outdir / "_band_counts.csv", index=False)
    per = (df_s01.groupby(["source","subject_id"])
           .agg(n=("score_cal","size"),
                score_mean=("score_cal","mean"),
                score_std=("score_cal","std"),
                high_pct=("band_cal", lambda s: (s.isin(["high","critical"])).mean()))
           .reset_index())
    per.to_csv(outdir / "_scores_by_subject.csv", index=False)

    print(f"[ok] İnferans tamam: {outdir.as_posix()}")
    print(" - features.parquet")
    print(" - scores_v01.parquet")
    print(" - scores_v02_calibrated.parquet  (auto-calibrated)")
    print(" - _band_counts.csv, _scores_by_subject.csv")
    print("\n[band_cal dağılımı]")
    print(bc)

if __name__ == "__main__":
    main()
