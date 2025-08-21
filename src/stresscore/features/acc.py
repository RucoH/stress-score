import numpy as np
import pandas as pd

def extract_acc_quality(acc: pd.DataFrame, fs: float, cfg: dict) -> dict:
    if acc is None or acc.empty:
        return {"qa_artifact_ratio": 0.0, "acc_mag_mean": np.nan, "acc_mag_std": np.nan}
    if "acc_mag" in acc.columns:
        m = acc["acc_mag"].astype(float)
    else:
        cols = [c for c in ["accx","accy","accz"] if c in acc.columns]
        if len(cols) < 2:
            return {"qa_artifact_ratio": 0.0, "acc_mag_mean": np.nan, "acc_mag_std": np.nan}
        m = np.sqrt((acc[cols].astype(float)**2).sum(axis=1))
    thr = float(cfg.get("movement_g", 1.4))
    ratio = float((m > thr).mean())
    return {
        "qa_artifact_ratio": ratio,
        "acc_mag_mean": float(m.mean()),
        "acc_mag_std": float(m.std(ddof=1)) if m.size > 1 else 0.0,
    }
