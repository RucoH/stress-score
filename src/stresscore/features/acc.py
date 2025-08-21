# src/stresscore/features/acc.py
import numpy as np
import pandas as pd

def extract_acc_quality(acc: pd.DataFrame, fs: float, cfg: dict) -> dict:
    """
    Gravity-compensated hareket oranı:
    - acc_mag = sqrt(x^2 + y^2 + z^2) ≈ 1g (yerçekimi) + dinamik.
    - 'dinamik büyüklük' = |acc_mag - median(acc_mag)|.
    - Oran = dinamik büyüklük > threshold (g) olan örneklerin yüzdesi.
    """
    if acc is None or acc.empty:
        return {"qa_artifact_ratio": 0.0, "acc_mag_mean": np.nan, "acc_mag_std": np.nan}

    if "acc_mag" in acc.columns:
        m = acc["acc_mag"].astype(float)
    else:
        cols = [c for c in ["accx", "accy", "accz"] if c in acc.columns]
        if len(cols) < 2:
            return {"qa_artifact_ratio": 0.0, "acc_mag_mean": np.nan, "acc_mag_std": np.nan}
        m = np.sqrt((acc[cols].astype(float) ** 2).sum(axis=1))

    m_med = float(np.median(m))
    dyn = (m - m_med).abs()

    # eşik: saf ivme farkı (g). yürüyüş ~0.1–0.3g, koşu >0.3g.
    thr_dyn = float(cfg.get("movement_dyn_g", 0.20))
    ratio = float((dyn > thr_dyn).mean())

    return {
        "qa_artifact_ratio": ratio,                 # override bu alanı kullanıyor
        "acc_mag_mean": float(m.mean()),
        "acc_mag_std": float(m.std(ddof=1)) if m.size > 1 else 0.0,
    }
