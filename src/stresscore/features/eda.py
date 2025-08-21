import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def extract_eda_feats(eda: pd.Series, fs: float, cfg: dict) -> dict:
    x = eda.astype(float).values
    n = len(x)
    if n == 0:
        return {"scl_median": np.nan, "scl_slope": np.nan, "scr_count": 0, "scr_amp_sum": 0.0}

    scl_median = float(np.median(x))

    # trend (lineer eğim)
    t = np.arange(n) / fs
    coeffs = np.polyfit(t, x, cfg.get("slope_poly_degree", 1))
    slope = float(coeffs[0])  # birim: µS/s (ya da dataset birimi)

    # SCR tepe sayımı (basitçe prominence ile)
    min_dist = max(1, int(cfg.get("min_distance_sec", 1.0) * fs))
    peaks, props = find_peaks(x, distance=min_dist, prominence=cfg.get("scr_prominence", 0.02))
    scr_count = int(len(peaks))
    scr_amp_sum = float(props["prominences"].sum()) if "prominences" in props else 0.0

    return {
        "scl_median": scl_median,
        "scl_slope": slope,
        "scr_count": scr_count,
        "scr_amp_sum": scr_amp_sum,
    }
