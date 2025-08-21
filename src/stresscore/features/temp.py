import numpy as np
import pandas as pd

def extract_temp_feats(temp: pd.Series, fs: float, cfg: dict) -> dict:
    x = temp.astype(float).values
    n = len(x)
    if n == 0:
        return {"temp_mean": np.nan, "temp_std": np.nan, "temp_slope": np.nan}
    mean = float(np.mean(x))
    std  = float(np.std(x, ddof=1)) if n > 1 else 0.0
    t = np.arange(n) / fs
    slope = float(np.polyfit(t, x, cfg.get("slope_poly_degree", 1))[0])
    return {"temp_mean": mean, "temp_std": std, "temp_slope": slope}
