import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def _ibi_from_hr(hr: pd.Series) -> np.ndarray:
    # HR bpm -> IBI (s)
    hr = hr.dropna().values
    if len(hr) == 0: 
        return np.array([])
    ibi = 60.0 / np.clip(hr, 1e-3, None)
    return ibi

def _ibi_from_bvp(bvp: pd.Series, fs: float, prom: float, min_hr: int, max_hr: int) -> np.ndarray:
    x = bvp.dropna().values
    if len(x) < int(fs*10):  # en az ~10 sn
        return np.array([])
    # minimum tepe aralığı (örnek)
    min_dist = max(1, int(fs * (60.0 / max_hr)))
    peaks, _ = find_peaks(x, distance=min_dist, prominence=prom)
    if len(peaks) < 2:
        return np.array([])
    # tepe araları -> IBI (s)
    d = np.diff(peaks) / fs
    # uçları kırp
    d = d[(d > 60.0/max_hr*0.5) & (d < 60.0/min_hr*2.0)]
    return d

def _time_domain_hrv(ibi_s: np.ndarray) -> dict:
    if ibi_s.size < 2:
        return {"sdnn": np.nan, "rmssd": np.nan, "pnn50": np.nan}
    sdnn  = float(np.std(ibi_s, ddof=1))      # s
    diff  = np.diff(ibi_s)
    rmssd = float(np.sqrt(np.mean(diff**2)))  # s
    pnn50 = float(np.mean(np.abs(diff) > 0.05))  # oran
    return {"sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50}

def extract_ppg_feats(win: dict, fs: float, cfg: dict) -> dict:
    """
    win: {"hr": pd.Series?, "bvp": pd.Series?}
    """
    feats = {"hr_mean": np.nan, "hr_std": np.nan, "sdnn": np.nan, "rmssd": np.nan, "pnn50": np.nan}
    have_hr  = "hr"  in win and isinstance(win["hr"],  pd.Series) and win["hr"].size  > 0
    have_bvp = "bvp" in win and isinstance(win["bvp"], pd.Series) and win["bvp"].size > 0

    ibi = np.array([])
    if cfg.get("use_hr_if_available", True) and have_hr:
        hrv = win["hr"].astype(float)
        feats["hr_mean"] = float(hrv.mean())
        feats["hr_std"]  = float(hrv.std(ddof=1)) if hrv.size > 1 else 0.0
        ibi = _ibi_from_hr(hrv)
    elif have_bvp:
        bvp = win["bvp"].astype(float)
        # HR tahmini için BVP tepe tespiti
        ibi = _ibi_from_bvp(
            bvp, fs,
            prom=cfg.get("peak_prominence", 0.01),
            min_hr=cfg.get("min_hr_bpm", 40),
            max_hr=cfg.get("max_hr_bpm", 180),
        )
        if ibi.size > 0:
            hr_from_ibi = 60.0 / ibi
            feats["hr_mean"] = float(np.mean(hr_from_ibi))
            feats["hr_std"]  = float(np.std(hr_from_ibi, ddof=1))
    # HRV (zaman alanı)
    feats.update(_time_domain_hrv(ibi))
    return feats
