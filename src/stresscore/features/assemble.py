import pandas as pd
from typing import Dict
from .ppg import extract_ppg_feats
from .eda import extract_eda_feats
from .temp import extract_temp_feats
from .acc import extract_acc_quality

def compute_window_features(win_data: Dict[str, pd.DataFrame], fs: float, cfg: dict) -> dict:
    out = {}
    # PPG/HR
    ppg_cfg = cfg.get("ppg", {})
    if "hr" in win_data or "bvp" in win_data:
        out.update(extract_ppg_feats({k: win_data.get(k) for k in ["hr","bvp"]}, fs, ppg_cfg))
    else:
        out.update({"hr_mean": None, "hr_std": None, "sdnn": None, "rmssd": None, "pnn50": None})

    # EDA
    if "eda" in win_data:
        out.update(extract_eda_feats(win_data["eda"].iloc[:,0] if isinstance(win_data["eda"], pd.DataFrame) else win_data["eda"], fs, cfg.get("eda", {})))
    else:
        out.update({"scl_median": None, "scl_slope": None, "scr_count": 0, "scr_amp_sum": 0.0})

    # Temp
    if "temp" in win_data:
        out.update(extract_temp_feats(win_data["temp"].iloc[:,0] if isinstance(win_data["temp"], pd.DataFrame) else win_data["temp"], fs, cfg.get("temp", {})))
    else:
        out.update({"temp_mean": None, "temp_std": None, "temp_slope": None})

    # ACC (QA)
    if "acc" in win_data:
        out.update(extract_acc_quality(win_data["acc"], fs, cfg.get("acc", {})))
    else:
        out.update({"qa_artifact_ratio": 0.0, "acc_mag_mean": None, "acc_mag_std": None})

    return out
