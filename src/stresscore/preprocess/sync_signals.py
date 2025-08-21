import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from .utils import (
    load_generic_csv, find_col, ensure_datetime_index, to_4hz,
    e4_key_from_filename, load_empatica_e4
)

def detect_columns(df: pd.DataFrame, hints: Dict) -> Dict[str, Optional[str]]:
    return {
        "time": find_col(df, hints["time"]),
        "bvp":  find_col(df, hints["bvp"]),
        "eda":  find_col(df, hints["eda"]),
        "temp": find_col(df, hints["temp"]),
        "hr":   find_col(df, hints["hr"]),
        "accx": find_col(df, hints["accx"]),
        "accy": find_col(df, hints["accy"]),
        "accz": find_col(df, hints["accz"]),
    }

def simple_smooth(series: pd.Series, win: int = 5) -> pd.Series:
    if win <= 1:
        return series
    return series.rolling(win, min_periods=1, center=True).mean()

def process_file(path: Path, fs_guess: Optional[float], hints: Dict, target_fs: float) -> Dict[str, pd.DataFrame]:
    """
    1) Dosya adı Empatica E4 formata uyuyorsa (BVP/EDA/TEMP/HR/ACC): E4-özel okuyucu.
    2) Aksi halde: genel CSV + column_hints.
    """
    out: Dict[str, pd.DataFrame] = {}

    # --- Empatica E4 otomatik algılama ---
    e4_key = e4_key_from_filename(path.name)
    if e4_key is not None:
        df_e4, e4_fs = load_empatica_e4(path, e4_key)
        # hafif smoothing ve 4Hz'e indir
        if e4_key == "bvp":
            df_e4[e4_key] = simple_smooth(df_e4[e4_key], win=7)
        if e4_key == "eda":
            df_e4[e4_key] = simple_smooth(df_e4[e4_key], win=5)

        # ACC varsa büyüklük kolonu ekleyelim (resample sonrası)
        if e4_key == "acc":
            acc = to_4hz(df_e4, target_fs)
            acc["acc_mag"] = np.sqrt((acc[["accx","accy","accz"]]**2).sum(axis=1))
            out["acc"] = acc
        else:
            out[e4_key] = to_4hz(df_e4, target_fs)

        return out  # E4 dosyaları tek sinyal döndürür

    # --- Genel CSV yolu ---
    df = load_generic_csv(path)

    # Başlık yok/tek kolon durumu: 'value' kolonunu dosya adına göre tahmin et
    if df.shape[1] == 1 and "value" in df.columns:
        lower = path.name.lower()
        if "bvp" in lower or "ppg" in lower:
            df = df.rename(columns={"value": "bvp"})
        elif "eda" in lower or "gsr" in lower:
            df = df.rename(columns={"value": "eda"})
        elif "temp" in lower or "temperature" in lower:
            df = df.rename(columns={"value": "temp"})
        elif "hr" in lower or "heart" in lower:
            df = df.rename(columns={"value": "hr"})

    cols = detect_columns(df, hints)
    time_col = cols["time"]
    df = ensure_datetime_index(df, time_col, fs_guess)

    # Tek kanallılar
    for key in ["bvp", "eda", "temp", "hr"]:
        c = cols[key] if cols[key] in df.columns else (key if key in df.columns else None)
        if c:
            sig = df[[c]].rename(columns={c: key})
            if key == "bvp":
                sig[key] = simple_smooth(sig[key], win=7)
            if key == "eda":
                sig[key] = simple_smooth(sig[key], win=5)
            out[key] = to_4hz(sig, target_fs)

    # ACC (vektör)
    cx, cy, cz = cols["accx"], cols["accy"], cols["accz"]
    if cx and cy and cz and all(c in df.columns for c in [cx, cy, cz]):
        acc = df[[cx, cy, cz]].rename(columns={cx:"accx", cy:"accy", cz:"accz"})
        acc = to_4hz(acc, target_fs)
        acc["acc_mag"] = np.sqrt((acc**2).sum(axis=1))
        out["acc"] = acc

    return out
