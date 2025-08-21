import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # regex eşleşme
    for k, v in cols.items():
        for cand in candidates:
            if re.fullmatch(cand.lower(), k):
                return v
    return None

def load_generic_csv(path: str | Path) -> pd.DataFrame:
    # virgül / noktalı virgül / tab ayırıcılarını sırayla dene
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 1:
                return df
        except Exception:
            pass
    # başlık yoksa tek kolonu 'value' kabul et
    try:
        df = pd.read_csv(path, header=None)
        if df.shape[1] == 1:
            df.columns = ["value"]
        return df
    except Exception as e:
        raise e

# -------------------- Empatica E4 yardımcıları --------------------

_E4_KEYS = {
    "bvp": ("bvp",),
    "eda": ("eda","gsr"),
    "temp": ("temp","temperature"),
    "hr": ("hr","heart"),
    "acc": ("acc","accelerometer"),
}

def e4_key_from_filename(name: str) -> Optional[str]:
    n = name.lower()
    # E4_BVP.csv, BVP.csv vb.
    if n.endswith(".csv"):
        n0 = n[:-4]
    else:
        n0 = n
    for key, hints in _E4_KEYS.items():
        for h in hints:
            if f"_{h}" in n0 or n0.endswith(h) or h in n0.split("_"):
                return key
    # "E4-" veya "-E4" içeren durumlar
    if "e4" in n:
        for key, hints in _E4_KEYS.items():
            for h in hints:
                if h in n:
                    return key
    return None

def load_empatica_e4(path: str | Path, key: str) -> Tuple[pd.DataFrame, float]:
    """
    E4 CSV biçimi:
      satır 1: epoch başlangıcı (unix saniye)
      satır 2: örnekleme hızı (fs)
      satır 3+: veriler (ACC için 3 kolon; diğerleri 1 kolon)
    """
    arr = pd.read_csv(path, header=None)
    if arr.shape[0] < 3:
        raise ValueError("E4 dosyası çok kısa veya boş.")

    t0 = float(arr.iloc[0, 0])
    fs = float(arr.iloc[1, 0])

    if key == "acc":
        # 3 kolon beklenir
        data = arr.iloc[2:, 0:3].astype(float)
        data.columns = ["accx", "accy", "accz"]
    else:
        data = arr.iloc[2:, 0].astype(float).to_frame(name=key)

    n = len(data)
    if n == 0:
        raise ValueError("E4 dosyasında veri yok.")
    times = t0 + (np.arange(n, dtype=float) / fs)
    idx = pd.to_datetime(times, unit="s", origin="unix", utc=True)

    data.index = pd.DatetimeIndex(idx)
    data = data[~data.index.duplicated(keep="first")].sort_index()
    return data, fs

# -------------------- Zaman ekseni --------------------

def _to_datetime_smart(s: pd.Series) -> pd.Series:
    """Unix zamanını akıllı birim tahminiyle datetime'a çevir."""
    s2 = s.copy()
    if pd.api.types.is_numeric_dtype(s2):
        m = pd.Series(s2).dropna()
        if len(m) == 0:
            return pd.to_datetime(s2, errors="coerce", utc=True, infer_datetime_format=True)
        med = float(m.median())
        # ns ~ 1e18, us ~ 1e15, ms ~ 1e12, s ~ 1e9
        if med > 1e17:
            unit = "ns"
        elif med > 1e14:
            unit = "us"
        elif med > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(s2, unit=unit, origin="unix", utc=True, errors="coerce")
    # string/datetime
    return pd.to_datetime(s2, errors="coerce", utc=True, infer_datetime_format=True)

def ensure_datetime_index(df: pd.DataFrame, time_col: Optional[str], fs_guess: Optional[float]) -> pd.DataFrame:
    """
    1) time_col varsa: akıllı şekilde datetime'a çevir; NaT'leri düş.
    2) Eğer %50'den fazlası NaT ise veya yoksa: fs_guess ile sentetik zaman.
    3) Index: UTC DatetimeIndex, sıralı ve tekil.
    """
    df = df.copy()

    ts = None
    if time_col and time_col in df.columns:
        ts_try = _to_datetime_smart(df[time_col])
        if ts_try.notna().mean() >= 0.5:
            df = df.loc[ts_try.notna()].copy()
            ts = ts_try[ts_try.notna()]

    if ts is None:
        if not fs_guess or fs_guess <= 0:
            raise ValueError("Zaman kolonu yok ve fs_guess bilinmiyor.")
        n = len(df)
        rng = pd.to_datetime(pd.Series(range(n)) / float(fs_guess), unit="s", origin="unix", utc=True)
        ts = rng

    df.index = pd.DatetimeIndex(ts)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df

def to_4hz(df: pd.DataFrame, target_fs: float, method: str = "linear") -> pd.DataFrame:
    # 'L' yerine 'ms'
    period_ms = int(round(1000.0 / target_fs))
    rule = f"{period_ms}ms"
    out = df.resample(rule).mean()
    out = out.interpolate(method=method, limit_direction="both")
    return out
