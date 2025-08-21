import pandas as pd
from typing import List, Tuple

def intersect_time(spans: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> Tuple[pd.Timestamp, pd.Timestamp] | None:
    if not spans: return None
    start = max(s for s,_ in spans)
    end   = min(e for _,e in spans)
    if end <= start: return None
    return start, end

def make_window_index(start: pd.Timestamp, end: pd.Timestamp, wsec: int, hsec: int) -> pd.DataFrame:
    idx = []
    t = start
    step = pd.Timedelta(seconds=hsec)
    w = pd.Timedelta(seconds=wsec)
    while t + w <= end:
        idx.append({"t_start": t, "t_end": t + w})
        t += step
    return pd.DataFrame(idx)
