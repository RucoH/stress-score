# scripts/11_make_wesad_label_intervals.py
import sys
from pathlib import Path
import pandas as pd
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]
WINDEX = ROOT / "data/interim/windows_index.csv"
OUTCSV = ROOT / "configs/wesad_intervals.csv"

DEF_DURS = [
    ("baseline",   20*60),  # saniye
    ("amusement",   5*60),
    ("stress",     10*60),
    ("meditation",  7*60),
]

def main():
    if not WINDEX.exists():
        print(f"[error] bulunamadı: {WINDEX}"); sys.exit(1)
    df = pd.read_csv(WINDEX, parse_dates=["t_start","t_end"])
    df = df[df["source"]=="wesad"].copy()
    if df.empty:
        print("[error] windows_index içinde 'wesad' yok."); sys.exit(1)

    rows=[]
    for sid, g in df.groupby("subject_id"):
        t0 = g["t_start"].min()
        cur = t0
        for label, dur_s in DEF_DURS:
            start = cur
            end   = cur + timedelta(seconds=int(dur_s))
            rows.append({
                "subject_id": sid,
                "label": label,
                "start_time": start.isoformat(),
                "end_time":   end.isoformat(),
                "start_offset_s": int((start - t0).total_seconds()),
                "end_offset_s":   int((end   - t0).total_seconds()),
            })
            cur = end

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)
    print(f"[ok] Şablon oluşturuldu → {OUTCSV.as_posix()}")
    print(">> Gerekirse bu dosyadaki 'start_time' / 'end_time' değerlerini düzelt ve kaydet.")

if __name__ == "__main__":
    main()
