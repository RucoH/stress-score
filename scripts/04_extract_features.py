# scripts/04_extract_features.py
import sys, yaml
from pathlib import Path
import pandas as pd

# src/ path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stresscore.features.assemble import compute_window_features

def load_cfg():
    with open(ROOT/"configs/features.yaml","r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_sync(source: str, subj: str, sync_dir: Path):
    # her sinyal için opsiyonel parquet yolu
    def rp(key: str):
        p = sync_dir / f"{source}_{subj}_{key}.parquet"
        return p if p.exists() else None
    paths = {k: rp(k) for k in ["bvp","eda","temp","hr","acc"]}
    dfs = {}
    for k,p in paths.items():
        if p is not None:
            dfs[k] = pd.read_parquet(p)
    return dfs

def slice_df(df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp):
    return df.loc[(df.index >= t0) & (df.index < t1)]

def main():
    cfg = load_cfg()
    fs = float(cfg.get("fs_hz", 4.0))
    widx_path = ROOT / cfg["io"]["windows_index"]
    sync_dir   = ROOT / cfg["io"]["sync_dir"]
    out_path   = ROOT / cfg["io"]["out_parquet"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    widx = pd.read_csv(widx_path, parse_dates=["t_start","t_end"])
    rows = []

    # cache: subject bazında sync dfs
    cache = {}

    for (source, subj), g in widx.groupby(["source","subject_id"]):
        subj = str(subj) if pd.notna(subj) and subj != "" else ""
        key = (source, subj)
        if key not in cache:
            cache[key] = load_sync(source, subj, sync_dir)
        syncs = cache[key]

        need_ok = True
        if cfg["required"]["need_eda"] and "eda" not in syncs: need_ok = False
        if cfg["required"]["need_temp"] and "temp" not in syncs: need_ok = False
        if cfg["required"]["need_ppg_or_hr"] and ("bvp" not in syncs and "hr" not in syncs): need_ok = False
        if not need_ok: 
            continue

        for _, r in g.iterrows():
            t0 = pd.to_datetime(r["t_start"], utc=True)
            t1 = pd.to_datetime(r["t_end"],   utc=True)

            win_data = {}
            for k, df in syncs.items():
                sl = slice_df(df, t0, t1)
                if not sl.empty:
                    # tek kolonlu DataFrame'leri Series olarak geçir
                    if isinstance(sl, pd.DataFrame) and sl.shape[1] == 1:
                        win_data[k] = sl.iloc[:,0]
                    else:
                        win_data[k] = sl

            # gerekli sinyallerin bu pencerede de bulunup bulunmadığını kontrol
            if ("eda" not in win_data) or ("temp" not in win_data) or (("bvp" not in win_data) and ("hr" not in win_data)):
                continue

            feats = compute_window_features(win_data, fs, cfg)

            feats_row = {
                "source": source,
                "subject_id": subj,
                "t_start": t0,
                "t_end": t1,
            }
            feats_row.update(feats)
            rows.append(feats_row)

    if not rows:
        print("[warn] Özellik çıkmadı. Gereken sinyaller pencerelerde bulunamamış olabilir.")
        return

    df_out = pd.DataFrame(rows)
    df_out.to_parquet(out_path, index=False)
    print(f"[ok] Özellikler yazıldı → {out_path.as_posix()}")
    print(f"[info] Satır sayısı: {len(df_out)} | Kolon: {len(df_out.columns)}")

if __name__ == "__main__":
    main()
