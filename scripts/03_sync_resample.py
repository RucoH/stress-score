# scripts/03_sync_resample.py  (v1.1)
import sys
from pathlib import Path

# --- src/ layout'ını ekle (kök/scripts/.. -> kök/src) ---
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd
import yaml
from pathlib import Path

# Artık 'stresscore' paketinden import ediyoruz
from stresscore.preprocess.sync_signals import process_file
from stresscore.preprocess.make_windows import intersect_time, make_window_index

CFG_PREP = "configs/preprocess.yaml"
INVENTORY = "data/interim/_inventory.csv"

def load_prep_cfg(path: str):
    with open(path,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    prep = load_prep_cfg(CFG_PREP)
    target_fs = float(prep["target_fs_hz"])
    wsec = int(prep["window_sec"]); hsec = int(prep["hop_sec"])
    hints = prep["column_hints"]
    sync_dir = Path(prep["io_layout"]["sync_dir"])
    sync_dir.mkdir(parents=True, exist_ok=True)

    inv = pd.read_csv(INVENTORY)
    inv = inv[inv["ext"].str.lower().eq("csv")]  # sadece csv

    outputs = []

    for (source, subj), g in inv.groupby(["source","subject_id"], dropna=False):
        if not subj:
            subj = ""
        print(f"[group] {source} / {subj or '(no-subject)'}")

        sig_store = {}

        for _,row in g.iterrows():
            rel = row["rel_path"]
            fs_guess = float(row["fs_guess_hz"]) if str(row["fs_guess_hz"]).strip() != "" else None
            path = ROOT / "data" / Path(rel)
            if not path.exists():
                continue
            try:
                sig_dfs = process_file(path, fs_guess, prep["column_hints"], target_fs)
            except Exception as e:
                print(f"[warn] skip {path.name}: {e}")
                continue

            for key, df in sig_dfs.items():
                prev = sig_store.get(key)
                if prev is None or len(df) > len(prev):
                    sig_store[key] = df

        if not sig_store:
            continue

        for key, df in sig_store.items():
            outp = sync_dir / f"{source}_{subj}_{key}.parquet"
            df.to_parquet(outp, index=True)

        spans = []
        need_bvp_or_hr = ("bvp" in sig_store) or ("hr" in sig_store)
        need_eda = "eda" in sig_store
        need_temp = "temp" in sig_store

        if need_bvp_or_hr and need_eda and need_temp:
            for key in ["bvp","hr","eda","temp"]:
                if key in sig_store:
                    spans.append((sig_store[key].index[0], sig_store[key].index[-1]))
        else:
            continue

        inter = intersect_time(spans)
        if inter is None:
            continue
        widx = make_window_index(inter[0], inter[1], wsec, hsec)
        if len(widx)==0:
            continue

        widx["source"] = source
        widx["subject_id"] = subj
        outputs.append(widx)

    if outputs:
        win_all = pd.concat(outputs, ignore_index=True)
        win_all = win_all[["source","subject_id","t_start","t_end"]]
        win_all.to_csv(Path(prep["io_layout"]["windows_index"]), index=False)
        print(f"[ok] windows_index → {prep['io_layout']['windows_index']}")
    else:
        print("[warn] pencere üretilemedi (yeterli sinyal bulunamadı).")

if __name__ == "__main__":
    main()
