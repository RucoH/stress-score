# scripts/02_inventory_raw.py
import os, re, argparse, csv
from pathlib import Path
from typing import Optional

# Sinyal adı tahmini (dosya adına göre)
SIG_PATTERNS = {
    "bvp": re.compile(r"\b(BVP|Bvp|bvp)\b|blood[_-]?volume|ppg", re.I),
    "eda": re.compile(r"\b(EDA|eda|gsr|electrodermal)\b", re.I),
    "temp": re.compile(r"\b(TEMP|Temp|temperature)\b", re.I),
    "hr": re.compile(r"\b(HR|hr|heart[_-]?rate)\b", re.I),
    "acc": re.compile(r"\b(ACC|Acc|accelerometer|accel)\b", re.I),
    "ibi": re.compile(r"\b(IBI|ibi|rr[_-]?interval)\b", re.I),
    "tags": re.compile(r"\btag[s]?\b", re.I),
    "meta": re.compile(r"\b(info|readme|read_me|about|meta)\b", re.I),
    "questionnaire": re.compile(r"\b(survey|questionnaire|form|demographic|toad)\b", re.I),
}

# Varsayılan örnekleme hızları (E4 tipik)
DEFAULT_FS = {
    "bvp": 64.0,
    "eda": 4.0,
    "temp": 4.0,
    "acc": 32.0,
    "hr": 1.0,
    # ibi olay tabanlı (Hz yok); None bırakıyoruz
}

SUBJECT_PAT = re.compile(r"[\\/](S\d{1,2})[\\/]", re.I)  # …/S2/… yakalamak için

def guess_signal(file_name: str) -> Optional[str]:
    for sig, pat in SIG_PATTERNS.items():
        if pat.search(file_name):
            return sig
    return None

def guess_fs(sig: Optional[str]) -> Optional[float]:
    if sig in DEFAULT_FS:
        return DEFAULT_FS[sig]
    return None

def guess_source(relpath: str) -> str:
    # data/raw/<source>/...
    parts = Path(relpath).parts
    if "wesad" in [p.lower() for p in parts]:
        return "wesad"
    if "biostress" in [p.lower() for p in parts]:
        return "biostress"
    return "unknown"

def extract_subject(relpath: str) -> Optional[str]:
    m = SUBJECT_PAT.search(relpath)
    return m.group(1) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data", help="Veri kökü (data/)")
    ap.add_argument("--save", default="data/interim/_inventory.csv", help="Çıktı CSV yolu")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    raw_root = data_root / "raw"
    rows = []

    for root, _, files in os.walk(raw_root):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), start=data_root)
            rel_posix = Path(rel).as_posix()
            ext = Path(fn).suffix.lower().lstrip(".")
            source = guess_source(rel_posix)
            subject = extract_subject(rel_posix)
            sig = guess_signal(fn)
            fs = guess_fs(sig)
            notes = ""
            if ext in {"zip"}:
                notes = "Compressed archive (zip) — çıkarılmalı."
            if sig is None and ext in {"csv", "pkl", "mat"}:
                notes = "Signal unknown (name-based). Will infer later by columns."

            rows.append({
                "source": source,
                "subject_id": subject or "",
                "rel_path": rel_posix,
                "file_name": fn,
                "ext": ext,
                "signal_guess": sig or "",
                "fs_guess_hz": fs if fs is not None else "",
                "notes": notes,
            })

    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source","subject_id","rel_path","file_name","ext",
            "signal_guess","fs_guess_hz","notes"
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Konsol özeti
    from collections import Counter
    c_src = Counter([r["source"] for r in rows])
    c_sig = Counter([r["signal_guess"] or "unknown" for r in rows])
    print(f"Envanter kaydedildi: {out_path.as_posix()}")
    print("Kaynak dağılımı:", dict(c_src))
    print("Sinyal tahmin dağılımı:", dict(c_sig))

if __name__ == "__main__":
    main()
