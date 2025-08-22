# scripts/16_batch_score.py
import sys
import argparse
from pathlib import Path
import subprocess

REQUIRED_ANY = [
    ("bvp", ["BVP.csv", "Bvp.csv", "*BVP*.csv"]),
]
REQUIRED_ALL = [
    ("eda",  ["EDA.csv", "*EDA*.csv"]),
    ("temp", ["TEMP.csv", "*TEMP*.csv"]),
]
OPTIONAL_ANY = [
    ("hr",   ["HR.csv", "*HR*.csv"]),
    ("acc",  ["ACC.csv", "*ACC*.csv"]),
]

def has_any(d: Path, patterns) -> bool:
    for pat in patterns:
        if list(d.glob(pat)):
            return True
    return False

def dir_is_e4_session(d: Path) -> bool:
    # EDA + TEMP zorunlu; BVP veya HR zorunlu
    ok_eda  = any(list(d.glob(p)) for p in ["EDA.csv", "*EDA*.csv"])
    ok_temp = any(list(d.glob(p)) for p in ["TEMP.csv", "*TEMP*.csv"])
    ok_bvp  = any(list(d.glob(p)) for p in ["BVP.csv", "Bvp.csv", "*BVP*.csv"])
    ok_hr   = any(list(d.glob(p)) for p in ["HR.csv", "*HR*.csv"])
    return ok_eda and ok_temp and (ok_bvp or ok_hr)

def subject_from_dir(d: Path, parent_depth: int) -> str:
    # parent_depth=1 -> d.parent.name (WESAD: S13)
    if parent_depth <= 0:
        return d.name
    if parent_depth > len(d.parents):
        return d.name
    return d.parents[parent_depth-1].name

def find_candidate_dirs(root: Path):
    # BVP olan klasörleri toplayıp E4 şartlarını filtrele
    dirs = {p.parent for p in root.rglob("BVP.csv")}
    # Eğer sadece HR var ise onu da yakala
    dirs.update({p.parent for p in root.rglob("HR.csv")})
    return sorted(d for d in dirs if dir_is_e4_session(d))

def main():
    ap = argparse.ArgumentParser(description="Batch score E4 sessions using stresscore CLI")
    ap.add_argument("--root", required=True, help="Kök klasör (altında SXX oturumları)")
    ap.add_argument("--source", default="e4infer", help="source etiketi")
    ap.add_argument("--subject-parent-depth", type=int, default=1,
                    help="subject için kaç parent yukarıdan alınacak (WESAD için 1 önerilir)")
    ap.add_argument("--report", action="store_true", help="Her oturum için rapor üret")
    ap.add_argument("--limit", type=int, default=0, help="İlk N klasörü işle (0=hepsi)")
    ap.add_argument("--dry", action="store_true", help="Komutları sadece yazdır, çalıştırma")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[error] root yok: {root}")
        sys.exit(1)

    dirs = find_candidate_dirs(root)
    if args.limit and args.limit > 0:
        dirs = dirs[:args.limit]
    if not dirs:
        print("[warn] E4 oturumu içeren klasör bulunamadı.")
        return

    print(f"[info] bulunmuş oturum klasörü sayısı: {len(dirs)}")
    for i, d in enumerate(dirs, 1):
        subj = subject_from_dir(d, args.subject_parent_depth)
        cmd = [sys.executable, "-m", "stresscore.cli", "score",
               "--input", str(d),
               "--subject", subj,
               "--source", args.source]
        if args.report:
            cmd.append("--report")
        print(f"[{i}/{len(dirs)}] {d}  →  subject={subj}")
        print("  $", " ".join(cmd))
        if not args.dry:
            subprocess.run(cmd, check=False)

    print("[ok] batch tamamlandı.")

if __name__ == "__main__":
    main()
