# scripts/19_pipeline_cohort.py
import sys, subprocess, datetime, zipfile
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=False).returncode

def zip_outputs(make_zip: bool, zip_name: str):
    if not make_zip: 
        return
    outdir = ROOT / "data" / "processed"
    cohort = outdir / "cohort"
    infer  = outdir / "infer"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zpath = (outdir / f"{zip_name}_{ts}.zip").resolve()

    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # cohort çıktıları
        if cohort.exists():
            for p in cohort.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(outdir))
        # infer altındaki rapor png/json/csv/parquet’ler
        if infer.exists():
            for p in infer.rglob("*"):
                if p.is_file():
                    # çok büyük ham dosya yok; hepsini koymak çoğu kez yararlı
                    zf.write(p, p.relative_to(outdir))
    print(f"[ok] ZIP paket hazır → {zpath.as_posix()}")

def main():
    ap = argparse.ArgumentParser(description="Batch → Cohort Summary → Excel tek komut pipeline")
    ap.add_argument("--root", required=True, help="WESAD/E4 root (içinde SXX klasörleri)")
    ap.add_argument("--source", default="e4infer", help="source etiketi (varsayılan: e4infer)")
    ap.add_argument("--subject-parent-depth", type=int, default=1, help="subject id için parent depth (WESAD=1)")
    ap.add_argument("--limit", type=int, default=0, help="ilk N klasörü işle (0=hepsi)")
    ap.add_argument("--report", action="store_true", help="Her oturum için PNG+JSON raporu üret")
    ap.add_argument("--skip-batch", action="store_true", help="Batch adımını atla")
    ap.add_argument("--skip-summary", action="store_true", help="Cohort özet adımını atla")
    ap.add_argument("--skip-excel", action="store_true", help="Excel çıktı adımını atla")
    ap.add_argument("--zip", dest="make_zip", action="store_true", help="Sonuçları tek ZIP olarak paketle")
    ap.add_argument("--zip-name", default="deliverable", help="ZIP ismi prefix’i (tarih eklenecek)")
    args = ap.parse_args()

    # 1) Batch skor
    if not args.skip_batch:
        cmd = [
            sys.executable, str(SCRIPTS / "16_batch_score.py"),
            "--root", args.root,
            "--source", args.source,
            "--subject-parent-depth", str(args.subject_parent_depth)
        ]
        if args.limit:   cmd += ["--limit", str(args.limit)]
        if args.report:  cmd += ["--report"]
        rc = run(cmd)
        if rc != 0:
            print("[warn] batch adımı hata kodu ile bitti:", rc)

    # 2) Cohort özet
    if not args.skip_summary:
        rc = run([sys.executable, str(SCRIPTS / "17_cohort_summary.py")])
        if rc != 0:
            print("[warn] cohort summary adımı hata kodu ile bitti:", rc)

    # 3) Excel
    if not args.skip_excel:
        rc = run([sys.executable, str(SCRIPTS / "18_export_excel.py")])
        if rc != 0:
            print("[warn] excel export adımı hata kodu ile bitti:", rc)

    # 4) ZIP paket
    zip_outputs(args.make_zip, args.zip_name)

    print("\n[ok] pipeline tamamlandı.")
    print(" - Batch skorlar → data/processed/infer/e4infer_*")
    print(" - Cohort özet  → data/processed/cohort/ (_subject_summary.csv, _cohort_band_counts.*, all_scores.parquet)")
    print(" - Excel rapor  → data/processed/cohort/summary.xlsx")

if __name__ == "__main__":
    main()
