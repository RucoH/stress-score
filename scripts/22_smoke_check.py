# scripts/22_smoke_check.py
from pathlib import Path
import argparse, json, subprocess, os, sys

ROOT = Path(__file__).resolve().parents[1]

def run(cmd):
    env = {**os.environ, "PYTHONUTF8":"1", "PYTHONIOENCODING":"utf-8"}
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", env=env)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise SystemExit(f"[ERR] Komut hata: {' '.join(cmd)}")
    return p.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="E4 oturum klasörü (BVP/EDA/TEMP csv)")
    ap.add_argument("--subject", default="Ssmoke")
    ap.add_argument("--source", default="e4api")
    args = ap.parse_args()

    outdir = ROOT / "data/processed/infer" / f"{args.source}_{args.subject}"
    if outdir.exists():
        # temizlemeden üzerine yazacağız; gerek yoksa geç
        pass

    print("[1/3] score çalışıyor...")
    run([sys.executable, "-m", "stresscore.cli", "score",
         "--input", args.input, "--subject", args.subject, "--source", args.source, "--report"])

    print("[2/3] quick_report...")
    scores_pq = outdir / "scores_v02_calibrated.parquet"
    run([sys.executable, str(ROOT/"scripts/07_quick_report.py"), "--path", str(scores_pq)])

    print("[3/3] özet kontrol...")
    summary_json = outdir / "_summary.json"
    assert summary_json.exists(), "_summary.json bulunamadı"
    meta = json.loads(summary_json.read_text(encoding="utf-8"))
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    print("\n[OK] Smoke test tamam.")
    print(f"Çıktılar: {outdir}")

if __name__ == "__main__":
    main()
