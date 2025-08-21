# scripts/01_validate_spec.py  (v2 — ayrıntılı çıktı + sağlam kontroller)
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_DEFAULT = ROOT / "configs" / "score_spec.yaml"

def load_yaml(p: Path):
    import yaml
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # arg: --path <yaml>
    cfg_path = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--path":
        cfg_path = Path(sys.argv[2])
    else:
        cfg_path = CFG_DEFAULT

    print(f"[info] CWD           : {Path.cwd()}")
    print(f"[info] Spec path    : {cfg_path.resolve()}")

    if not cfg_path.exists():
        print("[ERROR] score_spec.yaml bulunamadı.")
        sys.exit(1)

    try:
        spec = load_yaml(cfg_path)
    except Exception as e:
        print(f"[ERROR] YAML okunamadı: {e}")
        sys.exit(1)

    if not isinstance(spec, dict) or not spec:
        print("[ERROR] YAML boş veya sözlük değil. Dosyayı doldurmalısın.")
        sys.exit(1)

    # Zorunlu bölümler
    required = ["windowing", "normalization", "features", "bands", "outputs"]
    missing = [k for k in required if k not in spec]
    if missing:
        print(f"[ERROR] Eksik bölümler: {', '.join(missing)}")
        sys.exit(1)

    # logistic_transform veya logistic
    log_cfg = spec.get("logistic_transform") or spec.get("logistic")
    if not log_cfg:
        print("[ERROR] 'logistic_transform' (veya 'logistic') bölümü yok.")
        sys.exit(1)

    a = float(log_cfg.get("a", 0.9))
    b = float(log_cfg.get("b", 0.0))
    clip_z = float(spec.get("normalization", {}).get("clip_z", 3.0))

    # Ağırlıkların toplamı
    feats = spec["features"]
    weights = {k: float(feats[k].get("weight", 0.0)) for k in feats}
    wsum = sum(weights.values())

    # Bantlar
    bands = spec["bands"]
    # Çıktılar
    outputs = spec["outputs"]

    # Rapor
    print("[ok] YAML yüklendi.")
    print(f"[ok] logistic: a={a}, b={b}, clip_z={clip_z}")
    print(f"[ok] weights : {weights}  (sum={wsum:.3f})")
    if abs(wsum - 1.0) > 1e-6:
        print("[WARN] Özellik ağırlıkları 1.0 toplam yapmıyor. (çalışır, ama düzeltmen iyi olur)")

    print(f"[ok] bands   : {bands}")
    print(f"[ok] outputs : {outputs}")

    # Başarılı çıkış kodu
    print("OK: score_spec doğrulandı.")
    sys.exit(0)

if __name__ == "__main__":
    main()
