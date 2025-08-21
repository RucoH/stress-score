# src/stresscore/scoring/bands.py
from typing import Dict, Tuple

def to_band(score: float, bands_cfg: Dict[str, Tuple[float, float]]) -> str:
    """0–100 arası skoru, config'teki aralıklara göre banda çevirir."""
    try:
        s = float(score)
    except Exception:
        return "unknown"

    # Sözlük sırası garantili değilse alt sınırına göre sırala
    items = list(bands_cfg.items())
    try:
        items = sorted(items, key=lambda kv: float(kv[1][0]))
    except Exception:
        pass

    for name, rng in items:
        lo, hi = float(rng[0]), float(rng[1])
        if lo <= s <= hi:
            return str(name)
    return "unknown"

__all__ = ["to_band"]
