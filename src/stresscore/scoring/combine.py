# src/stresscore/scoring/combine.py
from typing import Dict

def weighted_mean(component_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    component_scores: {'ppg_hr': 72.3, 'eda': 63.1, 'temp': 55.0}  # 0–100
    weights         : {'ppg_hr': 0.35, 'eda': 0.35, 'temp': 0.30}
    Eksik bileşenlerde yalnızca mevcut anahtarlar kullanılır.
    """
    keys = [k for k in component_scores.keys() if k in weights]
    if not keys:
        return 0.0
    total_w = sum(weights[k] for k in keys)
    if total_w <= 0:
        # güvenli fallback: eşit ağırlık
        return sum(component_scores[k] for k in keys) / len(keys)
    s = sum(component_scores[k] * weights[k] for k in keys) / total_w
    return max(0.0, min(100.0, float(s)))
