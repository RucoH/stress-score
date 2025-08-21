# src/stresscore/scoring/overrides.py
from typing import Dict, List, Tuple

def apply_overrides(score: float,
                    temp_c: float | None,
                    hr_bpm: float | None,
                    artifact_ratio: float | None,
                    cfg: Dict) -> Tuple[float, str, List[str]]:
    """
    Model çıktısı 'score'u kural tabanlı güvenlik korumaları ile ayarlar.
    Döndürür: (yeni_skor, confidence, reasons_list)
    """
    reasons: List[str] = []
    conf = "high"
    ov = cfg.get("overrides", {})

    # Ateş/sıcaklık
    fv = ov.get("fever", {})
    if temp_c is not None:
        high_c = float(fv.get("high_temp_floor_c", 39.0))
        high_floor = float(fv.get("high_temp_floor_score", 85))
        if temp_c >= high_c:
            score = max(score, high_floor)
            reasons.append("fever_high")
            conf = "med"
        rng = fv.get("add_in_range_c", [38.0, 38.9])
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
            if lo <= temp_c <= hi:
                add = float(fv.get("add_in_range_score", 20))
                score = min(100.0, score + add)
                reasons.append("fever_moderate")
                conf = "med"
        low_c = float(fv.get("low_temp_flag_c", 35.0))
        if temp_c <= low_c:
            score = min(100.0, score + float(fv.get("low_temp_add_score", 15)))
            reasons.append("temp_low_flag")
            conf = "low"

    # Kalp atım hızı
    hr_cfg = ov.get("heart_rate", {})
    if hr_bpm is not None:
        if (hr_bpm >= float(hr_cfg.get("high_bpm", 120))) or (hr_bpm <= float(hr_cfg.get("low_bpm", 45))):
            score = min(100.0, score + float(hr_cfg.get("add_score", 15)))
            reasons.append("hr_out_of_range")
            conf = "med"

    # Veri kalitesi (ACC)
    q = ov.get("quality", {})
    if artifact_ratio is not None and artifact_ratio >= float(q.get("artifact_ratio_warn", 0.25)):
        reasons.append("low_quality")
        conf = "low"

    return float(score), conf, reasons

__all__ = ["apply_overrides"]
