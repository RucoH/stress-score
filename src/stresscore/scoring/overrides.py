# src/stresscore/scoring/overrides.py
import math

def _get(d, key, default=None):
    return (d or {}).get(key, default)

def apply_overrides(score, temp_c=None, hr_bpm=None, qa_artifact_ratio=None, spec=None):
    """
    Basit kural seti:
      - Ateş (fever): 38.0+ uyarı, 39.5+ yüksek → skora ekleme + güven = high
      - Kalp hızı (HR): aralık dışı → skora küçük ekleme + güven = med
      - Kalite (QA): artifact_ratio_warn eşik üstü → skoru değiştirme, güven = med (low'a düşürme!)
    Döndürür: (adjusted_score, confidence, reasons[])
    """
    s = float(score)
    conf = "med"     # <--- varsayılanı 'med' yaptık
    reasons = []

    spec = spec or {}
    ov = spec.get("overrides", {}) if isinstance(spec, dict) else {}

    # --- FEVER ---
    fever = _get(ov, "fever", {})
    t_warn = float(_get(fever, "warn_temp_c", float("inf")))
    t_high = float(_get(fever, "high_temp_c", float("inf")))
    add_warn = float(_get(fever, "add_score_warn", 0.0))
    add_high = float(_get(fever, "add_score_high", 0.0))
    conf_fever = _get(fever, "set_confidence", "high")

    if temp_c is not None and not (isinstance(temp_c, float) and math.isnan(temp_c)):
        if temp_c >= t_high:
            s = min(100.0, s + add_high)
            conf = conf_fever
            reasons.append("fever_high")
        elif temp_c >= t_warn:
            s = min(100.0, s + add_warn)
            conf = conf_fever
            reasons.append("fever_warn")

    # --- HEART RATE ---
    hr = _get(ov, "heart_rate", {})
    hi = float(_get(hr, "high_bpm", float("inf")))
    lo = float(_get(hr, "low_bpm", float("-inf")))
    add_hr = float(_get(hr, "add_score", 0.0))
    conf_hr = _get(hr, "set_confidence", "med")

    if hr_bpm is not None and not (isinstance(hr_bpm, float) and math.isnan(hr_bpm)):
        if hr_bpm >= hi or hr_bpm <= lo:
            s = min(100.0, s + add_hr)
            conf = conf_hr
            reasons.append("hr_out_of_range")

    # --- QUALITY (QA) ---
    # Not: Burada asla 'low' a düşürmüyoruz; med/higher kalıyor.
    qual = _get(ov, "quality", {})
    qa_thr = float(_get(qual, "artifact_ratio_warn", 1.01))  # 1.01 => pratikte kapalı
    conf_q = _get(qual, "set_confidence", "med")

    if qa_artifact_ratio is not None and not (isinstance(qa_artifact_ratio, float) and math.isnan(qa_artifact_ratio)):
        if qa_artifact_ratio >= qa_thr:
            conf = conf_q
            reasons.append("qa_warn")

    return s, conf, reasons
