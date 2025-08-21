# src/stresscore/scoring/logistic.py
import math

def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def clipped_logistic(z, a: float = 1.0, b: float = 0.0, clip_z: float = 3.0) -> float:
    # robust float dönüştürme
    try:
        z = float(z)
    except Exception:
        try:
            import numpy as np
            z = float(np.asarray(z, dtype="float64").ravel()[0])
        except Exception:
            return float("nan")
    a = float(a); b = float(b); clip_z = float(clip_z)
    # clip
    if z > clip_z:  z = clip_z
    if z < -clip_z: z = -clip_z
    return 100.0 * _stable_sigmoid(a * z + b)

__all__ = ["clipped_logistic"]
