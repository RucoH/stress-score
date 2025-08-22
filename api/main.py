# api/main.py — v0.3.1
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import unquote
import subprocess, sys, os, json
import mimetypes  # <-- eklendi

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# --------------------------------------------------------------------------------------
# Yol kurulumları
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INFER = PROCESSED / "infer"

app = FastAPI(title="Stress Score API", version="0.3.1")

# statik ve şablonlar
app.mount("/static", StaticFiles(directory=ROOT / "api" / "static"), name="static")
templates = Jinja2Templates(directory=str(ROOT / "api" / "templates"))


# --------------------------------------------------------------------------------------
# Yardımcılar
# --------------------------------------------------------------------------------------
def _run(cmd: List[str]):
    """CLI çalıştır (UTF-8 güvenli)."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env)


def _find_scores(outdir: Path) -> Path:
    """Çıktı klasöründe scores parquet bul."""
    for name in ("scores_v02_calibrated.parquet", "scores_v01.parquet"):
        p = outdir / name
        if p.exists():
            return p
    raise FileNotFoundError("scores parquet bulunamadı")


def summarize_scores(scores_path: Path) -> Dict[str, Any]:
    """Parquet dosyasından özet çıkar."""
    df = pd.read_parquet(scores_path)
    score_col = "score_cal" if "score_cal" in df.columns else "score"
    band_col = "band_cal" if "band_cal" in df.columns else "band"

    t0 = str(pd.to_datetime(df["t_start"]).min())
    t1 = str(pd.to_datetime(df["t_end"]).max())
    counts = df[band_col].value_counts(dropna=False).rename_axis("band").reset_index(name="count")
    counts = dict(zip(counts["band"].astype(str), counts["count"].astype(int)))

    return {
        "n_windows": int(len(df)),
        "time_range": [t0, t1],
        "score_col": score_col,
        "band_col": band_col,
        "band_counts": counts,
    }


def _session_dir_from_name(name: str) -> Optional[Path]:
    p = INFER / name
    return p if p.exists() and p.is_dir() else None


def _list_sessions() -> List[Dict[str, str]]:
    """infer/* altındaki oturumları listele."""
    out: List[Dict[str, str]] = []
    if not INFER.exists():
        return out
    for p in sorted(INFER.iterdir()):
        if not p.is_dir():
            continue
        # ad deseni: <source>_<subject>
        name = p.name
        if "_" not in name:
            continue
        source, subject = name.split("_", 1)
        item = {"name": name, "source": source, "subject": subject, "dir": str(p.relative_to(ROOT))}
        try:
            s = summarize_scores(_find_scores(p))
            item["n"] = str(s["n_windows"])
            item["range"] = f'{s["time_range"][0]} → {s["time_range"][1]}'
        except Exception:
            item["n"] = "?"
            item["range"] = "-"
        out.append(item)
    return out


def _norm_top_rows(top_csv: Path) -> List[Dict[str, Any]]:
    """_top_segments.csv'i esnek biçimde oku ve normalize et."""
    if not top_csv.exists():
        return []
    try:
        df = pd.read_csv(top_csv)
    except Exception:
        return []

    # muhtemel kolon adları
    score_cands = ["score", "score_cal", "score_raw"]
    band_cands = ["band", "band_cal"]
    reason_cands = ["reasons", "reason", "notes", "why"]

    score_col = next((c for c in score_cands if c in df.columns), None)
    band_col = next((c for c in band_cands if c in df.columns), None)
    reason_col = next((c for c in reason_cands if c in df.columns), None)

    # t_start / t_end yoksa alternatif isimlerden üret
    if "t_start" not in df.columns and "start" in df.columns:
        df["t_start"] = df["start"]
    if "t_end" not in df.columns and "end" in df.columns:
        df["t_end"] = df["end"]

    rows: List[Dict[str, Any]] = []
    for _, x in df.head(10).iterrows():
        val = None
        if score_col and pd.notna(x.get(score_col)):
            try:
                val = float(str(x.get(score_col)).strip().replace("%", ""))
            except Exception:
                val = None
        rows.append(
            {
                "t_start": str(x.get("t_start", "")),
                "t_end": str(x.get("t_end", "")),
                "score": val,
                "band": str(x.get(band_col, "")) if band_col else "",
                "reasons": str(x.get(reason_col, "")) if reason_col else "",
            }
        )
    return rows


# --------------------------------------------------------------------------------------
# JSON API’ler
# --------------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/score-e4")
def score_e4(
    input: str = Query(..., description="E4 CSV klasör yolu (BVP/EDA/TEMP)"),
    subject: str = Query("NEW"),
    source: str = Query("e4api"),
    report: bool = Query(True),
):
    input_dir = Path(input)
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Klasör bulunamadı: {input_dir}")

    outdir = PROCESSED / "infer" / f"{source}_{subject}"
    cmd = [
        sys.executable,
        "-m",
        "stresscore.cli",
        "score",
        "--input",
        str(input_dir),
        "--subject",
        subject,
        "--source",
        source,
    ]
    if report:
        cmd.append("--report")

    res = _run(cmd)
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail=f"CLI hata:\n{res.stderr or res.stdout}")

    scores = _find_scores(outdir)
    summary = summarize_scores(scores)

    # varsa _summary.json'u da iliştir
    summ_json = outdir / "_summary.json"
    if summ_json.exists():
        try:
            summary["session_summary"] = json.loads(summ_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    files = {
        "scores": str(scores.relative_to(ROOT)),
        "features": str((outdir / "features.parquet").relative_to(ROOT)) if (outdir / "features.parquet").exists() else None,
        "timeline_png": str((outdir / "_session_timeline.png").relative_to(ROOT)) if (outdir / "_session_timeline.png").exists() else None,
        "band_counts_csv": str((outdir / "_band_counts.csv").relative_to(ROOT)) if (outdir / "_band_counts.csv").exists() else None,
        "top_segments_csv": str((outdir / "_top_segments.csv").relative_to(ROOT)) if (outdir / "_top_segments.csv").exists() else None,
        "summary_json": str((outdir / "_summary.json").relative_to(ROOT)) if (outdir / "_summary.json").exists() else None,
    }

    return {
        "ok": True,
        "subject": subject,
        "source": source,
        "output_dir": str(outdir.relative_to(ROOT)),
        "summary": summary,
        "files": files,
    }


@app.post("/cohort-pipeline")
def cohort_pipeline(
    root: str = Query(..., description="WESAD/E4 kök klasörü (içinde SXX klasörleri)"),
    report: bool = Query(True, description="Her oturum için PNG/JSON rapor üret"),
    make_zip: bool = Query(True, description="Çıktıları ZIP paketle"),
    source: str = Query("e4api"),
    subject_parent_depth: int = Query(1, description="Subject id için parent depth (WESAD=1)"),
    limit: int = Query(0, description="İlk N klasörü işle (0=hepsi)"),
):
    root_p = Path(root)
    if not root_p.exists():
        raise HTTPException(status_code=404, detail=f"Kök klasör yok: {root_p}")

    cmd = [
        sys.executable,
        str((ROOT / "scripts" / "19_pipeline_cohort.py").resolve()),
        "--root",
        str(root_p),
        "--source",
        source,
        "--subject-parent-depth",
        str(subject_parent_depth),
    ]
    if report:
        cmd.append("--report")
    if make_zip:
        cmd.append("--zip")
    if limit:
        cmd += ["--limit", str(limit)]

    res = _run(cmd)
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Pipeline hata:\n{res.stderr or res.stdout}")

    cohort = PROCESSED / "cohort"
    summary_xlsx = cohort / "summary.xlsx"
    band_png = cohort / "_cohort_band_counts.png"
    subj_csv = cohort / "_subject_summary.csv"
    zips = sorted(PROCESSED.glob("deliverable_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    zip_rel = str(zips[0].relative_to(ROOT)) if zips else None

    return {
        "ok": True,
        "cohort_dir": str(cohort.relative_to(ROOT)),
        "downloads": {
            "summary_xlsx": str(summary_xlsx.relative_to(ROOT)) if summary_xlsx.exists() else None,
            "cohort_band_png": str(band_png.relative_to(ROOT)) if band_png.exists() else None,
            "subject_summary_csv": str(subj_csv.relative_to(ROOT)) if subj_csv.exists() else None,
            "zip": zip_rel,
        },
    }


@app.get("/download")
def download(relpath: str):
    """ROOT altındaki dosyayı güvenli indir + doğru dosya adıyla gönder."""
    rp = unquote(relpath).strip()
    if "relpath=" in rp:                 # olası çift sarmalamayı temizle
        rp = rp.split("relpath=", 1)[1]
    rp = rp.lstrip("/").replace("\\", "/")

    p = (ROOT / rp).resolve()
    if not p.exists() or not p.is_file() or ROOT not in p.parents:
        raise HTTPException(status_code=404, detail="Dosya bulunamadı.")

    media_type, _ = mimetypes.guess_type(str(p))
    return FileResponse(
        path=str(p),
        filename=p.name,                                  # doğru dosya adı
        media_type=media_type or "application/octet-stream",
    )


# --------------------------------------------------------------------------------------
# HTML Dashboard
# --------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return RedirectResponse(url="/ui/sessions")


@app.get("/ui/sessions", response_class=HTMLResponse)
def ui_sessions(request: Request):
    sessions = _list_sessions()
    return templates.TemplateResponse("session_list.html", {"request": request, "sessions": sessions})


@app.get("/ui/session", response_class=HTMLResponse)
def ui_session(request: Request, source: str, subject: str):
    outdir = INFER / f"{source}_{subject}"
    if not outdir.exists():
        raise HTTPException(status_code=404, detail=f"Oturum bulunamadı: {outdir}")

    scores = _find_scores(outdir)
    summary = summarize_scores(scores)

    timeline_png = outdir / "_session_timeline.png"
    band_csv = outdir / "_band_counts.csv"
    top_csv = outdir / "_top_segments.csv"
    summary_json = outdir / "_summary.json"

    top_rows = _norm_top_rows(top_csv)

    ctx = {
        "request": request,
        "source": source,
        "subject": subject,
        "outdir_rel": str(outdir.relative_to(ROOT)).replace("\\", "/"),
        "n_windows": summary["n_windows"],
        "time_range": summary["time_range"],
        "band_counts": summary["band_counts"],
        "timeline_rel": str(timeline_png.relative_to(ROOT)).replace("\\", "/") if timeline_png.exists() else None,
        "band_csv_rel": str(band_csv.relative_to(ROOT)).replace("\\", "/") if band_csv.exists() else None,
        "scores_rel": str(scores.relative_to(ROOT)).replace("\\", "/"),
        "top_segments": top_rows,
        "summary_json_rel": str(summary_json.relative_to(ROOT)).replace("\\", "/") if summary_json.exists() else None,
    }
    return templates.TemplateResponse("session.html", ctx)
