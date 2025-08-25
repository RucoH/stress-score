# 🧠 Stress Score — 0–100 Stress Scoring Pipeline

A compact, end-to-end pipeline that turns Empatica **E4** (BVP/EDA/TEMP/IBI) or **WESAD** sessions into a **0–100 stress score** with reports — featuring feature extraction, per-subject normalization, logistic fusion, rule-based overrides, distributional calibration, and a **FastAPI + mini Web UI**.

---

## ✨ Features

* 🔌 **Single-session or batch** scoring (point to an E4 folder or WESAD root)
* 📐 **Per-subject baselines** → robust z-scores
* 📈 **Logistic transform** + **weighted fusion** (PPG/HR, EDA, Temp)
* ⚠️ **Rule-based overrides** (e.g., fever ≥ 39.5 °C, very high HR, low quality)
* 🎛️ **Distributional calibration** → balanced 0–100 scale + **bands**
* 🖼️ **Reports**: timeline PNG, band counts CSV, top segments CSV/JSON
* 🌐 **FastAPI** service + **Web UI** (download scores/plots)

---

## 🚀 Getting Started

> **Python 3.11+** recommended. You don’t need to store raw data in the repo — just pass absolute paths to your E4/WESAD folders.

```bash
# 1) Install
pip install -U pip
pip install numpy pandas scipy pyarrow pyyaml matplotlib fastapi uvicorn jinja2 xlsxwriter kaggle
pip install -e .   # optional, editable install

# 2) Score a single E4 session (CLI)
python -m stresscore.cli score   --input "data\wesad\WESAD\S13\S13_E4_Data"   --subject S13 --source e4api --report
# Outputs → data/processed/infer/e4api_S13/
#   - scores_v02_calibrated.parquet, _session_timeline.png, _top_segments.csv, ...

# 3) Run the API + Web UI
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
# UI:     http://127.0.0.1:8000/ui/sessions
# Swagger: http://127.0.0.1:8000/docs
```

**Batch (WESAD root):**
```bash
python scripts/19_pipeline_cohort.py   --root "D:\data\wesad\WESAD" --source e4api --report --zip --subject-parent-depth 1
```

---

## 🎯 Purpose

Provide a clear, reproducible pipeline to quantify **physiological stress (0–100)** from PPG/EDA/TEMP signals for research, prototyping, and analytics dashboards.  
> Disclaimer: This is **not** a medical device and not intended for clinical diagnosis.

---

## 🛠️ Built With

* **Python** (NumPy, Pandas, SciPy, PyArrow)
* **FastAPI** + **Uvicorn** + **Jinja2**
* **Matplotlib** (plots), **XlsxWriter** (Excel)
* YAML configs for weights, overrides, bands

---

## 📂 Data Expectations (E4 / WESAD)

* E4 session folder should include: `BVP.csv` (64 Hz), `EDA.csv` (4 Hz), `TEMP.csv` (4 Hz); optional `IBI.csv`, `tags.csv`.  
  CSV format: line 1 = epoch (Unix s), line 2 = sampling rate (Hz), then values (IBI/tags have no rate line).
* WESAD layout: `.../WESAD/SXX/SXX_E4_Data`.

---

## 🧪 Common Commands

```bash
# Feature extraction / baseline / scoring (stepwise)
python scripts/04_extract_features.py
python scripts/05_build_baseline.py
python scripts/06_score_from_features.py
python scripts/08_calibrate_distributional.py
python scripts/07_quick_report.py --path data/processed/scores_v02_calibrated.parquet

# Single session inference + report
python scripts/09_infer_e4_session.py --input ".../S13_E4_Data" --subject S99 --source e4infer
python scripts/10_session_report.py --path data/processed/infer/e4infer_S99/scores_v02_calibrated.parquet

# Cohort summary + Excel
python scripts/17_cohort_summary.py
python scripts/18_export_excel.py   # → data/processed/cohort/summary.xlsx
```

---

## ❓ Troubleshooting

* `kaggle is not recognized` → `pip install kaggle` and place `~/.kaggle/kaggle.json`.  
* Parquet errors → `pip install pyarrow`.  
* Excel export → `pip install XlsxWriter`.  
* Web UI score column missing → templates handle it; update to latest `api/templates`.

---

## 📄 License

Distributed under the **[MIT License](LICENSE)**.

---

## 👤 Author

* GitHub: [@RucoH](https://github.com/RucoH)
* Live Site: [https://rucoh.github.io/](https://rucoh.github.io/)

---

Feel free to fork, open issues, or propose improvements!
