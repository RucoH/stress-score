# scripts/07_plot_examples.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "data/processed/scores_v01.parquet"

def main():
    df = pd.read_parquet(SCORES)

    # Histogram (bant değil, skorun dağılımı)
    plt.figure()
    df['score'].plot(kind='hist', bins=30, alpha=0.8)
    plt.title("Stres skoru dağılımı")
    plt.xlabel("score (0–100)")
    plt.ylabel("frekans")
    plt.tight_layout()
    plt.savefig(ROOT / "data/processed/_score_hist.png", dpi=160)

    # Örnek bir subject için zaman serisi
    ex = df.sort_values('t_start').groupby(['source','subject_id']).head(1).iloc[0]
    key = (ex['source'], ex['subject_id'])
    dsub = df[(df['source']==key[0]) & (df['subject_id']==key[1])].sort_values('t_start')

    plt.figure()
    plt.plot(dsub['t_start'], dsub['score'], marker='.', linewidth=1)
    plt.title(f"Zaman serisi — {key[0]} / {key[1]}")
    plt.xlabel("zaman")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig(ROOT / f"data/processed/_timeseries_{key[0]}_{key[1]}.png", dpi=160)

    print("[ok] Görseller kaydedildi → data/processed/_score_hist.png ve _timeseries_*.png")

if __name__ == "__main__":
    main()
