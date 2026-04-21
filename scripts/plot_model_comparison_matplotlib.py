from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(metrics_path: Path):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = data.get("model_results", [])
    if not rows:
        raise ValueError(f"No model_results found in {metrics_path}")

    # Always show the 6 baseline methods in a stable order.
    method_order = ["rf", "svm", "logreg", "dt", "knn", "gnb"]
    row_map = {str(r["model"]).lower(): r for r in rows}

    labels = [m.upper() for m in method_order]
    acc = [float(row_map[m]["accuracy"]) if m in row_map else float("nan") for m in method_order]
    macro_f1 = [float(row_map[m]["macro_f1"]) if m in row_map else float("nan") for m in method_order]
    return labels, acc, macro_f1


def plot_bar(labels, acc, macro_f1, out_path: Path, title: str):
    import matplotlib.pyplot as plt
    import numpy as np

    # Prefer Times New Roman; fall back to serif if unavailable.
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    # NaN values (missing methods) are rendered as 0-height bars and marked as N/A.
    import math
    acc_plot = [0.0 if math.isnan(v) else v for v in acc]
    f1_plot = [0.0 if math.isnan(v) else v for v in macro_f1]

    bars1 = ax.bar(x - width / 2, acc_plot, width, label="Accuracy", color="#4A7BD1")
    bars2 = ax.bar(x + width / 2, f1_plot, width, label="Macro-F1", color="#3CA46A")

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            idx = int(np.round(b.get_x() + b.get_width() / 2 - x[0]))
            # The index trick above is not robust with arbitrary bar spacing; fallback below.
            if bars is bars1:
                data_vals = acc
                bar_index = list(bars1).index(b)
            else:
                data_vals = macro_f1
                bar_index = list(bars2).index(b)

            if np.isnan(data_vals[bar_index]):
                label = "N/A"
            else:
                label = f"{h:.3f}"
            ax.annotate(
                label,
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("report/figures/model_comparison_matplotlib.png"),
    )
    parser.add_argument("--title", type=str, default="Model Comparison on Real Dataset")
    args = parser.parse_args()

    labels, acc, macro_f1 = load_results(args.metrics)
    plot_bar(labels, acc, macro_f1, args.out, args.title)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
