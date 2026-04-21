from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_confusion(metrics_path: Path):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    cm = data.get("confusion_matrix")
    actions = data.get("actions")
    if cm is None:
        raise ValueError(f"No confusion_matrix found in {metrics_path}")
    if not actions:
        raise ValueError(f"No actions found in {metrics_path}")
    return cm, actions


def plot_confusion_matrix(cm, actions, out_path: Path, title: str, normalize: bool):
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=180)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(actions)))
    ax.set_yticks(range(len(actions)))
    ax.set_xticklabels(actions, rotation=30, ha="right")
    ax.set_yticklabels(actions)

    thresh = cm.max() * 0.6 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
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
        default=Path("report/figures/confusion_matrix_matplotlib.png"),
    )
    parser.add_argument("--title", type=str, default="Confusion Matrix")
    parser.add_argument("--normalize", action="store_true", help="Plot row-normalized confusion matrix")
    args = parser.parse_args()

    cm, actions = load_confusion(args.metrics)
    plot_confusion_matrix(cm, actions, args.out, args.title, args.normalize)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
