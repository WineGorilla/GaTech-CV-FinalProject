from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_seed_confusions(runs_path: Path, method: str, model: str):
    data = json.loads(runs_path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    if not runs:
        raise ValueError(f"No runs found in {runs_path}")

    selected = []
    actions = None

    for run in runs:
        if str(run.get("method_variant", "")).lower() != method.lower():
            continue

        if actions is None:
            actions = run.get("actions")

        seed = run.get("seed")
        model_rows = run.get("model_results", [])
        row = next((r for r in model_rows if str(r.get("model", "")).lower() == model.lower()), None)
        if row is None:
            continue
        selected.append({"seed": seed, "cm": row.get("confusion_matrix")})

    if not selected:
        raise ValueError(f"No confusion matrices found for method={method}, model={model} in {runs_path}")
    if not actions:
        raise ValueError(f"No actions found in {runs_path}")

    selected.sort(key=lambda x: int(x["seed"]))
    return actions, selected


def plot_single_seed(actions, seed_item, out_path: Path, title: str, normalize: bool):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180)
    cm = np.asarray(seed_item["cm"], dtype=np.float64)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"{title} (Seed {seed_item['seed']})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(actions)))
    ax.set_yticks(range(len(actions)))
    ax.set_xticklabels(actions, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(actions, fontsize=9)

    thresh = cm.max() * 0.6 if cm.size else 0.0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            val = cm[r, c]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(
                c,
                r,
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
    parser.add_argument("--runs", type=Path, default=Path("outputs/experiments/runs.json"))
    parser.add_argument("--method", type=str, default="enhanced")
    parser.add_argument("--model", type=str, default="rf")
    parser.add_argument("--out_dir", type=Path, default=Path("report/figures/confusion_by_seed"))
    parser.add_argument("--title", type=str, default="RF Confusion Matrices Across Seeds")
    parser.add_argument("--normalize", action="store_true", help="Plot row-normalized confusion matrices")
    args = parser.parse_args()

    actions, per_seed = load_seed_confusions(args.runs, method=args.method, model=args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    suffix = "norm" if args.normalize else "raw"
    for item in per_seed:
        out_path = args.out_dir / f"cm_{args.model}_{args.method}_seed{item['seed']}_{suffix}.png"
        plot_single_seed(actions, item, out_path, args.title, args.normalize)
        saved.append(out_path)

    print("Saved:")
    for p in saved:
        print(f"- {p}")


if __name__ == "__main__":
    main()
