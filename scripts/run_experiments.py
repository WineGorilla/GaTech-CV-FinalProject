from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

# Allow running script directly via `python scripts/run_experiments.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mhi_demo.pipeline import run_training


def _parse_csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_seed_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("archive"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--methods", type=str, default="baseline,enhanced")
    parser.add_argument("--models", type=str, default="knn,svm,rf,logreg,dt,gnb")
    parser.add_argument("--seeds", type=str, default="42,52,62")
    parser.add_argument("--tau", type=int, default=20)
    parser.add_argument("--theta", type=int, default=25)
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    methods = _parse_csv_list(args.methods)
    seeds = _parse_seed_list(args.seeds)
    models_arg = ",".join(_parse_csv_list(args.models))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.out_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    runs = []

    for method in methods:
        if method not in {"baseline", "enhanced"}:
            raise ValueError(f"Unsupported method in --methods: {method}")
        for seed in seeds:
            run_id = f"{method}_seed{seed}"
            model_out = run_dir / f"{run_id}.joblib"
            metrics_out = run_dir / f"{run_id}.json"

            print(f"\n=== Running {run_id} ===")
            payload = run_training(
                data_dir=args.data_dir,
                model_out=model_out,
                metrics_out=metrics_out,
                tau=args.tau,
                theta=args.theta,
                max_frames=args.max_frames,
                k_neighbors=args.k,
                models_arg=models_arg,
                method_variant=method,
                seed=seed,
            )
            payload["run_id"] = run_id
            runs.append(payload)

    # Per-(method, model) aggregation
    grouped = defaultdict(lambda: {"accuracy": [], "macro_f1": [], "weighted_f1": []})
    for run in runs:
        method = run["method_variant"]
        for mr in run["model_results"]:
            m = mr["model"]
            grouped[(method, m)]["accuracy"].append(float(mr["accuracy"]))
            grouped[(method, m)]["macro_f1"].append(float(mr["macro_f1"]))
            grouped[(method, m)]["weighted_f1"].append(
                float(mr["classification_report"]["weighted avg"]["f1-score"])
            )

    model_summary = []
    for (method, model), vals in sorted(grouped.items()):
        acc_mean, acc_std = _mean_std(vals["accuracy"])
        mf_mean, mf_std = _mean_std(vals["macro_f1"])
        wf_mean, wf_std = _mean_std(vals["weighted_f1"])
        model_summary.append(
            {
                "method_variant": method,
                "model": model,
                "n_runs": len(vals["accuracy"]),
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "macro_f1_mean": mf_mean,
                "macro_f1_std": mf_std,
                "weighted_f1_mean": wf_mean,
                "weighted_f1_std": wf_std,
            }
        )

    # Method-level summary using best-per-run metrics
    method_group = defaultdict(lambda: {"best_accuracy": [], "best_macro_f1": []})
    for run in runs:
        method = run["method_variant"]
        method_group[method]["best_accuracy"].append(float(run["best_accuracy"]))
        method_group[method]["best_macro_f1"].append(float(run["best_macro_f1"]))

    method_summary = []
    for method, vals in sorted(method_group.items()):
        ba_mean, ba_std = _mean_std(vals["best_accuracy"])
        bm_mean, bm_std = _mean_std(vals["best_macro_f1"])
        method_summary.append(
            {
                "method_variant": method,
                "n_runs": len(vals["best_accuracy"]),
                "best_accuracy_mean": ba_mean,
                "best_accuracy_std": ba_std,
                "best_macro_f1_mean": bm_mean,
                "best_macro_f1_std": bm_std,
            }
        )

    summary = {
        "methods": methods,
        "models": _parse_csv_list(args.models),
        "seeds": seeds,
        "tau": args.tau,
        "theta": args.theta,
        "max_frames": args.max_frames,
        "model_summary": model_summary,
        "method_summary": method_summary,
    }

    # Save JSON artifacts
    runs_json = args.out_dir / "runs.json"
    summary_json = args.out_dir / "summary.json"
    runs_json.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save report-friendly CSVs
    method_csv = args.out_dir / "method_comparison.csv"
    with method_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method_variant",
                "n_runs",
                "best_accuracy_mean",
                "best_accuracy_std",
                "best_macro_f1_mean",
                "best_macro_f1_std",
            ],
        )
        w.writeheader()
        w.writerows(method_summary)

    model_csv = args.out_dir / "model_comparison.csv"
    with model_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method_variant",
                "model",
                "n_runs",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "weighted_f1_mean",
                "weighted_f1_std",
            ],
        )
        w.writeheader()
        w.writerows(model_summary)

    print("\nSaved:")
    print(f"- {runs_json}")
    print(f"- {summary_json}")
    print(f"- {method_csv}")
    print(f"- {model_csv}")


if __name__ == "__main__":
    main()
