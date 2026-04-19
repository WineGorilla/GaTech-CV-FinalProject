import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from .config import ACTIONS
from .data import collect_samples
from .models import build_model_specs, parse_requested_models


def run_training(
    data_dir: Path,
    model_out: Path,
    metrics_out: Path,
    tau: int,
    theta: int,
    max_frames: int,
    k_neighbors: int,
    models_arg: str,
):
    x, y, sample_paths = collect_samples(data_dir, tau, theta, max_frames)
    print(f"Loaded {len(y)} samples, feature_dim={x.shape[1]}")

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 classes in data_dir to train a classifier.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    model_specs = build_model_specs(k_neighbors)
    requested_models = parse_requested_models(models_arg, model_specs)

    results = []
    best = None
    best_model = None

    for name in requested_models:
        clf = model_specs[name]
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        acc = float(accuracy_score(y_test, pred))
        macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, pred).tolist()
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)

        model_result = {
            "model": name,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "confusion_matrix": cm,
            "classification_report": report,
        }
        results.append(model_result)

        if best is None or (acc, macro_f1) > (best["accuracy"], best["macro_f1"]):
            best = model_result
            best_model = clf

    results.sort(key=lambda r: (r["accuracy"], r["macro_f1"]), reverse=True)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "actions": ACTIONS,
        "tau": tau,
        "theta": theta,
        "max_frames": max_frames,
        "n_samples": int(len(y)),
        "sample_paths": sample_paths,
        "models_compared": requested_models,
        "model_results": results,
        "best_model": best["model"],
        "best_accuracy": best["accuracy"],
        "best_macro_f1": best["macro_f1"],
        "confusion_matrix": best["confusion_matrix"],
        "classification_report": best["classification_report"],
    }

    joblib.dump(
        {
            "model": best_model,
            "model_name": best["model"],
            "actions": ACTIONS,
            "tau": tau,
            "theta": theta,
        },
        model_out,
    )
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Model comparison (sorted by accuracy, macro_f1):")
    for r in results:
        print(f"  - {r['model']}: accuracy={r['accuracy']:.4f}, macro_f1={r['macro_f1']:.4f}")
    print(f"Best model: {best['model']}")
    print(f"Saved best model to: {model_out}")
    print(f"Saved metrics to: {metrics_out}")
    print("Best-model confusion matrix:")
    print(np.asarray(best["confusion_matrix"]))

    return payload
