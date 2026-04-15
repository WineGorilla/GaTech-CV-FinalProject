import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mhi_demo import build_mhi_from_video, extract_features_from_mhi

ACTIONS = ["walking", "jogging", "running", "boxing", "waving", "clapping"]


def collect_samples(data_dir: Path, tau: int, theta: int, max_frames: int | None):
    x_list = []
    y_list = []
    paths = []

    for label, action in enumerate(ACTIONS):
        action_dir = data_dir / action
        if not action_dir.exists():
            continue
        for video_path in sorted(action_dir.iterdir()):
            if video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
                continue
            mhi = build_mhi_from_video(
                str(video_path), tau=tau, theta=theta, resize_to=(160, 120), max_frames=max_frames
            )
            feat = extract_features_from_mhi(mhi)
            x_list.append(feat)
            y_list.append(label)
            paths.append(str(video_path))

    if not x_list:
        raise ValueError(f"No videos found in: {data_dir}")

    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.int64), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--model_out", type=Path, default=Path("models/mhi_best.joblib"))
    parser.add_argument("--metrics_out", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument("--tau", type=int, default=20)
    parser.add_argument("--theta", type=int, default=25)
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--models",
        type=str,
        default="knn,svm,rf,logreg",
        help="Comma-separated model names from: knn, svm, rf, logreg",
    )
    args = parser.parse_args()

    x, y, sample_paths = collect_samples(args.data_dir, args.tau, args.theta, args.max_frames)
    print(f"Loaded {len(y)} samples, feature_dim={x.shape[1]}")

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 classes in data_dir to train a classifier.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    model_specs = {
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=args.k)),
        "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3.0, gamma="scale")),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42),
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
        ),
    }

    requested_models = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    unknown_models = [name for name in requested_models if name not in model_specs]
    if unknown_models:
        raise ValueError(
            f"Unknown models in --models: {unknown_models}. Allowed: {list(model_specs.keys())}"
        )
    if not requested_models:
        raise ValueError("No models selected. Pass at least one model in --models.")

    results = []
    best = None
    best_bundle = None

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
            best_bundle = clf

    results.sort(key=lambda r: (r["accuracy"], r["macro_f1"]), reverse=True)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "actions": ACTIONS,
        "tau": args.tau,
        "theta": args.theta,
        "max_frames": args.max_frames,
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
            "model": best_bundle,
            "model_name": best["model"],
            "actions": ACTIONS,
            "tau": args.tau,
            "theta": args.theta,
        },
        args.model_out,
    )
    with args.metrics_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Model comparison (sorted by accuracy, macro_f1):")
    for r in results:
        print(
            f"  - {r['model']}: accuracy={r['accuracy']:.4f}, macro_f1={r['macro_f1']:.4f}"
        )
    print(f"Best model: {best['model']}")
    print(f"Saved best model to: {args.model_out}")
    print(f"Saved metrics to: {args.metrics_out}")
    print("Best-model confusion matrix:")
    print(np.asarray(best["confusion_matrix"]))


if __name__ == "__main__":
    main()
