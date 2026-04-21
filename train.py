import argparse
from pathlib import Path

from mhi_demo.pipeline import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("archive"))
    parser.add_argument("--model_out", type=Path, default=Path("models/mhi_best.joblib"))
    parser.add_argument("--metrics_out", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument("--tau", type=int, default=20)
    parser.add_argument("--theta", type=int, default=25)
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--method_variant", type=str, default="enhanced", choices=["baseline", "enhanced"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        type=str,
        default="knn,svm,rf,logreg,dt,gnb",
        help="Comma-separated model names from: knn, svm, rf, logreg, dt, gnb",
    )
    args = parser.parse_args()

    run_training(
        data_dir=args.data_dir,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        tau=args.tau,
        theta=args.theta,
        max_frames=args.max_frames,
        k_neighbors=args.k,
        models_arg=args.models,
        method_variant=args.method_variant,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
