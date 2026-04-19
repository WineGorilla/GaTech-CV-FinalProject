import argparse
from pathlib import Path

import joblib

from mhi_demo.viz import annotate_video, predict_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/mhi_best.joblib"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/pred_demo.mp4"))
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model_name = bundle.get("model_name", "unknown")
    label = predict_label(bundle, str(args.video))
    print(f"Model used: {model_name}")
    print(f"Predicted label: {label}")

    annotate_video(str(args.video), str(args.out), label)
    print(f"Annotated video saved: {args.out}")


if __name__ == "__main__":
    main()
