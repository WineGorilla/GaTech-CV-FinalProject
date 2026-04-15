import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np

from mhi_demo import build_mhi_from_video, extract_features_from_mhi


def predict_label(model_bundle, video_path: str):
    tau = model_bundle["tau"]
    theta = model_bundle["theta"]
    model = model_bundle["model"]
    actions = model_bundle["actions"]

    mhi = build_mhi_from_video(video_path, tau=tau, theta=theta, resize_to=(160, 120), max_frames=120)
    feat = extract_features_from_mhi(mhi)[None, :]
    pred_idx = int(model.predict(feat)[0])
    return actions[pred_idx]


def annotate_video(input_video: str, output_video: str, label: str):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(
            frame,
            f"Pred: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    cap.release()
    writer.release()


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
