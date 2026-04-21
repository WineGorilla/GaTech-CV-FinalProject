from pathlib import Path

import cv2

from .mhi import extract_features_from_video


def predict_label(model_bundle, video_path: str):
    feat = extract_features_from_video(
        video_path,
        base_tau=model_bundle["tau"],
        theta=model_bundle["theta"],
        resize_to=(160, 120),
        max_frames=120,
        method_variant=model_bundle.get("method_variant", "enhanced"),
    )[None, :]
    pred_idx = int(model_bundle["model"].predict(feat)[0])
    return model_bundle["actions"][pred_idx]


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
