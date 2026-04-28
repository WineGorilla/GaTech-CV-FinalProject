from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mhi_demo import build_mhi_from_video, extract_features_from_video
from mhi_demo.mhi import frame_diff_binary


CLASS_TO_DIR = {
    "walking": "walking",
    "jogging": "jogging",
    "running": "running",
    "boxing": "boxing",
    "waving": "handwaving",
    "clapping": "handclapping",
}


def _load_frames(video_path: Path, resize_to: tuple[int, int], max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, resize_to))
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from: {video_path}")
    return frames


def _motion_peak_index(frames: list[np.ndarray]) -> tuple[int, float]:
    if len(frames) < 2:
        return 0, 0.0
    scores = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        cur = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        score = float(np.mean(cv2.absdiff(cur, prev)))
        scores.append(score)
        prev = cur
    if not scores:
        return 0, 0.0
    best = int(np.argmax(scores)) + 1
    return best, float(scores[best - 1])


def _pick_class_video(data_dir: Path, class_name: str) -> Path:
    d = data_dir / CLASS_TO_DIR[class_name]
    videos = sorted([p for p in d.iterdir() if p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}])
    if not videos:
        raise ValueError(f"No videos in {d}")
    return videos[0]


def build_qualitative_grid(
    data_dir: Path,
    out_file: Path,
    model_bundle: dict,
    scan_frames: int,
):
    actions = model_bundle["actions"]
    model = model_bundle["model"]
    tau = model_bundle["tau"]
    theta = model_bundle["theta"]

    panels = []
    logs = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    for action in actions:
        video = _pick_class_video(data_dir, action)
        frames = _load_frames(video, (360, 240), scan_frames)
        idx, score = _motion_peak_index(frames)
        idx = min(max(0, idx), len(frames) - 1)
        frame = frames[idx].copy()

        feat = extract_features_from_video(
            str(video),
            base_tau=tau,
            theta=theta,
            resize_to=(160, 120),
            max_frames=120,
        )[None, :]
        pred = actions[int(model.predict(feat)[0])]
        color = (0, 255, 0) if pred == action else (0, 0, 255)

        cv2.putText(frame, f"True: {action}", (10, 24), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Pred: {pred}", (10, 52), font, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame={idx}", (10, 80), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        panels.append(frame)
        logs.append((action, video.name, idx, score))

    grid = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), grid)
    return logs


def build_pipeline_visualization(
    data_dir: Path,
    out_file: Path,
    scan_frames: int,
):
    video = _pick_class_video(data_dir, "running")
    frames = _load_frames(video, (320, 240), max(scan_frames, 120))
    idx, score = _motion_peak_index(frames[:scan_frames])
    idx = min(max(1, idx), len(frames) - 1)

    prev = cv2.cvtColor(frames[idx - 1], cv2.COLOR_BGR2GRAY)
    cur = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
    bt = frame_diff_binary(cv2.GaussianBlur(prev, (5, 5), 0), cv2.GaussianBlur(cur, (5, 5), 0), theta=25)

    mhi = build_mhi_from_video(str(video), tau=20, theta=25, resize_to=(320, 240), max_frames=120)
    mhi_u = (np.clip(mhi, 0, 1) * 255).astype(np.uint8)
    mei = ((mhi > 0.05).astype(np.uint8) * 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    p1 = frames[idx].copy()
    cv2.putText(p1, f"Input frame (idx={idx})", (8, 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    p2 = cv2.cvtColor(bt, cv2.COLOR_GRAY2BGR)
    cv2.putText(p2, "Binary motion Bt", (8, 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    p3 = cv2.applyColorMap(mhi_u, cv2.COLORMAP_JET)
    cv2.putText(p3, "MHI", (8, 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    p4 = cv2.cvtColor(mei, cv2.COLOR_GRAY2BGR)
    cv2.putText(p4, "MEI", (8, 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    panel = np.hstack([p1, p2, p3, p4])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), panel)
    return video.name, idx, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("archive"))
    parser.add_argument("--model", type=Path, default=Path("models/mhi_best.joblib"))
    parser.add_argument("--fig_dir", type=Path, default=Path("report/figures"))
    parser.add_argument("--scan_frames", type=int, default=120)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    q_path = args.fig_dir / "qualitative_grid.png"
    p_path = args.fig_dir / "pipeline_visualization.png"
    log_path = args.fig_dir / "frame_selection_log.txt"

    q_logs = build_qualitative_grid(args.data_dir, q_path, bundle, scan_frames=args.scan_frames)
    p_log = build_pipeline_visualization(args.data_dir, p_path, scan_frames=args.scan_frames)

    lines = ["# Motion-aware frame selection log", ""]
    lines.append("Qualitative grid:")
    for action, video, idx, score in q_logs:
        lines.append(f"- {action:8s} video={video} frame={idx} motion_score={score:.3f}")
    lines.append("")
    lines.append(
        f"Pipeline visualization: video={p_log[0]} frame={p_log[1]} motion_score={p_log[2]:.3f}"
    )
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
