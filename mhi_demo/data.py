from pathlib import Path

import numpy as np

from .config import ACTION_DIR_ALIASES, ACTIONS, VIDEO_SUFFIXES
from .mhi import extract_features_from_video


def collect_samples(data_dir: Path, tau: int, theta: int, max_frames: int | None, method_variant: str = "enhanced"):
    x_list = []
    y_list = []
    paths = []

    for label, action in enumerate(ACTIONS):
        candidate_dirs = [data_dir / name for name in ACTION_DIR_ALIASES.get(action, [action])]
        found_any_dir = False
        class_count = 0

        for action_dir in candidate_dirs:
            if not action_dir.exists():
                continue
            found_any_dir = True

            for video_path in sorted(action_dir.iterdir()):
                if video_path.suffix.lower() not in VIDEO_SUFFIXES:
                    continue

                feat = extract_features_from_video(
                    str(video_path),
                    base_tau=tau,
                    theta=theta,
                    resize_to=(160, 120),
                    max_frames=max_frames,
                    method_variant=method_variant,
                )
                x_list.append(feat)
                y_list.append(label)
                paths.append(str(video_path))
                class_count += 1

        if not found_any_dir:
            print(
                f"[WARN] Missing directory for class '{action}'. Checked: "
                + ", ".join(str(d) for d in candidate_dirs)
            )
        print(f"[INFO] Collected {class_count} videos for class '{action}'")

    if not x_list:
        raise ValueError(f"No videos found in: {data_dir}")

    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.int64), paths
