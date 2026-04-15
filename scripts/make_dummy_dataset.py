import math
from pathlib import Path

import cv2
import numpy as np

ACTIONS = ["walking", "jogging", "running", "boxing", "waving", "clapping"]


def draw_actor(canvas, center_x, center_y, arm_phase=0.0, leg_phase=0.0, scale=1.0):
    # Simple stick figure with moving limbs.
    head_r = int(8 * scale)
    body = int(24 * scale)
    arm = int(16 * scale)
    leg = int(18 * scale)

    head = (int(center_x), int(center_y - body - head_r))
    neck = (int(center_x), int(center_y - body))
    hip = (int(center_x), int(center_y))

    cv2.circle(canvas, head, head_r, (255, 255, 255), 2)
    cv2.line(canvas, neck, hip, (255, 255, 255), 2)

    left_arm = (int(center_x - arm * math.cos(arm_phase)), int(neck[1] + arm * math.sin(arm_phase)))
    right_arm = (int(center_x + arm * math.cos(arm_phase)), int(neck[1] + arm * math.sin(-arm_phase)))
    cv2.line(canvas, neck, left_arm, (255, 255, 255), 2)
    cv2.line(canvas, neck, right_arm, (255, 255, 255), 2)

    left_leg = (int(center_x - leg * math.cos(leg_phase)), int(hip[1] + leg * math.sin(abs(leg_phase)) + leg))
    right_leg = (int(center_x + leg * math.cos(leg_phase)), int(hip[1] + leg * math.sin(abs(-leg_phase)) + leg))
    cv2.line(canvas, hip, left_leg, (255, 255, 255), 2)
    cv2.line(canvas, hip, right_leg, (255, 255, 255), 2)


def action_params(action, t):
    if action == "walking":
        return 1.5, 0.9, 0.5, 0.35
    if action == "jogging":
        return 2.5, 1.3, 0.65, 0.45
    if action == "running":
        return 4.0, 1.8, 0.85, 0.55
    if action == "boxing":
        return 0.8, 2.2, 0.2, 0.1
    if action == "waving":
        return 0.6, 2.8, 0.15, 0.05
    if action == "clapping":
        return 0.3, 3.2, 0.05, 0.02
    return 1.0, 1.0, 0.3, 0.2


def make_video(path: Path, action: str, n_frames=80, w=320, h=240, fps=20):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    x = w // 3
    y = int(h * 0.62)

    for t in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        vx, freq, arm_amp, leg_amp = action_params(action, t)

        x_t = int(x + vx * t)
        x_t = max(40, min(w - 40, x_t))

        arm_phase = math.sin(t * 0.2 * freq) * math.pi * arm_amp
        leg_phase = math.sin(t * 0.2 * freq + math.pi / 2.0) * math.pi * leg_amp

        if action == "boxing":
            arm_phase = math.sin(t * 0.35 * freq) * math.pi * 0.9
            leg_phase = 0.0
        elif action == "waving":
            arm_phase = math.sin(t * 0.4 * freq) * math.pi * 0.95
            leg_phase = 0.0
        elif action == "clapping":
            arm_phase = math.sin(t * 0.55 * freq) * math.pi * 0.2
            leg_phase = 0.0

        draw_actor(frame, x_t, y, arm_phase=arm_phase, leg_phase=leg_phase, scale=1.25)
        cv2.putText(frame, action, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2)
        writer.write(frame)

    writer.release()


def main(root: Path = Path("data"), videos_per_class: int = 4):
    root.mkdir(parents=True, exist_ok=True)
    for action in ACTIONS:
        d = root / action
        d.mkdir(parents=True, exist_ok=True)
        for i in range(videos_per_class):
            out = d / f"{action}_{i:02d}.mp4"
            make_video(out, action)
            print(f"created {out}")


if __name__ == "__main__":
    main()
