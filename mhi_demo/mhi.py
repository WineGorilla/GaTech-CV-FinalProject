import cv2
import numpy as np

FEATURE_ORDERS = [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3), (2, 2)]
CANONICAL_SIZE = (96, 96)
EPS = 1e-8


def _preprocess_gray(gray: np.ndarray) -> np.ndarray:
    # Light smoothing makes frame differencing less sensitive to sensor/compression noise.
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _keep_largest_component(binary: np.ndarray, min_area: int = 80) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = int(np.argmax(areas)) + 1
    if stats[best_idx, cv2.CC_STAT_AREA] < min_area:
        return np.zeros_like(binary)

    out = np.zeros_like(binary)
    out[labels == best_idx] = 255
    return out


def frame_diff_binary(prev_gray: np.ndarray, gray: np.ndarray, theta: int = 25) -> np.ndarray:
    """Compute robust binary motion image Bt from frame differencing."""
    diff = cv2.absdiff(gray, prev_gray)

    # Adaptive threshold floor improves robustness to illumination/compression changes.
    adaptive_theta = int(np.mean(diff) + 1.5 * np.std(diff))
    thr = max(theta, adaptive_theta)
    bt = (diff >= thr).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    bt = cv2.morphologyEx(bt, cv2.MORPH_OPEN, kernel)
    bt = cv2.morphologyEx(bt, cv2.MORPH_CLOSE, kernel)
    bt = _keep_largest_component(bt)
    return bt


def build_mhi_from_video(
    video_path: str,
    tau: int = 20,
    theta: int = 25,
    resize_to: tuple[int, int] | None = (160, 120),
    max_frames: int | None = None,
) -> np.ndarray:
    """Build Motion History Image from a video file."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"Unable to read video: {video_path}")

    if resize_to is not None:
        frame = cv2.resize(frame, resize_to)

    prev_gray = _preprocess_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    mhi = np.zeros_like(prev_gray, dtype=np.float32)

    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if resize_to is not None:
            frame = cv2.resize(frame, resize_to)

        gray = _preprocess_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        bt = frame_diff_binary(prev_gray, gray, theta=theta)

        moving = bt > 0
        mhi[moving] = tau
        mhi[~moving] = np.maximum(mhi[~moving] - 1, 0)

        prev_gray = gray
        t += 1
        if max_frames is not None and t >= max_frames:
            break

    cap.release()

    if mhi.max() > 0:
        mhi = mhi / mhi.max()

    return mhi


def _raw_moment(img: np.ndarray, p: int, q: int) -> float:
    ys, xs = np.indices(img.shape)
    return float(np.sum((xs**p) * (ys**q) * img))


def _central_moment(img: np.ndarray, p: int, q: int) -> float:
    m00 = _raw_moment(img, 0, 0) + EPS
    x_bar = _raw_moment(img, 1, 0) / m00
    y_bar = _raw_moment(img, 0, 1) / m00

    ys, xs = np.indices(img.shape)
    return float(np.sum(((xs - x_bar) ** p) * ((ys - y_bar) ** q) * img))


def _scale_invariant_moment(img: np.ndarray, p: int, q: int) -> float:
    mu00 = _central_moment(img, 0, 0) + EPS
    mupq = _central_moment(img, p, q)
    gamma = 1.0 + (p + q) / 2.0
    return float(mupq / (mu00**gamma))


def _signed_log(v: float) -> float:
    return float(np.sign(v) * np.log1p(abs(v)))


def _hu_invariants(img: np.ndarray) -> list[float]:
    n20 = _scale_invariant_moment(img, 2, 0)
    n02 = _scale_invariant_moment(img, 0, 2)
    n11 = _scale_invariant_moment(img, 1, 1)
    n30 = _scale_invariant_moment(img, 3, 0)
    n12 = _scale_invariant_moment(img, 1, 2)
    n21 = _scale_invariant_moment(img, 2, 1)
    n03 = _scale_invariant_moment(img, 0, 3)

    h1 = n20 + n02
    h2 = (n20 - n02) ** 2 + 4 * (n11**2)
    h3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h5 = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + (
        3 * n21 - n03
    ) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    h6 = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (
        n21 + n03
    )
    h7 = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) - (
        n30 - 3 * n12
    ) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)

    return [_signed_log(h) for h in [h1, h2, h3, h4, h5, h6, h7]]


def _canonicalize_motion_region(img: np.ndarray, mei: np.ndarray) -> tuple[np.ndarray, float, float]:
    ys, xs = np.where(mei > 0)
    if xs.size == 0:
        return np.zeros(CANONICAL_SIZE, dtype=np.float32), 0.0, 0.0

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    w = x1 - x0 + 1
    h = y1 - y0 + 1
    pad_x = max(2, int(0.1 * w))
    pad_y = max(2, int(0.1 * h))

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(img.shape[1] - 1, x1 + pad_x)
    y1 = min(img.shape[0] - 1, y1 + pad_y)

    crop = img[y0 : y1 + 1, x0 : x1 + 1]
    crop = cv2.resize(crop, CANONICAL_SIZE, interpolation=cv2.INTER_LINEAR)

    aspect = float(w / (h + EPS))
    area_ratio = float((w * h) / (img.shape[0] * img.shape[1] + EPS))
    return crop.astype(np.float32), aspect, area_ratio


def _moment_block(img: np.ndarray) -> list[float]:
    feats: list[float] = []
    for p, q in FEATURE_ORDERS:
        feats.append(_signed_log(_central_moment(img, p, q)))
    for p, q in FEATURE_ORDERS:
        feats.append(_signed_log(_scale_invariant_moment(img, p, q)))
    feats.extend(_hu_invariants(img))
    return feats


def extract_features_from_mhi(mhi: np.ndarray) -> np.ndarray:
    """
    Robust feature extraction from MHI.

    Components:
    - Motion-region canonicalization (translation/scale robustness)
    - MHI moments + MEI moments (intensity + occupancy)
    - Manual Hu invariants (not using cv2.HuMoments)
    - Global motion statistics
    """
    mhi = np.asarray(mhi, dtype=np.float32)
    if mhi.max() > 0:
        mhi = mhi / (mhi.max() + EPS)

    mei = (mhi > 0.05).astype(np.float32)
    mhi_c, aspect, bbox_area_ratio = _canonicalize_motion_region(mhi, mei)
    mei_c, _, _ = _canonicalize_motion_region(mei, mei)

    feats: list[float] = []
    feats.extend(_moment_block(mhi_c))
    feats.extend(_moment_block(mei_c))

    motion_ratio = float(np.mean(mei))
    mean_energy = float(np.mean(mhi))
    std_energy = float(np.std(mhi))
    temporal_density = float(np.sum(mhi) / (np.sum(mei) + EPS))

    feats.extend([motion_ratio, mean_energy, std_energy, temporal_density, aspect, bbox_area_ratio])
    return np.asarray(feats, dtype=np.float32)
