import cv2
import numpy as np

FEATURE_ORDERS = [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3), (2, 2)]
CANONICAL_SIZE = (96, 96)
EPS = 1e-8


def _preprocess_gray(gray: np.ndarray) -> np.ndarray:
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
    adaptive_theta = int(np.mean(diff) + 1.5 * np.std(diff))
    thr = max(theta, adaptive_theta)
    bt = (diff >= thr).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    bt = cv2.morphologyEx(bt, cv2.MORPH_OPEN, kernel)
    bt = cv2.morphologyEx(bt, cv2.MORPH_CLOSE, kernel)
    bt = _keep_largest_component(bt)
    return bt


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


def _body_part_hu_features(mhi_c: np.ndarray, mei_c: np.ndarray) -> list[float]:
    # Body-part aware signal: upper/lower half for both MHI and MEI.
    h = mhi_c.shape[0]
    half = h // 2
    parts = [mhi_c[:half, :], mhi_c[half:, :], mei_c[:half, :], mei_c[half:, :]]
    feats: list[float] = []
    for p in parts:
        p_resized = cv2.resize(p, CANONICAL_SIZE, interpolation=cv2.INTER_LINEAR)
        feats.extend(_hu_invariants(p_resized))
    return feats


def extract_features_from_mhi(mhi: np.ndarray) -> np.ndarray:
    """
    Robust feature extraction from one MHI.

    Components:
    - Motion-region canonicalization (translation/scale robustness)
    - MHI moments + MEI moments (intensity + occupancy)
    - Manual Hu invariants (not using cv2.HuMoments)
    - Upper/lower body part Hu features (MHI + MEI)
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
    feats.extend(_body_part_hu_features(mhi_c, mei_c))

    motion_ratio = float(np.mean(mei))
    mean_energy = float(np.mean(mhi))
    std_energy = float(np.std(mhi))
    temporal_density = float(np.sum(mhi) / (np.sum(mei) + EPS))

    feats.extend([motion_ratio, mean_energy, std_energy, temporal_density, aspect, bbox_area_ratio])
    return np.asarray(feats, dtype=np.float32)


def _load_preprocessed_grays(
    video_path: str,
    resize_to: tuple[int, int] | None = (160, 120),
    max_frames: int | None = None,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    grays: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if resize_to is not None:
            frame = cv2.resize(frame, resize_to)
        gray = _preprocess_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        grays.append(gray)
        if max_frames is not None and len(grays) >= max_frames:
            break

    cap.release()
    if not grays:
        raise ValueError(f"No frames read from: {video_path}")
    return grays


def _build_mhi_from_grays(grays: list[np.ndarray], tau: int, theta: int, start: int, end: int) -> np.ndarray:
    shape = grays[0].shape
    mhi = np.zeros(shape, dtype=np.float32)
    if end - start < 2:
        return mhi

    prev = grays[start]
    for i in range(start + 1, end):
        cur = grays[i]
        bt = frame_diff_binary(prev, cur, theta=theta)
        moving = bt > 0
        mhi[moving] = tau
        mhi[~moving] = np.maximum(mhi[~moving] - 1, 0)
        prev = cur

    if mhi.max() > 0:
        mhi = mhi / mhi.max()
    return mhi


def _temporal_motion_stats(grays: list[np.ndarray], theta: int) -> list[float]:
    if len(grays) < 2:
        return [0.0, 0.0, 0.0, 0.0]

    seq = []
    prev = grays[0]
    for i in range(1, len(grays)):
        bt = frame_diff_binary(prev, grays[i], theta=theta)
        seq.append(float(np.mean(bt > 0)))
        prev = grays[i]

    arr = np.asarray(seq, dtype=np.float32)
    if len(arr) >= 2:
        a = arr[:-1]
        b = arr[1:]
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        corr = float(np.dot(a, b) / den) if den > 1e-12 else 0.0
    else:
        corr = 0.0
    return [float(arr.mean()), float(arr.std()), float(arr.max()), corr]


def _tau_set(base_tau: int) -> list[int]:
    # Multi-timescale temporal memory.
    candidates = [max(6, base_tau // 2), base_tau, max(base_tau + 8, base_tau * 2)]
    return sorted(set(int(x) for x in candidates))


def extract_features_from_video(
    video_path: str,
    base_tau: int = 20,
    theta: int = 25,
    resize_to: tuple[int, int] | None = (160, 120),
    max_frames: int | None = 120,
) -> np.ndarray:
    """
    Enhanced video descriptor with three innovations:
    1) Multi-τ MHI
    2) Temporal pyramid MHI (full + 3 segments)
    3) Body-part-aware moment features inside extract_features_from_mhi
    """
    grays = _load_preprocessed_grays(video_path, resize_to=resize_to, max_frames=max_frames)
    n = len(grays)

    # Full clip + temporal thirds.
    ranges = [
        (0, n),
        (0, max(2, n // 3)),
        (max(0, n // 3), max(2, 2 * n // 3)),
        (max(0, 2 * n // 3), n),
    ]

    feats: list[float] = []
    for tau in _tau_set(base_tau):
        for start, end in ranges:
            mhi = _build_mhi_from_grays(grays, tau=tau, theta=theta, start=start, end=end)
            feats.extend(extract_features_from_mhi(mhi).tolist())

    feats.extend(_temporal_motion_stats(grays, theta=theta))
    return np.asarray(feats, dtype=np.float32)


def build_mhi_from_video(
    video_path: str,
    tau: int = 20,
    theta: int = 25,
    resize_to: tuple[int, int] | None = (160, 120),
    max_frames: int | None = None,
) -> np.ndarray:
    """Backward-compatible helper to build one MHI from a full video."""
    grays = _load_preprocessed_grays(video_path, resize_to=resize_to, max_frames=max_frames)
    return _build_mhi_from_grays(grays, tau=tau, theta=theta, start=0, end=len(grays))
