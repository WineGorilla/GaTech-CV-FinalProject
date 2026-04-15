# CS6476 MHI Activity Classification Demo

This is a runnable baseline demo for activity classification using Motion History Images (MHI), without PyTorch/TensorFlow and without `cv2.HuMoments`.

## What this demo includes
- Manual MHI construction from frame differences (`Bt` + temporal decay).
- Manual moment-based feature extraction (central moments + scale-invariant moments).
- Multi-model comparison using `scikit-learn` (KNN, SVM, RandomForest, LogisticRegression).
- Simple prediction video export with per-frame label overlay.
- Synthetic dataset generator so you can run end-to-end immediately.

## 1) Install
```bash
python3 -m pip install -r requirements.txt
```

## 2) Generate a quick synthetic dataset
```bash
python3 scripts/make_dummy_dataset.py
```

This creates videos under:
```text
data/
  walking/
  jogging/
  running/
  boxing/
  waving/
  clapping/
```

## 3) Train baseline model
```bash
python3 train.py --data_dir data --model_out models/mhi_best.joblib --metrics_out outputs/metrics.json
```

Optional: choose model set
```bash
python3 train.py --models knn,svm,rf,logreg
```

## 4) Run prediction on one video and export labeled output
```bash
python3 infer_video.py --model models/mhi_best.joblib --video data/walking/walking_00.mp4 --out outputs/pred_demo.mp4
```

## Notes for your real project
- Replace synthetic data with your course dataset clips.
- Keep relative paths only.
- This baseline uses one label per video clip and overlays that label on each frame.
- Next optimization stage can add sliding-window MHI for true frame-level changing labels.
