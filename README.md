# CS6476 MHI Activity Classification Demo

This is a runnable baseline demo for activity classification using Motion History Images (MHI), without PyTorch/TensorFlow and without `cv2.HuMoments`.

## What this demo includes
- Manual MHI construction from frame differences (`Bt` + temporal decay).
- Manual moment-based feature extraction (central moments + scale-invariant moments).
- Multi-model comparison using `scikit-learn` (KNN, SVM, RandomForest, LogisticRegression, DecisionTree, GaussianNB).
- Simple prediction video export with per-frame label overlay.

## 1) Install
```bash
python3 -m pip install -r requirements.txt
```

## 2) Train model
```bash
python3 train.py --data_dir archive --model_out models/mhi_best.joblib --metrics_out outputs/metrics.json
```

Optional: choose model set
```bash
python3 train.py --models knn,svm,rf,logreg
```

## 3) Run prediction on one video and export labeled output
```bash
python3 infer_video.py --model models/mhi_best.joblib --video archive/walking/person01_walking_d1_uncomp.avi --out outputs/pred_demo.mp4
```

## Notes for your real project
- Keep relative paths only.
- This baseline uses one label per video clip and overlays that label on each frame.
- Next optimization stage can add sliding-window MHI for true frame-level changing labels.
