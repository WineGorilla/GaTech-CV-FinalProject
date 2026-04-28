# CS6476 MHI Activity Classification Demo

This is a runnable baseline demo for activity classification using Motion History Images (MHI)

## What this demo includes
- Manual MHI construction from frame differences.
- Manual moment-based feature extraction.
- Multi-model comparison using `scikit-learn` (KNN, SVM, RandomForest, LogisticRegression, DecisionTree, GaussianNB).
- Simple prediction video export with per-frame label overlay.

## 1) Install
```bash
python3 -m pip install -r requirements.txt
```

## 2) Train model
```bash
python3 train.py --data_dir archive
```

Optional: choose model set
```bash
python3 train.py --models knn,svm,rf,logreg
```

Optional: choose feature method + split seed
```bash
python3 train.py --method_variant enhanced --seed 42
python3 train.py --method_variant baseline --seed 42
```

## 3) Run prediction on one video and export labeled output
```bash
python3 infer_video.py --model models/mhi_best.joblib --video archive/walking/person01_walking_d1_uncomp.avi --out outputs/pred_demo.mp4
```

## Notes for your real project
- Keep relative paths only.
- This baseline uses one label per video clip and overlays that label on each frame.


## 4) Run 3-run experiments with mean/std
Run both methods (`baseline` and `enhanced`) across all models and 3 seeds:
```bash
python3 scripts/run_experiments.py \
  --data_dir archive \
  --methods baseline,enhanced \
  --models knn,svm,rf,logreg,dt,gnb \
  --seeds 42,52,62 \
  --max_frames 40
```

Generated files:
- `outputs/experiments/runs.json` (all per-run results)
- `outputs/experiments/summary.json` (aggregated mean/std)
- `outputs/experiments/method_comparison.csv` (baseline vs enhanced)
- `outputs/experiments/model_comparison.csv` (all model mean/std by method)
