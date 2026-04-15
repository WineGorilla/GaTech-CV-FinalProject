# Activity Classification using Motion History Images (MHI)

**Course:** CS 6476 Computer Vision  
**Project Topic:** Activity Classification using MHI  
**Author:** Ricardo (replace with full name)  
**Date:** April 2026

## 1. Introduction

This project implements a motion-based human activity classifier using Motion History Images (MHI). The target classes are:

- walking
- jogging
- running
- boxing
- waving
- clapping

Given an input video, the pipeline computes frame-wise motion, accumulates temporal motion evidence into an MHI, extracts moment-based descriptors, and predicts an activity label using a classical machine learning classifier.

This implementation follows the project constraints:

- No deep learning frameworks (no PyTorch / TensorFlow)
- No pre-trained models
- No direct use of `cv2.HuMoments`
- Relative paths only

## 2. Method

### 2.1 Binary motion signal

For consecutive grayscale frames \(I_t\) and \(I_{t-1}\), a binary motion map \(B_t\) is computed as:

\[
B_t(x,y)=
\begin{cases}
1, & |I_t(x,y)-I_{t-1}(x,y)| \ge \theta \\
0, & \text{otherwise}
\end{cases}
\]

where \(\theta\) is a threshold (used value: `theta = 25`). A morphological open operation is applied to suppress isolated noise pixels.

### 2.2 Motion History Image (MHI)

The MHI update rule is:

\[
M_t(x,y)=
\begin{cases}
\tau, & B_t(x,y)=1 \\
\max(M_{t-1}(x,y)-1, 0), & B_t(x,y)=0
\end{cases}
\]

where \(\tau\) controls temporal memory (used value: `tau = 20`). Motion pixels are refreshed to \(\tau\), and non-motion pixels decay by 1 per frame.

### 2.3 Moment-based features

To characterize motion shape in the MHI, the following are computed manually:

1. Spatial moments and centroid
2. Central moments \(\mu_{pq}\)
3. Scale-invariant moments:

\[
\nu_{pq} = \frac{\mu_{pq}}{\mu_{00}^{1 + (p+q)/2}}
\]

Feature orders used: \((p,q) \in \{(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),(2,2)\}\).  
The final feature vector concatenates unscaled central moments and scale-invariant moments.

### 2.4 Classifier

A KNN classifier (`k=3`) is trained on extracted feature vectors using a `StandardScaler + KNN` pipeline from scikit-learn.

## 3. Experimental Setup

### 3.1 Data used in this baseline run

For rapid end-to-end validation, this report run uses a synthetic dataset generator (stick-figure motion videos), with 24 total clips (4 per class).

**Important:** This baseline verifies pipeline correctness. Final submission should replace synthetic videos with real human action data and re-run all metrics.

### 3.2 Parameters

- `theta = 25`
- `tau = 20`
- `max_frames = 120`
- input resize: `160 x 120`
- train/test split: `70/30`, stratified

## 4. Results

### 4.1 Quantitative results (current baseline run)

From `outputs/metrics.json`:

- Number of samples: **24**
- Test accuracy: **1.00**
- Macro-F1: **1.00**

Confusion matrix (rows = true, cols = predicted; class order: walking, jogging, running, boxing, waving, clapping):

| True \\ Pred | walk | jog | run | box | wave | clap |
|---|---:|---:|---:|---:|---:|---:|
| walking | 1 | 0 | 0 | 0 | 0 | 0 |
| jogging | 0 | 1 | 0 | 0 | 0 | 0 |
| running | 0 | 0 | 1 | 0 | 0 | 0 |
| boxing  | 0 | 0 | 0 | 2 | 0 | 0 |
| waving  | 0 | 0 | 0 | 0 | 2 | 0 |
| clapping| 0 | 0 | 0 | 0 | 0 | 1 |

Classification error rate for this run is therefore `0.0`.

### 4.2 Qualitative results

The pipeline also exports annotated video predictions (example: `outputs/pred_demo.mp4`), where each frame is overlaid with the predicted action label.

For the final report, include:

- Sample input frames per action
- Corresponding MHI visualizations
- Predicted labels on video frames

## 5. Analysis: Positive and Negative Cases

### 5.1 Why the method works

The MHI representation captures spatiotemporal motion patterns compactly. Activities with distinct temporal signatures (e.g., running vs. waving) produce separable motion distributions in moment-feature space. Scale normalization improves robustness under moderate size variation.

### 5.2 Failure modes (expected on real data)

Even if the synthetic baseline is near-perfect, errors are expected on real videos due to:

- Camera motion and dynamic backgrounds corrupting frame differencing
- Subject scale and viewpoint changes beyond current normalization
- Similar upper-body motion (e.g., waving vs. boxing) causing feature overlap
- Action speed variability making fixed \(\tau\) suboptimal

### 5.3 Improvements for next iteration

- Per-class or adaptive `tau` selection
- Stronger foreground extraction before MHI
- Sliding-window prediction for temporal label transitions
- Classifier comparison (KNN vs. SVM / Random Forest)

## 6. Conclusion

This project demonstrates a complete MHI-based activity classification baseline under the assignment constraints. The implemented system computes motion signals, builds MHIs, extracts moment-based descriptors without banned APIs, and performs activity classification with classical ML.

The current synthetic-data run validates correctness. The next step is to evaluate on real human action videos and report both successful and failure cases with confusion matrices and error plots.

## References

1. A. F. Bobick and J. W. Davis, “The Recognition of Human Movement Using Temporal Templates,” *IEEE TPAMI*, 2001.  
2. M.-K. Hu, “Visual Pattern Recognition by Moment Invariants,” *IRE Transactions on Information Theory*, 1962.  
3. K. Simonyan and A. Zisserman, “Two-Stream Convolutional Networks for Action Recognition in Videos,” *NeurIPS*, 2014.  
4. J. Carreira and A. Zisserman, “Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset,” *CVPR*, 2017.  
5. C. Feichtenhofer, H. Fan, J. Malik, and K. He, “SlowFast Networks for Video Recognition,” *ICCV*, 2019.  
6. G. Bertasius, H. Wang, and L. Torresani, “Is Space-Time Attention All You Need for Video Understanding?,” *ICML*, 2021.  
7. H. Fan et al., “Multiscale Vision Transformers,” *ICCV*, 2021.  
8. Z. Liu et al., “Video Swin Transformer,” *CVPR*, 2022.
