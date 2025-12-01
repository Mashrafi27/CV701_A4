# CV701 Assignment 4 – Facial Keypoint Detection

This repository contains my implementation of Task 1 from CV701 Assignment 4: training a deep neural network that predicts 68 facial keypoints from the provided dataset and derives a rule-based emotion label (negative/neutral/positive) from the predicted landmarks. The code is written in PyTorch and supports training on CPU, Apple M‑series (MPS), and CUDA GPUs.

## Repository Structure

```
data/
├── dataset.py               # Dataset wrapper with optional indexing
├── training_frames_keypoints.csv
├── test_frames_keypoints.csv
artifacts/
└── task1_hpc/               # Latest HPC run metrics + predictions
src/
├── data/transforms.py       # Custom transforms (resize, flip, normalize)
├── models/keypoint_resnet.py# ResNet18 regression head
├── train_task1.py           # End-to-end training / evaluation CLI
└── utils/                   # Metrics + emotion classifier helpers
requirements.txt             # Python dependencies (numpy<2 for torch compatibility)
```

WandB run logs and large `.pt` checkpoints are ignored via `.gitignore` to keep the repository lightweight.

## Environment Setup

```bash
conda create -n cv701 python=3.11 -y
conda activate cv701
pip install -r requirements.txt
```

The requirements pin `numpy<2` to avoid ABI issues with the current torch/torchvision wheels.

## Training Command Examples

### Apple Silicon / CPU
```bash
python -m src.train_task1 \
  --pretrained \
  --device mps \
  --output-dir artifacts/task1_run_local
```

### HPC / CUDA (with WandB logging)
```bash
python -m src.train_task1 \
  --pretrained \
  --device cuda \
  --num-workers 8 \
  --output-dir artifacts/task1_hpc \
  --use-wandb --wandb-project cv701_a4 --wandb-run-name hpc_cuda
```

The script automatically writes `metrics.json` (loss, MAE, RMSE, NME curves) and `test_predictions.csv` (denormalized keypoints + emotion label) into the chosen output directory. The best validation checkpoint is stored as `best_model.pt`, but `.gitignore` excludes it from version control.

## Results Snapshot (HPC run)

* Validation NME ≈ **0.174** at epoch 34
* Test metrics: pixel MAE ≈ **4.84 px**, RMSE ≈ **7.62 px**, NME ≈ **0.148**
* Emotion distribution on test split: **negative = 738**, **neutral = 32** (rule-based classifier)

See `artifacts/task1_hpc/metrics.json` for the full history.

### Emotion Statistics Helper

After generating `test_predictions.csv`, run the helper script to inspect class balance and keypoint-derived features:

```bash
python scripts/compute_emotion_stats.py \
  --predictions artifacts/task1_hpc/test_predictions.csv \
  --test-csv data/test_frames_keypoints.csv \
  --test-root data/test
```

## Next Steps

Task 2 (real-time deployment/optimization) still needs to be implemented; Task 1 is ready for reporting with reproducible code, metrics, and prediction dumps.

## Task 2: Real-Time Deployment

Use the live deployment script to stream webcam frames, overlay keypoints, and estimate FPS (defaults to camera index 0 and 224×224 input):

```bash
python -m src.deploy_live \
  --checkpoint artifacts/task1_hpc/best_model.pt \
  --device mps \
  --backbone resnet34 \
  --show-emotion \
  --record-path demo.mp4 \
  --max-seconds 30 \
  --smooth-momentum 0.5 \
  --emotion-hold 15
```

Press `q` to exit the OpenCV window. `--smooth-momentum` applies exponential smoothing to stabilize keypoints, `--emotion-hold` keeps the dominant emotion over the last N frames, and `--record-path` saves annotated frames for the report/demo video. You can run the same command in a Jupyter notebook cell (prefix with `!`) to capture output alongside the saved video.
