# CV701 Assignment 4 – Facial Keypoint Detection & Deployment

This repository contains my end-to-end solution for CV701 Assignment 4. Task 1 covers training a deep neural network that predicts 68 facial keypoints from the provided dataset and derives a rule-based emotion label (negative/neutral/positive) from the predicted landmarks. Task 2 upgrades that model into a real-time deployment that streams webcam frames, renders landmarks, classifies emotions, and records live metrics. Everything is written in PyTorch and runs on CPU, Apple M‑series (MPS), or CUDA GPUs.

## Repository Structure

```
data/
├── dataset.py                   # Dataset wrapper with optional indexing
├── training_frames_keypoints.csv
├── test_frames_keypoints.csv
artifacts/
├── task1_resnet18/              # Final ResNet-18 run (metrics + predictions)
└── task1_hpc/                   # Previous CUDA baseline
report/                          # LaTeX report + figures
src/
├── data/transforms.py           # Custom transforms (resize, flip, normalize)
├── models/keypoint_resnet.py    # ResNet backbones + regression/heatmap heads
├── train_task1.py               # End-to-end training / evaluation CLI
├── deploy_live.py               # Real-time webcam deployment
└── utils/                       # Metrics, emotion classifier, keypoint helpers
demo_summary.json                # Latest live-run FPS/emotion log
final_demo.mp4                   # Recorded Task 2 demo video
requirements.txt                 # Python dependencies (numpy<2 for torch compatibility)
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

### Heatmap Head (optional)
```bash
python -m src.train_task1 \
  --head heatmap \
  --pretrained \
  --device cuda \
  --backbone resnet34 \
  --output-dir artifacts/task1_heatmap
```

The script automatically writes `metrics.json` (loss, MAE, RMSE, NME curves) and `test_predictions.csv` (denormalized keypoints + emotion label) into the chosen output directory. The best validation checkpoint is stored as `best_model.pt`, but `.gitignore` excludes it from version control.

## Results Snapshot (ResNet-18 final run)

* Validation NME ≈ **0.174** (epoch 34) with augmentation
* Test metrics: pixel MAE **4.29 px**, RMSE **6.33 px**, NME **0.130**, PCK@0.10 **48.2%**, PCK@0.05 **16.3%**, AUC@0.5 **0.748**
* Emotion distribution on test split: **negative = 574**, **neutral = 196** (dataset skew – positives are rare)

See `artifacts/task1_resnet18/metrics.json` for the full history plus WandB links.

### Emotion Statistics Helper

After generating `test_predictions.csv`, run the helper script to inspect class balance and keypoint-derived features:

```bash
python scripts/compute_emotion_stats.py \
  --predictions artifacts/task1_hpc/test_predictions.csv \
  --test-csv data/test_frames_keypoints.csv \
  --test-root data/test
```

## Next Steps

The current pipeline satisfies both Task 1 (training+analysis) and Task 2 (deployment). Future improvements could explore quantization, TorchScript/ONNX export, a lighter backbone (e.g., MobileNet), or learning the emotion classifier instead of using hand-crafted thresholds.

## Task 2: Real-Time Deployment

Use the live deployment script to stream webcam frames, overlay keypoints, and log FPS/emotion counts. The example below mirrors the final demo (Apple M2, ResNet-18 checkpoint, 30-second capture):

```bash
python -m src.deploy_live \
  --checkpoint artifacts/task1_resnet18/best_model.pt \
  --device mps \
  --backbone resnet18 \
  --head regression \
  --show-emotion \
  --record-path final_demo.mp4 \
  --max-seconds 30 \
  --smooth-momentum 0.4 \
  --emotion-hold 15 \
  --log-summary --log-path demo_summary.json
```

Press `q` to exit the OpenCV window. `--smooth-momentum` applies exponential smoothing to stabilize keypoints, and `--emotion-hold` keeps the dominant emotion over the last N frames. Optional flags: `--use-face-detector` (Haar crop for tighter framing), `--device cuda` for GPUs, `--device cpu` for laptops without accelerators, and `--head heatmap --heatmap-size 56` if you want to experiment with the deconvolutional head. For a GUI workflow, run `notebooks/task2_live_demo.ipynb` to start/stop/record from buttons.
