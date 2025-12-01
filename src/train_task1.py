"""End-to-end training pipeline for Task 1."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from data import FacialKeypointsDataset

from .data.transforms import (
    Compose,
    NormalizeImage,
    NormalizeKeypoints,
    RandomHorizontalFlip,
    Rescale,
    ToTensor,
)
from .models.keypoint_resnet import KeypointResNet
from .utils.emotion import EmotionClassifier
from .utils.keypoints import compute_metrics, denormalize_keypoints, flatten_keypoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a facial keypoint detector (Task 1)")
    parser.add_argument("--train-csv", default="data/training_frames_keypoints.csv")
    parser.add_argument("--train-root", default="data/training")
    parser.add_argument("--test-csv", default="data/test_frames_keypoints.csv")
    parser.add_argument("--test-root", default="data/test")
    parser.add_argument("--output-dir", default="artifacts/task1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of training data used for validation")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default="mps", help="mps | cuda | cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze the ResNet feature extractor")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretraining")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="cv701-task1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    name = name.lower()
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but not available")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(image_size: int):
    train_transform = Compose(
        [
            Rescale(image_size),
            RandomHorizontalFlip(prob=0.5),
            NormalizeKeypoints(),
            ToTensor(),
            NormalizeImage(),
        ]
    )
    eval_transform = Compose(
        [
            Rescale(image_size),
            NormalizeKeypoints(),
            ToTensor(),
            NormalizeImage(),
        ]
    )
    return train_transform, eval_transform


def split_indices(num_samples: int, val_fraction: float, seed: int):
    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(num_samples * val_fraction))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def create_dataloaders(args):
    baseline_dataset = FacialKeypointsDataset(args.train_csv, args.train_root)
    train_indices, val_indices = split_indices(len(baseline_dataset), args.val_split, args.seed)
    train_transform, eval_transform = build_transforms(args.image_size)

    train_dataset = FacialKeypointsDataset(
        args.train_csv, args.train_root, transform=train_transform, subset_indices=train_indices
    )
    val_dataset = FacialKeypointsDataset(
        args.train_csv, args.train_root, transform=eval_transform, subset_indices=val_indices
    )
    test_dataset = FacialKeypointsDataset(args.test_csv, args.test_root, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def create_model(args, device):
    model = KeypointResNet(pretrained=args.pretrained, dropout=args.dropout)
    if args.freeze_backbone:
        model.freeze_backbone(True)
    return model.to(device)


def train_one_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm):
    model.train()
    running_loss = 0.0
    total = 0
    for batch in dataloader:
        images = batch["image"].to(device)
        targets = flatten_keypoints(batch["keypoints"].to(device))

        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = criterion(preds, targets)
        loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    return running_loss / max(total, 1)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_sums = defaultdict(float)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["keypoints"].to(device)
            preds = model(images)
            loss = criterion(preds, flatten_keypoints(targets))

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            batch_metrics = compute_metrics(
                preds.view(batch_size, -1, 2),
                targets,
                batch["keypoint_normalization"].to(device),
            )
            for key, value in batch_metrics.items():
                metric_sums[key] += value * batch_size

    metrics = {key: metric_sums[key] / max(total_samples, 1) for key in metric_sums}
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def save_checkpoint(model, optimizer, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def run_inference(model, dataloader, device, output_dir: Path):
    model.eval()
    emotion_classifier = EmotionClassifier()
    records = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            names = batch["image_name"]
            preds = model(images).view(images.size(0), -1, 2)
            preds_px = denormalize_keypoints(
                preds, batch["keypoint_normalization"].to(device)
            ).cpu()

            for i, name in enumerate(names):
                kp = preds_px[i]
                row = {"image_name": name}
                for idx in range(kp.size(0)):
                    row[f"x{idx}"] = float(kp[idx, 0])
                    row[f"y{idx}"] = float(kp[idx, 1])
                row["emotion"] = emotion_classifier.predict(kp.numpy())
                records.append(row)

    df = pd.DataFrame(records)
    output_path = output_dir / "test_predictions.csv"
    df.to_csv(output_path, index=False)
    emotion_counts = df["emotion"].value_counts().to_dict()
    return output_path, emotion_counts


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = create_dataloaders(args)
    model = create_model(args, device)

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Please install it or disable --use-wandb.")
        run_name = args.wandb_run_name or f"task1-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )
        wandb.watch(model, log="all", log_freq=100)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = {"train": [], "val": []}
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.max_grad_norm)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train"].append({"epoch": epoch, "loss": train_loss})
        history["val"].append({"epoch": epoch, **val_metrics})

        print(
            f"Epoch {epoch}/{args.epochs} - train loss: {train_loss:.4f} - "
            f"val loss: {val_metrics['loss']:.4f} - val NME: {val_metrics.get('nme', 0):.4f}"
        )

        if wandb_run:
            log_payload = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_metrics["loss"],
            }
            for metric_key in ("nme", "pixel_mae", "pixel_rmse", "pck_0.05", "pck_0.10", "auc_0.5"):
                if metric_key in val_metrics:
                    log_payload[f"val/{metric_key}"] = val_metrics[metric_key]
            wandb.log(log_payload)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(model, optimizer, epoch, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics = evaluate(model, test_loader, criterion, device)
    predictions_csv, emotion_counts = run_inference(model, test_loader, device, output_dir)

    summary = {
        "best_val_loss": best_val,
        "test_metrics": test_metrics,
        "emotion_distribution": emotion_counts,
        "predictions_csv": str(predictions_csv),
        "history": history,
        "device": str(device),
        "config": vars(args),
    }
    if wandb_run:
        summary["wandb_run_id"] = wandb_run.id
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    if wandb_run:
        wandb_run.summary.update(
            {
                "best_val_loss": best_val,
                "test_metrics": test_metrics,
                "emotion_distribution": emotion_counts,
            }
        )
        wandb.finish()

    print(f"Saved best model to {best_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_csv}")


if __name__ == "__main__":
    main()
