"""Real-time deployment script for facial keypoint detection (Task 2)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from data import FacialKeypointsDataset  # noqa: F401 (ensures package import works)
from src.models.keypoint_resnet import KeypointResNet
from src.models.keypoint_heatmap import KeypointHeatmapNet
from src.utils.emotion import EmotionClassifier
from src.utils.heatmaps import heatmaps_to_keypoints
from src.utils.keypoints import denormalize_keypoints

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live facial keypoint detection.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--device", default="mps", help="cuda | mps | cpu")
    parser.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--heatmap-size", type=int, default=56)
    parser.add_argument("--head", choices=["regression", "heatmap"], default="regression")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--show-emotion", action="store_true")
    parser.add_argument("--fps-window", type=int, default=30)
    parser.add_argument("--record-path", default=None, help="Optional path to save annotated video")
    parser.add_argument("--record-fps", type=int, default=20)
    parser.add_argument("--max-seconds", type=int, default=0, help="Automatically stop after N seconds (0=manual)")
    parser.add_argument("--verbose-emotion", action="store_true", help="Print emotion features for debugging")
    parser.add_argument("--smooth-momentum", type=float, default=0.4, help="EMA smoothing factor [0,1)")
    parser.add_argument("--emotion-hold", type=int, default=10, help="Frames to hold last emotion label")
    parser.add_argument("--use-face-detector", action="store_true", help="Crop detected face before inference")
    parser.add_argument("--log-summary", action="store_true", help="Print session FPS/emotion counts when exiting")
    parser.add_argument("--log-path", default=None, help="Optional JSON file to store the session summary")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    name = name.lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_frame(frame: np.ndarray, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    resized = np.ascontiguousarray(resized)
    tensor = torch.as_tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor, resized


def detect_face_region(frame: np.ndarray, cascade: cv2.CascadeClassifier | None):
    if cascade is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    cx, cy = x + w // 2, y + h // 2
    side = int(max(w, h) * 1.4)
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(frame.shape[1], x1 + side)
    y2 = min(frame.shape[0], y1 + side)
    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if args.head == "heatmap":
        model = KeypointHeatmapNet(
            num_keypoints=68, backbone=args.backbone, pretrained=False, heatmap_size=args.heatmap_size
        )
    else:
        model = KeypointResNet(pretrained=False, dropout=0.0, backbone_name=args.backbone)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"], strict=False)
    model.to(device)
    model.eval()

    emotion_classifier = EmotionClassifier()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera_index}")

    norm_factors = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float32, device=device)
    frame_times: list[float] = []
    smoothed_keypoints = None
    emotion_history: list[str] = []

    writer = None
    if args.record_path:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            str(args.record_path), fourcc, args.record_fps, (args.image_size, args.image_size)
        )

    start_time = time.time()
    total_frames = 0
    emotion_counts: dict[str, int] = {}
    face_detector = None
    if args.use_face_detector:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_detector = cv2.CascadeClassifier(cascade_path)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from camera")
                break

            crop_region = frame
            if face_detector is not None:
                rect = detect_face_region(frame, face_detector)
                if rect:
                    x1, y1, x2, y2 = rect
                    crop_region = frame[y1:y2, x1:x2]

            pre_tensor, display_rgb = preprocess_frame(crop_region, args.image_size)
            pre_tensor = pre_tensor.unsqueeze(0).to(device)

            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(pre_tensor)
                if args.head == "heatmap":
                    preds = heatmaps_to_keypoints(
                        outputs, norm_factors.to(device), args.image_size
                    )
                else:
                    preds = outputs.view(1, -1, 2)
            torch.cuda.synchronize() if device.type == "cuda" else None
            inference_time = time.perf_counter() - start
            frame_times.append(inference_time)
            if len(frame_times) > args.fps_window:
                frame_times.pop(0)

            preds_px = denormalize_keypoints(preds, norm_factors).cpu().numpy()[0]
            if smoothed_keypoints is None:
                smoothed_keypoints = preds_px.copy()
            else:
                alpha = max(0.0, min(0.99, args.smooth_momentum))
                smoothed_keypoints = alpha * smoothed_keypoints + (1 - alpha) * preds_px
            display_points = smoothed_keypoints
            overlay = display_rgb.copy()
            for (x, y) in display_points:
                cv2.circle(overlay, (int(x), int(y)), 2, (0, 255, 0), -1)

            emotion_label = None
            if args.show_emotion:
                emotion_label = emotion_classifier.predict(display_points)
                emotion_history.append(emotion_label)
                if len(emotion_history) > max(1, args.emotion_hold):
                    emotion_history.pop(0)
                dominant = max(set(emotion_history), key=emotion_history.count)
                emotion_label = dominant
                cv2.putText(
                    overlay,
                    emotion_label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0)
                    if emotion_label == "positive"
                    else (0, 0, 255)
                    if emotion_label == "negative"
                    else (0, 255, 255),
                    2,
                )
                if args.verbose_emotion:
                    feats = EmotionClassifier._compute_features(display_points)
                    print(f"emotion={emotion_label} | " + ", ".join(f"{k}={v:.3f}" for k, v in feats.items()))
                emotion_counts[emotion_label] = emotion_counts.get(emotion_label, 0) + 1

            if frame_times:
                fps = 1.0 / (sum(frame_times) / len(frame_times))
                cv2.putText(overlay, f"FPS: {fps:.1f}", (10, args.image_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(
                overlay,
                "Press q to quit",
                (args.image_size - 180, args.image_size - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            bgr_overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            if writer:
                writer.write(bgr_overlay)

            cv2.imshow("Facial Keypoints", bgr_overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            total_frames += 1
            if args.max_seconds > 0 and (time.time() - start_time) >= args.max_seconds:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if writer:
            writer.release()
        elapsed = max(time.time() - start_time, 1e-6)
        summary = {
            "frames": total_frames,
            "avg_fps": total_frames / elapsed,
            "duration_sec": elapsed,
            "emotion_counts": emotion_counts,
        }
        if args.log_summary or args.log_path:
            print(f"[SUMMARY] {summary}")
        if args.log_path:
            try:
                Path(args.log_path).write_text(json.dumps(summary, indent=2))
            except OSError as exc:
                print(f"[WARN] Unable to write summary log: {exc}")


if __name__ == "__main__":
    main()
