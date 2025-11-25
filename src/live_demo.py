"""Simple webcam loop for experimenting with the gesture classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision import transforms

from model import GestureCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live gesture predictions.")
    parser.add_argument("--model-path", type=Path, default=Path("models/gesture_cnn.pt"))
    parser.add_argument("--label-path", type=Path, default=Path("models/class_to_idx.json"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def load_model(model_path: Path, class_to_idx_path: Path, device: torch.device) -> tuple[GestureCNN, dict[int, str]]:
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx")
    if class_to_idx is None and class_to_idx_path.exists():
        with class_to_idx_path.open("r", encoding="utf-8") as fp:
            class_to_idx = json.load(fp)
    if class_to_idx is None:
        raise RuntimeError("Could not determine class mappings.")

    num_classes = len(class_to_idx)
    model = GestureCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    return model, idx_to_class


def crop_center(frame):
    height, width = frame.shape[:2]
    size = min(height, width)
    start_y = (height - size) // 2
    start_x = (width - size) // 2
    return frame[start_y : start_y + size, start_x : start_x + size]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    model, idx_to_label = load_model(args.model_path, args.label_path, device)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    print("Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            square = crop_center(frame)
            rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            tensor = preprocess(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
            label = idx_to_label.get(pred.item(), "unknown")
            text = f"{label}: {conf.item():.2f}"
            cv2.putText(square, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(square, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Gesture", square)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
