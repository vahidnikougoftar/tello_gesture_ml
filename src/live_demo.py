"""Simple webcam loop for experimenting with the TensorFlow gesture classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live gesture predictions.")
    parser.add_argument("--model-path", type=Path, default=Path("models/gesture_cnn.keras"))
    parser.add_argument("--label-path", type=Path, default=Path("models/class_to_idx.json"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    return parser.parse_args()


def load_labels(mapping_path: Path) -> dict[int, str]:
    with mapping_path.open("r", encoding="utf-8") as fp:
        class_to_idx = json.load(fp)
    return {idx: label for label, idx in class_to_idx.items()}


def crop_center(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    size = min(height, width)
    start_y = (height - size) // 2
    start_x = (width - size) // 2
    return frame[start_y : start_y + size, start_x : start_x + size]


def preprocess_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size))
    tensor = resized.astype("float32") / 255.0
    return np.expand_dims(tensor, axis=0)


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path)
    idx_to_label = load_labels(args.label_path)

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
            tensor = preprocess_frame(square, args.image_size)
            probs = model.predict(tensor, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            label = idx_to_label.get(pred_idx, "unknown")
            text = f"{label}: {conf:.2f}"
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
