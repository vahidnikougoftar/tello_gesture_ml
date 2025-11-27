"""Evaluate the trained TensorFlow model against the validation split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate gesture classifier on a dataset split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--model-path", type=Path, default=Path("models/gesture_cnn.keras"))
    parser.add_argument("--label-path", type=Path, default=Path("models/class_to_idx.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def load_dataset(split_dir: Path, image_size: tuple[int, int], batch_size: int):
    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataset, dataset.class_names


def load_label_mapping(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as fp:
        class_to_idx = json.load(fp)
    return {idx: label for label, idx in class_to_idx.items()}


def collect_predictions(model: tf.keras.Model, dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch_images, batch_labels in dataset:
        probabilities = model.predict(batch_images, verbose=0)
        batch_preds = np.argmax(probabilities, axis=1)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(batch_preds.tolist())
    return np.array(y_true), np.array(y_pred)


def plot_confusion(cm: np.ndarray, class_names: list[str], save_path: Path) -> None:
    figure, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    figure.tight_layout()
    figure.savefig(save_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    split_dir = args.data_dir / args.split
    dataset, class_names = load_dataset(split_dir, (args.image_size, args.image_size), args.batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(args.model_path)
    idx_to_label = load_label_mapping(args.label_path)

    y_true, y_pred = collect_predictions(model, dataset)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    print("Evaluation split:", args.split)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cm_path = args.output_dir / f"confusion_matrix_{args.split}.png"
    plot_confusion(cm, class_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
