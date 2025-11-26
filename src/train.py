"""Train a TensorFlow CNN to classify drone gestures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from model import build_gesture_model


def load_datasets(data_dir: Path, batch_size: int, image_size=(64, 64), seed: int = 42):
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    class_names = raw_train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = raw_train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = raw_val_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, val_ds, class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gesture classifier with TensorFlow")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ds, val_ds, class_names = load_datasets(args.data_dir, args.batch_size, seed=args.seed)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    model = build_gesture_model(num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "gesture_cnn.keras"
    mapping_path = args.output_dir / "class_to_idx.json"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    print(
        f"Training finished. Best validation loss: {min(history.history['val_loss']):.4f}, "
        f"best validation accuracy: {max(history.history['val_accuracy']):.3f}"
    )

    with mapping_path.open("w", encoding="utf-8") as fp:
        json.dump(class_to_idx, fp, indent=2)
    print(f"Saved label mapping to {mapping_path}")


if __name__ == "__main__":
    main()
