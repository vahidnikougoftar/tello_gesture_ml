"""Train a TensorFlow CNN to classify drone gestures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import tensorflow as tf

from build_models import get_model_builder, list_available_models


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
    parser.add_argument(
        "--model-version",
        type=str,
        choices=list_available_models(),
        default="model_v1",
        help="Which internal architecture to use.",
    )
    parser.add_argument(
        "--pretrained-path",
        type=Path,
        default=None,
        help="Path to an existing .keras model to fine-tune.",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="Hugging Face repo id for downloading a .keras checkpoint.",
    )
    parser.add_argument(
        "--hf-filename",
        type=str,
        default=None,
        help="Target filename inside the Hugging Face repo.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for Hugging Face downloads.",
    )
    return parser.parse_args()


def resolve_pretrained(args) -> Optional[Path]:
    if args.pretrained_path and (args.hf_repo or args.hf_filename):
        raise ValueError("Choose either --pretrained-path or the Hugging Face options, not both.")
    if args.hf_repo or args.hf_filename:
        if not (args.hf_repo and args.hf_filename):
            raise ValueError("Both --hf-repo and --hf-filename must be provided together.")
        from huggingface_hub import hf_hub_download

        download_path = hf_hub_download(
            repo_id=args.hf_repo,
            filename=args.hf_filename,
            cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir else None,
        )
        return Path(download_path)
    return args.pretrained_path


def prepare_model(num_classes: int, args) -> tf.keras.Model:
    pretrained = resolve_pretrained(args)
    if pretrained:
        print(f"Loading pretrained model from {pretrained}")
        model = tf.keras.models.load_model(pretrained)
    else:
        builder = get_model_builder(args.model_version)
        model = builder(num_classes=num_classes)
    output_units = model.output_shape[-1]
    if output_units is None or output_units != num_classes:
        raise ValueError(
            f"Model output units ({output_units}) do not match dataset classes ({num_classes}). "
            "Adjust the model head or ensure the checkpoint matches the dataset."
        )
    return model


def main() -> None:
    args = parse_args()
    train_ds, val_ds, class_names = load_datasets(args.data_dir, args.batch_size, seed=args.seed)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    model = prepare_model(num_classes=len(class_names), args=args)
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
