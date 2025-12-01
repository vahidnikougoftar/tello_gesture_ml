"""Train a TensorFlow CNN to classify drone gestures."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import tensorflow as tf
import matplotlib.pyplot as plt

from build_models import get_model_builder, list_available_models


def save_debug_samples(
    dataset,
    class_names,
    split: str,
    output_dir: Path,
    num_images: int = 9,
) -> None:
    """Persist a small grid of sample images/labels for troubleshooting."""
    import numpy as np

    unbatched = dataset.unbatch()
    samples = []
    for image, label in unbatched.take(num_images):
        np_img = image.numpy()
        np_img = np.clip(np_img, 0, 255).astype("uint8")
        samples.append((np_img, class_names[int(label.numpy())]))
    if not samples:
        print(f"[debug] No samples found for split '{split}'.")
        return

    cols = min(3, len(samples))
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
    for ax, sample in zip(axes, samples):
        img, label = sample
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")
    for ax in axes[len(samples) :]:
        ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"debug_samples_{split}.png"
    fig.suptitle(f"{split} samples", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[debug] Saved sample grid for '{split}' to {path}")


def load_datasets(
    data_dir: Path,
    batch_size: int,
    image_size=(64, 64),
    seed: int = 42,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
    debug_num_images: int = 9,
    return_raw: bool = False,
):
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
    if debug:
        print("[debug] class names:", class_names)
        if debug_dir is None:
            debug_dir = data_dir / "debug"
        save_debug_samples(raw_train_ds, class_names, "train", debug_dir, num_images=debug_num_images)
        save_debug_samples(raw_val_ds, class_names, "val", debug_dir, num_images=debug_num_images)
    autotune = tf.data.AUTOTUNE
    train_ds = raw_train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = raw_val_ds.cache().prefetch(buffer_size=autotune)
    if return_raw:
        return train_ds, val_ds, class_names, raw_train_ds, raw_val_ds
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
        "--debug-data",
        action="store_true",
        help="Export sample images with labels to verify dataset ordering.",
    )
    parser.add_argument(
        "--debug-num-images",
        type=int,
        default=9,
        help="Number of images per split to export when --debug-data is set.",
    )
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


def plot_training_curves(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    metrics = history.history
    acc = metrics.get("accuracy")
    val_acc = metrics.get("val_accuracy")
    loss = metrics.get("loss")
    val_loss = metrics.get("val_loss")
    if not acc or not val_acc:
        print("[warn] accuracy metrics missing from history; skipping training curve plotting.")
        return

    epochs = range(1, len(acc) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, acc, label="train accuracy")
    axes[0].plot(epochs, val_acc, label="val accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    if loss and val_loss:
        axes[1].plot(epochs, loss, label="train loss")
        axes[1].plot(epochs, val_loss, label="val loss")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
    else:
        axes[1].axis("off")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "training_curves.png"
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)
    print(f"[info] Saved training curves to {curve_path}")


def main() -> None:
    args = parse_args()
    using_pca_logistic = args.model_version == "model_v3" and not (
        args.pretrained_path or args.hf_repo
    )
    need_raw = using_pca_logistic
    datasets = load_datasets(
        args.data_dir,
        args.batch_size,
        seed=args.seed,
        debug=args.debug_data,
        debug_dir=args.output_dir / "debug_samples",
        debug_num_images=args.debug_num_images,
        return_raw=need_raw,
    )
    if need_raw:
        train_ds, val_ds, class_names, raw_train_ds, raw_val_ds = datasets
    else:
        train_ds, val_ds, class_names = datasets
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "gesture_cnn.keras"
    mapping_path = args.output_dir / "class_to_idx.json"

    if using_pca_logistic:
        from types import SimpleNamespace
        from build_models.model_v3 import train_pca_logistic_pipeline

        history_data, model = train_pca_logistic_pipeline(
            raw_train_ds,
            raw_val_ds,
            class_names=class_names,
            input_shape=train_ds.element_spec[0].shape[1:],
            n_components=128,
        )
        history = SimpleNamespace(history=history_data)
    else:
        model = prepare_model(num_classes=len(class_names), args=args)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

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

    plot_training_curves(history, args.output_dir)
    if using_pca_logistic:
        model.save(model_path)

    with mapping_path.open("w", encoding="utf-8") as fp:
        json.dump(class_to_idx, fp, indent=2)
    print(f"Saved label mapping to {mapping_path}")


if __name__ == "__main__":
    main()
