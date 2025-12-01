"""Second version placeholder model for experimentation."""
from __future__ import annotations

import tensorflow as tf


def build_gesture_model(num_classes: int, input_shape: tuple[int, int, int] = (64, 64, 3)) -> tf.keras.Model:
    """Currently identical to v1 but kept separate for experimentation."""
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation_v2",
    )

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)
    for filters in (8,16,64):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(192, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="gesture_cnn_v2")
    return model
