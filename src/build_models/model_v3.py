"""PCA + Logistic Regression utilities."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


def _dataset_to_numpy(dataset):
    images = []
    labels = []
    for batch_images, batch_labels in dataset:
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())
    X = np.concatenate(images, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def _build_base_model(num_classes: int, input_shape, n_components: int) -> tf.keras.Model:
    flatten_dim = int(np.prod(input_shape))
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.Reshape((flatten_dim,)),
            tf.keras.layers.Dense(
                n_components,
                activation=None,
                use_bias=True,
                name="pca_projection",
                trainable=False,
            ),
            tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
                use_bias=True,
                name="logistic_head",
                trainable=False,
            ),
        ],
        name="pca_logistic",
    )
    return model


def build_gesture_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (64, 64, 3),
    n_components: int = 128,
) -> tf.keras.Model:
    """Return the PCA+logistic architecture (weights should be loaded afterwards)."""
    return _build_base_model(num_classes, input_shape, n_components)


def train_pca_logistic_pipeline(
    train_dataset,
    val_dataset,
    class_names,
    input_shape,
    n_components: int = 128,
):
    input_shape = tuple(int(dim) for dim in input_shape)
    X_train, y_train = _dataset_to_numpy(train_dataset)
    X_val, y_val = _dataset_to_numpy(val_dataset)

    num_classes = len(class_names)
    flatten_dim = int(np.prod(input_shape))
    X_train = X_train.astype("float32").reshape(len(X_train), flatten_dim) / 255.0
    X_val = X_val.astype("float32").reshape(len(X_val), flatten_dim) / 255.0

    n_components = min(n_components, flatten_dim)
    pca = PCA(n_components=n_components, whiten=False)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    clf.fit(X_train_pca, y_train)

    train_probs = clf.predict_proba(X_train_pca)
    val_probs = clf.predict_proba(X_val_pca)
    train_preds = np.argmax(train_probs, axis=1)
    val_preds = np.argmax(val_probs, axis=1)

    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    train_loss = log_loss(y_train, train_probs)
    val_loss = log_loss(y_val, val_probs)

    print(
        f"[model_v3] PCA+LogReg -> train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
    )

    model = _build_base_model(num_classes, input_shape, n_components)
    pca_layer = model.get_layer("pca_projection")
    logistic_layer = model.get_layer("logistic_head")

    weights_pca = pca.components_.T.astype("float32")
    bias_pca = (-pca.mean_ @ weights_pca).astype("float32")
    pca_layer.build((None, flatten_dim))
    pca_layer.set_weights([weights_pca, bias_pca])

    coef = clf.coef_.astype("float32")
    intercept = clf.intercept_.astype("float32")
    logistic_layer.build((None, n_components))
    logistic_layer.set_weights([coef.T, intercept])

    history = {
        "accuracy": [float(train_acc)],
        "val_accuracy": [float(val_acc)],
        "loss": [float(train_loss)],
        "val_loss": [float(val_loss)],
    }
    return history, model
