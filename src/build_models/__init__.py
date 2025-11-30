"""Model registry for gesture classifiers."""
from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict

MODEL_REGISTRY: Dict[str, str] = {
    "model_v1": "build_models.model_v1:build_gesture_model",
    "model_v2": "build_models.model_v2:build_gesture_model",
}


def list_available_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def get_model_builder(name: str) -> Callable[..., object]:
    target = MODEL_REGISTRY.get(name)
    if target is None:
        raise ValueError(f"Unknown model '{name}'. Available: {list_available_models()}")
    module_name, attr = target.split(":")
    module = import_module(module_name)
    builder = getattr(module, attr)
    return builder
