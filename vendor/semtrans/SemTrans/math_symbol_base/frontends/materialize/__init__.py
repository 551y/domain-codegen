# Adapted from NNSmith and modified for SemTrans; see THIRD_PARTY_NOTICES.md.

import os, sys
import importlib.util
import pickle, inspect, re
from os import PathLike
import numpy as np

from typing import Dict, TypeVar

# 1. First, import common interfaces from the registry to expose them for external use.
from SemTrans.testing_and_repair.logging import CONVERSION_LOG
from .registry import create_symbolnet, register_symbolnet, detect_input_ops

# 2. Import submodules to trigger the @register_symbolnet decorator
# This ensures all subclasses are automatically registered when the user imports materialize


"""
The test generation component based on NNsmith ASPLOS'23, GitHub repo: https://github.com/ise-uiuc/nnsmith.
"""

__all__ = ['create_symbolnet', 'detect_input_ops']


class Oracle:
    def __init__(
            self,
            input: Dict[str, np.ndarray],
            output: Dict[str, np.ndarray],
            provider: str = "unknown",
    ) -> None:
        self.input = input
        self.output = output
        self._provider = provider

    def __repr__(self) -> str:
        return f"input={self.input}, output={self.output}"

    @staticmethod
    def name() -> str:
        return "oracle.pkl"

    @property
    def provider(self) -> str:
        return self._provider

    def dump(self, path: PathLike) -> None:
        with open(path, "wb") as f:
            to_dump = {
                "input": self.input,
                "output": self.output,
                "provider": self.provider,
            }
            pickle.dump(to_dump, f)

    @staticmethod
    def load(path: PathLike) -> "Oracle":
        with open(path, "rb") as f:
            to_load = pickle.load(f)
            return Oracle(to_load["input"], to_load["output"], to_load["provider"])


MT = TypeVar("MT", bound="Model")


def load_model_class(file_path, class_name=None, init_args=None, init_kwargs=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    
    module_name = str(os.path.basename(file_path).replace(".py", ""))
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    
    if class_name:
        
        if hasattr(module, class_name):
            return getattr(module, class_name)

        
        
        for name, obj in module.__dict__.items():
            if isinstance(obj, type) and obj.__module__ == module_name:
                if name.lower() == class_name.lower():
                    CONVERSION_LOG.info(f"[DiffTest] Found '{name}' matching '{class_name}' (case-insensitive).")
                    return obj

        
        available_classes = [name for name, obj in module.__dict__.items()
                             if isinstance(obj, type) and obj.__module__ == module_name]
        raise AttributeError(f"Class '{class_name}' not found in {file_path}. Available classes: {available_classes}")

    
    found_classes = []
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and obj.__module__ == module_name:
            if any(k in name for k in ["Net", "Model", "Layer", "BERT", "ResNet", "VGG"]):
                found_classes.append(obj)

    if len(found_classes) == 1:
        return found_classes[0]
    elif len(found_classes) > 1:
        class_names = [cls.__name__ for cls in found_classes]
        
        CONVERSION_LOG.warning(f"[Loader Warning] Multiple classes found: {class_names}. Auto-selecting '{class_names[-1]}'.")
        return found_classes[-1]
    else:
        
        all_classes = [obj for name, obj in module.__dict__.items()
                       if isinstance(obj, type) and obj.__module__ == module_name]
        if len(all_classes) > 0:
            return all_classes[-1]  

        raise ValueError(f"No model class found in {file_path}.")


def load_model_instance(file_path, model_name, device, init_args=None, init_kwargs=None):
    """Load Model Instance"""
    if init_args is None: init_args = []
    if init_kwargs is None: init_kwargs = {}

    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.py':
        ModelClass = load_model_class(file_path, model_name)
        return ModelClass(*init_args, **init_kwargs)
    elif ext == '.onnx':
        import onnxruntime
        providers = ['CUDAExecutionProvider'] if 'gpu' in device else ['CPUExecutionProvider']
        return onnxruntime.InferenceSession(file_path, providers=providers)
    elif ext in ['.pt', '.pth']:
        import torch
        return torch.jit.load(file_path, map_location=device)
    elif ext == '.pdmodel':
        import paddle
        prefix = file_path.replace('.pdmodel', '')
        return paddle.jit.load(prefix)
    else:
        if os.path.isdir(file_path):
            import tensorflow as tf
            return tf.saved_model.load(file_path)
        raise ValueError(f"Unsupported file format: {ext}")


def diagnosis_init_loader(model_path, model_name, device, init_args, init_kwargs):
    return load_model_instance(
        model_path, model_name, device=device,
        init_args=init_args, init_kwargs=init_kwargs
    )


