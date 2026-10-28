import importlib
import inspect, re
import sys
from typing import Dict

from SemTrans.math_symbol_base.frontends.abstract import AbsTensor
from SemTrans.testing_and_repair.logging import CONVERSION_LOG
from SemTrans.testing_and_repair.util import detect_framework_type

_symbolnet_registry = {}


def register_symbolnet(framework_base_type):
    def decorator(symbolnet_class):
        _symbolnet_registry[framework_base_type] = symbolnet_class
        return symbolnet_class

    return decorator


_FRAMEWORK_ALIASES = {
    "torch": "pytorch",
    "trt": "tensorrt",
    "torch_tensorrt": "tensorrt",
    "relay": "tvm",
}

_SYMBOLNET_MODULES = {
    "pytorch": ".torch.torchnet",
    "paddle": ".paddle.paddlenet",
    "tensorflow": ".tensorflow.tfnet",
    "mindspore": ".mindspore.mindsporenet",
    "tensorrt": ".tensorrt.tensorrtnet",
    "tvm": ".tvm.tvmnet",
    "onnx": ".onnx.onnxnet",
    "tflite": ".tflite.tflitenet",
}


def create_symbolnet(model, use_gradient=False, input_map=None, framework_type=None, **kwargs):
    """
    Factory function: Dynamically loads the corresponding module based on the model type,
    and returns a SymbolNet instance.
    """
    # 1. Determine the framework type
    if framework_type is None:
        if isinstance(model, str):
            framework_type = model.lower()
        else:
            framework_type = detect_framework_type(model)
    framework_type = _FRAMEWORK_ALIASES.get(str(framework_type).lower(), str(framework_type).lower())

    # 2. [Core Modification] Lazy Import
    # Only import the corresponding file when a specific framework is actually needed.
    # Upon import, the @register_symbolnet decorator in the file executes, registering the class into _symbolnet_registry.

    module_name = _SYMBOLNET_MODULES.get(framework_type)
    if module_name is None:
        raise ValueError(f"Unsupported framework: {framework_type}. (Module import failed or decorator missing)")

    if framework_type not in _symbolnet_registry:
        importlib.import_module(module_name, package=__package__)

    if framework_type not in _symbolnet_registry:
        raise RuntimeError(f"SymbolNet module for {framework_type} did not register a frontend")

    return _symbolnet_registry[framework_type](
        model,
        framework_type,
        use_gradient=use_gradient,
        force_input_map=input_map,
        **kwargs,
    )


import inspect
import re
import sys
from typing import Dict


def detect_input_ops(model) -> Dict[str, 'AbsTensor']:
    """
    Performs input detection via strict signature parsing or native Keras properties.
    """
    params = {}

    # ==========================================

    # ==========================================
    if hasattr(model, 'inputs') and model.inputs:
        for i, inp in enumerate(model.inputs):

            # Keras may expose TensorShape, tuple/list, or a partially-known
            # shape depending on whether the model was built eagerly or from a
            # Functional graph.  Normalize all of them through one path.
            raw_shape_obj = getattr(inp, 'shape', None)
            try:
                if hasattr(raw_shape_obj, 'as_list'):
                    raw_shape = raw_shape_obj.as_list()
                elif raw_shape_obj is None:
                    raw_shape = [-1]
                else:
                    raw_shape = list(raw_shape_obj)
            except (TypeError, ValueError):
                raw_shape = [-1]

            shape = [-1 if s is None else int(s) for s in raw_shape]
            dtype_obj = getattr(inp, 'dtype', None)
            dtype = getattr(dtype_obj, 'name', None) or str(dtype_obj or 'float32')
            # Keras occasionally renders a dtype object as ``<dtype: ...>``;
            # keep the abstract frontend contract to a plain dtype name.
            if dtype.startswith('<dtype:') and "'" in dtype:
                dtype = dtype.split("'", 2)[1]

            name = inp.name.split(':')[0] if hasattr(inp, 'name') else f'input_{i}'
            params[name] = AbsTensor(shape, dtype)

        return params

    target_method = None
    if hasattr(model, 'forward'):
        target_method = model.forward
    elif hasattr(model, 'call'):
        target_method = model.call

    if target_method is None:
        raise ValueError("Model has no 'forward' or 'call' method, and no native 'inputs' property.")

    try:
        sig = inspect.signature(target_method)
    except Exception as e:
        raise ValueError(f"Cannot detect signature: {e}")

    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'training', 'kwargs', 'args'): continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD): continue

        annotation = param.annotation
        if annotation == inspect.Parameter.empty:
            continue

        anno_str = str(annotation)

        if 'Tensor' in anno_str:
            tensor_match = re.search(r'Tensor\s*[(\[<](.+?)[)\]>]', anno_str)

            if not tensor_match:
                CONVERSION_LOG.critical("=" * 80)
                CONVERSION_LOG.critical(" [CRITICAL ERROR] Ambiguous Input Definition")
                CONVERSION_LOG.critical("-" * 80)
                CONVERSION_LOG.critical(f" Reason: Invalid type annotation found for argument '{param_name}'.")
                CONVERSION_LOG.critical(f" Current Annotation: '{anno_str}'")
                CONVERSION_LOG.critical("-" * 80)
                CONVERSION_LOG.critical(" >>> HOW TO FIX:")
                CONVERSION_LOG.critical(" For SemTrans, you MUST specify the Tensor shape in:")
                CONVERSION_LOG.critical(" Forward / Call / __call__ / Construct methods.")
                CONVERSION_LOG.critical(" ")
                CONVERSION_LOG.critical(" [Allowed Examples] (Please use specific shapes/types):")
                CONVERSION_LOG.critical("  - 'Tensor(int, 3, 224, 224)'")
                CONVERSION_LOG.critical("  - 'Tensor[int, int]'")
                CONVERSION_LOG.critical(" ")
                CONVERSION_LOG.critical(" [Forbidden Examples] (Too vague):")
                CONVERSION_LOG.critical("  - 'Tensor'")
                CONVERSION_LOG.critical("  - 'xxx.Tensor'")
                CONVERSION_LOG.critical("=" * 80 + "\n")
                raise ValueError(
                    f"Ambiguous Tensor annotation for input '{param_name}': {anno_str}. "
                    "Provide an explicit tensor shape and dtype."
                )

            dtype = 'float32'  # default Dtype
            if tensor_match:
                content = tensor_match.group(1)
                parts = [p.strip() for p in content.split(',')]

                shapes = []

                for part in parts:
                    if part in ['float32', 'float64', 'int32', 'int64', 'bool']:
                        dtype = part
                        continue

                    if part == 'int':
                        shapes.append(-1)
                    elif part.isdigit():
                        shapes.append(int(part))
                    else:
                        shapes.append(-1)
            else:
                shapes = [1, 3, 224, 224]

            params[param_name] = AbsTensor(shapes, dtype)

    return params

