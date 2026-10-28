import ast
import copy
import itertools
import time
import numpy as np
import sympy, os, yaml, re, sys
import math
import hashlib
import traceback
from collections import namedtuple
from contextlib import contextmanager

from typing import List, Any, Generator, Optional, Dict, Tuple
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_equals_signs
from sympy.printing.pycode import PythonCodePrinter
from sympy.printing.numpy import NumPyPrinter

from SemTrans.math_symbol_base.symbolic_system.custom_symbolic_ops import S_None
from SemTrans.math_symbol_base.symbolic_system.math_symbol_space import SymbolSpace, SmartTensor, string_token_transform, \
    keyword_none_transform, SymStr, apply_symbolic_lowering
from SemTrans.math_symbol_base.backends import *


def _print_GetItem(self, expr):
    tensor_name = self._print(expr.args[0])
    indices = [self._print(idx) for idx in expr.args[1:]]
    return f"{tensor_name}[{', '.join(indices)}]"


PythonCodePrinter._print_GetItem = _print_GetItem
MatchResult = namedtuple('MatchResult', ['type', 'src_block', 'tgt_block', 'gap_info'])


def _safe_numpy(value):
    """Convert framework tensors to detached CPU arrays without gradients."""
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)


def _tflite_fused_activation(value, code):
    code = int(code)
    if code == 0:
        return value
    if code == 1:
        return np.maximum(value, 0)
    if code == 2:
        return np.clip(value, -1, 1)
    if code == 3:
        return np.clip(value, 0, 6)
    if code == 4:
        return np.tanh(value)
    raise ValueError(f'unqualified TFLite fused activation code {code}')


def preprocess_yaml_expr(expr_input):
    """Normalize YAML math expressions before SymPy parsing."""
    if isinstance(expr_input, (list, tuple)):
        return [preprocess_yaml_expr(e) for e in expr_input]

    if not isinstance(expr_input, str):
        return expr_input

    expr_str = expr_input

    # Lower simple equality checks into SymPy Eq calls.
    pattern_eq = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*==\s*([a-zA-Z_][a-zA-Z0-9_]*|True|False)"
    expr_str = re.sub(pattern_eq, r"Eq(\1, \2)", expr_str)

    expr_str = re.sub(r"\blen\s*\(\s*(\w+)\.shape\s*\)", r"Rank(\1)", expr_str)
    expr_str = re.sub(r"(\w+)\.shape(?![\w\(])", r"Shape(\1)", expr_str)
    expr_str = re.sub(r"(\w+)\.T\b", r"Transpose(\1)", expr_str)
    expr_str = re.sub(r"(\w+)\.cast\s*\(", r"Cast(\1, ", expr_str)

    # Lower attribute access such as X.dtype into GetAttr(X, 'dtype').
    pattern_attr = r"(?<!['\"])\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w+)\b(?!['\"])"

    def getattr_replacer(match):
        obj, attr = match.groups()
        if attr in ['shape', 'T']: return match.group(0)
        return f"GetAttr({obj}, '{attr}')"

    expr_str = re.sub(pattern_attr, getattr_replacer, expr_str)
    return expr_str


def _lower_tensor_bitwise_expr(expr_str):
    bitwise_funcs = {
        ast.BitAnd: 'BitwiseAnd',
        ast.BitOr: 'BitwiseOr',
        ast.BitXor: 'BitwiseXor',
    }

    class TensorBitwiseLowerer(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            for op_type, func_name in bitwise_funcs.items():
                if isinstance(node.op, op_type):
                    return ast.copy_location(
                        ast.Call(func=ast.Name(id=func_name, ctx=ast.Load()),
                                 args=[node.left, node.right], keywords=[]),
                        node
                    )
            return node

        def visit_UnaryOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.Invert):
                return ast.copy_location(
                    ast.Call(func=ast.Name(id='BitwiseNot', ctx=ast.Load()),
                             args=[node.operand], keywords=[]),
                    node
                )
            return node

    try:
        tree = ast.parse(expr_str, mode='eval')
        lowered = TensorBitwiseLowerer().visit(tree)
        ast.fix_missing_locations(lowered)
        return ast.unparse(lowered)
    except Exception:
        return expr_str


def _lower_control_flow_for_numpy_backend(expr):
    """Route symbolic control flow through NumpyBackend instead of NumPyPrinter."""
    if isinstance(expr, sympy.Piecewise):
        pairs = [
            sympy.Tuple(
                _lower_control_flow_for_numpy_backend(pair.expr),
                _lower_control_flow_for_numpy_backend(pair.cond),
            )
            for pair in expr.args
        ]
        return sympy.Function('__ST_Piecewise')(*pairs)
    if isinstance(expr, sympy.And):
        return sympy.Function('__ST_And')(
            *[_lower_control_flow_for_numpy_backend(arg) for arg in expr.args]
        )
    if isinstance(expr, sympy.Or):
        return sympy.Function('__ST_Or')(
            *[_lower_control_flow_for_numpy_backend(arg) for arg in expr.args]
        )
    if isinstance(expr, sympy.Not):
        return sympy.Function('__ST_Not')(
            _lower_control_flow_for_numpy_backend(expr.args[0])
        )
    if not isinstance(expr, sympy.Basic) or not expr.args:
        return expr

    lowered_args = tuple(
        _lower_control_flow_for_numpy_backend(arg) for arg in expr.args
    )
    try:
        return expr.func(*lowered_args)
    except Exception:
        return expr


class RuntimeMathCompiler:
    def __init__(self):
        self._compiled_kernel_cache = {}

    @staticmethod
    def _diagnostic_value(value, depth=0):
        """Keep MathIR evidence structural and bounded for audit/prompt use."""
        if depth > 3:
            return "<nested>"
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, sympy.Basic):
            return str(value)
        if isinstance(value, np.ndarray):
            return {"dtype": str(value.dtype), "shape": list(value.shape)}
        if isinstance(value, dict):
            return {
                str(key): RuntimeMathCompiler._diagnostic_value(item, depth + 1)
                for key, item in list(value.items())[:64]
            }
        if isinstance(value, (list, tuple)):
            return [RuntimeMathCompiler._diagnostic_value(item, depth + 1) for item in value[:64]]
        if hasattr(value, "shape"):
            return {"dtype": str(getattr(value, "dtype", "unknown")), "shape": list(value.shape)}
        return str(value)

    @staticmethod
    def _canonical_formula_expr(expr, op_instance, op_def):
        """Render spatial formulas in the common N,C,H,W coordinate convention.

        The executable MathIR remains the framework-native definition. This
        canonical view is derived from that definition and is used solely for
        cross-frontend formula alignment/evidence; it never replaces native
        framework execution.
        """
        attrs = getattr(op_instance, "attrs", {}) or {}
        # Frontends annotate the physical tensor layout once while parsing
        # the native operator.  The shared evaluator consumes that semantic
        # token; it must not infer a conversion route from an operation name
        # (for example, ``Conv2D`` versus ``Pooling``).
        layout = str(attrs.get("semantic_layout") or attrs.get("data_format") or "").upper()
        layout_aliases = {
            "CHANNELS_LAST": "NHWC",
            "CHANNELS_FIRST": "NCHW",
            "NWC": "NWC",
            "NCW": "NCW",
            "NDHWC": "NDHWC",
            "NCDHW": "NCDHW",
        }
        layout = layout_aliases.get(layout, layout)
        if not layout.endswith("HWC") and layout not in {"NHWC", "NWC"}:
            return str(expr)

        op_name = str(getattr(op_instance, "op_type", "")).lower()

        rename = {
            "Pad_Top": "P_h", "Pad_Left": "P_w",
            "filters": "cout", "strides": "S_h", "dilation_rate": "D_h",
            "use_bias": "bias",
        }

        def transform(node):
            if isinstance(node, sympy.Indexed):
                label = getattr(node.base, "label", None)
                label_name = getattr(label, "name", str(label))
                indices = list(node.indices)
                if len(indices) == 4 and label_name in {"X", "Y", "Output"}:
                    indices = [indices[0], indices[3], indices[1], indices[2]]
                elif len(indices) == 4 and label_name in {"W", "weight"} and (
                    str(attrs.get("semantic_weight_layout", "")).upper() == "HWIO"
                    or "conv" in op_name
                ):
                    # A frontend may declare its kernel layout explicitly.
                    # Keep the old Conv* fallback for existing definitions;
                    # no model or route is selected here.
                    indices = [indices[3], indices[2], indices[0], indices[1]]
                try:
                    return node.base[tuple(indices)]
                except Exception:
                    return node
            if isinstance(node, sympy.Symbol) and node.name in rename:
                return sympy.Symbol(rename[node.name])
            if not isinstance(node, sympy.Basic) or not node.args:
                return node
            args = tuple(transform(arg) for arg in node.args)
            try:
                return node.func(*args)
            except Exception:
                return node

        return transform(expr)

    def _formula_diagnostic(self, op_instance, op_def, specialized_expr, subs_map, symnet):
        attrs = getattr(op_instance, "attrs", {}) or {}
        param_bindings = {}
        name_by_symbol = {
            getattr(symbol, "name", str(symbol)): value
            for symbol, value in (subs_map or {}).items()
            if not isinstance(value, dict)
        }
        for attr_name, mapped in (op_def.get("param_map", {}) or {}).items():
            names = mapped if isinstance(mapped, (list, tuple)) else [mapped]
            for name in names:
                if name in name_by_symbol:
                    param_bindings[str(name)] = self._diagnostic_value(name_by_symbol[name])
                elif attr_name in attrs:
                    param_bindings[str(name)] = self._diagnostic_value(attrs[attr_name])
        derived_bindings = {}
        for name in (op_def.get("derived_variables", {}) or {}):
            if name in name_by_symbol:
                derived_bindings[name] = self._diagnostic_value(name_by_symbol[name])

        # Normalize framework vocabulary into the MathIR comparison namespace.
        # These are symbol aliases, not source-code or model-specific rules.
        canonical_aliases = {
            "filters": "cout",
            "use_bias": "bias",
            # Paddle Linear calls the boolean bias switch ``bias_attr`` and
            # the MathIR definition names it ``ba``.  Both represent the
            # same semantic ``bias`` predicate used by the PyTorch formula.
            "ba": "bias",
            "Pad_Top": "P_h",
            "Pad_Left": "P_w",
        }
        canonical_bindings = {}
        for name, value in param_bindings.items():
            canonical_bindings[canonical_aliases.get(name, name)] = value
        for name, value in derived_bindings.items():
            canonical_name = canonical_aliases.get(name)
            if canonical_name is not None:
                canonical_bindings[canonical_name] = value
        bias_binding = canonical_bindings.get("bias")
        if bias_binding is None or bias_binding == "None":
            canonical_bindings["bias"] = False
        # Reduce numeric derived arithmetic (for example TensorFlow SAME's
        # ``floor(Max(0, 5) / 2)``) before rendering evidence.
        concrete_expr = specialized_expr
        try:
            def reduce_numeric(node):
                if not isinstance(node, sympy.Basic) or not node.args:
                    return node
                args = tuple(reduce_numeric(arg) for arg in node.args)
                name = getattr(getattr(node, "func", None), "__name__", "")
                if name == "Max" and all(not getattr(arg, "free_symbols", set()) for arg in args):
                    return sympy.Max(*args)
                if name == "Min" and all(not getattr(arg, "free_symbols", set()) for arg in args):
                    return sympy.Min(*args)
                if name in {"floor", "ceiling"} and len(args) == 1 and not getattr(args[0], "free_symbols", set()):
                    return sympy.floor(args[0]) if name == "floor" else sympy.ceiling(args[0])
                try:
                    return node.func(*args)
                except Exception:
                    return node
            concrete_expr = reduce_numeric(specialized_expr)
        except Exception:
            pass
        canonical = self._canonical_formula_expr(concrete_expr, op_instance, op_def)
        return {
            "operation": str(getattr(op_instance, "op_type", type(op_instance).__name__)),
            "framework": str(getattr(symnet, "framework_type", "unknown")),
            "native_formula": str(concrete_expr),
            "canonical_formula": str(canonical),
            "parameter_bindings": param_bindings,
            "derived_bindings": derived_bindings,
            "canonical_bindings": canonical_bindings,
            "layout": {
                "data_format": attrs.get("data_format"),
                "semantic_layout": attrs.get("semantic_layout"),
                "canonical_layout": "NCHW" if str(attrs.get("data_format", "")).lower().endswith("last") else None,
            },
        }

    def _freeze_cache_value(self, value):
        if isinstance(value, dict):
            return tuple(sorted((k, self._freeze_cache_value(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_cache_value(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(self._freeze_cache_value(v) for v in value))
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
            try:
                shape = tuple(
                    None if dimension is None else int(dimension)
                    for dimension in value.shape
                )
            except (TypeError, ValueError):
                shape = str(value.shape)
            return ('array_like', shape, str(value.dtype))
        return value

    def _make_compilation_cache_key(self, op_def, op_instance, inputs, extra_kwargs):
        trace_progress = os.getenv('SEMTRANS_TRACE_PROGRESS', '') == '1'

        def freeze_component(label, value):
            started = time.perf_counter()
            frozen = self._freeze_cache_value(value)
            if trace_progress:
                print(
                    f"[RuntimeMathCompilerCacheKey] component={label} "
                    f"elapsed={time.perf_counter() - started:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )
            return frozen

        input_shapes = tuple(
            tuple(inp.shape) if hasattr(inp, 'shape') and isinstance(inp.shape, (list, tuple)) else
            (inp.shape if hasattr(inp, 'shape') else None)
            for inp in inputs
        )
        op_type = getattr(op_instance, 'op_type', type(op_instance).__name__)
        semantic_definition = {
            key: op_def.get(key)
            for key in (
                'math_symbol', 'derived_variables', 'output_shape',
                'param_map', 'default_values', 'mechanisms',
            )
        }
        op_attrs = getattr(op_instance, 'attrs', {}) or {}
        frozen_semantic_definition = freeze_component(
            'semantic_definition', semantic_definition
        )
        if trace_progress:
            frozen_attr_items = []
            for attr_name, attr_value in op_attrs.items():
                frozen_attr_items.append((
                    attr_name,
                    freeze_component(f'op_attrs.{attr_name}', attr_value),
                ))
            frozen_op_attrs = tuple(sorted(frozen_attr_items))
        else:
            frozen_op_attrs = self._freeze_cache_value(op_attrs)
        frozen_kwargs = freeze_component('extra_kwargs', extra_kwargs)
        return (
            op_def.get('name', op_type),
            input_shapes,
            frozen_semantic_definition,
            frozen_op_attrs,
            frozen_kwargs,
        )

    @contextmanager
    def _timer(self, name):
        start = time.time()
        yield
        end = time.time()
        stream = (
            sys.stderr
            if os.getenv('SEMTRANS_TRACE_PROGRESS', '') == '1'
            else sys.stdout
        )
        print(
            f"[Profiler] [{name}] Cost: {end - start:.4f}s",
            file=stream,
            flush=True,
        )

    def _eval_int_bound(self, expr, subs_map=None):
        value = expr
        if subs_map is not None and hasattr(value, 'subs'):
            value = value.subs(subs_map)
        if hasattr(value, 'doit'):
            value = value.doit()
        try:
            value = sympy.simplify(value)
        except Exception:
            pass
        if hasattr(value, 'evalf'):
            try:
                value = value.evalf()
            except Exception:
                pass
        try:
            return int(float(value))
        except Exception:
            if getattr(value, 'free_symbols', None):
                raise
            evaluator = sympy.lambdify(
                (),
                value,
                modules=[{
                    'ceiling': lambda x: math.ceil(x),
                    'floor': lambda x: math.floor(x),
                    'Max': lambda *xs: max(xs),
                    'Min': lambda *xs: min(xs),
                    'Mod': lambda a, b: a % b,
                }, 'math']
            )
            return int(float(evaluator()))

    def _is_none_symbol(self, value):
        return value is None or value is S_None or type(value).__name__ == 'NoneSymbol'

    def _lookup_subs_value(self, node, subs_map, default=None):
        if isinstance(node, sympy.Symbol):
            for sym, value in subs_map.items():
                if hasattr(sym, 'name') and sym.name == node.name:
                    return value
        return default

    def _static_truth_value(self, node, subs_map):
        if self._is_none_symbol(node):
            return False
        if node is True or node is sympy.true:
            return True
        if node is False or node is sympy.false:
            return False
        if isinstance(node, sympy.Symbol):
            missing = object()
            value = self._lookup_subs_value(node, subs_map, missing)
            if value is missing:
                return None
            if self._is_none_symbol(value):
                return False
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, float, np.integer, np.floating)):
                return bool(value)
            return None
        if isinstance(node, sympy.Basic):
            scalar_subs = {}
            for sym, value in subs_map.items():
                if not hasattr(sym, 'name'):
                    continue
                if self._is_none_symbol(value):
                    scalar_subs[sym] = S_None
                elif isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
                    scalar_subs[sym] = value
            try:
                reduced = node.subs(scalar_subs).doit()
                if reduced is sympy.true:
                    return True
                if reduced is sympy.false:
                    return False
                if self._is_none_symbol(reduced):
                    return False
                if not getattr(reduced, 'free_symbols', None):
                    return bool(reduced)
            except Exception:
                pass
        return None

    def _static_scalar_value(self, node, tensor_map, subs_map):
        if isinstance(node, (int, np.integer)):
            return int(node)
        if isinstance(node, sympy.Integer):
            return int(node)
        if isinstance(node, sympy.Symbol):
            value = self._lookup_subs_value(node, subs_map)
            if isinstance(value, (int, np.integer)):
                return int(value)
        if isinstance(node, sympy.Function):
            func_name = node.func.__name__
            if func_name == 'Rank':
                shape = self._static_shape_for_node(node.args[0], tensor_map, subs_map)
                if shape is not None:
                    return len(shape)
            if func_name == 'make_swap_perm':
                rank = self._static_scalar_value(node.args[0], tensor_map, subs_map)
                dim_a = self._static_scalar_value(node.args[1], tensor_map, subs_map)
                dim_b = self._static_scalar_value(node.args[2], tensor_map, subs_map)
                if rank is not None and dim_a is not None and dim_b is not None:
                    return self._make_swap_perm(rank, dim_a, dim_b)
        return None

    def _static_shape_for_node(self, tensor_node, tensor_map, subs_map):
        t_name = tensor_node.name if hasattr(tensor_node, 'name') else str(tensor_node)
        if t_name in tensor_map:
            obj = tensor_map[t_name]
            shape = getattr(obj, 'shape', None)
            if shape is None and hasattr(obj, 'data'):
                shape = getattr(obj.data, 'shape', None)
            if shape is not None:
                return tuple(int(dim) for dim in shape)

        label_node = getattr(tensor_node, 'label', None)
        if label_node is not None and label_node is not tensor_node:
            label_shape = self._static_shape_for_node(label_node, tensor_map, subs_map)
            if label_shape is not None:
                return label_shape

        if hasattr(tensor_node, 'args') and len(getattr(tensor_node, 'args', ())) == 1:
            inner_node = tensor_node.args[0]
            if inner_node is not tensor_node:
                inner_shape = self._static_shape_for_node(inner_node, tensor_map, subs_map)
                if inner_shape is not None:
                    return inner_shape

        if isinstance(tensor_node, sympy.Symbol):
            value = self._lookup_subs_value(tensor_node, subs_map)
            if hasattr(value, 'shape'):
                return tuple(int(dim) for dim in value.shape)
            if isinstance(value, (list, tuple, np.ndarray)):
                try:
                    return tuple(int(dim) for dim in value)
                except Exception:
                    pass

        if isinstance(tensor_node, sympy.Piecewise):
            fallback_shape = None
            for value, condition in tensor_node.args:
                truth_value = self._static_truth_value(condition, subs_map)
                branch_shape = self._static_shape_for_node(value, tensor_map, subs_map)
                if truth_value is True:
                    return branch_shape
                if truth_value is False:
                    continue
                if fallback_shape is None:
                    fallback_shape = branch_shape
            return fallback_shape

        if isinstance(tensor_node, sympy.Function):
            func_name = tensor_node.func.__name__
            if func_name == 'Transpose':
                base_shape = self._static_shape_for_node(tensor_node.args[0], tensor_map, subs_map)
                if base_shape is None:
                    return None
                if len(tensor_node.args) == 1:
                    return tuple(reversed(base_shape))
                if len(tensor_node.args) == 2:
                    axes = self._static_scalar_value(tensor_node.args[1], tensor_map, subs_map)
                    if isinstance(axes, (list, tuple)):
                        return tuple(base_shape[int(axis)] for axis in axes)
                if len(tensor_node.args) == 3:
                    dim_a = self._static_scalar_value(tensor_node.args[1], tensor_map, subs_map)
                    dim_b = self._static_scalar_value(tensor_node.args[2], tensor_map, subs_map)
                    if dim_a is not None and dim_b is not None:
                        rank = len(base_shape)
                        if dim_a < 0:
                            dim_a += rank
                        if dim_b < 0:
                            dim_b += rank
                        perm = list(range(rank))
                        perm[dim_a], perm[dim_b] = perm[dim_b], perm[dim_a]
                        return tuple(base_shape[idx] for idx in perm)
        return None

    def _select_static_piecewise_branch(self, expr, subs_map):
        if isinstance(expr, sympy.Piecewise):
            unresolved = []
            for value, condition in expr.args:
                truth_value = self._static_truth_value(condition, subs_map)
                if truth_value is True:
                    return value
                if truth_value is False:
                    continue
                unresolved.append((value, condition))
            if not unresolved:
                return sympy.Integer(0)
            if len(unresolved) != len(expr.args):
                return sympy.Piecewise(*unresolved)
        return expr

    def _simplify_indexed_conditionals(self, expr, subs_map):
        def simplify_indexed(indexed_node):
            base = indexed_node.base
            label = getattr(base, 'label', None)
            if label is None and hasattr(base, 'args') and len(getattr(base, 'args', ())) == 1:
                label = base.args[0]
            selected = self._select_static_piecewise_branch(label, subs_map) if label is not None else label
            if selected is None or selected == label:
                return indexed_node
            new_base = selected if hasattr(selected, '__getitem__') else SmartTensor(selected)
            try:
                return new_base[tuple(indexed_node.indices)]
            except Exception:
                return indexed_node

        return expr.replace(lambda node: isinstance(node, sympy.Indexed), simplify_indexed)

    def _resolve_shape_macros(self, expr, resolution_ctx):
        inputs = resolution_ctx.get('inputs', [])
        input_names = resolution_ctx.get('input_names', [])
        out_shape = resolution_ctx.get('out_shape')
        out_idx = resolution_ctx.get('out_idx', 0)
        subs_map = resolution_ctx.get('subs_map', {})

        tensor_map = {}
        # Seed common input aliases.
        alias_to_idx_in = {
            'X': 0, 'X1': 0, 'A': 1, 'X2': 1, 'Indices': 1,
            'X3': 2, 'X4': 3, 'X5': 4, 'X6': 5,
            'Memory': 1,
            'Query': 0, 'Key': 1, 'Value': 2
        }
        for alias, idx in alias_to_idx_in.items():
            if idx < len(inputs):
                tensor_map[alias] = inputs[idx]

        for i, name in enumerate(input_names):
            if i < len(inputs):
                tensor_map[name] = inputs[i]

        # =========================================================
        # 2. Handle Weights (W)
        # =========================================================
        op_instance = resolution_ctx.get('op_instance')
        symnet = resolution_ctx.get('symnet')

        if op_instance and hasattr(op_instance, 'attrs'):
            for attr_name, attr_val in op_instance.attrs.items():
                if self._is_tensor(attr_val):
                    tensor_map[attr_name] = attr_val

                    if attr_name.startswith('weight_'):
                        tensor_map['W_' + attr_name[7:]] = attr_val
                    elif attr_name.startswith('bias_'):
                        tensor_map['B_' + attr_name[5:]] = attr_val
                    elif attr_name == 'weight':
                        tensor_map['W'] = attr_val
                    elif attr_name == 'bias':
                        tensor_map['B'] = attr_val

        if op_instance and symnet and hasattr(symnet, 'adapter'):
            try:
                lookup_keys = {'weight', 'W', 'bias', 'B'}
                for node in expr.atoms(sympy.Indexed):
                    label = getattr(node.base, 'label', None)
                    if hasattr(label, 'name'):
                        lookup_keys.add(label.name)
                for node in expr.atoms(sympy.Function):
                    if node.func.__name__ in {'Dim', 'Shape', 'Rank'} and node.args:
                        tensor_node = node.args[0]
                        if hasattr(tensor_node, 'name'):
                            lookup_keys.add(tensor_node.name)
                for key in lookup_keys:
                    if key not in tensor_map:
                        w = symnet.adapter.get_named_tensor(op_instance, key)
                        if w is not None: tensor_map[key] = w
            except:
                pass

        # =========================================================
        # 3.  (Y, Y1, Output...)
        # =========================================================
        if out_shape:
            out_proxy = type('OutputProxy', (object,), {'shape': out_shape})
            alias_to_idx_out = {
                'Y': out_idx,
                'Output': out_idx,
                f'Y{out_idx + 1}': out_idx
            }
            for alias in alias_to_idx_out:
                tensor_map[alias] = out_proxy

        replacements = {}
        for node in expr.atoms(sympy.Function):
            func_name = type(node).__name__

            if func_name == 'Shape':
                tensor_node = node.args[0]
                shape = self._static_shape_for_node(tensor_node, tensor_map, subs_map)
                if shape is not None:
                    composed_shape = sympy.Tuple(*[sympy.Integer(dim) for dim in shape])
                    replacements[node] = composed_shape
                    continue
            elif func_name == 'Dim':
                tensor_node, dim_idx = node.args
                t_name = tensor_node.name if hasattr(tensor_node, 'name') else str(tensor_node)

                try:
                    idx = self._static_scalar_value(dim_idx, tensor_map, subs_map)
                    if idx is None:
                        idx = int(dim_idx)
                    shape = self._static_shape_for_node(tensor_node, tensor_map, subs_map)
                    if shape is not None and -len(shape) <= idx < len(shape):
                        replacements[node] = sympy.Integer(shape[idx])
                        continue

                    sym_name = f"DIM_{t_name}_{idx}"
                    replacements[node] = sympy.Symbol(sym_name, integer=True)
                except Exception:
                    pass

            elif func_name == 'Rank':
                tensor_node = node.args[0]
                t_name = tensor_node.name if hasattr(tensor_node, 'name') else str(tensor_node)
                shape = self._static_shape_for_node(tensor_node, tensor_map, subs_map)
                if shape is not None:
                    replacements[node] = sympy.Integer(len(shape))
                    continue

                found_in_subs = None
                for s, v in subs_map.items():
                    if hasattr(s, 'name') and s.name == t_name:
                        found_in_subs = v
                        break

                if found_in_subs is not None:
                    ndim = len(found_in_subs) if isinstance(found_in_subs, (list, tuple)) else 0
                    replacements[node] = sympy.Integer(ndim)
                    continue

                if t_name in tensor_map:
                    obj = tensor_map[t_name]
                    ndim = getattr(obj, 'ndim', None) or (len(obj.shape) if hasattr(obj, 'shape') else None)
                    if ndim is not None:
                        replacements[node] = sympy.Integer(ndim)
                        continue
                replacements[node] = sympy.Symbol(f"RANK_{t_name}", integer=True)

        return expr.subs(replacements)

    def _should_preserve_reduction_structure(self, expr_new, body, limits, op_def=None):
        if not limits:
            return False

        loop_vars = [item[0] for item in limits if isinstance(item, (tuple, list, sympy.Tuple)) and len(item) >= 1]
        indexed_count = len(body.atoms(sympy.Indexed))
        piecewise_count = len(body.atoms(sympy.Piecewise))
        minmax_count = len(body.atoms(sympy.Max)) + len(body.atoms(sympy.Min))
        relation_count = sum(
            len(body.atoms(rel_type))
            for rel_type in (
                sympy.Equality,
                sympy.GreaterThan,
                sympy.StrictGreaterThan,
                sympy.LessThan,
                sympy.StrictLessThan,
            )
        )

        if len(limits) >= 2 and indexed_count > 0 and (piecewise_count > 0 or minmax_count > 0):
            return True

        for indexed_node in body.atoms(sympy.Indexed):
            for index_expr in indexed_node.indices:
                free_syms = getattr(index_expr, "free_symbols", set())
                if any(loop_var in free_syms for loop_var in loop_vars):
                    if not isinstance(index_expr, sympy.Symbol):
                        return True

        structural_cost = (
            indexed_count * 8
            + piecewise_count * 12
            + minmax_count * 6
            + relation_count * 4
            + len(loop_vars) * 10
        )
        if structural_cost >= 24:
            return True

        if op_def:
            desc = str(op_def.get('description', '')).lower()
            keywords = [
                'convolution',
                'pool',
                'pooling',
                'window',
                'unpool',
                'local response normalization',
            ]
            if any(keyword in desc for keyword in keywords):
                return True

        return False

    def _estimate_expr_cost(self, expr):
        if not isinstance(expr, sympy.Basic):
            return 1
        try:
            op_cost = int(sympy.count_ops(expr, visual=False))
        except Exception:
            op_cost = 0
        indexed_cost = len(expr.atoms(sympy.Indexed)) * 4
        piecewise_cost = len(expr.atoms(sympy.Piecewise)) * 8
        minmax_cost = (len(expr.atoms(sympy.Max)) + len(expr.atoms(sympy.Min))) * 3
        relation_cost = sum(
            len(expr.atoms(rel_type))
            for rel_type in (
                sympy.Equality,
                sympy.GreaterThan,
                sympy.StrictGreaterThan,
                sympy.LessThan,
                sympy.StrictLessThan,
            )
        ) * 2
        reduction_cost = 0
        for node in expr.atoms(sympy.Function):
            name = getattr(node.func, "__name__", "")
            if name in {"TensorSum", "TensorProd", "TensorMax", "TensorMin", "Sum", "Product", "Max", "Min"}:
                reduction_cost += 10
        return op_cost + indexed_cost + piecewise_cost + minmax_cost + relation_cost + reduction_cost

    def _infer_reduction_axis_from_body(self, body, loop_var):
        if not isinstance(loop_var, sympy.Symbol):
            return None
        candidate_axes = []
        for indexed_node in body.atoms(sympy.Indexed):
            for axis_idx, index_expr in enumerate(indexed_node.indices):
                free_syms = getattr(index_expr, "free_symbols", set())
                if loop_var in free_syms:
                    candidate_axes.append(axis_idx)
        if candidate_axes and len(set(candidate_axes)) == 1:
            return candidate_axes[0]
        return None

    def _factor_common_terms(self, expr, memo=None):
        if memo is None:
            memo = {}
        if not isinstance(expr, sympy.Basic):
            return expr
        if expr in memo:
            return memo[expr]
        if not expr.args:
            memo[expr] = expr
            return expr

        factored_args = [self._factor_common_terms(arg, memo=memo) for arg in expr.args]
        try:
            expr_new = expr.func(*factored_args)
        except Exception:
            expr_new = expr

        expr_name = type(expr_new).__name__
        if expr_name in {"TensorSum", "TensorProd", "TensorMax", "TensorMin"}:
            memo[expr] = expr_new
            return expr_new

        candidate = expr_new
        if getattr(expr_new, "is_Add", False) or getattr(expr_new, "is_Mul", False):
            try:
                candidate = sympy.factor_terms(expr_new)
            except Exception:
                candidate = expr_new

        old_cost = self._estimate_expr_cost(expr_new)
        new_cost = self._estimate_expr_cost(candidate)
        if new_cost <= old_cost:
            memo[expr] = candidate
            return candidate

        memo[expr] = expr_new
        return expr_new

    def _merge_piecewise_masks(self, expr, memo=None):
        if memo is None:
            memo = {}
        if not isinstance(expr, sympy.Basic):
            return expr
        if expr in memo:
            return memo[expr]
        if not expr.args:
            memo[expr] = expr
            return expr

        new_args = [self._merge_piecewise_masks(arg, memo=memo) for arg in expr.args]
        try:
            expr_new = expr.func(*new_args)
        except Exception:
            expr_new = expr

        if not getattr(expr_new, "is_Mul", False):
            memo[expr] = expr_new
            return expr_new

        other_factors = []
        payload_factors = []
        mask_conditions = []

        for factor in expr_new.args:
            if isinstance(factor, sympy.Piecewise) and len(factor.args) == 2:
                first_case, second_case = factor.args
                first_expr, first_cond = first_case.expr, first_case.cond
                second_expr, second_cond = second_case.expr, second_case.cond

                if second_cond == sympy.true and second_expr == 0:
                    mask_conditions.append(first_cond)
                    if first_expr != 1:
                        payload_factors.append(first_expr)
                    continue

            other_factors.append(factor)

        if not mask_conditions:
            memo[expr] = expr_new
            return expr_new

        merged_cond = sympy.And(*mask_conditions) if len(mask_conditions) > 1 else mask_conditions[0]
        merged_payload = sympy.Mul(*(other_factors + payload_factors), evaluate=False) if (other_factors or payload_factors) else sympy.Integer(1)
        candidate = sympy.Piecewise((merged_payload, merged_cond), (sympy.Integer(0), sympy.true))

        if self._estimate_expr_cost(candidate) <= self._estimate_expr_cost(expr_new):
            memo[expr] = candidate
            return candidate

        memo[expr] = expr_new
        return expr_new

    def _build_symbol_bounds(self, binding, out_shape, subs_map):
        bounds = {}
        for sym_name, dim_idx in binding.items():
            try:
                upper = int(out_shape[dim_idx]) - 1
                bounds[sym_name] = (0, upper)
            except Exception:
                pass

        reduction_key = sympy.Symbol('_reduction_map')
        reduction_map = subs_map.get(reduction_key, {}) if isinstance(subs_map, dict) else {}
        for sym_name, spec in reduction_map.items():
            try:
                start = spec.get('start')
                end = spec.get('end')
                if start is not None and end is not None:
                    bounds[sym_name] = (int(start), int(end))
            except Exception:
                pass
        return bounds

    def _interval_of_expr(self, expr, bounds):
        if isinstance(expr, (sympy.Integer, sympy.Float)):
            value = float(expr)
            return value, value
        if isinstance(expr, sympy.Symbol):
            return bounds.get(expr.name)

        if getattr(expr, 'is_Add', False):
            current = (0.0, 0.0)
            for arg in expr.args:
                interval = self._interval_of_expr(arg, bounds)
                if interval is None:
                    return None
                current = (current[0] + interval[0], current[1] + interval[1])
            return current

        if getattr(expr, 'is_Mul', False):
            coeff, remainder = expr.as_coeff_Mul()
            if remainder == expr:
                return None
            if remainder == 1:
                value = float(coeff)
                return value, value
            base_interval = self._interval_of_expr(remainder, bounds)
            if base_interval is None:
                return None
            coeff_value = float(coeff)
            candidates = [coeff_value * base_interval[0], coeff_value * base_interval[1]]
            return min(candidates), max(candidates)

        if isinstance(expr, sympy.floor):
            base_interval = self._interval_of_expr(expr.args[0], bounds)
            if base_interval is None:
                return None
            return np.floor(base_interval[0]), np.floor(base_interval[1])

        if isinstance(expr, sympy.Mod):
            lhs_interval = self._interval_of_expr(expr.args[0], bounds)
            rhs_interval = self._interval_of_expr(expr.args[1], bounds)
            if lhs_interval is None or rhs_interval is None or rhs_interval[0] != rhs_interval[1]:
                return None
            modulus = int(rhs_interval[0])
            if modulus <= 0:
                return None
            if lhs_interval[0] >= 0 and lhs_interval[1] < modulus:
                return lhs_interval
            return 0, modulus - 1

        return None

    def _simplify_by_bounds(self, expr, symbol_bounds, memo=None):
        if memo is None:
            memo = {}
        if not isinstance(expr, sympy.Basic):
            return expr
        if expr in memo:
            return memo[expr]
        if not expr.args:
            memo[expr] = expr
            return expr

        try:
            new_args = [self._simplify_by_bounds(arg, symbol_bounds, memo=memo) for arg in expr.args]
        except RecursionError:
            memo[expr] = expr
            return expr
        try:
            expr_new = expr.func(*new_args)
        except Exception:
            expr_new = expr

        if isinstance(expr_new, sympy.Mod):
            try:
                lhs_interval = self._interval_of_expr(expr_new.args[0], symbol_bounds)
                rhs_interval = self._interval_of_expr(expr_new.args[1], symbol_bounds)
            except RecursionError:
                lhs_interval = rhs_interval = None
            if lhs_interval is not None and rhs_interval is not None and rhs_interval[0] == rhs_interval[1]:
                modulus = int(rhs_interval[0])
                if modulus > 0 and lhs_interval[0] >= 0 and lhs_interval[1] < modulus:
                    memo[expr] = expr_new.args[0]
                    return expr_new.args[0]

        if isinstance(expr_new, sympy.floor):
            target = expr_new.args[0]
            if getattr(target, 'is_Mul', False):
                coeff, remainder = target.as_coeff_Mul()
                if remainder != 1:
                    try:
                        interval = self._interval_of_expr(remainder, symbol_bounds)
                    except RecursionError:
                        interval = None
                    if interval is not None:
                        coeff_value = float(coeff)
                        if coeff_value != 0 and abs(coeff_value) < 1:
                            upper = interval[1] * coeff_value
                            lower = interval[0] * coeff_value
                            if lower >= 0 and upper < 1:
                                memo[expr] = sympy.Integer(0)
                                return sympy.Integer(0)

        if isinstance(expr_new, (sympy.GreaterThan, sympy.StrictGreaterThan, sympy.LessThan, sympy.StrictLessThan, sympy.Equality)):
            try:
                lhs_interval = self._interval_of_expr(expr_new.lhs, symbol_bounds)
                rhs_interval = self._interval_of_expr(expr_new.rhs, symbol_bounds)
            except RecursionError:
                lhs_interval = rhs_interval = None
            if lhs_interval is not None and rhs_interval is not None:
                l0, l1 = lhs_interval
                r0, r1 = rhs_interval
                result = None
                if isinstance(expr_new, sympy.GreaterThan):
                    if l0 >= r1:
                        result = sympy.true
                    elif l1 < r0:
                        result = sympy.false
                elif isinstance(expr_new, sympy.StrictGreaterThan):
                    if l0 > r1:
                        result = sympy.true
                    elif l1 <= r0:
                        result = sympy.false
                elif isinstance(expr_new, sympy.LessThan):
                    if l1 <= r0:
                        result = sympy.true
                    elif l0 > r1:
                        result = sympy.false
                elif isinstance(expr_new, sympy.StrictLessThan):
                    if l1 < r0:
                        result = sympy.true
                    elif l0 >= r1:
                        result = sympy.false
                elif isinstance(expr_new, sympy.Equality):
                    if l0 == l1 == r0 == r1:
                        result = sympy.true
                    elif l1 < r0 or r1 < l0:
                        result = sympy.false
                if result is not None:
                    memo[expr] = result
                    return result

        memo[expr] = expr_new
        return expr_new

    def _optimize_reduction_structure(self, expr, memo=None):
        if memo is None:
            memo = {}
        if not isinstance(expr, sympy.Basic):
            return expr
        if expr in memo:
            return memo[expr]
        if not expr.args:
            memo[expr] = expr
            return expr

        new_args = [self._optimize_reduction_structure(arg, memo=memo) for arg in expr.args]
        try:
            expr_new = expr.func(*new_args)
        except Exception:
            expr_new = expr

        expr_name = type(expr_new).__name__
        if expr_name not in {"TensorSum", "TensorProd"}:
            memo[expr] = expr_new
            return expr_new

        body = expr_new.args[0]
        reduction_args = list(expr_new.args[1:])

        inner_name = type(body).__name__
        if inner_name == expr_name and len(expr_new.args) >= 2:
            nested_body = expr_new.args[0].args[0]
            nested_reduction_args = list(expr_new.args[0].args[1:])
            reduction_args = nested_reduction_args + list(expr_new.args[1:])
            expr_new = expr_new.func(nested_body, *reduction_args)
            body = nested_body

        if expr_name == "TensorSum" and getattr(body, "is_Mul", False):
            reduction_symbols = set()
            for item in reduction_args:
                if isinstance(item, sympy.Symbol):
                    reduction_symbols.add(item)
                elif isinstance(item, (tuple, list, sympy.Tuple)) and len(item) >= 1 and isinstance(item[0], sympy.Symbol):
                    reduction_symbols.add(item[0])

            invariant_factors = []
            variant_factors = []
            for factor in body.args:
                free = getattr(factor, "free_symbols", set())
                if free.isdisjoint(reduction_symbols):
                    invariant_factors.append(factor)
                else:
                    variant_factors.append(factor)

            if invariant_factors and variant_factors:
                invariant_part = sympy.Mul(*invariant_factors, evaluate=False)
                variant_part = sympy.Mul(*variant_factors, evaluate=False)
                candidate = invariant_part * expr_new.func(variant_part, *reduction_args)
                if self._estimate_expr_cost(candidate) <= self._estimate_expr_cost(expr_new):
                    expr_new = candidate

        memo[expr] = expr_new
        return expr_new

    def _unroll_reductions(self, expr, subs_map, memo=None, op_def=None, res_ctx=None):
        if memo is None: memo = {}
        if not isinstance(expr, sympy.Basic): return expr
        if expr in memo: return memo[expr]

        if not expr.args:
            memo[expr] = expr
            return expr

        new_args = [
            self._unroll_reductions(arg, subs_map, memo, op_def=op_def, res_ctx=res_ctx)
            for arg in expr.args
        ]

        try:
            expr_new = expr.func(*new_args)
        except:
            expr_new = expr

        target_ops = {'Max', 'Min', 'ArgMax', 'ArgMin', 'Sum', 'Product', 'TensorSum', 'TensorProd'}
        op_name = type(expr_new).__name__
        func_name = getattr(expr_new.func, 'name', '')
        is_target = (func_name in target_ops) or (op_name in target_ops)

        result = expr_new

        if is_target:
            body = expr_new.args[0]
            raw_limits = expr_new.args[1:]

            is_real_reduction = False
            if raw_limits:
                first_lim = raw_limits[0]
                if isinstance(first_lim, (tuple, list, sympy.Tuple)) or isinstance(first_lim, sympy.Symbol):
                    is_real_reduction = True

            if not is_real_reduction and op_name in ['Max', 'Min']:
                memo[expr] = expr_new
                return expr_new

            limits = []
            actual_axes = []

            for item in raw_limits:
                print(f"\n[PROBE_AST] Parsing raw limit item: {item}")
                try:
                    substituted = item.subs(subs_map) if hasattr(item, 'subs') else item

                    if not isinstance(substituted, (tuple, list, sympy.Tuple)):
                        dim_val = self._eval_int_bound(substituted)
                        if dim_val < 0:
                            rank = None
                            if res_ctx is not None:
                                inputs = res_ctx.get('inputs') or []
                                if inputs and hasattr(inputs[0], 'ndim'):
                                    rank = int(inputs[0].ndim)
                            if rank is None:
                                rank = self._eval_int_bound(subs_map.get('Rank', 4))
                            dim_val = rank + dim_val
                        actual_axes.append(dim_val)

                        loop_var = sympy.Symbol(f'idx_dim_{dim_val}')
                        dim_limit = (loop_var, 0, sympy.Symbol(f'Dim(X, {dim_val})') - 1)
                        limits.append(dim_limit)
                    else:
                        limits.append(item)
                        end_expr = item[2]
                        print(f"[PROBE_AST] end_expr = {end_expr}, Type = {type(end_expr)}")
                        for atom in end_expr.atoms(sympy.Function):
                            if atom.func.__name__ == 'Dim' and len(atom.args) > 1:
                                actual_axes.append(int(atom.args[1]))
                                break
                except:
                    limits.append(item)

            print(f"[DEBUG_REDUCTION] Normalized Limits: {limits}")
            if limits:
                unrollable_limits = []
                total_items = 1
                can_static_unroll = True

                for item in limits:
                    if isinstance(item, (tuple, sympy.Tuple)) and len(item) == 3:
                        loop_var, start, end = item
                        try:
                            s_val = self._eval_int_bound(start, subs_map)
                            e_val = self._eval_int_bound(end, subs_map)
                            count = e_val - s_val + 1

                            if 0 < count <= 1024:
                                unrollable_limits.append((loop_var, s_val, e_val))
                                total_items *= count
                            else:
                                print(f"[DEBUG_REDUCTION] Skip unroll: count {count} > 1024")
                                can_static_unroll = False
                                break
                        except:
                            can_static_unroll = False
                            break
                    else:
                        print(f"[DEBUG_REDUCTION] Skip unroll: limit {item} is not a 3-tuple range")
                        can_static_unroll = False
                        break

                if self._should_preserve_reduction_structure(expr_new, body, limits, op_def=op_def):
                    can_static_unroll = False

                if can_static_unroll:
                    body_complexity = len(body.args) if hasattr(body, 'args') and body.args else 1
                    estimated_ast_size = total_items * body_complexity
                    if total_items > 16 or estimated_ast_size > 1024:
                        can_static_unroll = False

                if can_static_unroll:
                    ranges = [range(s, e + 1) for _, s, e in unrollable_limits]
                    vars_order = [v for v, _, _ in unrollable_limits]
                    all_body_atoms = list(body.atoms(sympy.Symbol))
                    target_atoms = {v: [a for a in all_body_atoms if a.name == v.name] + [v] for v in vars_order}

                    flat_args = []
                    for indices in itertools.product(*ranges):
                        exact_subs = {}
                        for var_sym, idx_val in zip(vars_order, indices):
                            for atom in target_atoms[var_sym]:
                                exact_subs[atom] = idx_val
                        term = body.xreplace(exact_subs)
                        flat_args.append(term)

                    if getattr(expr_new, 'is_Sum', False) or op_name in ['Sum', 'TensorSum']:
                        result = sympy.Add(*flat_args, evaluate=False)
                    elif getattr(expr_new, 'is_Product', False) or op_name in ['Product', 'TensorProd']:
                        result = sympy.Mul(*flat_args, evaluate=False)
                    else:
                        try:
                            result = expr_new.func(*flat_args, evaluate=False)
                        except TypeError:
                            result = expr_new.func(*flat_args)

                elif op_name in ['TensorSum', 'Sum']:
                    print(f"[DEBUG_REDUCTION] >>> Fallback to Symbolic Function: TensorSum")
                    print(f"[PROBE_MAP] Ready to map! actual_axes: {actual_axes}, raw_limits: {raw_limits}")
                    reduction_args = []
                    scalar_subs_map = {k: v for k, v in subs_map.items() if not isinstance(v, dict)}
                    masked_body = body
                    for i, item in enumerate(raw_limits):
                        if isinstance(item, (tuple, sympy.Tuple, list)) and len(item) >= 1 and isinstance(item[0], sympy.Symbol):
                            loop_var = item[0]
                            inferred_axis = self._infer_reduction_axis_from_body(masked_body, loop_var)
                            if len(item) >= 3:
                                start_expr = item[1]
                                end_expr = item[2]
                                is_static_bounds = False
                                try:
                                    self._eval_int_bound(start_expr, scalar_subs_map)
                                    self._eval_int_bound(end_expr, scalar_subs_map)
                                    is_static_bounds = True
                                except Exception:
                                    is_static_bounds = False

                                if not is_static_bounds and inferred_axis is not None:
                                    mask_cond = sympy.And(loop_var >= start_expr, loop_var <= end_expr)
                                    masked_body = sympy.Piecewise((masked_body, mask_cond), (sympy.Integer(0), sympy.true))
                                    axis_symbol = sympy.Symbol(f"REDUCE_AXIS_{loop_var.name}")
                                    reduction_key = sympy.Symbol('_reduction_map')
                                    if reduction_key not in subs_map:
                                        subs_map[reduction_key] = {}
                                    subs_map[reduction_key][loop_var.name] = {
                                        'axis': inferred_axis,
                                        'axis_symbol': axis_symbol.name,
                                        'start': 0,
                                        'end': None,
                                        'size': None
                                    }
                                    reduction_args.append(axis_symbol)
                                    continue

                            reduction_args.append(loop_var)
                        elif i < len(actual_axes):
                            reduction_args.append(sympy.Integer(actual_axes[i]))

                    result = sympy.Function('TensorSum')(masked_body, *reduction_args)

                    reduction_key = sympy.Symbol('_reduction_map')
                    if reduction_key not in subs_map:
                        subs_map[reduction_key] = {}

                    for i, item in enumerate(raw_limits):
                        if isinstance(item, (tuple, sympy.Tuple, list)) and len(item) >= 1:
                            loop_var = item[0]
                            if isinstance(loop_var, sympy.Symbol):
                                existing_spec = subs_map[reduction_key].get(loop_var.name, {})
                                reduction_spec = {
                                    'axis': existing_spec.get('axis', actual_axes[i] if i < len(actual_axes) else None),
                                    'axis_symbol': existing_spec.get('axis_symbol'),
                                    'start': existing_spec.get('start', 0),
                                    'end': existing_spec.get('end', None),
                                    'size': existing_spec.get('size', None)
                                }

                                if len(item) >= 3:
                                    try:
                                        start_val = self._eval_int_bound(item[1], scalar_subs_map)
                                        end_val = self._eval_int_bound(item[2], scalar_subs_map)
                                        reduction_spec['start'] = start_val
                                        reduction_spec['end'] = end_val
                                        reduction_spec['size'] = max(0, end_val - start_val + 1)
                                    except Exception:
                                        pass

                                subs_map[reduction_key][loop_var.name] = reduction_spec

                elif op_name in ['Max', 'TensorMax', 'ArgMax', 'ArgMin']:
                    reduction_args = []
                    scalar_subs_map = {k: v for k, v in subs_map.items() if not isinstance(v, dict)}
                    masked_body = body
                    reduction_key = sympy.Symbol('_reduction_map')
                    if reduction_key not in subs_map:
                        subs_map[reduction_key] = {}
                    for i, item in enumerate(raw_limits):
                        if isinstance(item, (tuple, sympy.Tuple, list)) and len(item) >= 1 and isinstance(item[0], sympy.Symbol):
                            loop_var = item[0]
                            inferred_axis = self._infer_reduction_axis_from_body(masked_body, loop_var)
                            axis_symbol = None
                            if inferred_axis is not None:
                                axis_symbol = sympy.Symbol(f"REDUCE_AXIS_{loop_var.name}")
                            if len(item) >= 3:
                                start_expr = item[1]
                                end_expr = item[2]
                                is_static_bounds = False
                                try:
                                    start_int = self._eval_int_bound(start_expr, scalar_subs_map)
                                    end_int = self._eval_int_bound(end_expr, scalar_subs_map)
                                    is_static_bounds = True
                                except Exception:
                                    start_int = end_int = None
                                    is_static_bounds = False

                                if not is_static_bounds and inferred_axis is not None:
                                    mask_cond = sympy.And(loop_var >= start_expr, loop_var <= end_expr)
                                    masked_body = sympy.Piecewise((masked_body, mask_cond), (-sympy.Float('1.0e10'), sympy.true))
                                    subs_map[reduction_key][loop_var.name] = {
                                        'axis': inferred_axis,
                                        'axis_symbol': axis_symbol.name,
                                        'start': 0,
                                        'end': None,
                                        'size': None,
                                    }
                                    reduction_args.append(axis_symbol)
                                    continue

                                if is_static_bounds:
                                    subs_map[reduction_key][loop_var.name] = {
                                        'axis': inferred_axis,
                                        'axis_symbol': axis_symbol.name if axis_symbol is not None else None,
                                        'start': start_int,
                                        'end': end_int,
                                        'size': max(0, end_int - start_int + 1),
                                    }
                            if axis_symbol is not None:
                                reduction_args.append(axis_symbol)
                            else:
                                reduction_args.append(loop_var)
                        elif i < len(actual_axes):
                            reduction_args.append(sympy.Integer(actual_axes[i]))

                    if op_name in ['ArgMax', 'ArgMin']:
                        target_name = op_name
                    else:
                        target_name = 'TensorMax'
                    result = sympy.Function(target_name)(masked_body, *reduction_args)
                elif op_name in ['Min', 'TensorMin']:
                    result = sympy.Function('TensorMin')(body, *[sympy.Integer(a) for a in actual_axes])

        memo[expr] = result
        print(f"[DEBUG_REDUCTION] Final Result Type: {type(result)}")
        return result

    def _bind_dynamic_dimensions(self, final_syms, out_shape, lhs_indices_hint, res_ctx, ignore_syms=None):
        if ignore_syms is None: ignore_syms = set()
        binding = {}

        if lhs_indices_hint and len(lhs_indices_hint) == len(out_shape):
            for dim_i, idx_name in enumerate(lhs_indices_hint):
                binding[idx_name] = dim_i
            return binding

        needed_sym_names = {s.name for s in final_syms if s.name not in ignore_syms}
        rank = len(out_shape)

        batch_first_val = None
        if 'subs_map' in res_ctx:
            for key, value in res_ctx['subs_map'].items():
                if hasattr(key, 'name') and key.name == 'batch_first':
                    if isinstance(value, (bool, np.bool_)):
                        batch_first_val = bool(value)
                    break
        if batch_first_val is None:
            op = res_ctx.get('op_instance')
            attrs = getattr(op, 'attrs', {})
            if isinstance(attrs.get('batch_first'), (bool, np.bool_)):
                batch_first_val = bool(attrs['batch_first'])
            elif hasattr(op, 'module') and hasattr(op.module, 'batch_first'):
                batch_first_val = bool(op.module.batch_first)

        if rank == 3 and batch_first_val is not None:
            if batch_first_val:
                if 'n' in needed_sym_names:
                    binding['n'] = 0
                if 'l' in needed_sym_names:
                    binding['l'] = 1
            else:
                if 'l' in needed_sym_names:
                    binding['l'] = 0
                if 'n' in needed_sym_names:
                    binding['n'] = 1

        if rank >= 1 and 'n' in needed_sym_names and 'n' not in binding:
            binding['n'] = 0

        already_bound_dims = set(binding.values())
        avail_dims = [i for i in range(rank) if i not in already_bound_dims]

        is_channels_last = False
        data_format_val = None

        if 'subs_map' in res_ctx:
            for v in res_ctx['subs_map'].values():
                s_val = str(v).lower()
                if 'channels_last' in s_val:
                    data_format_val = 'channels_last'
                    break

        if data_format_val is None:
            op = res_ctx.get('op_instance')
            attrs = getattr(op, 'attrs', {})
            df = attrs.get('data_format', '')

            if not df and hasattr(op, 'data_format'):
                df = op.data_format

            if df:
                data_format_val = str(df).lower()

        if data_format_val and 'last' in data_format_val:
            is_channels_last = True

        if is_channels_last:
            candidates = ['d', 'h', 'w', 'l', 'out', 'c']
        else:
            candidates = ['out', 'c', 'd', 'h', 'w', 'l']

        for sym in candidates:
            if sym in needed_sym_names and sym not in binding and avail_dims:
                binding[sym] = avail_dims.pop(0)

        generic_candidates = ['i', 'j', 'k', 'm']
        for sym in generic_candidates:
            if sym in needed_sym_names and sym not in binding and avail_dims:
                binding[sym] = avail_dims.pop(0)

        return binding

    def _make_swap_perm(self, rank, dim_a, dim_b):
        rank = int(rank)
        dim_a = int(dim_a)
        dim_b = int(dim_b)
        if dim_a < 0:
            dim_a += rank
        if dim_b < 0:
            dim_b += rank
        perm = list(range(rank))
        perm[dim_a], perm[dim_b] = perm[dim_b], perm[dim_a]
        return tuple(perm)

    def _is_input_runtime_symbol(self, sname, input_names=None):
        input_names = set(input_names or [])
        if sname in input_names:
            return True
        if sname in {'X', 'A', 'Indices', 'Query', 'Key', 'Value', 'Memory'}:
            return True
        return sname.startswith('X') and sname[1:].isdigit()

    def _uses_channels_last_layout(self, res_ctx=None):
        if not res_ctx:
            return False

        data_format_val = None
        for value in (res_ctx.get('subs_map') or {}).values():
            text = str(value).lower()
            if 'channels_last' in text or text in {'nhwc', 'ndhwc', 'nlc'}:
                data_format_val = text
                break

        if data_format_val is None:
            op = res_ctx.get('op_instance')
            attrs = getattr(op, 'attrs', {}) if op is not None else {}
            df = attrs.get('data_format') or attrs.get('data_format_str') or ''
            if not df and op is not None and hasattr(op, 'data_format'):
                df = op.data_format
            if df:
                data_format_val = str(df).lower()

        if not data_format_val:
            return False
        return (
            'last' in data_format_val
            or data_format_val in {'nhwc', 'ndhwc', 'nlc'}
        )

    def _preferred_1d_broadcast_axis(self, size, out_shape, res_ctx=None):
        out_shape = tuple(int(x) for x in out_shape)
        rank = len(out_shape)
        if rank <= 1:
            return None

        candidates = [axis for axis, dim in enumerate(out_shape) if int(dim) == int(size)]
        if not candidates:
            return None

        if self._uses_channels_last_layout(res_ctx):
            preferred = [rank - 1]
        elif rank == 2:
            preferred = [rank - 1, 1]
        else:
            preferred = [1, rank - 1]

        preferred.extend(axis for axis in range(1, rank) if axis not in preferred)
        preferred.append(0)
        for axis in preferred:
            if axis in candidates:
                return axis
        return candidates[0]

    def _align_runtime_arg_for_output(
        self,
        sname,
        arg_val,
        out_shape,
        sim_rank,
        input_names=None,
        res_ctx=None
    ):
        indexed_runtime_symbols = set((res_ctx or {}).get('indexed_runtime_symbols', ()))
        if sname in indexed_runtime_symbols:
            return arg_val

        if (
            isinstance(arg_val, np.ndarray)
            and arg_val.ndim == 1
            and len(out_shape) > 1
            and not self._is_input_runtime_symbol(sname, input_names)
        ):
            axis = self._preferred_1d_broadcast_axis(arg_val.size, out_shape, res_ctx)
            if axis is not None and axis < sim_rank:
                new_shape = [1] * sim_rank
                new_shape[axis] = arg_val.size
                return arg_val.reshape(new_shape)
        return arg_val

    def _safe_scalar_int(self, value):
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise TypeError("only length-1 arrays can be converted to Python scalars")
            value = value.reshape(-1)[0]
        elif hasattr(value, 'item') and not isinstance(value, (list, tuple)):
            value = value.item()
        return int(value)

    def _safe_scalar_float(self, value):
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise TypeError("only length-1 arrays can be converted to Python scalars")
            value = value.reshape(-1)[0]
        elif hasattr(value, 'item') and not isinstance(value, (list, tuple)):
            value = value.item()
        return float(value)

    def _safe_scalar_bool(self, value):
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return bool(np.all(value))
            value = value.reshape(-1)[0]
        elif hasattr(value, 'item') and not isinstance(value, (list, tuple)):
            value = value.item()
        return bool(value)

    def _safe_eval_intrinsics(self):
        return {
            'int': self._safe_scalar_int,
            'float': self._safe_scalar_float,
            'bool': self._safe_scalar_bool,
            'round': round,
            'len': len,
            'range': range,
            'tuple': tuple,
            'list': list,
        }

    def _runtime_transpose(self, x, *axes):
        if hasattr(x, 'shape') and not isinstance(x, sympy.Basic):
            if len(axes) == 1 and isinstance(axes[0], (list, tuple, np.ndarray)):
                return np.transpose(x, axes[0])
            if len(axes) == 2 and not isinstance(axes[0], (list, tuple, np.ndarray)):
                return np.swapaxes(x, int(axes[0]), int(axes[1]))
            if axes:
                return np.transpose(x, axes)
            return np.transpose(x)
        return sympy.Function('Transpose')(x, *axes)

    def _runtime_piecewise(self, *pairs):
        for val, cond in pairs:
            if cond is True or cond is sympy.true:
                return val
            if cond is False or cond is None or cond is S_None or type(cond).__name__ == 'NoneSymbol' or cond is sympy.false:
                continue
            if hasattr(cond, 'all'):
                try:
                    if bool(cond.all()):
                        return val
                    continue
                except Exception:
                    pass
            try:
                if bool(cond):
                    return val
            except TypeError:
                return sympy.Piecewise(*pairs)
        return 0

    def _install_runtime_derived_helpers(self, eval_ctx):
        backend_ops = NumpyBackend.get_ops()
        eval_ctx.update(backend_ops)
        eval_ctx['make_swap_perm'] = self._make_swap_perm
        eval_ctx['Transpose'] = self._runtime_transpose
        eval_ctx['Piecewise'] = self._runtime_piecewise

    def _compute_derived_variables(self, op_def, eval_ctx, overwrite=False):
        if 'derived_variables' in op_def:
            self._install_runtime_derived_helpers(eval_ctx)
            eval_ctx.update(self._safe_eval_intrinsics())
            class DefaultZeroDict(dict):
                def __missing__(self, key):
                    return 0

            local_ctx = DefaultZeroDict(eval_ctx)
            for name, formula in op_def['derived_variables'].items():
                if name in eval_ctx and not overwrite:
                    continue
                try:
                    val = eval(str(formula), {"__builtins__": None}, local_ctx)
                    eval_ctx[name] = val
                    local_ctx[name] = val
                except Exception as e:
                    print(f"[Warn] Failed to derive variable {name}: {e}")
                    pass

    def compile_and_run(self, math_prog, op_instance, inputs, symnet, **extra_kwargs):
        raw_expr_container = math_prog['expr']
        op_def = math_prog.get('op_def', {})
        self.last_jit_contexts = []
        progress_enabled = os.getenv('SEMTRANS_TRACE_PROGRESS', '') == '1'
        compile_started = time.perf_counter()
        op_label = getattr(op_instance, 'op_type', type(op_instance).__name__)

        def report_compile_stage(stage):
            if progress_enabled:
                print(
                    f"[RuntimeMathCompilerStage] operation={op_label} "
                    f"stage={stage} elapsed={time.perf_counter() - compile_started:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )

        report_compile_stage('start')

        def _shortcut_result(formula: str, value: Any, bindings: Optional[Dict[str, Any]] = None):
            """Record exact tensor-composition semantics without JIT binding noise."""
            self.last_jit_context = {
                'formula_diagnostic': {
                    'operation': str(op_label),
                    'framework': str(getattr(symnet, 'framework_type', 'unknown')),
                    'native_formula': formula,
                    'canonical_formula': formula,
                    'parameter_bindings': dict(bindings or {}),
                    'derived_bindings': {},
                    'canonical_bindings': dict(bindings or {}),
                    'layout': {},
                }
            }
            self.last_jit_contexts = [self.last_jit_context]
            output_shape_limit = extra_kwargs.get('_semtrans_output_shape_limit')
            if output_shape_limit is not None:
                array = np.asarray(value)
                if array.ndim == len(output_shape_limit):
                    value = array[
                        tuple(
                            slice(0, min(int(limit), size))
                            for limit, size in zip(output_shape_limit, array.shape)
                        )
                    ]
            return value

        # These operations have no hidden framework state.  Evaluating them
        # directly on the resolved NumPy tensors is the executable MathIR
        # definition and avoids accidental alias binding (e.g. a GraphDef
        # constant axis being treated as a second runtime tensor).
        label = str(op_label).lower()
        comparison_shortcuts = {
            'equal': ('Equal(X1, X2)', np.equal),
            'notequal': ('NotEqual(X1, X2)', np.not_equal),
            'greaterthan': ('GreaterThan(X1, X2)', np.greater),
            'greaterequal': ('GreaterEqual(X1, X2)', np.greater_equal),
            'lessthan': ('LessThan(X1, X2)', np.less),
            'lessequal': ('LessEqual(X1, X2)', np.less_equal),
        }
        if len(inputs) == 2 and label in comparison_shortcuts:
            formula, function = comparison_shortcuts[label]
            value = function(np.asarray(inputs[0]), np.asarray(inputs[1]))
            return _shortcut_result(formula, value)
        if len(inputs) == 2 and label == 'tfliteembeddinglookup':
            value = NumpyBackend.get_ops()['EmbeddingLookup'](
                np.asarray(inputs[1]), np.asarray(inputs[0]), -1
            )
            return _shortcut_result(
                'EmbeddingLookup(weight, indices, padding_idx=-1)',
                value,
                {
                    'input_schema': '[indices, weight]',
                    'weight_shape': list(np.asarray(inputs[1]).shape),
                },
            )
        if inputs and label in {'pythongetitem'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            index_spec = attrs.get('index_spec')
            value = NumpyBackend.get_ops()['PythonGetItem'](
                np.asarray(inputs[0]), index_spec, *inputs[1:]
            )
            return _shortcut_result(
                'PythonGetItem(X, index_spec)',
                value,
                {
                    'index_spec': index_spec,
                    'dynamic_input_count': max(0, len(inputs) - 1),
                },
            )
        if inputs and label in {'range'}:
            value = NumpyBackend.get_ops()['Range'](inputs[0])
            return _shortcut_result('Range(X)', value)
        if inputs and label in {'einsum'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            equation = attrs.get('equation', attrs.get('arg_0'))
            if equation is None:
                raise ValueError('Einsum equation is missing from target attrs')
            value = np.einsum(
                str(equation).strip("'\""),
                *[np.asarray(item) for item in inputs],
            )
            return _shortcut_result(
                'Einsum(X1, X2, equation)',
                value,
                {'equation': str(equation)},
            )
        if inputs and label in {'einsumdense', 'traildense'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            equation = attrs.get('equation', attrs.get('einsum_expr'))
            module = getattr(op_instance, 'module', None)
            weight = attrs.get('weight')
            bias = attrs.get('bias')
            if weight is None and module is not None:
                weight = getattr(module, 'weight', None)
            if bias is None and module is not None:
                bias = getattr(module, 'bias', None)
            if equation is None or weight is None:
                raise ValueError(
                    'EinsumDense requires equation and weight provenance'
                )
            if hasattr(weight, 'numpy'):
                weight = weight.numpy()
            if hasattr(bias, 'numpy'):
                bias = bias.numpy()
            value = np.einsum(
                str(equation), np.asarray(inputs[0]), np.asarray(weight)
            )
            if bias is not None:
                value = value + np.asarray(bias)
            return _shortcut_result(
                'Einsum(X, W, equation) + B',
                value,
                {
                    'equation': str(equation),
                    'weight_shape': list(np.asarray(weight).shape),
                    'bias_present': bias is not None,
                },
            )
        if inputs and label in {'semtrans_lstm', 'semtranslstm'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            if bool(attrs.get('stateful', False)):
                raise ValueError('SemTrans_LSTM does not qualify stateful recurrence')
            if float(attrs.get('dropout', 0.0)) != 0.0 or float(
                attrs.get('recurrent_dropout', 0.0)
            ) != 0.0:
                raise ValueError('SemTrans_LSTM requires deterministic zero dropout')
            activation = str(attrs.get('activation', 'tanh')).strip("'\"").lower()
            recurrent_activation = str(
                attrs.get('recurrent_activation', 'sigmoid')
            ).strip("'\"").lower()
            if activation != 'tanh' or recurrent_activation != 'sigmoid':
                raise ValueError(
                    'SemTrans_LSTM currently qualifies tanh/sigmoid gates only'
                )
            value = np.asarray(inputs[0])
            if value.ndim != 3:
                raise ValueError(
                    f'SemTrans_LSTM requires batch-major rank-3 input, got {value.shape}'
                )

            def tensor_attr(names):
                for name in names:
                    candidate = attrs.get(name)
                    if candidate is None:
                        candidate = getattr(getattr(op_instance, 'module', None), name, None)
                    if candidate is None:
                        continue
                    try:
                        return np.asarray(
                            candidate.numpy()
                            if hasattr(candidate, 'numpy')
                            else candidate
                        )
                    except (TypeError, ValueError):
                        continue
                return None

            kernel = tensor_attr(('weight_ih', 'kernel'))
            recurrent_kernel = tensor_attr(('weight_hh', 'recurrent_kernel'))
            bias = tensor_attr(('bias',))
            if kernel is None or recurrent_kernel is None:
                raise ValueError('SemTrans_LSTM kernel provenance is incomplete')
            if kernel.ndim != 2 or recurrent_kernel.ndim != 2:
                raise ValueError('SemTrans_LSTM kernels must be matrices')
            units = int(recurrent_kernel.shape[0])
            if kernel.shape[1] != 4 * units or recurrent_kernel.shape[1] != 4 * units:
                raise ValueError('SemTrans_LSTM requires Keras IFCO gate layout')
            if bias is None:
                bias = np.zeros((4 * units,), dtype=value.dtype)
            bias = np.asarray(bias).reshape(-1)
            if bias.size == 8 * units:
                bias = bias[: 4 * units] + bias[4 * units :]
            if bias.size != 4 * units:
                raise ValueError('SemTrans_LSTM bias gate layout is invalid')

            # Keras' artifact schema exposes optional initial state tensors as
            # inputs [X, h0, c0].  Preserve those values instead of silently
            # replacing them with zeros; doing so is required for decoder and
            # stateful-in-the-graph (but non-stateful layer) LSTM regions.
            initial_state_provided = len(inputs) >= 3
            if len(inputs) == 2:
                raise ValueError(
                    'SemTrans_LSTM requires both hidden and cell initial states'
                )
            if len(inputs) > 3:
                raise ValueError(
                    f'SemTrans_LSTM received unexpected input count: {len(inputs)}'
                )

            def _resolve_initial_state(index, name):
                state = np.asarray(inputs[index])
                if state.ndim == 1 and value.shape[0] == 1:
                    state = state.reshape(1, -1)
                if state.ndim != 2 or tuple(state.shape) != (value.shape[0], units):
                    raise ValueError(
                        f'SemTrans_LSTM {name} must have shape '
                        f'[{value.shape[0]}, {units}], got {state.shape}'
                    )
                return state.astype(value.dtype, copy=False)

            if initial_state_provided:
                hidden = _resolve_initial_state(1, 'initial hidden state')
                cell = _resolve_initial_state(2, 'initial cell state')
                initial_hidden = hidden.copy()
                initial_cell = cell.copy()
            else:
                hidden = np.zeros((value.shape[0], units), dtype=value.dtype)
                cell = np.zeros_like(hidden)
                initial_hidden = None
                initial_cell = None

            sigmoid = lambda item: 1.0 / (1.0 + np.exp(-item))
            time_indices = list(range(value.shape[1]))
            if bool(attrs.get('go_backwards', False)):
                time_indices.reverse()
            outputs = []
            for time_index in time_indices:
                gates = (
                    np.matmul(value[:, time_index, :], kernel)
                    + np.matmul(hidden, recurrent_kernel)
                    + bias
                )
                input_gate, forget_gate, candidate, output_gate = np.split(
                    gates, 4, axis=-1
                )
                input_gate = sigmoid(input_gate)
                forget_gate = sigmoid(forget_gate)
                candidate = np.tanh(candidate)
                output_gate = sigmoid(output_gate)
                cell = forget_gate * cell + input_gate * candidate
                hidden = output_gate * np.tanh(cell)
                outputs.append(hidden.copy())
            sequence = np.stack(outputs, axis=1)
            primary = sequence if bool(attrs.get('return_sequences', False)) else hidden
            result = (
                (primary, hidden, cell)
                if bool(attrs.get('return_state', False))
                else primary
            )
            bindings = {
                'timesteps': int(value.shape[1]),
                'features': int(value.shape[2]),
                'units': units,
                'go_backwards': bool(attrs.get('go_backwards', False)),
                'return_sequences': bool(attrs.get('return_sequences', False)),
                'return_state': bool(attrs.get('return_state', False)),
                'activation': activation,
                'recurrent_activation': recurrent_activation,
                'gate_layout': 'IFCO',
                'initial_state_provided': initial_state_provided,
            }
            if initial_state_provided:
                for name, state in (
                    ('initial_hidden', initial_hidden),
                    ('initial_cell', initial_cell),
                ):
                    bindings[f'{name}_shape'] = list(state.shape)
                    bindings[f'{name}_dtype'] = str(state.dtype)
                    bindings[f'{name}_sha256'] = hashlib.sha256(
                        np.ascontiguousarray(state).tobytes()
                    ).hexdigest()
            return _shortcut_result(
                'LSTMRecurrenceIFCO(X, kernel, recurrent_kernel, bias, h0, c0)'
                if initial_state_provided
                else 'LSTMRecurrenceIFCO(X, kernel, recurrent_kernel, bias)',
                result,
                bindings,
            )
        if inputs and label in {'semtrans_gru', 'semtransgru'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            if bool(attrs.get('stateful', False)) or float(attrs.get('dropout', 0.0)) != 0.0 or float(attrs.get('recurrent_dropout', 0.0)) != 0.0:
                raise ValueError('SemTrans_GRU requires non-stateful zero-dropout recurrence')
            if str(attrs.get('activation', 'tanh')).strip("'\"").lower() != 'tanh' or str(attrs.get('recurrent_activation', 'sigmoid')).strip("'\"").lower() != 'sigmoid':
                raise ValueError('SemTrans_GRU qualifies tanh/sigmoid activations only')
            value = np.asarray(inputs[0])
            if value.ndim != 3:
                raise ValueError(f'SemTrans_GRU requires batch-major rank-3 input, got {value.shape}')
            def tensor_attr(names):
                for name in names:
                    candidate = attrs.get(name)
                    if candidate is None:
                        candidate = getattr(getattr(op_instance, 'module', None), name, None)
                    if candidate is not None:
                        return np.asarray(candidate.numpy() if hasattr(candidate, 'numpy') else candidate)
                return None
            kernel = tensor_attr(('weight_ih', 'kernel'))
            recurrent_kernel = tensor_attr(('weight_hh', 'recurrent_kernel'))
            bias = tensor_attr(('bias',))
            units = int(recurrent_kernel.shape[0])
            if kernel is None or recurrent_kernel is None or kernel.shape[1] != 3 * units or recurrent_kernel.shape[1] != 3 * units:
                raise ValueError('SemTrans_GRU kernel provenance or gate layout is invalid')
            if bias is None:
                input_bias = recurrent_bias = np.zeros((3 * units,), dtype=value.dtype)
            elif bias.ndim == 2 and bias.shape[0] == 2:
                input_bias, recurrent_bias = bias[0], bias[1]
            else:
                input_bias = np.asarray(bias).reshape(-1)
                recurrent_bias = np.zeros_like(input_bias)
            hidden = np.zeros((value.shape[0], units), dtype=value.dtype)
            time_indices = list(range(value.shape[1]))
            if bool(attrs.get('go_backwards', False)):
                time_indices.reverse()
            outputs = []
            for time_index in time_indices:
                x_proj = np.matmul(value[:, time_index, :], kernel) + input_bias
                h_proj = np.matmul(hidden, recurrent_kernel) + recurrent_bias
                z = 1.0 / (1.0 + np.exp(-x_proj[:, :units] - h_proj[:, :units]))
                r = 1.0 / (1.0 + np.exp(-x_proj[:, units:2 * units] - h_proj[:, units:2 * units]))
                candidate = np.tanh(x_proj[:, 2 * units:] + r * h_proj[:, 2 * units:])
                hidden = z * hidden + (1.0 - z) * candidate
                outputs.append(hidden.copy())
            sequence = np.stack(outputs, axis=1)
            result = sequence if bool(attrs.get('return_sequences', False)) else hidden
            bindings = {'timesteps': int(value.shape[1]), 'features': int(value.shape[2]), 'units': units, 'go_backwards': bool(attrs.get('go_backwards', False)), 'return_sequences': bool(attrs.get('return_sequences', False)), 'gate_layout': 'ZRH', 'reset_after': bool(attrs.get('reset_after', True))}
            return _shortcut_result('GRURecurrenceResetAfter(X, kernel, recurrent_kernel, bias)', result, bindings)
        if inputs and label == 'attention':
            attrs = getattr(op_instance, 'attrs', {}) or {}
            module = getattr(op_instance, 'module', None)
            score_mode = str(attrs.get('score_mode', 'dot')).strip("'\"").lower()
            if score_mode != 'dot':
                raise ValueError(
                    'Attention currently qualifies dot-product score mode only'
                )
            training = bool(attrs.get('training', False))
            dropout = float(attrs.get('dropout', 0.0))
            if training and dropout != 0.0:
                raise ValueError(
                    'Attention requires deterministic inference-time dropout'
                )
            if len(inputs) not in {2, 3}:
                raise ValueError(
                    'Attention expects [query, value] or [query, value, key]'
                )
            query = np.asarray(inputs[0])
            value = np.asarray(inputs[1])
            key = np.asarray(inputs[2]) if len(inputs) == 3 else value
            if query.ndim != 3 or value.ndim != 3 or key.ndim != 3:
                raise ValueError(
                    'Attention requires batch-major rank-3 query/value/key tensors'
                )
            if query.shape[0] != value.shape[0] or value.shape[0] != key.shape[0]:
                raise ValueError('Attention batch dimensions must agree')
            if query.shape[-1] != key.shape[-1]:
                raise ValueError('Attention query/key feature dimensions must agree')
            if value.shape[1] != key.shape[1]:
                raise ValueError('Attention value/key sequence dimensions must agree')

            use_scale = bool(attrs.get('use_scale', False))
            scale = 1.0
            if use_scale:
                scale_value = attrs.get('scale')
                if scale_value is None and module is not None:
                    scale_value = getattr(module, 'scale', None)
                if scale_value is None:
                    raise ValueError('Attention scale provenance is incomplete')
                if hasattr(scale_value, 'numpy'):
                    scale_value = scale_value.numpy()
                scale_array = np.asarray(scale_value).reshape(-1)
                if scale_array.size != 1:
                    raise ValueError('Attention scale must be scalar')
                scale = float(scale_array[0])

            scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale
            causal = bool(
                attrs.get('causal', attrs.get('use_causal_mask', False))
            )
            if causal:
                query_positions = np.arange(query.shape[1])[:, None]
                key_positions = np.arange(key.shape[1])[None, :]
                causal_mask = key_positions <= query_positions
                scores = np.where(causal_mask[None, :, :], scores, -1e9)
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
            output = np.matmul(probabilities, value)
            return _shortcut_result(
                'DotProductAttention(Query, Value, Key, scale, causal)',
                output,
                {
                    'score_mode': score_mode,
                    'use_scale': use_scale,
                    'scale': scale,
                    'causal': causal,
                    'dropout': dropout,
                    'training': training,
                    'query_length': int(query.shape[1]),
                    'key_length': int(key.shape[1]),
                },
            )
        if inputs and label in {'multiheadattention', 'multiheadattn'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            module = getattr(op_instance, 'module', None)
            query_dense = getattr(module, '_query_dense', None) or getattr(module, 'query_dense', None)
            key_dense = getattr(module, '_key_dense', None) or getattr(module, 'key_dense', None)
            value_dense = getattr(module, '_value_dense', None) or getattr(module, 'value_dense', None)
            output_dense = getattr(module, '_output_dense', None) or getattr(module, 'output_dense', None)
            if any(layer is None for layer in (query_dense, key_dense, value_dense, output_dense)):
                raise ValueError('MultiHeadAttention projection provenance is incomplete')
            def weights(layer):
                values = list(layer.get_weights())
                return np.asarray(values[0]), (np.asarray(values[1]) if len(values) > 1 else None)
            q_kernel, q_bias = weights(query_dense)
            k_kernel, k_bias = weights(key_dense)
            v_kernel, v_bias = weights(value_dense)
            o_kernel, o_bias = weights(output_dense)
            query = np.asarray(inputs[0]); key = np.asarray(inputs[1]); value = np.asarray(inputs[2])
            def project(x, kernel, bias):
                output = np.einsum('btd,dhk->bthk', x, kernel)
                return output + (bias.reshape((1, 1) + tuple(bias.shape)) if bias is not None else 0.0)
            q = project(query, q_kernel, q_bias)
            k = project(key, k_kernel, k_bias)
            v = project(value, v_kernel, v_bias)
            scores = np.einsum('bthk,bshk->bhts', q, k) / np.sqrt(float(q_kernel.shape[-1]))
            if len(inputs) > 3 and inputs[3] is not None:
                mask = np.asarray(inputs[3]).astype(bool)
                while mask.ndim < scores.ndim:
                    mask = np.expand_dims(mask, axis=0)
                scores = np.where(mask, scores, -1e9)
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            attention = np.exp(scores)
            attention = attention / np.sum(attention, axis=-1, keepdims=True)
            attended = np.einsum('bhts,bshv->bthv', attention, v)
            output = np.einsum('bthv,hvo->bto', attended, o_kernel)
            if o_bias is not None:
                output = output + o_bias
            return _shortcut_result(
                'MultiHeadAttention(Q, K, V, mask)', output,
                {
                    'num_heads': int(q_kernel.shape[1]),
                    'key_dim': int(q_kernel.shape[2]),
                    'value_dim': int(v_kernel.shape[2]),
                    'mask_provided': len(inputs) > 3 and inputs[3] is not None,
                },
            )
        if inputs and label in {'abs', 'sqrt', 'rsqrt', 'exp', 'log', 'tan', 'round', 'neg'}:
            value = np.asarray(inputs[0])
            unary = {
                'abs': np.abs,
                'sqrt': np.sqrt,
                'rsqrt': lambda item: np.reciprocal(np.sqrt(item)),
                'exp': np.exp,
                'log': np.log,
                'tan': np.tan,
                'round': np.round,
                'neg': np.negative,
            }[label]
            return _shortcut_result(f'{label}(X)', unary(value))
        if inputs and label in {'div', 'divide'} and len(inputs) >= 2:
            return _shortcut_result(
                'divide(X1, X2)',
                np.divide(np.asarray(inputs[0]), np.asarray(inputs[1])),
            )
        if inputs and label in {'add', 'addv2', 'biasadd', 'sub', 'subtract', 'mul', 'multiply', 'minimum', 'maximum'} and len(inputs) >= 2:
            left, right = np.asarray(inputs[0]), np.asarray(inputs[1])
            attrs = getattr(op_instance, 'attrs', {}) or {}
            if label in {'add', 'addv2'} and 'alpha' in attrs:
                right = right * float(attrs['alpha'])
            binary = {
                'add': np.add, 'addv2': np.add, 'biasadd': np.add,
                'sub': np.subtract, 'subtract': np.subtract,
                'mul': np.multiply, 'multiply': np.multiply,
                'minimum': np.minimum, 'maximum': np.maximum,
            }[label]
            return _shortcut_result(f'{label}(X1, X2)', binary(left, right))
        if inputs and label in {'matmul', 'batchmatmul', 'batchmatmulv2'} and len(inputs) >= 2:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            left, right = np.asarray(inputs[0]), np.asarray(inputs[1])
            transpose_left = bool(
                attrs.get('transpose_x', attrs.get('transpose_a', False))
            )
            transpose_right = bool(
                attrs.get('transpose_y', attrs.get('transpose_b', False))
            )
            if transpose_left:
                left = np.swapaxes(left, -1, -2)
            if transpose_right:
                right = np.swapaxes(right, -1, -2)
            return _shortcut_result(
                'MatMul(X1, X2)',
                np.matmul(left, right),
                {
                    'transpose_x': transpose_left,
                    'transpose_y': transpose_right,
                },
            )
        if inputs and label in {'leakyrelu'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            slope = float(attrs.get('negative_slope', attrs.get('alpha', 0.2)))
            value = np.asarray(inputs[0])
            return _shortcut_result('LeakyReLU(X, negative_slope)', np.where(value >= 0, value, slope * value), {'negative_slope': slope})
        if inputs and label in {
            'batchnorm', 'batchnorm1d', 'batchnorm2d', 'batchnorm3d',
            'batchnormalization',
        }:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            value = np.asarray(inputs[0])
            raw_axis = attrs.get('axis')
            if isinstance(raw_axis, (list, tuple)) and raw_axis:
                channel_axis = int(raw_axis[0])
            elif raw_axis is not None:
                channel_axis = int(raw_axis)
            else:
                layout = str(
                    attrs.get('data_layout', attrs.get('data_format', ''))
                ).strip("'\"").upper()
                channel_axis = value.ndim - 1 if layout in {
                    'NHWC', 'NDHWC', 'NWC', 'CHANNELS_LAST'
                } else 1
            if channel_axis < 0:
                channel_axis += value.ndim
            if channel_axis < 0 or channel_axis >= value.ndim:
                raise ValueError(
                    f'BatchNorm channel axis {channel_axis} is invalid for {value.shape}'
                )
            channel_count = int(value.shape[channel_axis])
            module = getattr(op_instance, 'module', None)

            def parameter(names, default):
                for name in names:
                    candidate = getattr(module, name, None) if module is not None else None
                    if candidate is None:
                        candidate = attrs.get(name)
                    if candidate is None:
                        continue
                    try:
                        array = _safe_numpy(candidate).reshape(-1)
                    except (TypeError, ValueError):
                        continue
                    if array.size == channel_count:
                        return array
                return np.full((channel_count,), default, dtype=value.dtype)

            training = bool(attrs.get('training', False))
            use_global_stats = attrs.get('use_global_stats', None)
            use_running = bool(use_global_stats is True) or (
                use_global_stats is None and not training
            )
            if use_running:
                mean = parameter(('moving_mean', 'running_mean', '_mean'), 0.0)
                variance = parameter(
                    ('moving_variance', 'running_var', '_variance'), 1.0
                )
            else:
                reduce_axes = tuple(axis for axis in range(value.ndim) if axis != channel_axis)
                mean = np.mean(value, axis=reduce_axes).reshape(-1)
                variance = np.mean(
                    (value - mean.reshape([1 if axis != channel_axis else channel_count for axis in range(value.ndim)]) ) ** 2,
                    axis=reduce_axes,
                ).reshape(-1)
            weight = parameter(('gamma', 'weight'), 1.0)
            bias = parameter(('beta', 'bias'), 0.0)
            shape = [1] * value.ndim
            shape[channel_axis] = channel_count
            epsilon = float(attrs.get('epsilon', 1e-5))
            output = (
                (value - mean.reshape(shape))
                / np.sqrt(variance.reshape(shape) + epsilon)
                * weight.reshape(shape)
                + bias.reshape(shape)
            )
            return _shortcut_result(
                'BatchNormInference(X, mean, variance, weight, bias, epsilon, channel_axis)',
                output,
                {
                    'channel_axis': channel_axis,
                    'channel_count': channel_count,
                    'epsilon': epsilon,
                    'training': training,
                    'use_global_stats': use_global_stats,
                },
            )
        if inputs and label in {'less'} and len(inputs) >= 2:
            return _shortcut_result('Less(X1, X2)', np.less(np.asarray(inputs[0]), np.asarray(inputs[1])))
        if inputs and label in {'logicalnot', 'logical_not'}:
            return _shortcut_result(
                'Not(X)', np.logical_not(np.asarray(inputs[0]))
            )
        if inputs and label in {'greater', 'greaterthan'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            right = np.asarray(inputs[1]) if len(inputs) >= 2 else np.asarray(attrs.get('y', 0))
            return _shortcut_result(
                'Greater(X1, X2)',
                np.greater(np.asarray(inputs[0]), right),
            )
        if inputs and label in {'argmax', 'argmaxv2', 'arg_max'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            axis = int(attrs.get('axis', -1))
            return _shortcut_result(
                'ArgMax(X, axis)',
                np.argmax(np.asarray(inputs[0]), axis=axis).astype(
                    np.dtype(str(attrs.get('output_type', 'int64')).replace('tf.', ''))
                ),
                {'axis': axis},
            )
        if inputs and label in {'repeat'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            axis = attrs.get('axis')
            axis = None if axis is None else int(axis)
            return _shortcut_result(
                'Repeat(X, repeats, axis)',
                NumpyBackend.get_ops()['Repeat'](
                    inputs[0], inputs[1], axis=axis
                ),
                {'axis': axis},
            )
        if inputs and label in {'oneslike', 'ones_like'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            dtype = attrs.get('dtype')
            if isinstance(dtype, str):
                dtype = dtype.replace('tf.', '')
            return _shortcut_result(
                'OnesLike(X, dtype)',
                np.ones_like(np.asarray(inputs[0]), dtype=dtype),
                {'dtype': str(dtype) if dtype is not None else None},
            )
        if inputs and label in {'zeroslike', 'zeros_like'}:
            return _shortcut_result(
                'ZerosLike(X)', np.zeros_like(np.asarray(inputs[0]))
            )
        if inputs and label in {'scatterndupdate'}:
            return _shortcut_result(
                'ScatterNDUpdate(X, indices, updates)',
                NumpyBackend.get_ops()['ScatterNDUpdate'](
                    inputs[0], inputs[1], inputs[2]
                ),
            )
        if inputs and label in {'scatternd'}:
            return _shortcut_result(
                'ScatterND(indices, updates, shape)',
                NumpyBackend.get_ops()['ScatterND'](
                    inputs[0], inputs[1], inputs[2]
                ),
            )
        if inputs and label in {'scatter'} and len(inputs) >= 3:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            return _shortcut_result(
                'Scatter(X, indices, updates, overwrite)',
                NumpyBackend.get_ops()['Scatter'](
                    inputs[0], inputs[1], inputs[2],
                    bool(attrs.get('overwrite', True)),
                ),
                {'overwrite': bool(attrs.get('overwrite', True))},
            )
        if inputs and label in {'expand', 'expanddims'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            shape = inputs[1] if len(inputs) >= 2 else attrs.get('shape')
            if shape is None:
                raise ValueError('Expand target shape is missing')
            return _shortcut_result(
                'Expand(X, shape)',
                NumpyBackend.get_ops()['Expand'](inputs[0], shape),
            )
        if inputs and label in {'broadcastto', 'broadcast_to'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            shape = inputs[1] if len(inputs) >= 2 else attrs.get('shape')
            if shape is None:
                raise ValueError('BroadcastTo target shape is missing')
            return _shortcut_result(
                'BroadcastTo(X, shape)',
                NumpyBackend.get_ops()['BroadcastTo'](inputs[0], shape),
            )
        if inputs and label in {'takealongaxis', 'take_along_axis'} and len(inputs) >= 2:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            axis = int(attrs.get('axis', -1))
            return _shortcut_result(
                'TakeAlongAxis(X, indices, axis)',
                NumpyBackend.get_ops()['TakeAlongAxis'](
                    inputs[0], inputs[1], axis
                ),
                {'axis': axis},
            )
        if inputs and label in {'gatherv2', 'gather'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            axis = int(attrs.get('axis', 0))
            indices = np.asarray(inputs[1]) if len(inputs) >= 2 else np.asarray(attrs.get('indices'))
            if int(attrs.get('batch_dims', 0)) == 1:
                return _shortcut_result(
                    'Gather(X, indices, axis, batch_dims)',
                    np.take_along_axis(
                        np.asarray(inputs[0]),
                        np.asarray(indices)[..., None] if np.asarray(indices).ndim == np.asarray(inputs[0]).ndim - 1 else np.asarray(indices),
                        axis=axis,
                    ),
                    {'axis': axis, 'batch_dims': 1},
                )
            return _shortcut_result('Gather(X, indices, axis)', np.take(np.asarray(inputs[0]), indices, axis=axis), {'axis': axis})
        if inputs and label in {'reducemax', 'max'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            axis = attrs.get('axis', 0)
            keepdims = bool(attrs.get('keepdims', attrs.get('keep_dims', False)))
            axis_tuple = tuple(int(item) for item in np.asarray(axis).reshape(-1))
            return _shortcut_result('ReduceMax(X, axis, keepdims)', np.max(np.asarray(inputs[0]), axis=axis_tuple, keepdims=keepdims), {'axis': list(axis_tuple), 'keepdims': keepdims})
        if inputs and label in {'reducesum', 'sum'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            axis = attrs.get('axis', attrs.get('dim'))
            keepdims = bool(
                attrs.get(
                    'keepdims',
                    attrs.get('keepdim', attrs.get('keep_dims', False)),
                )
            )
            if axis is None or (
                isinstance(axis, str) and axis.strip("'\"").lower() in {'all', 'none'}
            ):
                axis_tuple = None
            else:
                axis_tuple = tuple(int(item) for item in np.asarray(axis).reshape(-1))
            return _shortcut_result(
                'ReduceSum(X, axis, keepdims)',
                np.sum(np.asarray(inputs[0]), axis=axis_tuple, keepdims=keepdims),
                {'axis': None if axis_tuple is None else list(axis_tuple), 'keepdims': keepdims},
            )
        if inputs and label in {'splitv', 'split'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            if (
                str(getattr(symnet, 'framework_type', '')).lower() == 'tflite'
                and label == 'split'
                and len(inputs) >= 2
            ):
                axis = int(np.asarray(inputs[0]).reshape(-1)[0])
                value = np.asarray(inputs[1])
                num_splits = int(attrs.get('num_splits', 0))
                if num_splits <= 0 or value.shape[axis] % num_splits:
                    raise ValueError(
                        'TFLite Split axis is not divisible by NumSplits'
                    )
                pieces = tuple(np.split(value, num_splits, axis=axis))
                return _shortcut_result(
                    'TFLiteSplit(axis, X, num_splits)',
                    pieces,
                    {'axis': axis, 'num_splits': num_splits},
                )
            axis = int(attrs.get('axis', 0))
            sizes = [int(item) for item in (attrs.get('size_splits') or [])]
            value = np.asarray(inputs[0])
            if sizes and -1 in sizes:
                missing = int(value.shape[axis]) - sum(item for item in sizes if item != -1)
                sizes = [missing if item == -1 else item for item in sizes]
            pieces = tuple(np.split(value, np.cumsum(sizes)[:-1], axis=axis)) if sizes else tuple(np.array_split(value, len(getattr(op_instance, 'output_names', ()) or ()), axis=axis))
            return _shortcut_result('SplitV(X, size_splits, axis)', pieces, {'axis': axis, 'size_splits': sizes})
        if inputs and label in {'tile'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            repeat_times = attrs.get('repeat_times')
            repeat_template = attrs.get('repeat_template')
            if repeat_times is None and isinstance(repeat_template, (list, tuple)):
                resolved = []
                for item in repeat_template:
                    if isinstance(item, dict) and 'input_index' in item:
                        input_index = int(item['input_index'])
                        if input_index >= len(inputs):
                            raise ValueError(
                                f'Tile repeat template input {input_index} is missing'
                            )
                        resolved.append(
                            int(np.asarray(inputs[input_index]).reshape(-1)[0])
                        )
                    else:
                        resolved.append(int(item))
                repeat_times = resolved
            if repeat_times is None and len(inputs) > 1:
                repeat_times = inputs[1]
            if repeat_times is None:
                raise ValueError('Tile repeat_times is missing')
            repeats = tuple(
                int(item) for item in np.asarray(repeat_times).reshape(-1)
            )
            value = NumpyBackend.get_ops()['Tile'](inputs[0], repeats)
            return _shortcut_result(
                'Tile(X, repeat_times)',
                value,
                {
                    'repeat_times': list(repeats),
                    'repeat_template': repeat_template,
                },
            )
        if inputs and label in {'stridedslice', 'strided_slice'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            starts = [int(item) for item in (attrs.get('starts') or attrs.get('begin') or [])]
            ends = [int(item) for item in (attrs.get('ends') or attrs.get('end') or [])]
            strides = [int(item) for item in (attrs.get('strides') or [1] * len(starts))]
            if str(getattr(symnet, 'framework_type', '')).lower() == 'tflite':
                begin_mask = int(attrs.get('begin_mask', 0))
                end_mask = int(attrs.get('end_mask', 0))
                ellipsis_mask = int(attrs.get('ellipsis_mask', 0))
                new_axis_mask = int(attrs.get('new_axis_mask', 0))
                shrink_axis_mask = int(attrs.get('shrink_axis_mask', 0))
                if bool(attrs.get('offset', False)):
                    raise ValueError('TFLite StridedSlice offset semantics are not qualified')
                value = np.asarray(inputs[0])
                slices = []
                input_axis = 0
                for index, (start, end, stride) in enumerate(zip(starts, ends, strides)):
                    bit = 1 << index
                    if ellipsis_mask & bit:
                        remaining_specs = sum(
                            not (new_axis_mask & (1 << future))
                            and not (ellipsis_mask & (1 << future))
                            for future in range(index + 1, len(starts))
                        )
                        remaining_axes = value.ndim - input_axis
                        fill = max(remaining_axes - remaining_specs, 0)
                        slices.extend([slice(None)] * fill)
                        input_axis += fill
                        continue
                    if new_axis_mask & bit:
                        slices.append(None)
                        continue
                    if shrink_axis_mask & bit:
                        selected = 0 if begin_mask & bit else start
                        slices.append(int(selected))
                        input_axis += 1
                        continue
                    step = int(stride)
                    slices.append(
                        slice(
                            None if begin_mask & bit else int(start),
                            None if end_mask & bit else int(end),
                            step,
                        )
                    )
                    input_axis += 1
                slices.extend([slice(None)] * (value.ndim - input_axis))
                return _shortcut_result(
                    'TFLiteStridedSlice(X, begin, end, strides, masks)',
                    value[tuple(slices)],
                    {
                        'starts': starts,
                        'ends': ends,
                        'strides': strides,
                        'begin_mask': begin_mask,
                        'end_mask': end_mask,
                        'ellipsis_mask': ellipsis_mask,
                        'new_axis_mask': new_axis_mask,
                        'shrink_axis_mask': shrink_axis_mask,
                    },
                )
            # TensorFlow GraphDef StridedSlice uses the same mask encoding as
            # TFLite.  Preserve open-ended Python slices (for example ``:3``)
            # instead of treating the encoded sentinel end as a literal zero.
            begin_mask = int(attrs.get('begin_mask', 0))
            end_mask = int(attrs.get('end_mask', 0))
            ellipsis_mask = int(attrs.get('ellipsis_mask', 0))
            new_axis_mask = int(attrs.get('new_axis_mask', 0))
            shrink_axis_mask = int(attrs.get('shrink_axis_mask', 0))
            if begin_mask or end_mask or ellipsis_mask or new_axis_mask or shrink_axis_mask:
                value = np.asarray(inputs[0])
                slices = []
                input_axis = 0
                for index, (start, end, stride) in enumerate(zip(starts, ends, strides)):
                    bit = 1 << index
                    if ellipsis_mask & bit:
                        remaining_specs = sum(
                            not (new_axis_mask & (1 << future))
                            and not (ellipsis_mask & (1 << future))
                            for future in range(index + 1, len(starts))
                        )
                        remaining_axes = value.ndim - input_axis
                        fill = max(remaining_axes - remaining_specs, 0)
                        slices.extend([slice(None)] * fill)
                        input_axis += fill
                        continue
                    if new_axis_mask & bit:
                        slices.append(None)
                        continue
                    if shrink_axis_mask & bit:
                        slices.append(0 if begin_mask & bit else int(start))
                        input_axis += 1
                        continue
                    slices.append(slice(
                        None if begin_mask & bit else int(start),
                        None if end_mask & bit else int(end),
                        int(stride),
                    ))
                    input_axis += 1
                slices.extend([slice(None)] * (value.ndim - input_axis))
                return _shortcut_result(
                    'StridedSlice(X, begin, end, strides, masks)',
                    value[tuple(slices)],
                    {
                        'begin': starts,
                        'end': ends,
                        'strides': strides,
                        'begin_mask': begin_mask,
                        'end_mask': end_mask,
                        'ellipsis_mask': ellipsis_mask,
                        'new_axis_mask': new_axis_mask,
                        'shrink_axis_mask': shrink_axis_mask,
                    },
                )
            axes = [int(item) for item in (attrs.get('axes') or range(len(starts)))]
            slices = [slice(None)] * np.asarray(inputs[0]).ndim
            for axis, start, end, stride in zip(axes, starts, ends, strides):
                slices[axis] = slice(start, end, stride)
            return _shortcut_result(
                'StridedSlice(X, starts, ends, strides)',
                np.asarray(inputs[0])[tuple(slices)],
                {'axes': axes, 'starts': starts, 'ends': ends, 'strides': strides},
            )
        if inputs and label in {'slice'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            axes = [int(item) for item in (attrs.get('axes') or [])]
            starts = [int(item) for item in (attrs.get('starts') or [])]
            ends = [int(item) for item in (attrs.get('ends') or [])]
            value = NumpyBackend.get_ops()['Slice'](
                np.asarray(inputs[0]), axes, starts, ends
            )
            return _shortcut_result(
                'Slice(X, axes, starts, ends)',
                value,
                {'axes': axes, 'starts': starts, 'ends': ends},
            )
        if inputs and label in {'resizenearestneighbor', 'resizebilinear'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            raw_size = attrs.get('size', inputs[1] if len(inputs) >= 2 else [])
            size = [int(item) for item in np.asarray(raw_size).reshape(-1)]
            value = np.asarray(inputs[0])
            if len(size) != 2 or value.ndim != 4 or any(item <= 0 for item in size):
                raise ValueError('TFLite Resize requires NHWC rank-4 input and two positive sizes')
            align_corners = bool(attrs.get('align_corners', False))
            half_pixel_centers = bool(attrs.get('half_pixel_centers', False))

            def coordinates(output_size, input_size):
                if align_corners and output_size > 1:
                    return np.arange(output_size, dtype=np.float64) * (
                        (input_size - 1) / (output_size - 1)
                    )
                if half_pixel_centers:
                    return (
                        (np.arange(output_size, dtype=np.float64) + 0.5)
                        * input_size / output_size
                        - (0.0 if label == 'resizenearestneighbor' else 0.5)
                    )
                return np.arange(output_size, dtype=np.float64) * input_size / output_size

            y = coordinates(size[0], value.shape[1])
            x_coord = coordinates(size[1], value.shape[2])
            if label == 'resizenearestneighbor':
                if align_corners:
                    yi = np.floor(y + 0.5).astype(np.int64)
                    xi = np.floor(x_coord + 0.5).astype(np.int64)
                else:
                    yi = np.floor(y).astype(np.int64)
                    xi = np.floor(x_coord).astype(np.int64)
                yi = np.clip(yi, 0, value.shape[1] - 1)
                xi = np.clip(xi, 0, value.shape[2] - 1)
                output = value[:, yi[:, None], xi[None, :], :]
                formula = 'ResizeNearestNeighbor(X, size, align_corners, half_pixel_centers)'
            else:
                y = np.clip(y, 0.0, value.shape[1] - 1)
                x_coord = np.clip(x_coord, 0.0, value.shape[2] - 1)
                y0 = np.floor(y).astype(np.int64)
                x0 = np.floor(x_coord).astype(np.int64)
                y1 = np.minimum(y0 + 1, value.shape[1] - 1)
                x1 = np.minimum(x0 + 1, value.shape[2] - 1)
                wy = y - y0
                wx = x_coord - x0
                output = (
                    value[:, y0[:, None], x0[None, :], :]
                    * (1 - wy)[None, :, None, None]
                    * (1 - wx)[None, None, :, None]
                    + value[:, y0[:, None], x1[None, :], :]
                    * (1 - wy)[None, :, None, None]
                    * wx[None, None, :, None]
                    + value[:, y1[:, None], x0[None, :], :]
                    * wy[None, :, None, None]
                    * (1 - wx)[None, None, :, None]
                    + value[:, y1[:, None], x1[None, :], :]
                    * wy[None, :, None, None]
                    * wx[None, None, :, None]
                ).astype(value.dtype, copy=False)
                formula = 'ResizeBilinear(X, size, align_corners, half_pixel_centers)'
            return _shortcut_result(
                formula,
                output,
                {
                    'size': size,
                    'align_corners': align_corners,
                    'half_pixel_centers': half_pixel_centers,
                },
            )
        if inputs and label in {'upsampling2d', 'interpolate'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            mode = str(attrs.get('interpolation', attrs.get('mode', 'nearest'))).strip("'\"").lower()
            raw_size = attrs.get('size') or attrs.get('scale_factor')
            if raw_size is None:
                raise ValueError('upsampling/interpolate scale factor is missing')
            factors = tuple(float(item) for item in np.asarray(raw_size).reshape(-1))
            if len(factors) != 2 or any(item <= 0 for item in factors):
                raise ValueError(f'upsampling/interpolate factors are invalid: {factors}')
            value = np.asarray(inputs[0])
            if value.ndim != 4:
                raise ValueError(f'2D upsampling requires rank-4 input, got {value.shape}')
            framework = str(getattr(symnet, 'framework_type', '')).lower()
            data_format = str(
                attrs.get(
                    'data_format',
                    'channels_last' if framework == 'tensorflow' else 'channels_first',
                )
            ).strip("'\"").lower()
            align_corners = bool(attrs.get('align_corners', False))
            align_mode = int(attrs.get('align_mode', 0))
            output = NumpyBackend.get_ops()['Resize2D'](
                value,
                factors,
                mode,
                align_corners,
                align_mode,
                data_format,
            )
            return _shortcut_result(
                'Resize2D(X, scale_factor, mode, align_corners, align_mode, data_format)',
                output,
                {
                    'mode': mode,
                    'scale_factor': list(factors),
                    'align_corners': align_corners,
                    'align_mode': align_mode,
                    'data_format': data_format,
                },
            )
        if inputs and label in {'averagepooling2d', 'avgpool2d'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            kernel = [int(item) for item in (attrs.get('pool_size') or attrs.get('kernel_size') or [1, 1])]
            stride = [int(item) for item in (attrs.get('strides') or kernel)]
            padding_attr = attrs.get('padding', 'valid')
            padding = str(padding_attr).strip("'\"").lower()
            value = np.asarray(inputs[0])
            if value.ndim == 4:
                data_format = str(attrs.get('data_format', '')).strip("'\"").lower()
                channels_first = data_format in {'channels_first', 'nchw'}
                if channels_first:
                    value = np.transpose(value, (0, 2, 3, 1))
                kh, kw = kernel; sh, sw = stride
                source_height, source_width = value.shape[1], value.shape[2]
                ceil_mode = bool(attrs.get('ceil_mode', False))
                exclusive = bool(attrs.get('exclusive', True))
                divisor_override = attrs.get('divisor_override')
                if divisor_override is not None:
                    try:
                        divisor_override = float(np.asarray(divisor_override).reshape(-1)[0])
                    except (TypeError, ValueError, IndexError):
                        divisor_override = None
                pad_top = pad_bottom = pad_left = pad_right = 0
                explicit_pad_top = explicit_pad_bottom = explicit_pad_left = explicit_pad_right = 0
                if isinstance(padding_attr, (list, tuple)):
                    explicit_padding = [int(item) for item in padding_attr]
                    if len(explicit_padding) == 2:
                        # Paddle's two-value form is the amount on each side.
                        pad_top = pad_bottom = explicit_padding[0]
                        pad_left = pad_right = explicit_padding[1]
                    elif len(explicit_padding) == 4:
                        pad_top, pad_bottom, pad_left, pad_right = explicit_padding
                    else:
                        raise ValueError(f'AveragePooling2D padding rank is invalid: {padding_attr!r}')
                    explicit_pad_top, explicit_pad_bottom = pad_top, pad_bottom
                    explicit_pad_left, explicit_pad_right = pad_left, pad_right
                    numerator_h = value.shape[1] + pad_top + pad_bottom - kh
                    numerator_w = value.shape[2] + pad_left + pad_right - kw
                    out_h = int(np.ceil(numerator_h / sh) + 1) if ceil_mode else int(np.floor(numerator_h / sh) + 1)
                    out_w = int(np.ceil(numerator_w / sw) + 1) if ceil_mode else int(np.floor(numerator_w / sw) + 1)
                    if ceil_mode:
                        pad_bottom += max((out_h - 1) * sh + kh - (value.shape[1] + pad_top + pad_bottom), 0)
                        pad_right += max((out_w - 1) * sw + kw - (value.shape[2] + pad_left + pad_right), 0)
                elif padding == 'same':
                    out_h = int(np.ceil(value.shape[1] / sh)); out_w = int(np.ceil(value.shape[2] / sw))
                    total_h = max((out_h - 1) * sh + kh - value.shape[1], 0); total_w = max((out_w - 1) * sw + kw - value.shape[2], 0)
                    pad_top, pad_bottom = total_h // 2, total_h - total_h // 2
                    pad_left, pad_right = total_w // 2, total_w - total_w // 2
                    explicit_pad_top, explicit_pad_bottom = pad_top, pad_bottom
                    explicit_pad_left, explicit_pad_right = pad_left, pad_right
                else:
                    numerator_h = value.shape[1] + pad_top + pad_bottom - kh
                    numerator_w = value.shape[2] + pad_left + pad_right - kw
                    out_h = int(np.ceil(numerator_h / sh) + 1) if ceil_mode else int(np.floor(numerator_h / sh) + 1)
                    out_w = int(np.ceil(numerator_w / sw) + 1) if ceil_mode else int(np.floor(numerator_w / sw) + 1)
                    if ceil_mode:
                        pad_bottom += max((out_h - 1) * sh + kh - (value.shape[1] + pad_top + pad_bottom), 0)
                        pad_right += max((out_w - 1) * sw + kw - (value.shape[2] + pad_left + pad_right), 0)
                valid_mask = np.ones(value.shape, dtype=np.float32)
                value = np.pad(value, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
                valid_mask = np.pad(valid_mask, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
                value = np.stack([
                    np.stack([
                        value[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :].sum(axis=(1, 2))
                        / np.maximum(
                            float(divisor_override)
                            if divisor_override is not None
                            else (
                                valid_mask[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :].sum(axis=(1, 2))
                                if exclusive
                                else float(
                                    (max(0, min(i * sh - pad_top + kh, source_height) - max(i * sh - pad_top, 0))
                                     + min(max(-(i * sh - pad_top), 0), explicit_pad_top)
                                     + min(max(i * sh - pad_top + kh - source_height, 0), explicit_pad_bottom))
                                    * (max(0, min(j * sw - pad_left + kw, source_width) - max(j * sw - pad_left, 0))
                                       + min(max(-(j * sw - pad_left), 0), explicit_pad_left)
                                       + min(max(j * sw - pad_left + kw - source_width, 0), explicit_pad_right))
                                )
                            ),
                            1.0,
                        )
                        for j in range(out_w)
                    ], axis=1)
                    for i in range(out_h)
                ], axis=1)
                if channels_first:
                    value = np.transpose(value, (0, 3, 1, 2))
            return _shortcut_result('AveragePooling2D(X, pool_size, strides, padding)', value, {'pool_size': kernel, 'strides': stride, 'padding': padding})
        if (
            inputs
            and str(getattr(symnet, 'framework_type', '')).lower() == 'tflite'
            and label in {'maxpool2d'}
        ):
            attrs = getattr(op_instance, 'attrs', {}) or {}
            value = np.asarray(inputs[0])
            if value.ndim != 4:
                raise ValueError(f'TFLite MaxPool2D requires NHWC rank-4 input, got {value.shape}')
            kernel = [int(item) for item in attrs.get('pool_size', [1, 1])]
            stride = [int(item) for item in attrs.get('strides', kernel)]
            if len(kernel) != 2 or len(stride) != 2:
                raise ValueError('TFLite MaxPool2D filter/stride options are invalid')
            kh, kw = kernel
            sh, sw = stride
            padding_code = int(attrs.get('padding_code', 0))
            if padding_code == 0:
                out_h = int(math.ceil(value.shape[1] / sh))
                out_w = int(math.ceil(value.shape[2] / sw))
                pad_h = max((out_h - 1) * sh + kh - value.shape[1], 0)
                pad_w = max((out_w - 1) * sw + kw - value.shape[2], 0)
            elif padding_code == 1:
                pad_h = pad_w = 0
                out_h = (value.shape[1] - kh) // sh + 1
                out_w = (value.shape[2] - kw) // sw + 1
            else:
                raise ValueError(f'unqualified TFLite padding enum {padding_code}')
            minimum = (
                -np.inf
                if np.issubdtype(value.dtype, np.floating)
                else np.iinfo(value.dtype).min
            )
            padded = np.pad(
                value,
                ((0, 0), (pad_h // 2, pad_h - pad_h // 2),
                 (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                mode='constant',
                constant_values=minimum,
            )
            windows = np.lib.stride_tricks.sliding_window_view(
                padded, (kh, kw), axis=(1, 2)
            )[:, ::sh, ::sw]
            output = np.max(windows, axis=(-1, -2))
            output = _tflite_fused_activation(
                output, attrs.get('fused_activation', 0)
            )
            return _shortcut_result(
                'TFLiteMaxPool2D_NHWC(X)',
                output,
                {
                    'pool_size': kernel,
                    'strides': stride,
                    'padding_code': padding_code,
                    'fused_activation': int(attrs.get('fused_activation', 0)),
                },
            )
        if inputs and label in {'mean', 'reducemean'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            raw_axis = attrs.get('axis', attrs.get('dim', None))
            if raw_axis is None or (
                isinstance(raw_axis, str) and raw_axis.strip("'\"").lower() in {'all', 'none'}
            ):
                axis_tuple = None
            else:
                axis_tuple = tuple(int(item) for item in np.asarray(raw_axis).reshape(-1))
            keepdims = bool(attrs.get('keepdim', attrs.get('keepdims', attrs.get('keep_dims', False))))
            return _shortcut_result(
                'ReduceMean(X, axis, keepdims)',
                np.mean(np.asarray(inputs[0]), axis=axis_tuple, keepdims=keepdims),
                {'axis': None if axis_tuple is None else list(axis_tuple), 'keepdims': keepdims},
            )
        if inputs and label in {'concat', 'concatv2'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            axis = int(attrs.get('axis', 0))
            return _shortcut_result('Concat(X1, X2, axis)', np.concatenate([np.asarray(item) for item in inputs], axis=axis), {'axis': axis})
        if inputs and label in {'stack'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            axis = int(attrs.get('axis', 0))
            return _shortcut_result(
                'Stack(X1, ..., Xn, axis)',
                np.stack([np.asarray(item) for item in inputs], axis=axis),
                {'axis': axis, 'input_count': len(inputs)},
            )
        if len(inputs) >= 2 and label in {'reverse'}:
            axes = tuple(
                int(item) for item in np.asarray(inputs[1]).reshape(-1)
            )
            return _shortcut_result(
                'Reverse(X, axis)',
                np.flip(np.asarray(inputs[0]), axis=axes),
                {'axis': list(axes)},
            )
        if len(inputs) >= 3 and label in {'range3'}:
            start = np.asarray(inputs[0]).reshape(-1)[0]
            limit = np.asarray(inputs[1]).reshape(-1)[0]
            delta = np.asarray(inputs[2]).reshape(-1)[0]
            if delta == 0:
                raise ValueError('Range3 delta must be non-zero')
            dtype = np.result_type(
                np.asarray(inputs[0]).dtype,
                np.asarray(inputs[1]).dtype,
                np.asarray(inputs[2]).dtype,
            )
            return _shortcut_result(
                'Range3(start, limit, delta)',
                np.arange(start, limit, delta, dtype=dtype),
                {'start': start.item() if hasattr(start, 'item') else start,
                 'limit': limit.item() if hasattr(limit, 'item') else limit,
                 'delta': delta.item() if hasattr(delta, 'item') else delta},
            )
        if inputs and label in {'addn'}:
            output = np.asarray(inputs[0]).copy()
            for value in inputs[1:]:
                output = output + np.asarray(value)
            return _shortcut_result(
                'AddN(X1, ..., Xn)', output, {'input_count': len(inputs)}
            )
        if len(inputs) >= 2 and label in {'fill'}:
            shape = tuple(
                int(item) for item in np.asarray(inputs[0]).reshape(-1)
            )
            value = np.asarray(inputs[1]).reshape(-1)
            if value.size != 1:
                raise ValueError('Fill value must be scalar')
            return _shortcut_result(
                'Fill(shape, value)',
                np.full(shape, value[0], dtype=value.dtype),
                {'shape': list(shape), 'dtype': str(value.dtype)},
            )
        if len(inputs) >= 3 and label in {'spacetobatchnd'}:
            value = np.asarray(inputs[0])
            block = np.asarray(inputs[1]).reshape(-1).astype(np.int64)
            paddings = np.asarray(inputs[2]).reshape(-1, 2).astype(np.int64)
            spatial_rank = int(block.size)
            if spatial_rank == 0 or paddings.shape != (spatial_rank, 2):
                raise ValueError('SpaceToBatchND block/padding schema is invalid')
            if value.ndim < spatial_rank + 1 or np.any(block <= 0):
                raise ValueError('SpaceToBatchND input rank or block shape is invalid')
            padded = np.pad(
                value,
                [(0, 0)] + [tuple(item) for item in paddings]
                + [(0, 0)] * (value.ndim - spatial_rank - 1),
            )
            spatial = padded.shape[1:1 + spatial_rank]
            if any(size % factor for size, factor in zip(spatial, block)):
                raise ValueError('SpaceToBatchND padded spatial shape is not divisible by block shape')
            remaining = padded.shape[1 + spatial_rank:]
            reshape_shape = [padded.shape[0]]
            for size, factor in zip(spatial, block):
                reshape_shape.extend([size // int(factor), int(factor)])
            reshape_shape.extend(remaining)
            reshaped = padded.reshape(reshape_shape)
            block_axes = [2 + 2 * index for index in range(spatial_rank)]
            spatial_axes = [1 + 2 * index for index in range(spatial_rank)]
            tail_axes = list(range(1 + 2 * spatial_rank, reshaped.ndim))
            permuted = np.transpose(
                reshaped, block_axes + [0] + spatial_axes + tail_axes
            )
            output_shape = [padded.shape[0] * int(np.prod(block))]
            output_shape.extend(
                size // int(factor) for size, factor in zip(spatial, block)
            )
            output_shape.extend(remaining)
            return _shortcut_result(
                'SpaceToBatchND(X, block_shape, paddings)',
                permuted.reshape(output_shape),
                {'block_shape': block.tolist(), 'paddings': paddings.tolist()},
            )
        if len(inputs) >= 3 and label in {'batchtospacend'}:
            value = np.asarray(inputs[0])
            block = np.asarray(inputs[1]).reshape(-1).astype(np.int64)
            crops = np.asarray(inputs[2]).reshape(-1, 2).astype(np.int64)
            spatial_rank = int(block.size)
            block_product = int(np.prod(block))
            if spatial_rank == 0 or crops.shape != (spatial_rank, 2):
                raise ValueError('BatchToSpaceND block/crop schema is invalid')
            if block_product <= 0 or value.shape[0] % block_product:
                raise ValueError('BatchToSpaceND batch is not divisible by block product')
            output_batch = value.shape[0] // block_product
            spatial = value.shape[1:1 + spatial_rank]
            remaining = value.shape[1 + spatial_rank:]
            reshaped = value.reshape(
                list(block) + [output_batch] + list(spatial) + list(remaining)
            )
            batch_axis = spatial_rank
            spatial_axes = list(range(spatial_rank + 1, 2 * spatial_rank + 1))
            tail_axes = list(range(2 * spatial_rank + 1, reshaped.ndim))
            permutation = [batch_axis]
            for index in range(spatial_rank):
                permutation.extend([spatial_axes[index], index])
            permutation.extend(tail_axes)
            permuted = np.transpose(reshaped, permutation)
            expanded_shape = [output_batch]
            expanded_shape.extend(
                int(size) * int(factor) for size, factor in zip(spatial, block)
            )
            expanded_shape.extend(remaining)
            expanded = permuted.reshape(expanded_shape)
            slices = [slice(None)]
            for size, crop in zip(expanded_shape[1:1 + spatial_rank], crops):
                slices.append(slice(int(crop[0]), size - int(crop[1])))
            slices.extend([slice(None)] * len(remaining))
            return _shortcut_result(
                'BatchToSpaceND(X, block_shape, crops)',
                expanded[tuple(slices)],
                {'block_shape': block.tolist(), 'crops': crops.tolist()},
            )
        if inputs and label in {'real'}:
            return _shortcut_result('Real(X)', np.real(np.asarray(inputs[0])))
        if inputs and label in {'square'}:
            return _shortcut_result('Square(X)', np.square(np.asarray(inputs[0])))
        if inputs and label in {'fft2d'}:
            value = np.asarray(inputs[0])
            if value.ndim < 2 or not np.issubdtype(value.dtype, np.complexfloating):
                raise ValueError('FFT2D requires a complex tensor with rank at least two')
            output = np.fft.fft2(value, axes=(-2, -1)).astype(value.dtype, copy=False)
            return _shortcut_result(
                'FFT2D(X)',
                output,
                {'axes': [-2, -1], 'dtype': str(value.dtype)},
            )
        if len(inputs) >= 3 and label in {'unsortedsegmentprod'}:
            data = np.asarray(inputs[0])
            segment_ids = np.asarray(inputs[1]).astype(np.int64)
            num_segments = int(np.asarray(inputs[2]).reshape(-1)[0])
            if num_segments < 0:
                raise ValueError('UnsortedSegmentProd num_segments must be non-negative')
            if tuple(data.shape[:segment_ids.ndim]) != tuple(segment_ids.shape):
                raise ValueError(
                    'UnsortedSegmentProd segment_ids shape must prefix the data shape'
                )
            if np.any(segment_ids >= num_segments):
                raise ValueError('UnsortedSegmentProd segment id is out of range')
            suffix = data.shape[segment_ids.ndim:]
            output = np.ones((num_segments,) + suffix, dtype=data.dtype)
            flat_data = data.reshape((segment_ids.size,) + suffix)
            for value, segment in zip(flat_data, segment_ids.reshape(-1)):
                if segment >= 0:
                    output[int(segment)] *= value
            return _shortcut_result(
                'UnsortedSegmentProd(X, segment_ids, num_segments)',
                output,
                {
                    'num_segments': num_segments,
                    'segment_id_rank': int(segment_ids.ndim),
                },
            )
        if inputs and label in {'onehot'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            depth = int(
                attrs.get(
                    'depth',
                    np.asarray(inputs[1]).reshape(-1)[0]
                    if len(inputs) >= 2
                    else 0,
                )
            )
            if depth <= 0:
                raise ValueError('OneHot requires positive static depth')
            axis = int(attrs.get('axis', -1))
            indices = np.asarray(inputs[0]).astype(np.int64)
            on_value = np.asarray(inputs[2]).reshape(-1)[0] if len(inputs) >= 3 else 1.0
            off_value = np.asarray(inputs[3]).reshape(-1)[0] if len(inputs) >= 4 else 0.0
            dtype = np.result_type(on_value, off_value)
            output = np.full(indices.shape + (depth,), off_value, dtype=dtype)
            valid = (indices >= 0) & (indices < depth)
            coordinates = np.nonzero(valid)
            output[coordinates + (indices[coordinates],)] = on_value
            if axis != -1 and axis != output.ndim - 1:
                output = np.moveaxis(output, -1, axis)
            return _shortcut_result(
                'OneHot(indices, depth, on_value, off_value, axis)', output,
                {
                    'depth': depth,
                    'axis': axis,
                    'dtype': str(output.dtype),
                    'on_value': on_value.item() if hasattr(on_value, 'item') else on_value,
                    'off_value': off_value.item() if hasattr(off_value, 'item') else off_value,
                },
            )
        if inputs and label in {'arange'}:
            end = int(np.asarray(inputs[-1]).reshape(-1)[0])
            return _shortcut_result('Arange(end)', np.arange(end, dtype=np.int64))
        if inputs and label in {'mod', 'remainder'} and len(inputs) >= 2:
            return _shortcut_result('Mod(X, Y)', np.mod(np.asarray(inputs[0]), np.asarray(inputs[1])))
        if inputs and label in {'bilinearcross'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}

            def tensor_attr(name):
                value = attrs.get(name)
                if value is None:
                    value = getattr(getattr(op_instance, 'module', None), name, None)
                return np.asarray(value.numpy() if hasattr(value, 'numpy') else value)

            u = tensor_attr('u')
            v = tensor_attr('v')
            bias = tensor_attr('bias')
            if u.ndim != 2 or v.ndim != 2 or bias.ndim != 1:
                raise ValueError('BilinearCross schema requires u/v matrices and rank-1 bias')
            value = np.asarray(inputs[0]) * np.matmul(np.matmul(np.asarray(inputs[1]), u), v)
            value = value + bias + np.asarray(inputs[1])
            return _shortcut_result(
                'BilinearCross(X0, X, u, v, bias)',
                value,
                {
                    'semantic_extension': 'bilinear_cross',
                    'u_shape': list(u.shape),
                    'v_shape': list(v.shape),
                    'bias_shape': list(bias.shape),
                },
            )
        if inputs and label in {'factorizationinteraction'}:
            value = np.asarray(inputs[0])
            if value.ndim != 3:
                raise ValueError(f'FactorizationInteraction requires rank-3 input, got {value.shape}')
            square_of_sum = np.square(np.sum(value, axis=1))
            sum_of_square = np.sum(np.square(value), axis=1)
            output = 0.5 * np.sum(square_of_sum - sum_of_square, axis=-1, keepdims=True)
            return _shortcut_result(
                'FactorizationInteraction(X)',
                output,
                {'input_rank': int(value.ndim), 'field_axis': 1, 'scale': 0.5},
            )
        if inputs and label in {'scale'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            scale = float(attrs.get('scale', 1.0))
            bias = float(attrs.get('bias', attrs.get('bias_value', 0.0)))
            return _shortcut_result(
                'Scale(X, scale, bias)',
                np.asarray(inputs[0]) * scale + bias,
                {'scale': scale, 'bias': bias},
            )
        if inputs and label in {'reshape'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            shape = attrs.get('target_shape')
            if shape is None:
                shape = attrs.get('shape')
            if shape is None and len(inputs) > 1:
                shape = inputs[1]
            if shape is not None:
                shape_tuple = tuple(int(item) for item in np.asarray(shape).reshape(-1))
                value = np.asarray(inputs[0])
                if str(getattr(symnet, 'framework_type', '')).lower() == 'paddle':
                    shape_tuple = tuple(
                        int(value.shape[index]) if dimension == 0 else dimension
                        for index, dimension in enumerate(shape_tuple)
                    )
                if bool(attrs.get('keras_preserve_batch', False)):
                    shape_tuple = (int(value.shape[0]),) + shape_tuple
                return _shortcut_result(
                    'Reshape(X, target_shape)',
                    np.reshape(value, shape_tuple),
                    {
                        'target_shape': list(shape_tuple),
                        'keras_preserve_batch': bool(
                            attrs.get('keras_preserve_batch', False)
                        ),
                    },
                )
        if inputs and label in {'permute', 'transpose'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            perm = attrs.get('perm') or attrs.get('dims')
            if perm is not None:
                perm_tuple = tuple(int(item) for item in np.asarray(perm).reshape(-1))
                value = np.asarray(inputs[0])
                framework = str(getattr(symnet, 'framework_type', '')).lower()
                # Keras Permute uses 1-based feature axes and omits the batch
                # axis; native transpose requires the explicit batch prefix.
                if framework == 'tensorflow' and len(perm_tuple) == value.ndim - 1:
                    perm_tuple = (0,) + perm_tuple
                return _shortcut_result('Transpose(X, perm)', np.transpose(value, perm_tuple), {'perm': list(perm_tuple)})
        if inputs and label in {'squeeze'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            axis = attrs.get('axis', attrs.get('dim', attrs.get('arg_1')))
            value = NumpyBackend.get_ops()['Squeeze'](np.asarray(inputs[0]), axis)
            return _shortcut_result('Squeeze(X, axis)', value, {'axis': axis})
        if inputs and label in {'expanddims', 'unsqueeze'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            raw_axis = attrs.get('axis', inputs[1] if len(inputs) > 1 else 0)
            axes = tuple(int(item) for item in np.asarray(raw_axis).reshape(-1))
            axis = axes[0] if len(axes) == 1 else axes
            return _shortcut_result('ExpandDims(X, axis)', np.expand_dims(np.asarray(inputs[0]), axis=axis), {'axis': list(axes)})
        if inputs and label in {'prod', 'reduceprod'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            raw_axis = attrs.get('axis')
            axis = None if raw_axis is None else tuple(int(item) for item in np.asarray(raw_axis).reshape(-1))
            if axis == ():
                axis = None
            keepdim = bool(attrs.get('keepdim', attrs.get('keepdims', False)))
            return _shortcut_result(
                'ReduceProd(X, axis, keepdim)',
                np.prod(np.asarray(inputs[0]), axis=axis, keepdims=keepdim),
                {'axis': None if axis is None else list(axis), 'keepdim': keepdim},
            )
        if inputs and label in {'cast'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            dtype = attrs.get('dtype', 'float32')
            dtype_text = str(dtype).lower()
            dtype_match = re.search(r'(?:float|bfloat|int|uint|bool)\d*', dtype_text)
            dtype_name = dtype_match.group(0) if dtype_match else dtype_text.split('.')[-1].replace('dt_', '').strip(" <>:'\"")
            return _shortcut_result('Cast(X, dtype)', np.asarray(inputs[0]).astype(dtype_name), {'dtype': dtype_name})
        if inputs and label in {'astype'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            raw_dtype = attrs.get('dtype', attrs.get('arg_1', 'int32'))
            if isinstance(raw_dtype, (int, np.integer)):
                dtype_name = {0: 'bool', 1: 'uint8', 2: 'int8', 3: 'int32', 4: 'int64', 5: 'float16', 6: 'float32', 7: 'float64'}.get(int(raw_dtype), 'int32')
            else:
                dtype_text = str(raw_dtype).lower()
                dtype_match = re.search(r'(?:float|bfloat|int|uint|bool)\d*', dtype_text)
                dtype_name = dtype_match.group(0) if dtype_match else 'int32'
            return _shortcut_result('Astype(X, dtype)', np.asarray(inputs[0]).astype(dtype_name), {'dtype': dtype_name})
        if inputs and label in {'l2normalize'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            axis = int(attrs.get('axis', -1))
            value = np.asarray(inputs[0])
            denominator = np.sqrt(np.maximum(np.sum(np.square(value), axis=axis, keepdims=True), np.finfo(value.dtype).tiny))
            return _shortcut_result('L2Normalize(X, axis)', value / denominator, {'axis': axis})
        if inputs and label in {'assign'}:
            return _shortcut_result('Assign(X)', np.asarray(inputs[0]))
        if inputs and label in {'relu'}:
            return _shortcut_result('ReLU(X)', np.maximum(np.asarray(inputs[0]), 0))
        if inputs and label in {'clip'} and len(inputs) >= 3:
            value = np.asarray(inputs[0])
            minimum = np.asarray(inputs[1])
            maximum = np.asarray(inputs[2])
            return _shortcut_result(
                'Clip(X, min_value, max_value)',
                np.clip(value, minimum, maximum),
                {
                    'min_value': minimum.tolist() if minimum.ndim else float(minimum),
                    'max_value': maximum.tolist() if maximum.ndim else float(maximum),
                },
            )
        if inputs and label in {'cumsum'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            raw_axis = attrs.get('axis', inputs[1] if len(inputs) > 1 else 0)
            axis = int(np.asarray(raw_axis).reshape(-1)[0])
            return _shortcut_result(
                'Cumsum(X, axis)',
                np.cumsum(np.asarray(inputs[0]), axis=axis),
                {'axis': axis},
            )
        if inputs and label in {'sigmoid'}:
            value = np.asarray(inputs[0])
            return _shortcut_result('Sigmoid(X)', 1.0 / (1.0 + np.exp(-value)))
        if inputs and label in {'tanh'}:
            return _shortcut_result('Tanh(X)', np.tanh(np.asarray(inputs[0])))
        if inputs and label in {'softmax'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            axis = int(attrs.get('axis', -1))
            beta = float(attrs.get('beta', 1.0))
            value = np.asarray(inputs[0]) * beta
            shifted = value - np.max(value, axis=axis, keepdims=True)
            exp_value = np.exp(shifted)
            return _shortcut_result(
                'Softmax(beta * X, axis)',
                exp_value / np.sum(exp_value, axis=axis, keepdims=True),
                {'axis': axis, 'beta': beta},
            )
        if inputs and label in {'gelu'}:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            value = np.asarray(inputs[0])
            approximate = attrs.get('approximate', False)
            approximate = (
                bool(approximate)
                if isinstance(approximate, (bool, np.bool_))
                else str(approximate).strip("'\"").lower() in {'true', 'tanh', 'approximate'}
            )
            if approximate:
                output = 0.5 * value * (
                    1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (value + 0.044715 * value ** 3))
                )
            else:
                output = 0.5 * value * (
                    1.0 + np.vectorize(math.erf)(value / np.sqrt(2.0))
                )
            return _shortcut_result('Gelu(X)', output.astype(value.dtype, copy=False))
        if inputs and label in {'linear'} and str(getattr(symnet, 'framework_type', '')).lower() == 'tflite' and len(inputs) >= 2:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            value = np.asarray(inputs[0])
            weight = np.asarray(inputs[1])
            if weight.ndim != 2 or value.shape[-1] != weight.shape[1]:
                raise ValueError(
                    f'TFLite FULLY_CONNECTED expects OI weight, got input {value.shape} and weight {weight.shape}'
                )
            keep_num_dims = bool(attrs.get('keep_num_dims', False))
            working = value if keep_num_dims else value.reshape(-1, value.shape[-1])
            output = np.matmul(working, np.swapaxes(weight, -1, -2))
            if len(inputs) >= 3 and inputs[2] is not None:
                output = output + np.asarray(inputs[2])
            activation = int(attrs.get('fused_activation', 0))
            if activation == 1:
                output = np.maximum(output, 0)
            elif activation == 2:
                output = np.clip(output, -1, 1)
            elif activation == 3:
                output = np.clip(output, 0, 6)
            elif activation == 4:
                output = np.tanh(output)
            elif activation not in {0}:
                raise ValueError(f'unqualified TFLite fused activation code {activation}')
            return _shortcut_result(
                'MatMul(X, Transpose(W_OI)) + B',
                output,
                {
                    'weight_layout': 'OI',
                    'keep_num_dims': keep_num_dims,
                    'fused_activation': activation,
                },
            )

        if (
            inputs
            and str(getattr(symnet, 'framework_type', '')).lower() == 'tflite'
            and label in {'conv2d', 'depthwiseconv2d'}
            and len(inputs) >= 2
        ):
            attrs = getattr(op_instance, 'attrs', {}) or {}
            value = np.asarray(inputs[0])
            weight = np.asarray(inputs[1])
            if value.ndim != 4 or weight.ndim != 4 or not np.issubdtype(
                value.dtype, np.floating
            ) or not np.issubdtype(weight.dtype, np.floating):
                raise ValueError(
                    'TFLite convolution shortcut currently requires float32 NHWC/OHWI tensors'
                )
            stride_h = int(attrs.get('stride_h', 1))
            stride_w = int(attrs.get('stride_w', 1))
            dilation_h = int(attrs.get('dilation_h', 1))
            dilation_w = int(attrs.get('dilation_w', 1))
            kh = int(weight.shape[1])
            kw = int(weight.shape[2])
            effective_kh = (kh - 1) * dilation_h + 1
            effective_kw = (kw - 1) * dilation_w + 1
            padding_code = int(attrs.get('padding_code', 0))
            if padding_code == 0:  # TFLite SAME
                out_h = int(math.ceil(value.shape[1] / stride_h))
                out_w = int(math.ceil(value.shape[2] / stride_w))
                pad_h = max((out_h - 1) * stride_h + effective_kh - value.shape[1], 0)
                pad_w = max((out_w - 1) * stride_w + effective_kw - value.shape[2], 0)
            elif padding_code == 1:  # TFLite VALID
                pad_h = pad_w = 0
                out_h = (value.shape[1] - effective_kh) // stride_h + 1
                out_w = (value.shape[2] - effective_kw) // stride_w + 1
            else:
                raise ValueError(f'unqualified TFLite padding enum {padding_code}')
            if out_h <= 0 or out_w <= 0:
                raise ValueError('TFLite convolution output has non-positive shape')
            output_channels = int(weight.shape[3])
            if label == 'conv2d':
                if int(weight.shape[0]) != output_channels or int(weight.shape[3]) != value.shape[3]:
                    # TFLite OHWI: [out, height, width, in].
                    output_channels = int(weight.shape[0])
                    if int(weight.shape[3]) != value.shape[3]:
                        raise ValueError(
                            f'TFLite Conv2D expects OHWI input channels, got {weight.shape} for {value.shape}'
                        )
            if value.shape[0] * out_h * out_w * output_channels > 50_000_000:
                raise ValueError('TFLite convolution temporary window exceeds audit limit')
            if pad_h or pad_w:
                value_padded = np.pad(
                    value,
                    (
                        (0, 0),
                        (pad_h // 2, pad_h - pad_h // 2),
                        (pad_w // 2, pad_w - pad_w // 2),
                        (0, 0),
                    ),
                    mode='constant',
                )
            else:
                value_padded = value
            windows = np.lib.stride_tricks.sliding_window_view(
                value_padded, (effective_kh, effective_kw), axis=(1, 2)
            )
            windows = windows[:, ::stride_h, ::stride_w, :, ::dilation_h, ::dilation_w]
            if label == 'conv2d':
                # OHWI filter layout; windows are NHWCKL.
                output = np.einsum(
                    'nhwckl,oklc->nhwo', windows, weight, optimize=True
                )
            else:
                # Depthwise filter layout is [1, H, W, channel_multiplier].
                channels = int(value.shape[3])
                if int(weight.shape[0]) != 1 or int(weight.shape[3]) % channels:
                    raise ValueError(
                        f'TFLite DepthwiseConv2D filter shape is not channel-multiplier form: {weight.shape}'
                    )
                multiplier = int(weight.shape[3]) // channels
                depthwise_weight = weight.reshape(kh, kw, channels, multiplier)
                output = np.einsum(
                    'nhwckl,klcm->nhwcm', windows, depthwise_weight, optimize=True
                ).reshape(value.shape[0], out_h, out_w, channels * multiplier)
            if len(inputs) >= 3 and inputs[2] is not None:
                output = output + np.asarray(inputs[2]).reshape((1, 1, 1, -1))
            output = _tflite_fused_activation(
                output, attrs.get('fused_activation', 0)
            )
            return _shortcut_result(
                'TFLiteConv2D_NHWC_OHWI(X, W, B)',
                output.astype(value.dtype, copy=False),
                {
                    'layout': 'NHWC/OHWI',
                    'stride': [stride_h, stride_w],
                    'dilation': [dilation_h, dilation_w],
                    'padding_code': padding_code,
                    'fused_activation': int(attrs.get('fused_activation', 0)),
                },
            )

        if (
            inputs
            and str(getattr(symnet, 'framework_type', '')).lower() == 'tflite'
            and label in {'conv3d'}
            and len(inputs) >= 2
        ):
            attrs = getattr(op_instance, 'attrs', {}) or {}
            value = np.asarray(inputs[0])
            weight = np.asarray(inputs[1])
            if value.ndim != 5 or weight.ndim != 5:
                raise ValueError('TFLite Conv3D requires NDHWC input and DHWIO filter')
            kd, kh, kw, input_channels, output_channels = weight.shape
            if input_channels != value.shape[-1]:
                raise ValueError(
                    f'TFLite Conv3D DHWIO channels differ: {weight.shape} versus {value.shape}'
                )
            strides = [
                int(attrs.get('stride_d', 1)),
                int(attrs.get('stride_h', 1)),
                int(attrs.get('stride_w', 1)),
            ]
            dilations = [
                int(attrs.get('dilation_d', 1)),
                int(attrs.get('dilation_h', 1)),
                int(attrs.get('dilation_w', 1)),
            ]
            effective = [
                (size - 1) * dilation + 1
                for size, dilation in zip((kd, kh, kw), dilations)
            ]
            spatial = value.shape[1:4]
            padding_code = int(attrs.get('padding_code', 0))
            if padding_code == 0:
                output_spatial = [
                    int(math.ceil(size / stride))
                    for size, stride in zip(spatial, strides)
                ]
                total_padding = [
                    max((out - 1) * stride + kernel - size, 0)
                    for out, stride, kernel, size in zip(
                        output_spatial, strides, effective, spatial
                    )
                ]
            elif padding_code == 1:
                total_padding = [0, 0, 0]
                output_spatial = [
                    (size - kernel) // stride + 1
                    for size, kernel, stride in zip(spatial, effective, strides)
                ]
            else:
                raise ValueError(f'unqualified TFLite padding enum {padding_code}')
            if any(size <= 0 for size in output_spatial):
                raise ValueError('TFLite Conv3D output has non-positive shape')
            if value.shape[0] * int(np.prod(output_spatial)) * output_channels > 50_000_000:
                raise ValueError('TFLite Conv3D output exceeds audit limit')
            padded = np.pad(
                value,
                [(0, 0)]
                + [(pad // 2, pad - pad // 2) for pad in total_padding]
                + [(0, 0)],
            )
            output = np.zeros(
                (value.shape[0], *output_spatial, output_channels),
                dtype=np.result_type(value.dtype, weight.dtype),
            )
            for depth_index in range(kd):
                for height_index in range(kh):
                    for width_index in range(kw):
                        sampled = padded[
                            :,
                            depth_index * dilations[0]:depth_index * dilations[0] + output_spatial[0] * strides[0]:strides[0],
                            height_index * dilations[1]:height_index * dilations[1] + output_spatial[1] * strides[1]:strides[1],
                            width_index * dilations[2]:width_index * dilations[2] + output_spatial[2] * strides[2]:strides[2],
                            :,
                        ]
                        output += np.tensordot(
                            sampled,
                            weight[depth_index, height_index, width_index],
                            axes=([-1], [0]),
                        )
            if len(inputs) >= 3 and inputs[2] is not None:
                output += np.asarray(inputs[2]).reshape((1, 1, 1, 1, -1))
            output = _tflite_fused_activation(
                output, attrs.get('fused_activation', 0)
            )
            return _shortcut_result(
                'TFLiteConv3D_NDHWC_DHWIO(X, W, B)',
                output.astype(value.dtype, copy=False),
                {
                    'layout': 'NDHWC/DHWIO',
                    'strides': strides,
                    'dilations': dilations,
                    'padding_code': padding_code,
                    'fused_activation': int(attrs.get('fused_activation', 0)),
                },
            )

        if (
            len(inputs) >= 2
            and str(getattr(symnet, 'framework_type', '')).lower() == 'tflite'
            and label in {'pad'}
        ):
            value = np.asarray(inputs[0])
            paddings = np.asarray(inputs[1]).reshape(-1, 2).astype(np.int64)
            if paddings.shape[0] != value.ndim:
                raise ValueError(
                    f'TFLite Pad rank mismatch: {paddings.shape[0]} versus {value.ndim}'
                )
            fill = np.asarray(inputs[2]).reshape(-1)[0] if len(inputs) >= 3 else 0
            return _shortcut_result(
                'TFLitePad(X, paddings, value)',
                np.pad(
                    value,
                    [tuple(item) for item in paddings],
                    mode='constant',
                    constant_values=fill,
                ),
                {'paddings': paddings.tolist(), 'value': fill.item() if hasattr(fill, 'item') else fill},
            )

        # Padding is a tensor-shape operation whose static paddings are
        # already captured from the GraphDef.  Route this exact, explicit
        # MathIR definition through the tested NumPy primitive instead of the
        # generic pointwise kernel, which cannot vectorize a rank-2 paddings
        # parameter.  The diagnostic is still recorded as a formula result.
        if str(op_label).lower() in {'identity'} and inputs:
            self.last_jit_context = {
                'formula_diagnostic': {
                    'operation': str(op_label),
                    'framework': str(getattr(symnet, 'framework_type', 'unknown')),
                    'native_formula': 'X',
                    'canonical_formula': 'X',
                    'parameter_bindings': {},
                    'derived_bindings': {},
                    'canonical_bindings': {},
                    'layout': {},
                }
            }
            self.last_jit_contexts = [self.last_jit_context]
            return inputs[0]

        if str(op_label) in {'Pad', 'PadV2'} and inputs:
            attrs = getattr(op_instance, 'attrs', {}) or {}
            pad_impl = NumpyBackend.get_ops().get('Pad')
            if pad_impl is not None:
                output = pad_impl(
                    inputs[0],
                    attrs.get('paddings', []),
                    attrs.get('constant_values', 0),
                )
                self.last_jit_context = {
                    'formula_diagnostic': {
                        'operation': str(op_label),
                        'framework': str(getattr(symnet, 'framework_type', 'unknown')),
                        'native_formula': str(raw_expr_container),
                        'canonical_formula': str(raw_expr_container),
                        'parameter_bindings': {
                            'paddings': attrs.get('paddings'),
                            'constant_values': attrs.get('constant_values', 0),
                        },
                        'derived_bindings': {},
                        'canonical_bindings': {},
                        'layout': {},
                    }
                }
                self.last_jit_contexts = [self.last_jit_context]
                return output

        if isinstance(raw_expr_container, (list, tuple)):
            exprs_list = raw_expr_container
            is_multi_output = True
        else:
            exprs_list = [raw_expr_container]
            is_multi_output = False

        # =========================================================
        # Cache Key
        # =========================================================
        cache_key = None
        try:
            cache_key = self._make_compilation_cache_key(op_def, op_instance, inputs, extra_kwargs)
            report_compile_stage('cache_key_ready')
            hash(cache_key)
            report_compile_stage('cache_hash_ready')

            if cache_key in self._compiled_kernel_cache:
                report_compile_stage('cache_hit')
                cached_results = self._compiled_kernel_cache[cache_key]
                final_results = []
                for cached_item in cached_results:
                    kernel_func, final_syms, binding, out_shape, res_ctx_lite = cached_item[:5]
                    functional_expr = cached_item[5] if len(cached_item) > 5 else None
                    naive_final_syms = cached_item[6] if len(cached_item) > 6 else final_syms
                    naive_functional_expr = cached_item[7] if len(cached_item) > 7 else functional_expr
                    runtime_data_map = cached_item[8] if len(cached_item) > 8 else {}
                    reduction_map = cached_item[9] if len(cached_item) > 9 else {}
                    formula_diagnostic = cached_item[10] if len(cached_item) > 10 else None
                    self.last_jit_context = {
                        'final_syms': final_syms,
                        'functional_expr': functional_expr,
                        'naive_final_syms': naive_final_syms,
                        'naive_functional_expr': naive_functional_expr,
                        'binding': binding,
                        'out_shape': out_shape,
                        'runtime_data_map': runtime_data_map,
                        'reduction_map': reduction_map,
                        'formula_diagnostic': formula_diagnostic,
                    }
                    self.last_jit_contexts.append(self.last_jit_context)
                    execution_ctx = dict(res_ctx_lite)
                    execution_ctx['reduction_map'] = reduction_map
                    output_np = self._execute_compiled_kernel(
                        kernel_func, final_syms, binding, out_shape,
                        inputs, op_instance, symnet, execution_ctx
                    )
                    final_results.append(output_np)
                report_compile_stage('cache_execution_complete')
                return tuple(final_results) if is_multi_output else final_results[0]
        except Exception:
            report_compile_stage('cache_path_fallback')
            pass

        # =========================================================
        # 1. Context Setup
        # =========================================================
        with self._timer("Step 1: Context Setup"):
            op_inputs_metadata = getattr(op_instance, 'inputs', inputs)
            num_inputs = len(op_inputs_metadata)
            default_names = ['X'] if num_inputs <= 1 else [f'X{i + 1}' for i in range(num_inputs)]
            actual_input_names = op_def.get('input_names', default_names)

            res_ctx = {
                'inputs': inputs, 'op_instance': op_instance, 'op_def': op_def,
                'symnet': symnet, 'input_names': actual_input_names,
                'extra_kwargs': extra_kwargs
            }

        # =========================================================
        # 2. Automated Symbol Role Detection (AST Inspection)
        # =========================================================
        with self._timer("Step 2: Automated Symbol Role Detection"):
            tensor_names = set(actual_input_names)
            num_outputs = len(exprs_list)
            if num_outputs <= 1:
                tensor_names.add('Y')
                tensor_names.add('Output')
            else:
                for i in range(num_outputs):
                    tensor_names.add(f'Y{i + 1}')

            loop_names = set()

            for e in exprs_list:
                for node in e.atoms(sympy.Indexed):
                    lbl = node.base.label
                    if hasattr(lbl, 'name'):
                        tensor_names.add(lbl.name)
                    else:
                        for sym in node.base.free_symbols:
                            if hasattr(sym, 'name'):
                                tensor_names.add(sym.name)

                for node in e.atoms(sympy.Sum) | e.atoms(sympy.Product):
                    for limit in node.limits:
                        if hasattr(limit[0], 'name'):
                            loop_names.add(limit[0].name)

                target_smart_ops = {'TensorSum', 'TensorProd', 'Sum', 'Prod', 'TensorMax', 'TensorMin', 'Max', 'Min'}
                for node in e.atoms(sympy.Function):
                    func_name = node.func.__name__ if hasattr(node.func, '__name__') else str(node.func)

                    if func_name in target_smart_ops:
                        for arg in node.args[1:]:
                            if isinstance(arg, (tuple, sympy.Tuple)) and len(arg) >= 1:
                                loop_var = arg[0]
                                if hasattr(loop_var, 'name'):
                                    loop_names.add(loop_var.name)

                for node in e.atoms(sympy.Function):
                    if node.func.__name__ in ['Max', 'Min', 'ArgMax', 'ArgMin']:
                        for arg in node.args:
                            if isinstance(arg, (tuple, sympy.Tuple)) and len(arg) == 3:
                                loop_var = arg[0]
                                if isinstance(loop_var, sympy.Symbol):
                                    loop_names.add(loop_var.name)

            standard_axes = {'n', 'c', 'out', 'l', 'd', 'h', 'w', 'i'}
            index_names = (loop_names | standard_axes) - tensor_names
            blacklist = tensor_names | index_names | \
                        {'Dim', 'Mod', 'Abs', 'floor', 'ceiling', 'Min', 'Max', 'Piecewise'}

        # ===================================================================
        # 3. Static Parameter Specialization (Constructing a clean subs_map)
        # ===================================================================
        with self._timer("Step 3: Static Parameter Specialization (Constructing a clean subs_map)"):
            subs_map = {}
            expr_symbols = set()
            for e in exprs_list: expr_symbols.update(e.atoms(sympy.Symbol))
            name_to_sym = {s.name: s for s in expr_symbols}
            all_atoms = set()
            all_atoms.update(expr_symbols)

            if 'param_map' in op_def:
                for math_name in op_def['param_map'].values():
                    targets = math_name if isinstance(math_name, (list, tuple)) else [math_name]
                    for mn in targets:
                        if isinstance(mn, str):
                            if mn in name_to_sym:
                                all_atoms.add(name_to_sym[mn])
                            else:
                                all_atoms.add(sympy.Symbol(mn))

            derived_names = set(op_def.get('derived_variables', {}).keys())

            for sym in all_atoms:
                if not isinstance(sym, sympy.Symbol): continue
                name = sym.name
                if name in derived_names or sym in subs_map: continue
                if name in blacklist or "'" in name or '"' in name:
                    continue
                try:
                    val, category = self._resolve_symbol(sym, res_ctx)
                    if category == 'static' and val is not None:
                        actual_val = val.item() if hasattr(val, 'item') and not isinstance(val, (
                            list, tuple, np.ndarray)) else val
                        if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 1: actual_val = val[0]
                        if isinstance(actual_val, str):
                            actual_val = SymStr(actual_val)
                        subs_map[sym] = actual_val
                except Exception:
                    pass

            # Materialize scalar derived variables early so reduction bounds such
            # as S_kv_ext - 1 can be recognized as static integers later.
            for derived_name in sorted(derived_names):
                derived_sym = name_to_sym.get(derived_name, sympy.Symbol(derived_name))
                if derived_sym in subs_map:
                    continue
                try:
                    val, category = self._resolve_symbol(derived_sym, res_ctx)
                    if category != 'static' or val is None:
                        continue
                    if self._is_tensor(val) or isinstance(val, np.ndarray):
                        continue
                    actual_val = val.item() if hasattr(val, 'item') and not isinstance(val, (list, tuple)) else val
                    if isinstance(actual_val, sympy.Basic) and getattr(actual_val, 'free_symbols', None):
                        continue
                    subs_map[derived_sym] = actual_val
                except Exception:
                    pass
        print("\n" + "=" * 50)
        print("[PROBE 1] Step 3 Output (subs_map):")
        print(f"  PyTorch GT layer.bias: {getattr(op_instance, 'bias', None)}")
        for k, v in subs_map.items():
            if hasattr(k, 'name') and k.name in ['bias', 'B']:
                val_repr = str(v) if not hasattr(v, 'shape') else f"Tensor{v.shape} = {v}"
                print(f"  > Symbol '{k.name}' -> {type(v)}: {val_repr}")
        print("=" * 50)
        # =========================================================
        # Step 3.5. Mechanism Guard
        # =========================================================
        with self._timer("Step 3.5: Mechanism Guard"):
            self._check_mechanisms_lazy(op_def, subs_map, res_ctx, tensor_names)

        logic_context_base = {}
        for sym, val in subs_map.items():
            if not isinstance(val, SymStr): logic_context_base[sym] = val

        def force_evaluate_logic(node, shape_context=None, memo=None):
            if shape_context is None: shape_context = {}
            if memo is None: memo = {}

            if node in memo: return memo[node]
            if isinstance(node, sympy.Symbol):
                context_hit = False
                context_val = None
                if node in shape_context:
                    context_hit = True
                    context_val = shape_context[node]
                elif node.name in shape_context:
                    context_hit = True
                    context_val = shape_context[node.name]

                if context_hit:
                    if self._is_none_symbol(context_val):
                        return sympy.false
                    if isinstance(context_val, (bool, np.bool_)):
                        return sympy.true if bool(context_val) else sympy.false
                    if isinstance(context_val, (int, float, np.integer, np.floating)):
                        return sympy.sympify(context_val)
                return node
            if isinstance(node, (int, float, str, sympy.Integer, sympy.Float)):
                return node

            result = node
            if self._is_none_symbol(result):
                return sympy.false

            if shape_context and hasattr(node, 'free_symbols'):
                free = node.free_symbols
                if free:
                    subs_subset = {s: shape_context[s] for s in free if s in shape_context}
                    if subs_subset:
                        try:
                            result = result.subs(subs_subset)
                        except:
                            pass

            if isinstance(result, sympy.Equality):
                lhs, rhs = result.args
                if not lhs.free_symbols and not rhs.free_symbols:
                    l_val = lhs.name if hasattr(lhs, 'name') else str(lhs).strip("'\"")
                    r_val = rhs.name if hasattr(rhs, 'name') else str(rhs).strip("'\"")
                    ans = sympy.true if l_val == r_val else sympy.false
                    memo[node] = ans
                    return ans

            elif isinstance(result, sympy.Piecewise):
                new_args = []
                for val, cond in result.args:
                    new_cond = force_evaluate_logic(cond, shape_context, memo)
                    if new_cond == sympy.true:
                        new_val = force_evaluate_logic(val, shape_context, memo)
                        new_args.append((new_val, sympy.true))
                        break
                    elif new_cond == sympy.false:
                        continue
                    else:
                        new_val = force_evaluate_logic(val, shape_context, memo)
                        new_args.append((new_val, new_cond))
                if not new_args:
                    ans = sympy.Integer(0)
                elif len(new_args) == 1 and new_args[0][1] == sympy.true:
                    ans = new_args[0][0]
                else:
                    has_true = any(c == sympy.true for v, c in new_args)
                    if not has_true and new_args:
                        new_args.append((sympy.Integer(0), sympy.true))
                    ans = sympy.Piecewise(*new_args)
                memo[node] = ans
                return ans

            elif isinstance(result, (sympy.And, sympy.Or, sympy.Not)):
                new_args = [force_evaluate_logic(arg, shape_context, memo) for arg in result.args]
                ans = result.func(*new_args)
                memo[node] = ans
                return ans

            elif hasattr(result, 'args') and result.args:
                if getattr(result, 'is_Add', False) or getattr(result, 'is_Mul', False):
                    try:
                        ans = result.func(*[force_evaluate_logic(arg, shape_context, memo) for arg in result.args],
                                          evaluate=False)
                    except:
                        ans = result.func(*[force_evaluate_logic(arg, shape_context, memo) for arg in result.args])
                elif isinstance(result, (sympy.Sum, sympy.Product)):
                    new_func = force_evaluate_logic(result.function, shape_context, memo)
                    ans = result.func(new_func, *result.limits)
                else:
                    new_args = [force_evaluate_logic(arg, shape_context, memo) for arg in result.args]
                    try:
                        ans = result.func(*new_args)
                    except:
                        ans = result
                memo[node] = ans
                return ans

            memo[node] = result
            return result

        # =======================================================================
        # 3.6: Traversal and Independent Compilation of Multi-Output Expressions
        # =======================================================================
        final_results = []

        with self._timer("Step 3.6: Traversal and Independent Compilation of Multi-Output Expressions"):
            for out_idx, raw_expr in enumerate(exprs_list):
                computation_expr = raw_expr
                lhs_indices_hint = []

                # =========================================================
                # 3.7. Output Shape Inference
                # =========================================================
                with self._timer(f"Step 3.7: Output Shape Inference"):
                    shape_infer_ctx = {
                        'op_def': op_def, 'math_prog': math_prog, 'subs_map': subs_map,
                        'runtime_data': {},
                        'input_sym_map': {},
                        'inputs': inputs, 'res_ctx': res_ctx, 'symnet': symnet, 'op_instance': op_instance
                    }
                    out_shape = self._infer_output_shape(shape_infer_ctx, output_index=out_idx)

                    output_shape_limit = extra_kwargs.get('_semtrans_output_shape_limit')
                    if output_shape_limit is not None:
                        limit = tuple(int(size) for size in output_shape_limit)
                        if len(limit) != len(out_shape):
                            raise ValueError(
                                "SemTrans output probe rank does not match inferred output rank"
                            )
                        out_shape = [
                            min(int(actual), requested)
                            for actual, requested in zip(out_shape, limit)
                        ]

                    res_ctx['out_shape'] = out_shape
                    res_ctx['subs_map'] = subs_map

                # =========================================================
                # 4. Symbol Substitution and Pre-parsing
                # =========================================================
                with self._timer("Step 4: Symbol Substitution and Pre-parsing"):
                    def clean_value(val):
                        if callable(val):
                            return None

                        if isinstance(val, str):
                            try:
                                return SymStr(val)
                            except NameError:
                                return sympy.Symbol(val)

                        if isinstance(val, (list, tuple)):
                            cleaned = [clean_value(x) for x in val]
                            return tuple(x for x in cleaned if x is not None)
                        if isinstance(val, dict):
                            return val
                        return val

                    clean_subs_map = {}
                    for k, v in subs_map.items():
                        cv = clean_value(v)
                        if cv is not None: clean_subs_map[k] = cv

                    subs_map = clean_subs_map
                    specialized_expr = self._resolve_shape_macros(computation_expr, res_ctx)
                    name_subs = {
                        sympy.Symbol(s.name): v
                        for s, v in subs_map.items()
                        if hasattr(s, 'name') and v is not None and not self._is_none_symbol(v) and not isinstance(v, dict)
                    }
                    specialized_expr = specialized_expr.subs(name_subs).doit()
                    specialized_expr = self._simplify_indexed_conditionals(specialized_expr, subs_map)

                    def _simplify_is_list(node):
                        if hasattr(node, 'func') and node.func.__name__ == 'is_list':
                            arg = node.args[0]
                            if isinstance(arg, (list, tuple, sympy.Tuple)): return sympy.true
                            if type(arg).__name__ == 'SymStr': return sympy.false
                            if isinstance(arg, (int, float, str)): return sympy.false
                        return node

                    specialized_expr = specialized_expr.replace(
                        lambda x: isinstance(x, sympy.Function) and x.func.__name__ == 'is_list',
                        _simplify_is_list
                    )

                    # Keep an unoptimized expression for benchmark reporting.
                    naive_unoptimized_expr = force_evaluate_logic(specialized_expr, shape_context=logic_context_base)
                    naive_functional_expr = naive_unoptimized_expr.replace(
                        sympy.Indexed, lambda *args: sympy.Function('GetItem')(*args)
                    )

                    unique_naive_syms = {s.name: s for s in naive_functional_expr.free_symbols}
                    reduction_symbols_naive = set()
                    for node in naive_functional_expr.atoms(sympy.Sum) | naive_functional_expr.atoms(sympy.Product):
                        for limit in node.limits:
                            if isinstance(limit[0], sympy.Symbol): reduction_symbols_naive.add(limit[0].name)

                    naive_final_syms = sorted(
                        [
                            s for name, s in unique_naive_syms.items()
                            if name not in reduction_symbols_naive and not isinstance(s, SymStr)
                        ],
                        key=lambda s: s.name)
                    # ===================================================================

                    specialized_expr = self._apply_algebraic_lowering(specialized_expr)

                    logic_context = logic_context_base.copy()

                    for sym in specialized_expr.atoms(sympy.Symbol):
                        if sym.name.startswith("DIM_"):
                            try:
                                parts = sym.name.split("_")
                                t_name, d_idx = parts[1], int(parts[2])
                                t_sym = sympy.Symbol(t_name)
                                val, stype = self._resolve_symbol(t_sym, res_ctx)
                                real_shape_val = None
                                if stype == 'dynamic':
                                    res_dict = symnet.to_numpy(val)
                                    if res_dict:
                                        arr = next(iter(res_dict.values()))
                                        if hasattr(arr, 'shape'): real_shape_val = arr.shape[d_idx]
                                elif stype == 'static':
                                    if hasattr(val, 'shape'):
                                        real_shape_val = val.shape[d_idx]
                                    elif isinstance(val, (list, tuple, np.ndarray)):
                                        try:
                                            if d_idx < len(val):
                                                real_shape_val = val[d_idx]
                                        except Exception:
                                            pass
                                if real_shape_val is not None: logic_context[sym] = int(real_shape_val)
                            except:
                                pass
                        elif sym.name not in index_names and sym.name not in tensor_names:
                            try:
                                val, category = self._resolve_symbol(sym, res_ctx)
                                if category == 'static':
                                    actual_val = val.item() if hasattr(val, 'item') else val
                                    if isinstance(actual_val, (int, float)):
                                        logic_context[sym] = actual_val
                            except:
                                pass

                # =========================================================
                # 5. Pruning & Expansion
                # =========================================================
                with self._timer("Step 5: Pruning & Expansion"):
                    eval_memo = {}
                    specialized_expr = force_evaluate_logic(specialized_expr, shape_context=logic_context,
                                                            memo=eval_memo)
                    specialized_expr = specialized_expr.doit()
                    pre_unroll_cost = self._estimate_expr_cost(specialized_expr)
                    unroll_memo = {}
                    specialized_expr = self._unroll_reductions(
                        specialized_expr,
                        subs_map,
                        memo=unroll_memo,
                        op_def=op_def,
                        res_ctx=res_ctx
                    )
                    eval_memo.clear()
                    specialized_expr = force_evaluate_logic(specialized_expr, shape_context=logic_context,
                                                            memo=eval_memo)
                    symbol_bounds = self._build_symbol_bounds(binding={}, out_shape=out_shape, subs_map=subs_map)
                    bound_memo = {}
                    specialized_expr = self._simplify_by_bounds(specialized_expr, symbol_bounds, memo=bound_memo)
                    reduction_memo = {}
                    specialized_expr = self._optimize_reduction_structure(specialized_expr, memo=reduction_memo)
                    piecewise_memo = {}
                    specialized_expr = self._merge_piecewise_masks(specialized_expr, memo=piecewise_memo)
                    pre_factor_cost = self._estimate_expr_cost(specialized_expr)
                    factor_memo = {}
                    specialized_expr = self._factor_common_terms(specialized_expr, memo=factor_memo)
                    post_factor_cost = self._estimate_expr_cost(specialized_expr)

                    print("\n[PROBE 2] Step 5 Output (specialized_expr):")
                    expr_str = str(specialized_expr)
                    print(f"  > Cost before reduction policy: {pre_unroll_cost}")
                    print(f"  > Cost before factorization: {pre_factor_cost}")
                    print(f"  > Cost after factorization: {post_factor_cost}")
                    if 'bias' in expr_str or 'B' in expr_str:
                        print(f"  > Formula contains Bias-related terms!")
                        print(f"  > Snippet: {expr_str[-150:]}")
                    else:
                        print(f"  > Formula has NO Bias-related terms!")

                    if isinstance(specialized_expr, sympy.Equality):
                        curr_lhs = specialized_expr.lhs
                        lhs_str = str(curr_lhs)
                        if 'Y' in lhs_str or 'Output' in lhs_str:
                            if isinstance(curr_lhs, sympy.Indexed):
                                lhs_indices_hint.clear()
                                for dim_idx, idx_node in enumerate(curr_lhs.indices):
                                    if not isinstance(idx_node, (sympy.Symbol, sympy.Integer)) and not (
                                            hasattr(idx_node, 'is_Number') and idx_node.is_Number):
                                        raise ValueError(
                                            f"[Compiler Rule Violation] Invalid LHS index '{idx_node}' in '{curr_lhs}'. "
                                        )
                                    lhs_indices_hint.append(str(idx_node))
                            specialized_expr = specialized_expr.rhs

                # =========================================================
                # 6. JIT Kernel Compilation (Per Output)
                # =========================================================
                with self._timer("Step 6: JIT Kernel Compilation"):
                    # ============================================================================
                    # AST Normalization & Printer Override Pass:
                    # Convert native SymPy objects into generic undefined SymPy Functions.
                    # This prevents SymPy's default code generators (e.g., NumPyPrinter) from
                    # applying hardcoded, unsafe translations (like translating sympy.Max into
                    # numpy.amax, which flattens multi-dimensional tensors).
                    # By forcing the printer to output standard IR names ("GetItem", "Max", "Min"),
                    # we ensure these operations are strictly routed to our custom, broadcast-safe
                    # backend implementations.

                    functional_expr = specialized_expr.replace(
                        sympy.Indexed, lambda *args: sympy.Function('GetItem')(*args)
                    )
                    functional_expr = functional_expr.replace(
                        sympy.Max, lambda *args: sympy.Function('Max')(*args)
                    )
                    functional_expr = functional_expr.replace(
                        sympy.Min, lambda *args: sympy.Function('Min')(*args)
                    )
                    functional_expr = _lower_control_flow_for_numpy_backend(functional_expr)
                    indexed_runtime_symbols = {
                        node.args[0].name
                        for node in functional_expr.atoms(sympy.Function)
                        if (
                            node.func.__name__ == 'GetItem'
                            and node.args
                            and isinstance(node.args[0], sympy.Symbol)
                        )
                    }
                    res_ctx['indexed_runtime_symbols'] = tuple(sorted(indexed_runtime_symbols))
                    # ============================================================================

                    unique_syms_map = {}
                    for s in functional_expr.free_symbols:
                        unique_syms_map[s.name] = s

                    reduction_key = sympy.Symbol('_reduction_map')
                    active_reduction_map = subs_map.get(reduction_key, {})

                    reduction_symbols = set()
                    for node in specialized_expr.atoms(sympy.Sum) | specialized_expr.atoms(sympy.Product):
                        for limit in node.limits:
                            if isinstance(limit[0], sympy.Symbol):
                                s_name = limit[0].name
                                if s_name not in active_reduction_map:
                                    reduction_symbols.add(s_name)

                    for node in specialized_expr.atoms(sympy.Function):
                        if node.func.__name__ in ['TensorMax', 'TensorMin', 'Max', 'Min', 'ArgMax', 'ArgMin']:
                            for arg in node.args[1:]:
                                if isinstance(arg, (tuple, sympy.Tuple)) and len(arg) >= 1:
                                    if isinstance(arg[0], sympy.Symbol):
                                        s_name = arg[0].name
                                        if s_name not in active_reduction_map:
                                            reduction_symbols.add(s_name)

                    final_syms_list = []
                    for name, s in unique_syms_map.items():
                        if isinstance(s, SymStr):
                            continue
                        if name in active_reduction_map:
                            final_syms_list.append(s)
                        elif name not in reduction_symbols:
                            final_syms_list.append(s)

                    final_syms = sorted(final_syms_list, key=lambda s: s.name)

                    out_rank_local = len(out_shape)
                    try:
                        backend_ops = NumpyBackend.get_ops(context={'out_rank': out_rank_local})
                    except TypeError:
                        backend_ops = NumpyBackend.get_ops()
                    except NameError:
                        raise RuntimeError("Backend not found!")

                    def robust_get_item(obj, *indices):
                        if isinstance(obj, (int, float, np.number)): return obj
                        if not isinstance(obj, np.ndarray):
                            if hasattr(obj, 'numpy'):
                                obj = obj.numpy()
                            else:
                                obj = np.array(obj)
                        if obj.ndim == 0: return obj.item()

                        try:
                            clean_indices = []
                            has_array_index = False
                            for i, x in enumerate(indices):
                                dim_limit = obj.shape[i] if i < len(obj.shape) else 1
                                if isinstance(x, np.ndarray):
                                    safe_x = np.clip(x.astype(np.intp), 0, dim_limit - 1)
                                    clean_indices.append(safe_x)
                                    has_array_index = True
                                else:
                                    val = int(x)
                                    safe_val = max(0, min(val, dim_limit - 1))
                                    clean_indices.append(safe_val)

                            if has_array_index:
                                index_arrays = [
                                    idx if isinstance(idx, np.ndarray) else np.asarray(idx, dtype=np.intp)
                                    for idx in clean_indices
                                ]
                                broadcasted_indices = np.broadcast_arrays(*index_arrays)
                                projected_shape = broadcasted_indices[0].shape
                                projected_items = int(np.prod(projected_shape, dtype=np.int64))
                                if projected_items > 100_000_000:
                                    raise MemoryError(
                                        f"Projected GetItem broadcast too large: shape={projected_shape}, "
                                        f"items={projected_items}"
                                    )

                                flat_indices = np.ravel_multi_index(
                                    tuple(arr.reshape(-1) for arr in broadcasted_indices),
                                    obj.shape[:len(clean_indices)],
                                    mode='clip'
                                )
                                indexed = obj[tuple(broadcasted_indices)]
                                return np.asarray(indexed).reshape(projected_shape + obj.shape[len(clean_indices):])

                            return obj[tuple(clean_indices)]
                        except Exception as e:
                            print(f"\n[ GetItem Crash! ] Obj Shape: {obj.shape} | Indices: {indices}")
                            raise e

                    modules_list = [{
                        'GetItem': robust_get_item,
                        '__ST_Piecewise': backend_ops['Piecewise'],
                        '__ST_And': backend_ops['And'],
                        '__ST_Or': backend_ops['Or'],
                        '__ST_Not': backend_ops['Not'],
                    }, backend_ops, 'numpy', 'math']

                    def _print_NoneSymbol(self, expr):
                        return "None"

                    if not hasattr(NumPyPrinter, '_print_NoneSymbol'):
                        NumPyPrinter._print_NoneSymbol = _print_NoneSymbol
                    if not hasattr(PythonCodePrinter, '_print_NoneSymbol'):
                        PythonCodePrinter._print_NoneSymbol = _print_NoneSymbol

                    has_dynamic_loop = len(specialized_expr.atoms(sympy.Sum, sympy.Product)) > 0

                    try:
                        safe_cse = not has_dynamic_loop
                        kernel_func = sympy.lambdify(
                            final_syms,
                            functional_expr,
                            modules=modules_list,
                            cse=safe_cse
                        )
                    except TypeError:
                        print("[Warn] SymPy outdated, cse=True failed. Update SymPy for speedup.")
                        kernel_func = sympy.lambdify(
                            final_syms,
                            functional_expr,
                            modules=modules_list
                        )
                    except Exception as e:
                        print(f"[Error] Lambdify failed: {e}")
                        raise e

                # =========================================================
                # 7. Runtime Data Preparation
                # =========================================================
                with self._timer("Step 7: Runtime Data Preparation"):
                    res_ctx_lite = {
                        'input_names': res_ctx.get('input_names', []),
                        'op_def': op_def,
                        'indexed_runtime_symbols': res_ctx.get('indexed_runtime_symbols', ()),
                    }

                    if cache_key:
                        if cache_key not in self._compiled_kernel_cache:
                            self._compiled_kernel_cache[cache_key] = []

                        if len(self._compiled_kernel_cache[cache_key]) <= out_idx:
                            self._compiled_kernel_cache[cache_key].append(
                                (kernel_func, final_syms, {}, out_shape, res_ctx_lite)
                            )

                    runtime_data_map = {}
                    input_names = res_ctx.get('input_names', [])
                    input_name_to_idx = {name: i for i, name in enumerate(input_names)}

                    for sym in final_syms:
                        name = sym.name
                        if name.startswith("DIM_") or name.startswith("RANK_"): continue

                        val = None
                        target_idx = -1

                        if name in input_name_to_idx:
                            target_idx = input_name_to_idx[name]
                        elif name == 'X' and len(inputs) > 0:
                            target_idx = 0
                        elif name.startswith('X') and name[1:].isdigit():
                            try:
                                idx = int(name[1:]) - 1
                                if 0 <= idx < len(inputs): target_idx = idx
                            except:
                                pass

                        if target_idx >= 0:
                            raw_val = inputs[target_idx]
                            if not isinstance(raw_val, np.ndarray):
                                res = symnet.to_numpy(raw_val)
                                raw_val = next(iter(res.values())) if isinstance(res, dict) and res else raw_val

                            if name in ['Indices', 'A'] or (
                                    hasattr(raw_val, 'dtype') and np.issubdtype(raw_val.dtype, np.integer)):
                                val = raw_val
                            else:
                                val = raw_val.astype(np.float32) if hasattr(raw_val, 'astype') else raw_val

                            runtime_data_map[name] = val
                            continue

                        if val is None:
                            try:
                                resolved, stype = self._resolve_symbol(sym, res_ctx)
                                if resolved is not None:
                                    if stype == 'static':
                                        val = resolved[0] if isinstance(resolved, (tuple, list)) else resolved
                                    elif stype == 'dynamic':
                                        if isinstance(resolved, np.ndarray):
                                            val = resolved
                                        else:
                                            res = symnet.to_numpy(resolved)
                                            val = next(iter(res.values())) if res else resolved
                            except Exception:
                                pass

                        if val is not None:
                            if not isinstance(val, np.ndarray) and hasattr(val, 'numpy'):
                                if hasattr(val, 'detach'):
                                    val = val.detach()
                                val = val.numpy()
                            runtime_data_map[name] = val

                        elif name == 'B' or name == 'bias':
                            runtime_data_map[name] = 0.0

                    for sym in final_syms:
                        if sym.name.startswith("DIM_"):
                            try:
                                parts = sym.name.split("_")
                                d_idx = int(parts[-1])
                                t_name = "_".join(parts[1:-1])

                                if t_name in runtime_data_map:
                                    obj = runtime_data_map[t_name]
                                    if hasattr(obj, 'shape'):
                                        runtime_data_map[sym.name] = obj.shape[d_idx]
                            except:
                                pass

                print("\n[PROBE 3] Step 7 Output (runtime_data_map):")
                for k, v in runtime_data_map.items():
                    if k in ['B', 'bias']:
                        val_repr = str(v) if not hasattr(v, 'shape') else f"Array{v.shape} = {v}"
                        print(f"  > Data '{k}' -> {type(v)}: {val_repr}")
                # =========================================================
                # 8. Dynamic Dimension Binding (Using LHS Hint)
                # =========================================================
                with self._timer(f"Step 8: Dynamic Dimension Binding"):
                    if lhs_indices_hint and len(lhs_indices_hint) != len(out_shape):
                        if (
                            len(lhs_indices_hint) + 1 == len(out_shape) and
                            lhs_indices_hint == ['i']
                        ):
                            lhs_indices_hint = ['n', 'i']
                        else:
                            raise ValueError(
                                f"[Compiler Rule Violation] LHS Rank Mismatch! "
                                f"Equation defines {len(lhs_indices_hint)} indices ({lhs_indices_hint}), "
                                f"but physical output shape {out_shape} has Rank {len(out_shape)}. "
                                f"They must be exactly equal."
                            )

                    reduction_key = sympy.Symbol('_reduction_map')
                    active_reduction_map = subs_map.get(reduction_key, {})
                    known_reduction_vars = set(active_reduction_map.keys()) | reduction_symbols

                    binding = self._bind_dynamic_dimensions(
                        final_syms,
                        out_shape,
                        lhs_indices_hint,
                        res_ctx,
                        ignore_syms=known_reduction_vars
                    )

                    post_bind_bounds = self._build_symbol_bounds(binding=binding, out_shape=out_shape, subs_map=subs_map)
                    post_bind_memo = {}
                    specialized_expr = self._simplify_by_bounds(
                        specialized_expr,
                        post_bind_bounds,
                        memo=post_bind_memo
                    )

                    formula_diagnostic = self._formula_diagnostic(
                        op_instance, op_def, specialized_expr, subs_map, symnet
                    )
                    if cache_key:
                        self._compiled_kernel_cache[cache_key][-1] = (
                            kernel_func, final_syms, binding, out_shape, res_ctx_lite,
                            functional_expr, naive_final_syms, naive_functional_expr,
                            runtime_data_map, active_reduction_map, formula_diagnostic
                        )
                # =========================================================
                # Jit Compile Record
                # =========================================================
                self.last_jit_context = {
                    'final_syms': final_syms,
                    'functional_expr': functional_expr,
                    'naive_final_syms': naive_final_syms,
                    'naive_functional_expr': naive_functional_expr,
                    'binding': binding,
                    'out_shape': out_shape,
                    'runtime_data_map': runtime_data_map,
                    'reduction_map': active_reduction_map,
                    'formula_diagnostic': formula_diagnostic,
                }
                self.last_jit_contexts.append(self.last_jit_context)

                print(f"DEBUG: Functional Expression Body: {functional_expr}")
                print(f"DEBUG: Final Symbols for JIT: {[s.name for s in final_syms]}")
                # =========================================================
                # 9. Generic Multi-dimensional Loop Simulation (Execution)
                # =========================================================
                with self._timer(f"Step 9: Generic Multi-dimensional Loop Simulation"):
                    reduction_key = sympy.Symbol('_reduction_map')
                    active_reduction_map = subs_map.get(reduction_key, {})
                    has_dynamic_reduction = any(
                        isinstance(spec, dict) and spec.get('axis_symbol')
                        for spec in active_reduction_map.values()
                    )

                    if has_dynamic_reduction:
                        exec_res_ctx = dict(res_ctx_lite)
                        exec_res_ctx['reduction_map'] = active_reduction_map
                        output_np = self._execute_compiled_kernel(
                            kernel_func, final_syms, binding, out_shape, inputs, op_instance, symnet, exec_res_ctx
                        )
                    else:
                        reduction_vars = []
                        reduction_to_dim = {}
                        axis_symbol_bindings = {}
                        try:
                            def _normalize_reduction_spec(spec):
                                if isinstance(spec, dict):
                                    axis = spec.get('axis')
                                    start = int(spec.get('start', 0))
                                    axis_symbol = spec.get('axis_symbol')
                                    if spec.get('size') is not None:
                                        size = int(spec['size'])
                                        end = int(spec.get('end', start + size - 1))
                                    else:
                                        raw_end = spec.get('end', start)
                                        end = start if raw_end is None else int(raw_end)
                                        size = max(0, end - start + 1)
                                    return {'axis': axis, 'axis_symbol': axis_symbol, 'start': start, 'end': end, 'size': size}

                                axis = None if spec is None else int(spec)
                                return {'axis': axis, 'axis_symbol': None, 'start': 0, 'end': None, 'size': None}

                            normalized_reduction_map = {
                                sname: _normalize_reduction_spec(spec)
                                for sname, spec in active_reduction_map.items()
                            }

                            missing_reduction_vars = []
                            for s in final_syms:
                                sname = s.name
                                if sname not in runtime_data_map and sname not in binding and sname not in ['B', 'bias']:
                                    if sname.startswith('REDUCE_AXIS_'):
                                        continue
                                    if sname not in normalized_reduction_map:
                                        missing_reduction_vars.append(sname)

                            if missing_reduction_vars:
                                default_axis = len(inputs[0].shape) - 1
                                for sname in missing_reduction_vars:
                                    dim_size = inputs[0].shape[default_axis]
                                    normalized_reduction_map[sname] = {
                                        'axis': default_axis,
                                        'start': 0,
                                        'end': dim_size - 1,
                                        'size': dim_size
                                    }

                            out_rank = len(out_shape)
                            reduction_vars = list(normalized_reduction_map.keys())
                            sim_rank = out_rank + len(reduction_vars)

                            full_sim_shape = list(out_shape)
                            reduction_to_dim = {}
                            for i, sname in enumerate(reduction_vars):
                                spec = normalized_reduction_map[sname]
                                dim_size = spec['size']
                                if dim_size is None:
                                    target_axis_in_input = spec['axis']
                                    if target_axis_in_input is None:
                                        raise ValueError(f"Reduction variable '{sname}' is missing axis/size metadata.")
                                    dim_size = inputs[0].shape[target_axis_in_input]
                                    spec['start'] = 0
                                    spec['end'] = dim_size - 1
                                    spec['size'] = dim_size
                                full_sim_shape.append(dim_size)
                                reduction_to_dim[sname] = {
                                    'sim_dim': out_rank + i,
                                    'start': spec['start'],
                                    'size': spec['size']
                                }
                                axis_symbol = spec.get('axis_symbol')
                                if axis_symbol:
                                    axis_symbol_bindings[axis_symbol] = out_rank + i

                            projected_items = int(np.prod(full_sim_shape, dtype=np.int64))
                            if projected_items > 100_000_000:
                                raise MemoryError(
                                    f"Projected vectorized simulation too large: shape={tuple(full_sim_shape)}, "
                                    f"items={projected_items}"
                                )

                            vectorized_args = []
                            for sym in final_syms:
                                sname = sym.name

                                if sname in runtime_data_map:
                                    vectorized_args.append(
                                        self._align_runtime_arg_for_output(
                                            sname,
                                            runtime_data_map[sname],
                                            out_shape,
                                            sim_rank,
                                            input_names,
                                            res_ctx
                                        )
                                    )

                                elif sname in axis_symbol_bindings:
                                    vectorized_args.append(np.int32(axis_symbol_bindings[sname]))

                                elif sname in binding:
                                    dim_idx = binding[sname]
                                    dim_size = out_shape[dim_idx]

                                    reshape_shape = [1] * sim_rank
                                    reshape_shape[dim_idx] = dim_size
                                    arg_val = np.arange(dim_size, dtype=np.int32).reshape(reshape_shape)
                                    vectorized_args.append(arg_val)

                                elif sname in reduction_to_dim:
                                    target_dim = reduction_to_dim[sname]['sim_dim']
                                    dim_size = full_sim_shape[target_dim]
                                    start = reduction_to_dim[sname]['start']

                                    reshape_shape = [1] * sim_rank
                                    reshape_shape[target_dim] = dim_size
                                    arg_val = (start + np.arange(dim_size, dtype=np.int32)).reshape(reshape_shape)
                                    vectorized_args.append(arg_val)

                                else:
                                    vectorized_args.append(0)

                            output_raw = kernel_func(*vectorized_args)
                            output_np = np.asarray(output_raw)

                            if output_np.shape != tuple(out_shape):
                                try:
                                    if output_np.size == np.prod(out_shape):
                                        output_np = output_np.reshape(tuple(out_shape))
                                    else:
                                        output_np = np.broadcast_to(output_np, tuple(out_shape))
                                except:
                                    raise ValueError(f"Shape mismatch: {output_np.shape} vs {out_shape}")

                        except Exception as e:
                            print(f"[Warn] Vectorized execution failed: {e}. Falling back to slow loop.")
                            output_np = np.zeros(out_shape, dtype=np.float64)
                            reduction_grids = {}
                            reduce_rank = len(reduction_vars)
                            for ridx, sname in enumerate(reduction_vars):
                                spec = reduction_to_dim[sname]
                                reshape_shape = [1] * reduce_rank
                                reshape_shape[ridx] = spec['size']
                                reduction_grids[sname] = (
                                    spec['start'] + np.arange(spec['size'], dtype=np.int32)
                                ).reshape(reshape_shape)

                            chunked_done = False
                            if out_shape:
                                try:
                                    chunk_axis = int(np.argmax(out_shape))
                                    fixed_volume = 1
                                    for dim_idx, dim_size in enumerate(out_shape):
                                        if dim_idx != chunk_axis:
                                            fixed_volume *= max(1, dim_size)
                                    for spec in reduction_to_dim.values():
                                        fixed_volume *= max(1, spec['size'])

                                    target_items = 20_000_000
                                    chunk_size = max(1, min(out_shape[chunk_axis], target_items // max(1, fixed_volume)))

                                    if chunk_size > 0:
                                        for start_idx in range(0, out_shape[chunk_axis], chunk_size):
                                            stop_idx = min(out_shape[chunk_axis], start_idx + chunk_size)
                                            block_out_shape = list(out_shape)
                                            block_out_shape[chunk_axis] = stop_idx - start_idx
                                            block_sim_rank = len(block_out_shape) + reduce_rank

                                            block_args = []
                                            for sym in final_syms:
                                                sname = sym.name
                                                if sname in runtime_data_map:
                                                    block_args.append(
                                                        self._align_runtime_arg_for_output(
                                                            sname,
                                                            runtime_data_map[sname],
                                                            block_out_shape,
                                                            block_sim_rank,
                                                            input_names,
                                                            res_ctx
                                                        )
                                                    )
                                                elif sname in axis_symbol_bindings:
                                                    block_args.append(np.int32(axis_symbol_bindings[sname]))
                                                elif sname in binding:
                                                    dim_idx = binding[sname]
                                                    dim_size = block_out_shape[dim_idx]
                                                    reshape_shape = [1] * block_sim_rank
                                                    reshape_shape[dim_idx] = dim_size
                                                    base_idx = start_idx if dim_idx == chunk_axis else 0
                                                    block_args.append(
                                                        (base_idx + np.arange(dim_size, dtype=np.int32)).reshape(reshape_shape)
                                                    )
                                                elif sname in reduction_to_dim:
                                                    spec = reduction_to_dim[sname]
                                                    reshape_shape = [1] * block_sim_rank
                                                    reshape_shape[len(block_out_shape) + reduction_vars.index(sname)] = spec['size']
                                                    block_args.append(
                                                        (spec['start'] + np.arange(spec['size'], dtype=np.int32)).reshape(
                                                            reshape_shape
                                                        )
                                                    )
                                                else:
                                                    block_args.append(0)

                                            block_raw = kernel_func(*block_args)
                                            block_np = np.asarray(block_raw, dtype=np.float64)
                                            if block_np.shape != tuple(block_out_shape):
                                                if block_np.size == np.prod(block_out_shape):
                                                    block_np = block_np.reshape(tuple(block_out_shape))
                                                else:
                                                    block_np = np.broadcast_to(block_np, tuple(block_out_shape))

                                            block_slice = [slice(None)] * len(out_shape)
                                            block_slice[chunk_axis] = slice(start_idx, stop_idx)
                                            output_np[tuple(block_slice)] = block_np

                                        chunked_done = True
                                except Exception as chunk_err:
                                    print(f"[Warn] Chunked fallback failed: {chunk_err}. Falling back to pointwise loop.")

                            if not chunked_done:
                                for multi_idx in np.ndindex(tuple(out_shape)):
                                    current_args = []
                                    for sym in final_syms:
                                        sname = sym.name
                                        if sname in runtime_data_map:
                                            current_args.append(runtime_data_map[sname])
                                        elif sname in axis_symbol_bindings:
                                            current_args.append(np.int32(axis_symbol_bindings[sname]))
                                        elif sname in binding:
                                            current_args.append(multi_idx[binding[sname]])
                                        elif sname in reduction_grids:
                                            current_args.append(reduction_grids[sname])
                                        else:
                                            current_args.append(0)
                                    res = kernel_func(*current_args)
                                    output_np[multi_idx] = float(np.asarray(res).reshape(-1)[0])

                final_results.append(output_np)
        attrs = getattr(op_instance, 'attrs', {})
        op_type = getattr(op_instance, 'op_type', "")

        if is_multi_output and attrs.get('return_state') is False:
            return final_results[0]

        return tuple(final_results) if is_multi_output else final_results[0]

    def _prepare_runtime_data_map(self, final_syms, inputs, op_instance, symnet, res_ctx):
        runtime_data_map = {}
        input_names = res_ctx.get('input_names', [])
        input_name_to_idx = {name: i for i, name in enumerate(input_names)}

        for sym in final_syms:
            name = sym.name
            if name.startswith("DIM_") or name.startswith("RANK_"): continue

            val = None

            target_idx = -1
            if name in input_name_to_idx:
                target_idx = input_name_to_idx[name]
            elif name == 'X' and len(inputs) > 0:
                target_idx = 0
            elif name.startswith('X') and name[1:].isdigit():
                try:
                    idx = int(name[1:]) - 1
                    if 0 <= idx < len(inputs): target_idx = idx
                except:
                    pass

            if target_idx >= 0:
                raw_val = inputs[target_idx]
                if not isinstance(raw_val, np.ndarray):
                    res = symnet.to_numpy(raw_val)
                    raw_val = next(iter(res.values())) if isinstance(res, dict) and res else raw_val

                if name in ['Indices', 'A'] or (hasattr(raw_val, 'dtype') and np.issubdtype(raw_val.dtype, np.integer)):
                    val = raw_val
                else:
                    val = raw_val.astype(np.float32) if hasattr(raw_val, 'astype') else raw_val
                runtime_data_map[name] = val
                continue

            if val is None:
                try:
                    resolved, stype = self._resolve_symbol(sym, res_ctx)
                    if resolved is not None:
                        if stype == 'static':
                            val = resolved[0] if isinstance(resolved, (tuple, list)) else resolved
                        elif stype == 'dynamic':
                            if isinstance(resolved, np.ndarray):
                                val = resolved
                            else:
                                res = symnet.to_numpy(resolved)
                                val = next(iter(res.values())) if res else resolved
                except:
                    pass

            if val is not None:
                if not isinstance(val, np.ndarray) and hasattr(val, 'numpy'):
                    val = val.numpy()
                runtime_data_map[name] = val
            elif name == 'B' or name == 'bias':
                runtime_data_map[name] = 0.0

        for sym in final_syms:
            if sym.name.startswith("DIM_"):
                try:
                    parts = sym.name.split("_")
                    d_idx = int(parts[-1])
                    t_name = "_".join(parts[1:-1])
                    if t_name in runtime_data_map:
                        obj = runtime_data_map[t_name]
                        if hasattr(obj, 'shape'):
                            runtime_data_map[sym.name] = obj.shape[d_idx]
                except:
                    pass
        return runtime_data_map

    def _build_vectorized_args(
        self,
        final_syms,
        binding,
        out_shape,
        runtime_data_map,
        reduction_map=None,
        input_names=None,
        res_ctx=None
    ):
        reduction_map = reduction_map or {}
        out_rank = len(out_shape)
        reduction_vars = list(reduction_map.keys())
        sim_rank = out_rank + len(reduction_vars)
        reduction_to_dim = {}
        axis_symbol_bindings = {}
        for i, sname in enumerate(reduction_vars):
            spec = reduction_map[sname]
            sim_dim = out_rank + i
            reduction_to_dim[sname] = {
                'sim_dim': sim_dim,
                'start': int(spec.get('start', 0)),
                'size': spec.get('size'),
                'axis': spec.get('axis'),
            }
            axis_symbol = spec.get('axis_symbol')
            if axis_symbol:
                axis_symbol_bindings[axis_symbol] = sim_dim

        vectorized_args = []
        for sym in final_syms:
            sname = sym.name
            if sname in runtime_data_map:
                arg_val = runtime_data_map[sname]
                arg_val = self._align_runtime_arg_for_output(
                    sname,
                    arg_val,
                    out_shape,
                    sim_rank,
                    input_names,
                    res_ctx
                )
                vectorized_args.append(arg_val)
            elif sname in axis_symbol_bindings:
                vectorized_args.append(int(axis_symbol_bindings[sname]))
            elif sname in binding:
                dim_idx = binding[sname]
                dim_size = out_shape[dim_idx]
                reshape_shape = [1] * sim_rank
                reshape_shape[dim_idx] = dim_size
                arg_val = np.arange(dim_size, dtype=np.int32).reshape(reshape_shape)
                vectorized_args.append(arg_val)
            elif sname in reduction_to_dim:
                spec = reduction_to_dim[sname]
                dim_size = spec['size']
                if dim_size is None:
                    axis = spec['axis']
                    if axis is None:
                        vectorized_args.append(0)
                        continue
                    dim_size = runtime_data_map['X'].shape[axis] if 'X' in runtime_data_map else out_shape[-1]
                reshape_shape = [1] * sim_rank
                reshape_shape[spec['sim_dim']] = dim_size
                start = spec['start']
                arg_val = (start + np.arange(dim_size, dtype=np.int32)).reshape(reshape_shape)
                vectorized_args.append(arg_val)
            else:
                vectorized_args.append(0)
        return vectorized_args

    def _materialize_kernel_output(self, output_raw, out_shape):
        output_np = np.asarray(output_raw, dtype=np.float32)
        if output_np.shape != tuple(out_shape):
            squeezed = np.squeeze(output_np)
            if np.size(squeezed) == np.prod(out_shape):
                output_np = np.reshape(squeezed, tuple(out_shape))
            else:
                output_np = np.broadcast_to(output_np, tuple(out_shape))
        return output_np

    def _execute_compiled_kernel(self, kernel_func, final_syms, binding, out_shape, inputs, op_instance, symnet,
                                 res_ctx):
        exec_res_ctx = dict(res_ctx or {})
        exec_res_ctx.update({
            'op_instance': op_instance,
            'inputs': inputs,
            'symnet': symnet,
        })
        runtime_data_map = self._prepare_runtime_data_map(final_syms, inputs, op_instance, symnet, exec_res_ctx)

        try:
            reduction_map = exec_res_ctx.get('reduction_map', {})
            vectorized_args = self._build_vectorized_args(
                final_syms,
                binding,
                out_shape,
                runtime_data_map,
                reduction_map,
                exec_res_ctx.get('input_names', []),
                exec_res_ctx
            )
            output_raw = kernel_func(*vectorized_args)
            return self._materialize_kernel_output(output_raw, out_shape)

        except Exception as e:
            raise RuntimeError(f"Cached MathIR kernel execution failed: {e}") from e

    def build_prepared_runner(self, math_prog, op_instance, inputs, symnet, **extra_kwargs):
        op_def = math_prog.get('op_def', {})
        op_type = getattr(op_instance, 'op_type', type(op_instance).__name__)
        cache_key = self._make_compilation_cache_key(op_def, op_instance, inputs, extra_kwargs)

        if cache_key not in self._compiled_kernel_cache:
            self.compile_and_run(math_prog, op_instance, inputs, symnet, **extra_kwargs)

        cached_results = self._compiled_kernel_cache.get(cache_key)
        if not cached_results:
            raise RuntimeError(f"No compiled kernel cache found for {op_type}")

        kernel_func, final_syms, binding, out_shape, res_ctx_lite = cached_results[0][:5]
        prepared_res_ctx = dict(res_ctx_lite)
        prepared_res_ctx.update({
            'op_instance': op_instance,
            'inputs': inputs,
            'symnet': symnet,
        })
        runtime_data_map = self._prepare_runtime_data_map(
            final_syms, inputs, op_instance, symnet, prepared_res_ctx
        )
        reduction_map = cached_results[0][9] if len(cached_results[0]) > 9 else {}
        vectorized_args = None
        axis_symbol_names = {
            spec.get('axis_symbol')
            for spec in reduction_map.values()
            if isinstance(spec, dict) and spec.get('axis_symbol')
        }
        unresolved_syms = [
            s.name for s in final_syms
            if s.name not in runtime_data_map
            and s.name not in binding
            and s.name not in reduction_map
            and s.name not in axis_symbol_names
            and s.name not in ['B', 'bias']
        ]
        if not unresolved_syms:
            res_ctx_lite = prepared_res_ctx
            vectorized_args = self._build_vectorized_args(
                final_syms,
                binding,
                out_shape,
                runtime_data_map,
                reduction_map,
                res_ctx_lite.get('input_names', []),
                res_ctx_lite
            )

        def runner():
            if vectorized_args is None:
                exec_res_ctx = dict(res_ctx_lite)
                exec_res_ctx['reduction_map'] = reduction_map
                return self._execute_compiled_kernel(
                    kernel_func, final_syms, binding, out_shape, inputs, op_instance, symnet, exec_res_ctx
                )
            output_raw = kernel_func(*vectorized_args)
            return self._materialize_kernel_output(output_raw, out_shape)

        return runner

    def _check_mechanisms_lazy(self, op_def, subs_map, res_ctx, candidate_tensors):
        mechanisms = op_def.get('mechanisms', [])
        if not mechanisms: return

        class DefaultZeroDict(dict):
            def __missing__(self, key):
                return 0

        op_instance = res_ctx.get('op_instance')
        module = getattr(op_instance, 'module', None) if op_instance else None
        if not module and hasattr(op_instance, 'op'):
            module = getattr(op_instance.op, 'module', op_instance.op)

        # -----------------------------------------------------
        # Internal Helpers
        # -----------------------------------------------------
        def sanitize(val):
            if val is S_None or type(val).__name__ == 'NoneSymbol':
                return None
            if type(val).__name__ == 'SymStr':
                return str(val)
            try:
                if isinstance(val, (np.generic, int, float, bool)):
                    return val.item() if hasattr(val, 'item') else val
            except:
                pass
            return val

        def is_valid_value(val):
            if isinstance(val, sympy.Symbol): return False
            if val is None: return False
            return True

        def inject(name, val):
            if val is None: return
            clean_val = sanitize(val)
            if is_valid_value(clean_val):
                eval_ctx[name] = clean_val
                is_tensor = hasattr(clean_val, 'shape') or hasattr(clean_val, 'data_ptr')
                if not is_tensor:
                    subs_map[sympy.Symbol(name)] = clean_val

        def dim_impl(obj, axis):
            axis = int(axis)
            if hasattr(obj, 'shape'):
                shape = obj.shape
                return shape[axis] if -len(shape) <= axis < len(shape) else 0
            if isinstance(obj, (list, tuple)):
                return obj[axis] if -len(obj) <= axis < len(obj) else 0
            if hasattr(obj, 'data') and hasattr(obj.data, 'shape'):
                shape = obj.data.shape
                return shape[axis] if -len(shape) <= axis < len(shape) else 0
            return 0

        def _get_item_impl(obj, *args):
            if not isinstance(obj, (list, tuple, np.ndarray)): return obj
            try:
                if len(args) == 1:
                    res = obj[int(args[0])]
                else:
                    indices = tuple(int(x) for x in args)
                    res = obj[indices]
                if hasattr(res, 'item'): return res.item()
                if isinstance(res, np.ndarray) and res.size == 1: return res.item()
                return res
            except Exception:
                return obj

        def _is_list_impl(obj):
            return isinstance(obj, (list, tuple, np.ndarray))

        def piecewise_impl(*args):
            for val, cond in args:
                is_true = cond
                if hasattr(cond, 'all'): is_true = cond.all()

                if is_true:
                    return val
            return 0

        def _join_impl(*args):
            res = ()
            for arg in args:
                if isinstance(arg, (list, tuple, np.ndarray)):
                    res = res + tuple(arg) if not (isinstance(arg, np.ndarray) and arg.ndim == 0) else res + (
                        arg.item(),)
                else:
                    res = res + (arg,)
            return res

        rank_impl = lambda x: (
            int(x.ndim)
            if hasattr(x, 'ndim')
            else (
                len(x.shape)
                if hasattr(x, 'shape')
                else (len(x) if isinstance(x, (list, tuple)) else 0)
            )
        )

        eval_ctx = {}
        eval_ctx.update({
            'abs': np.abs, 'Abs': np.abs,
            'min': np.minimum, 'Min': np.minimum,
            'max': np.maximum, 'Max': np.maximum,
            'floor': np.floor, 'ceiling': np.ceil,
            'ceil': np.ceil, 'Round': np.round, 'sqrt': np.sqrt, 'Sqrt': np.sqrt, 'Mod': np.mod,
            'And': lambda *args: np.logical_and.reduce(args) if args else True,
            'Or': lambda *args: np.logical_or.reduce(args) if args else False,
            'Not': np.logical_not,
            'int': self._safe_scalar_int,
            'float': self._safe_scalar_float,
            'bool': self._safe_scalar_bool,
            'Eq': lambda a, b: (a == b),
            'Ne': lambda a, b: (a != b),
            'Gt': lambda a, b: (a > b),
            'Lt': lambda a, b: (a < b),
            'S_None': S_None,
            'len': len,
            'range': range,
            'tuple': tuple,
            'list': list,
            'Piecewise': piecewise_impl,
            'True': True, 'False': False,
            'Dim': dim_impl, 'Rank': rank_impl,
            'Shape': lambda x: tuple(dim_impl(x, i) for i in range(rank_impl(x))),
            'GetItem': _get_item_impl,
            'is_list': _is_list_impl,
            'join': _join_impl,
            'prod': np.prod
        })

        # 1. Inject subs_map
        initial_items = list(subs_map.items())
        for sym, val in initial_items:
            if isinstance(sym, sympy.Symbol): inject(sym.name, val)

        # 2. Inject Inputs
        inputs = res_ctx['inputs']
        alias_to_idx = {
            'X': 0, 'X1': 0, 'A': 1, 'X2': 1, 'Indices': 1,
            'Memory': 1,
            'Query': 0, 'Key': 1, 'Value': 2
        }
        for alias, idx in alias_to_idx.items():
            if idx < len(inputs): eval_ctx[alias] = inputs[idx]

        input_names = op_def.get('input_names', ['X'])
        for idx, name in enumerate(input_names):
            if idx < len(inputs): eval_ctx[name] = inputs[idx]

        # 3. Inject Outputs Placeholder
        output_aliases = {'Y', 'Output', 'Y1', 'Y2'}
        if any(name in candidate_tensors for name in output_aliases):
            if 'out_shape' not in res_ctx:
                tmp_ctx = {
                    'op_def': op_def, 'subs_map': subs_map, 'inputs': inputs,
                    'res_ctx': res_ctx, 'symnet': res_ctx.get('symnet'),
                    'op_instance': op_instance, 'input_sym_map': {}
                }
                try:
                    res_ctx['out_shape'] = self._infer_output_shape(tmp_ctx)
                except:
                    res_ctx['out_shape'] = None

            if res_ctx.get('out_shape'):
                out_proxy = type('OutputProxy', (object,), {
                    'shape': res_ctx['out_shape'], 'ndim': len(res_ctx['out_shape'])
                })
                for alias in output_aliases: eval_ctx[alias] = out_proxy

        # 4. Inject Candidate Tensors (W, B)
        fallback_targets = ['W', 'B'] + list(candidate_tensors)
        for name in fallback_targets:
            if name in eval_ctx: continue
            try:
                sym = sympy.Symbol(name)
                val, stype = self._resolve_symbol(sym, res_ctx)
                inject(name, val)
            except:
                pass

        # 5. Inject Param Map
        if 'param_map' in op_def:
            for py_attr, sym_names in op_def['param_map'].items():
                targets = sym_names if isinstance(sym_names, (list, tuple)) else [sym_names]
                raw_val = getattr(op_instance, 'attrs', {}).get(py_attr)
                if raw_val is None:
                    raw_val = getattr(module, py_attr, getattr(op_instance, py_attr, None))

                if raw_val is not None:
                    if isinstance(raw_val, (list, tuple, np.ndarray)):
                        should_preserve = False
                        if len(targets) == 1:
                            t_name = targets[0]
                            if any(k in t_name.lower() for k in ['shape', 'size', 'cropping', 'kernel', 'stride']):
                                should_preserve = True

                        if should_preserve:
                            val_to_inject = tuple(raw_val)
                            inject(targets[0], val_to_inject)
                        elif len(raw_val) <= len(targets):
                            for t_name, item in zip(targets, raw_val):
                                inject(t_name, item)
                            # Variable-rank definitions may reserve symbolic
                            # names for a larger verified rank. Guarded formula
                            # branches do not consume the trailing names.
                            for t_name in targets[len(raw_val):]:
                                inject(t_name, 0)
                        elif len(targets) == 1:
                            if len(raw_val) == 1:
                                inject(targets[0], raw_val[0])
                            else:
                                inject(targets[0], tuple(raw_val))

                    elif isinstance(raw_val, (int, float, bool, str, np.generic)):
                        for t_name in targets:
                            inject(t_name, raw_val)

                for t_name in targets:
                    if t_name not in eval_ctx:
                        try:
                            resolved_val, _ = self._resolve_symbol(sympy.Symbol(t_name), res_ctx)
                            inject(t_name, resolved_val)
                        except:
                            pass

        self._compute_derived_variables(op_def, eval_ctx, overwrite=True)

        safe_eval_ctx = DefaultZeroDict(eval_ctx)

        for mech in mechanisms:
            desc = mech.get('description', 'Unknown Check')
            expr_str = mech.get('check')

            try:
                py_expr_str = re.sub(r'\bAnd\b(?!\s*\()', ' and ', expr_str)
                py_expr_str = re.sub(r'\bOr\b(?!\s*\()', ' or ', py_expr_str)
                py_expr_str = re.sub(r'\bNot\b(?!\s*\()', ' not ', py_expr_str)
                result = eval(py_expr_str, {}, safe_eval_ctx)

                if hasattr(result, 'all'): result = result.all()
                if not result:
                    safe_ctx_print = {k: (f"Tensor{v.shape}" if hasattr(v, 'shape') else v)
                                      for k, v in eval_ctx.items() if not callable(v)}
                    raise ValueError(
                        f"Mechanism check failed: {desc}\nCondition: {expr_str}\nContext: {safe_ctx_print}")
            except NameError as e:
                missing_var = str(e).split("'")[1] if "'" in str(e) else str(e)
                raise ValueError(f"Mechanism check failed: Variable '{missing_var}' not found. (Check: {desc})")
            except Exception as e:
                if "Mechanism check failed:" in str(e): raise e
                raise ValueError(f"Mechanism check failed (execution error): {e}")

    def _infer_output_shape(self, ctx, output_index=0):
        op_def = ctx['op_def']
        res_ctx = ctx.get('res_ctx', {})

        extra_kwargs = res_ctx.get('extra_kwargs', {})
        op_instance = ctx.get('op_instance')
        val = extra_kwargs.get('output_size', getattr(op_instance, 'output_size', None))
        if val is not None:
            spatial_dims = [val] if isinstance(val, (int, float, np.integer)) else list(val)
            inputs = ctx.get('inputs', [])
            if inputs and hasattr(inputs[0], 'shape'):
                in_shape = inputs[0].shape
                if len(in_shape) > len(spatial_dims):
                    prefix = list(in_shape)[:len(in_shape) - len(spatial_dims)]
                    return [int(s) for s in (prefix + spatial_dims)]
            return [int(s) for s in spatial_dims]

        eval_ctx = dict(SymbolSpace.get_env())
        for k, v in ctx['subs_map'].items():
            if hasattr(k, 'name'): eval_ctx[k.name] = v

        if 'output_size' not in eval_ctx or eval_ctx.get('output_size') is None:
            eval_ctx['output_size'] = [-1] * 10

        def get_rank(obj):
            if hasattr(obj, 'shape') and obj.shape is not None: return len(obj.shape)
            if hasattr(obj, 'ndim'): return obj.ndim
            if isinstance(obj, (list, tuple)): return len(obj)
            return 0

        def get_dim(obj, axis):
            axis = int(axis)
            rank = get_rank(obj)
            if axis < 0:
                axis += rank
            # Python evaluates all values passed to Piecewise before the
            # selected branch is known.  Keep dimensions from unselected
            # rank alternatives symbolic instead of raising or silently
            # replacing them with a concrete value.
            invalid = sympy.Symbol(f"__invalid_dim_{axis}")
            if hasattr(obj, 'shape') and obj.shape is not None:
                if axis < 0 or axis >= len(obj.shape): return invalid
                return obj.shape[axis]
            if hasattr(obj, 'data') and hasattr(obj.data, 'shape'):
                if axis < 0 or axis >= len(obj.data.shape): return invalid
                return obj.data.shape[axis]
            if isinstance(obj, (list, tuple)):
                if axis < 0 or axis >= len(obj): return invalid
                return obj[axis]
            return invalid

        def _get_item_impl(obj, *args):
            if obj is None: return 0
            if not isinstance(obj, (list, tuple, np.ndarray)): return obj
            try:
                if len(args) == 1:
                    res = obj[int(args[0])]
                else:
                    indices = tuple(int(x) for x in args)
                    res = obj[indices]
                if hasattr(res, 'item'): return res.item()
                if isinstance(res, np.ndarray) and res.size == 1: return res.item()
                return res
            except Exception:
                return 0

        def _is_list_impl(obj):
            return isinstance(obj, (list, tuple, np.ndarray))

        def _join_shape_impl(*parts):
            """Evaluate the existing MathIR ``Join`` shape constructor."""
            result = []
            for part in parts:
                if isinstance(part, np.ndarray):
                    result.extend(part.reshape(-1).tolist())
                elif isinstance(part, (list, tuple, sympy.Tuple)):
                    result.extend(list(part))
                else:
                    result.append(part)
            return tuple(result)

        shape_intrinsics = {
            'Rank': get_rank,
            'Dim': get_dim,
            'GetItem': _get_item_impl,
            'is_list': _is_list_impl,
            'Join': _join_shape_impl,
            'join': _join_shape_impl,
            'int': self._safe_scalar_int,
            'float': self._safe_scalar_float,
            'bool': self._safe_scalar_bool,
            'Shape': lambda obj: [get_dim(obj, i) for i in range(get_rank(obj))],
            'Piecewise': self._runtime_piecewise,
        }
        eval_ctx.update(shape_intrinsics)

        for k, v in ctx['subs_map'].items():
            name = k.name if hasattr(k, 'name') else str(k)
            if isinstance(v, (bool, np.bool_)):
                eval_ctx[name] = bool(v)
            elif isinstance(v, (int, np.integer)):
                eval_ctx[name] = int(v)
            elif isinstance(v, (float, np.floating)):
                # Shape inference also sees scalar semantic attributes such
                # as epsilon and Slice/FILL values.  Coercing every float to
                # int both loses semantics and raises on legitimate +/-inf.
                eval_ctx[name] = float(v)

        inputs = ctx['inputs']
        alias_to_idx = {
            'X': 0, 'X1': 0, 'A': 1, 'X2': 1, 'Indices': 1,
            'Memory': 1,
            'Query': 0, 'Key': 1, 'Value': 2
        }
        for alias, idx in alias_to_idx.items():
            if idx < len(inputs): eval_ctx[alias] = inputs[idx]

        input_names = res_ctx.get('input_names', ['X'])
        for idx, name in enumerate(input_names):
            if idx < len(inputs): eval_ctx[name] = inputs[idx]

        extra_kwargs = res_ctx.get('extra_kwargs', {})
        for key, value in extra_kwargs.items():
            eval_ctx[key] = value

        main_input = eval_ctx.get('X1', eval_ctx.get('X'))
        if main_input is not None and hasattr(main_input, 'shape'):
            actual_shape = main_input.shape
            if len(actual_shape) > 0:
                eval_ctx['n'] = int(actual_shape[0])
            dim_map = {'c': 1, 'd': 2, 'h': 3, 'w': 4}
            for name, axis in dim_map.items():
                if axis < len(actual_shape): eval_ctx[name] = int(actual_shape[axis])
            if len(actual_shape) == 3: eval_ctx['l'] = int(actual_shape[2])

        if 'runtime_data' in ctx: eval_ctx.update(ctx['runtime_data'])
        eval_ctx.update(ctx['input_sym_map'])

        module = getattr(op_instance, 'module', op_instance)
        if 'param_map' in op_def:
            for py_attr, sym_names in op_def['param_map'].items():
                targets = sym_names if isinstance(sym_names, (list, tuple)) else [sym_names]
                if all(t in eval_ctx for t in targets): continue

                val = getattr(module, py_attr, None)
                if val is None: val = getattr(op_instance, 'attrs', {}).get(py_attr)

                if val is not None:
                    if isinstance(val, (list, tuple)) and len(val) <= len(targets):
                        for target, item in zip(targets, val):
                            eval_ctx[target] = item
                        for target in targets[len(val):]:
                            eval_ctx[target] = 0
                    elif isinstance(val, (int, float, str, bool, np.generic)):
                        for t in targets: eval_ctx[t] = val

        if op_instance and hasattr(op_instance, 'attrs'):
            for attr_name, attr_val in op_instance.attrs.items():
                if self._is_tensor(attr_val):
                    eval_ctx[attr_name] = attr_val
                    if attr_name == 'weight':
                        eval_ctx['W'] = attr_val
                    elif attr_name == 'bias':
                        eval_ctx['B'] = attr_val
                    elif attr_name.startswith('weight_'):
                        eval_ctx['W_' + attr_name[7:]] = attr_val
                    elif attr_name.startswith('bias_'):
                        eval_ctx['B_' + attr_name[5:]] = attr_val

        # Shape programs may derive dimensions from parameter tensors that
        # are exposed under framework-native names (for example a Keras RNN
        # cell's ``kernel``). Discover those references from the MathIR and
        # resolve them through the same frontend adapter used by formula
        # execution.
        semantic_shape_sources = [
            op_def.get('derived_variables', {}),
            op_def.get('output_shape', []),
        ]

        def iter_semantic_strings(value):
            if isinstance(value, dict):
                for nested in value.values():
                    yield from iter_semantic_strings(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    yield from iter_semantic_strings(nested)
            elif isinstance(value, str):
                yield value

        referenced_tensors = set()
        for source in semantic_shape_sources:
            for expression in iter_semantic_strings(source):
                referenced_tensors.update(re.findall(
                    r'\b(?:Dim|Shape|Rank)\(\s*([A-Za-z_]\w*)',
                    expression,
                ))

        adapter = getattr(ctx.get('symnet'), 'adapter', None)
        if adapter is not None and hasattr(adapter, 'get_named_tensor'):
            for tensor_name in referenced_tensors:
                if tensor_name in {'X', 'Y', 'Output'}:
                    continue
                tensor_value = adapter.get_named_tensor(op_instance, tensor_name)
                if tensor_value is not None:
                    eval_ctx[tensor_name] = tensor_value

        self._compute_derived_variables(op_def, eval_ctx, overwrite=True)
        # Derived-variable evaluation installs backend operators into the
        # shared context. Restore the local bounds-safe shape intrinsics so
        # unselected Piecewise rank branches cannot index outside the actual
        # tensor rank during output-shape inference.
        eval_ctx.update(shape_intrinsics)

        math_expr = ctx.get('math_prog', {}).get('expr')
        if (
            isinstance(math_expr, sympy.Function)
            and math_expr.func.__name__ == 'Transpose'
            # This shortcut infers the output from one input permutation. A
            # composite expression such as Shuffle applies earlier shape
            # transforms, so its declared output-shape program must be used.
            and bool(getattr(math_expr.args[0], 'is_Atom', False))
            and inputs
            and hasattr(inputs[0], 'shape')
        ):
            try:
                param_map = op_def.get('param_map', {}) or {}
                attrs = getattr(op_instance, 'attrs', {}) or {}

                def attribute_for_symbol(symbol_name):
                    for attribute_name, mapped in param_map.items():
                        mapped_names = mapped if isinstance(mapped, (list, tuple)) else [mapped]
                        if symbol_name in mapped_names and attribute_name in attrs:
                            return attrs[attribute_name]
                    return None

                input_shape = tuple(int(dim) for dim in inputs[0].shape)
                direct_perm = attribute_for_symbol('perm')
                if direct_perm is None:
                    # TVM Relay names the full permutation ``axes``.  Treat it
                    # as the same semantic parameter instead of falling
                    # through to an unevaluable ``output_dimensions`` token.
                    direct_perm = attribute_for_symbol('axes')
                if isinstance(direct_perm, (list, tuple, np.ndarray)):
                    perm = tuple(int(axis) for axis in direct_perm)
                    if sorted(perm) == list(range(len(input_shape))):
                        return [input_shape[axis] for axis in perm]

                dim0 = attribute_for_symbol('dim0')
                dim1 = attribute_for_symbol('dim1')
                if dim0 is not None and dim1 is not None:
                    perm = self._make_swap_perm(len(input_shape), dim0, dim1)
                    return [input_shape[axis] for axis in perm]

                resolved_expr = self._resolve_shape_macros(math_expr, res_ctx)
                name_subs = {
                    sympy.Symbol(symbol.name): value
                    for symbol, value in ctx['subs_map'].items()
                    if hasattr(symbol, 'name') and value is not None
                }
                resolved_expr = resolved_expr.subs(name_subs)
                perm_args = resolved_expr.args[1:]
                if len(perm_args) == 1:
                    perm_expr = perm_args[0]
                    perm = sympy.lambdify(
                        sorted(perm_expr.free_symbols, key=lambda item: item.name),
                        perm_expr,
                        modules=[NumpyBackend.get_ops(), np],
                    )()
                else:
                    perm = tuple(int(arg) for arg in perm_args)
                perm = tuple(int(axis) for axis in perm)
                if sorted(perm) == list(range(len(input_shape))):
                    return [input_shape[axis] for axis in perm]
            except Exception:
                pass

        if (
            isinstance(math_expr, sympy.Function)
            and math_expr.func.__name__ == 'Squeeze'
            and inputs
            and hasattr(inputs[0], 'shape')
        ):
            try:
                input_shape = [int(size) for size in inputs[0].shape]
                param_map = op_def.get('param_map', {}) or {}
                attrs = getattr(op_instance, 'attrs', {}) or {}
                squeeze_axes = None
                for attribute_name, mapped in param_map.items():
                    mapped_names = mapped if isinstance(mapped, (list, tuple)) else [mapped]
                    if any(name in {'dim', 'axis'} for name in mapped_names) and attribute_name in attrs:
                        squeeze_axes = attrs[attribute_name]
                        break
                if squeeze_axes is None:
                    defaults = op_def.get('default_values', {}) or {}
                    squeeze_axes = defaults.get('dim', defaults.get('axis'))

                squeeze_all = (
                    squeeze_axes is None
                    or type(squeeze_axes).__name__ == 'NoneSymbol'
                    or (
                        isinstance(squeeze_axes, str)
                        and squeeze_axes.strip("'\"").lower() == 'all'
                    )
                )
                if squeeze_all:
                    return [size for size in input_shape if size != 1]

                raw_axes = np.asarray(squeeze_axes).reshape(-1)
                normalized_axes = []
                for raw_axis in raw_axes:
                    axis = int(raw_axis)
                    if axis < 0:
                        axis += len(input_shape)
                    if axis < 0 or axis >= len(input_shape):
                        raise IndexError(
                            f"Squeeze dim {axis} is outside tensor rank {len(input_shape)}"
                        )
                    normalized_axes.append(axis)
                axes_to_remove = {
                    axis for axis in normalized_axes if input_shape[axis] == 1
                }
                return [
                    size for axis, size in enumerate(input_shape)
                    if axis not in axes_to_remove
                ]
            except Exception:
                pass

        if (
            isinstance(math_expr, sympy.Function)
            and math_expr.func.__name__ == 'ReduceMean'
            and inputs
            and hasattr(inputs[0], 'shape')
        ):
            try:
                input_shape = [int(size) for size in inputs[0].shape]
                param_map = op_def.get('param_map', {}) or {}
                attrs = getattr(op_instance, 'attrs', {}) or {}

                def value_for_symbols(symbol_names, default=None):
                    for attribute_name, mapped in param_map.items():
                        mapped_names = mapped if isinstance(mapped, (list, tuple)) else [mapped]
                        if any(name in symbol_names for name in mapped_names) and attribute_name in attrs:
                            return attrs[attribute_name]
                    defaults = op_def.get('default_values', {}) or {}
                    for name in symbol_names:
                        if name in defaults:
                            return defaults[name]
                    return default

                raw_axes = value_for_symbols({'axis', 'dim'}, 'all')
                keepdim = bool(value_for_symbols({'keepdim'}, False))
                reduce_all = (
                    raw_axes is None
                    or type(raw_axes).__name__ == 'NoneSymbol'
                    or (isinstance(raw_axes, str) and raw_axes.strip("'\"").lower() == 'all')
                )
                if reduce_all:
                    axes = list(range(len(input_shape)))
                else:
                    axes = []
                    for raw_axis in np.asarray(raw_axes).reshape(-1):
                        axis = int(raw_axis)
                        if axis < 0:
                            axis += len(input_shape)
                        if axis < 0 or axis >= len(input_shape):
                            raise IndexError(
                                f"ReduceMean axis {axis} is outside tensor rank {len(input_shape)}"
                            )
                        axes.append(axis)
                axes = sorted(set(axes))
                if keepdim:
                    return [1 if axis in axes else size for axis, size in enumerate(input_shape)]
                return [size for axis, size in enumerate(input_shape) if axis not in axes]
            except Exception:
                pass

        if (
            isinstance(math_expr, sympy.Function)
            and math_expr.func.__name__ == 'Concat'
            and inputs
        ):
            try:
                input_shapes = [tuple(int(size) for size in item.shape) for item in inputs]
                rank = len(input_shapes[0])
                if any(len(shape) != rank for shape in input_shapes):
                    raise ValueError("Concat input ranks differ")

                param_map = op_def.get('param_map', {}) or {}
                attrs = getattr(op_instance, 'attrs', {}) or {}
                raw_axis = None
                for attribute_name, mapped in param_map.items():
                    mapped_names = mapped if isinstance(mapped, (list, tuple)) else [mapped]
                    if 'axis' in mapped_names and attribute_name in attrs:
                        raw_axis = attrs[attribute_name]
                        break
                if raw_axis is None:
                    raw_axis = (op_def.get('default_values', {}) or {}).get('axis', 0)
                axis = int(np.asarray(raw_axis).reshape(-1)[0])
                if axis < 0:
                    axis += rank
                if axis < 0 or axis >= rank:
                    raise IndexError(f"Concat axis {axis} is outside tensor rank {rank}")

                reference = input_shapes[0]
                for shape in input_shapes[1:]:
                    for dim, (left, right) in enumerate(zip(reference, shape)):
                        if dim != axis and left != right:
                            raise ValueError(
                                f"Concat non-axis dimension {dim} differs: {left} != {right}"
                            )
                output_shape = list(reference)
                output_shape[axis] = sum(shape[axis] for shape in input_shapes)
                return output_shape
            except Exception:
                pass

        op_type = str(getattr(op_instance, 'op_type', ''))
        if op_type == 'TensorUnfold' and inputs:
            input_shape = [int(size) for size in np.asarray(inputs[0]).shape]
            attrs = getattr(op_instance, 'attrs', {}) or {}
            module = getattr(op_instance, 'module', op_instance)
            raw_dimension = attrs.get('dimension', getattr(module, 'dimension', None))
            size = attrs.get('size', getattr(module, 'size', None))
            step = attrs.get('step', getattr(module, 'step', None))
            if raw_dimension is None or size is None or step is None:
                raise ValueError('TensorUnfold requires static dimension, size, and step')
            dimension = int(raw_dimension)
            if dimension < 0:
                dimension += len(input_shape)
            if dimension < 0 or dimension >= len(input_shape):
                raise IndexError(
                    f'TensorUnfold dimension {raw_dimension} is outside rank {len(input_shape)}'
                )
            size = int(size)
            step = int(step)
            if size <= 0 or step <= 0 or size > input_shape[dimension]:
                raise ValueError(
                    f'invalid TensorUnfold size/step: size={size}, step={step}, '
                    f'input_extent={input_shape[dimension]}'
                )
            output_shape = list(input_shape)
            output_shape[dimension] = (
                input_shape[dimension] - size
            ) // step + 1
            output_shape.append(size)
            return output_shape

        if 'output_shape' in op_def:
            try:
                raw_shapes = op_def['output_shape']
                math_outputs = ctx.get('math_prog', {}).get('expr')
                is_multi_output = isinstance(math_outputs, (list, tuple))
                if is_multi_output:
                    selected_shape = raw_shapes[output_index]
                    exprs = selected_shape if isinstance(selected_shape, (list, tuple)) else [selected_shape]
                elif isinstance(raw_shapes, list) and raw_shapes and isinstance(raw_shapes[0], list):
                    exprs = raw_shapes[output_index]
                else:
                    exprs = raw_shapes

                final_shape = []
                for s in exprs:
                    res = eval(str(s), {"__builtins__": None}, eval_ctx)

                    if hasattr(res, 'subs'):
                        free_syms = res.free_symbols
                        subs_dict = {sym: eval_ctx[sym.name] for sym in free_syms if sym.name in eval_ctx}
                        res = res.subs(subs_dict).doit()

                    def to_int_or_sym(v):
                        try:
                            return int(v)
                        except:
                            return v

                    if isinstance(res, (list, tuple, sympy.Tuple)):
                        for dim in res: final_shape.append(to_int_or_sym(dim))
                    else:
                        if res == sympy.nan:
                            final_shape.append(1)
                        else:
                            final_shape.append(to_int_or_sym(res))

                concrete_shape = [int(v) for v in final_shape]
                inferred_axes = [
                    axis for axis, size in enumerate(concrete_shape) if size == -1
                ]
                if inferred_axes:
                    if len(inferred_axes) != 1:
                        raise ValueError(
                            "output shape contains more than one inferred dimension"
                        )
                    if not inputs or not hasattr(inputs[0], 'shape'):
                        raise ValueError(
                            "cannot resolve inferred dimension without an input shape"
                        )
                    known_volume = math.prod(
                        size for size in concrete_shape if size != -1
                    )
                    input_volume = math.prod(int(size) for size in inputs[0].shape)
                    if known_volume <= 0 or input_volume % known_volume != 0:
                        raise ValueError(
                            f"cannot infer output dimension for input volume {input_volume} "
                            f"and known output volume {known_volume}"
                        )
                    concrete_shape[inferred_axes[0]] = input_volume // known_volume
                if any(size < 0 for size in concrete_shape):
                    raise ValueError(f"output shape contains a negative dimension: {concrete_shape}")
                return concrete_shape
            except Exception as exc:
                raise ValueError(
                    f"MathIR output_shape evaluation failed for "
                    f"{getattr(op_instance, 'op_type', type(op_instance).__name__)}: {exc}"
                ) from exc

        symnet = ctx['symnet']
        inputs = ctx['inputs']

        if hasattr(symnet, 'adapter') and hasattr(symnet.adapter, 'run_reference_inference'):
            try:
                ready_inputs = [symnet.to_tensor(i) for i in inputs]
                real_out = symnet.adapter.run_reference_inference(op_instance, ready_inputs, symnet)
                if real_out is not None:
                    if isinstance(real_out, (tuple, list)):
                        out_item = real_out[output_index] if output_index < len(real_out) else real_out[0]
                    else:
                        out_item = real_out
                    if hasattr(out_item, 'shape'): return list(out_item.shape)
            except Exception:
                pass

        try:
            inferred = self._infer_shape_from_symbolic_access(ctx)
            if inferred: return inferred
        except Exception:
            pass

        if inputs:
            t = symnet.to_tensor(inputs[0])
            if hasattr(t, 'shape'): return list(t.shape)

        return [1]

    def _infer_shape_from_symbolic_access(self, ctx):
        import sympy

        expr = ctx['specialized_expr']
        res_ctx = ctx['res_ctx']

        input_names = res_ctx.get('input_names', ['X'])
        if not input_names: return None
        main_input_name = input_names[0]

        if main_input_name not in ctx['input_sym_map']: return None
        input_shape = ctx['input_sym_map'][main_input_name].shape

        dims_max = {}

        atoms = expr.atoms(sympy.Indexed, sympy.Function)

        found_access = False
        for node in atoms:
            indices = None

            if isinstance(node, sympy.Indexed) and str(node.base.label) == main_input_name:
                indices = node.indices

            elif isinstance(node, sympy.Function) and type(node).__name__ == 'GetItem':
                if str(node.args[0]) == main_input_name:
                    indices = node.args[1:]

            if not indices: continue
            found_access = True

            for dim_i, idx_expr in enumerate(indices):
                if dim_i >= len(input_shape): break
                input_limit = input_shape[dim_i]

                free_syms = idx_expr.free_symbols
                target_syms = [s for s in free_syms if s.name in ['d', 'h', 'w', 'l']]

                for sym in target_syms:
                    try:
                        probe_map = {s: 0 for s in free_syms if s != sym}
                        isolated_expr = idx_expr.subs(probe_map)

                        stride = isolated_expr.coeff(sym)
                        if stride <= 0: continue

                        offset = isolated_expr.subs(sym, 0)

                        raw_max_idx = (input_limit - 1 - offset) / stride

                        is_loose_mode = False

                        for k, v in ctx['subs_map'].items():
                            k_name = str(k).lower()
                            if v is True and ('ceil' in k_name or 'round' in k_name):
                                is_loose_mode = True
                                break

                        if not is_loose_mode:
                            op = ctx.get('op_instance')
                            module = getattr(op, 'module', op)
                            if getattr(module, 'ceil_mode', False):
                                is_loose_mode = True
                            elif hasattr(op, 'attrs') and op.attrs.get('ceil_mode', False):
                                is_loose_mode = True

                        if is_loose_mode:
                            est_max = sympy.ceiling(raw_max_idx) + 1
                        else:
                            est_max = sympy.floor(raw_max_idx) + 1

                        if sym.name not in dims_max:
                            dims_max[sym.name] = est_max
                        else:
                            if hasattr(est_max, 'is_number') and est_max.is_number and \
                                    hasattr(dims_max[sym.name], 'is_number') and dims_max[sym.name].is_number:
                                dims_max[sym.name] = sympy.Min(dims_max[sym.name], est_max)
                            else:
                                dims_max[sym.name] = sympy.Min(dims_max[sym.name], est_max)

                    except Exception:
                        pass

        if not found_access or not dims_max:
            return None

        final_shape = list(input_shape)
        rank = len(final_shape)

        axis_map = {}
        if rank == 5:  # NCDHW (3D)
            axis_map = {'d': 2, 'h': 3, 'w': 4}
        elif rank == 4:  # NCHW (2D)
            axis_map = {'h': 2, 'w': 3}
        elif rank == 3:  # NCW (1D)
            axis_map = {'l': 2, 'w': 2}  # l and w

        for name, val in dims_max.items():
            if name in axis_map:
                try:
                    # Convert Python int
                    final_shape[axis_map[name]] = int(val)
                except:
                    final_shape[axis_map[name]] = val

        return final_shape

    def _analyze_loop_ranges(self, expr, res_ctx):
        loop_max_map = {}

        sum_nodes = list(expr.atoms(sympy.Sum))
        if not sum_nodes:
            return {}

        for node in sum_nodes:
            for limit in node.limits:
                if len(limit) < 3:
                    continue

                loop_sym = limit[0]
                end = limit[2]

                try:
                    end_val = int(end)
                except Exception:
                    temp_subs = {}
                    for s in end.free_symbols:
                        try:
                            val, stype = self._resolve_symbol(s, res_ctx)
                            if hasattr(val, 'item'):
                                val = val.item()
                            elif hasattr(val, 'shape'):
                                val = 0

                            temp_subs[s] = val
                        except Exception:
                            pass

                    try:
                        subbed_end = end.subs(temp_subs)
                        end_val = int(subbed_end)
                    except Exception:
                        end_val = 0

                loop_max_map[loop_sym.name] = end_val

        return loop_max_map

    def _execute_vectorized_core(self, kernel_func, final_syms, runtime_data, binding, out_shape, fixed_values=None):
        if fixed_values is None: fixed_values = {}
        vectorized_args = []

        for sym in final_syms:
            sname = sym.name
            if sname in runtime_data and sname not in binding:
                arg_val = runtime_data[sname]
                if isinstance(arg_val, np.ndarray) and arg_val.ndim == 1 and len(out_shape) > 1:
                    for dim_i, size in enumerate(out_shape):
                        if arg_val.size == size:
                            new_shape = [1] * len(out_shape)
                            new_shape[dim_i] = arg_val.size
                            arg_val = arg_val.reshape(new_shape)
                            break

                vectorized_args.append(arg_val)

            elif sname in fixed_values:
                vectorized_args.append(fixed_values[sname])

            elif sname in binding:
                dim_idx = binding[sname]
                dim_size = int(out_shape[dim_idx])
                reshape_shape = [1] * len(out_shape)
                reshape_shape[dim_idx] = dim_size
                coord_array = np.arange(dim_size, dtype=np.int32).reshape(reshape_shape)
                vectorized_args.append(coord_array)

            else:
                raise ValueError(
                    f"Unbound symbol {sname} detected in vectorized core, fallback to Step 9 generic simulation.")

        return kernel_func(*vectorized_args)

    def _resolve_symbol(self, sym, ctx):
        name = sym.name
        op_instance = ctx['op_instance']
        input_names = ctx.get('input_names', ['X'])
        inputs = ctx['inputs']
        symnet = ctx['symnet']

        # Framework definitions are authoritative for parameter binding.  The
        # generic W/B aliases below are only fallbacks for older definitions.
        attr_key = name
        tuple_idx = None
        param_map = ctx.get('op_def', {}).get('param_map', {}) or {}
        for framework_attr, math_name in param_map.items():
            if isinstance(math_name, (list, tuple)):
                if name in math_name:
                    attr_key = framework_attr
                    tuple_idx = math_name.index(name)
                    break
            elif math_name == name:
                attr_key = framework_attr
                break

        derived_defs = ctx.get('op_def', {}).get('derived_variables', {}) or {}
        if name in derived_defs:
            try:
                base_env = dict(SymbolSpace.get_env())
                transformations = standard_transformations + (
                    string_token_transform, keyword_none_transform, convert_equals_signs,)
                processed_expr = preprocess_yaml_expr(derived_defs[name])
                parsed_expr = parse_expr(
                    processed_expr,
                    local_dict=base_env,
                    transformations=transformations,
                    evaluate=False
                )

                out_rank = None
                if inputs:
                    first_input = inputs[0]
                    if hasattr(first_input, 'shape'):
                        out_rank = len(first_input.shape)
                eval_ctx = dict(NumpyBackend.get_ops(context={'out_rank': out_rank}))
                eval_ctx.update({
                    'True': True,
                    'False': False,
                    'S_None': S_None,
                })
                eval_ctx.update(self._safe_eval_intrinsics())
                self._install_runtime_derived_helpers(eval_ctx)

                alias_to_idx = {
                    'X': 0, 'X1': 0, 'A': 1, 'X2': 1, 'Indices': 1,
                    'X3': 2, 'X4': 3, 'X5': 4, 'X6': 5, 'X7': 6, 'X8': 7,
                    'Memory': 1,
                    'Query': 0, 'Key': 1, 'Value': 2
                }
                for alias, idx in alias_to_idx.items():
                    if idx < len(inputs):
                        eval_ctx[alias] = inputs[idx]
                for idx, input_name in enumerate(input_names):
                    if idx < len(inputs):
                        eval_ctx[input_name] = inputs[idx]

                dependency_nodes = {}
                for free_sym in sorted(parsed_expr.free_symbols, key=lambda s: s.name):
                    if free_sym.name != name:
                        dependency_nodes[free_sym.name] = free_sym
                for tensor_node in parsed_expr.atoms(sympy.IndexedBase):
                    tensor_name = getattr(tensor_node, 'name', None)
                    if tensor_name and tensor_name != name:
                        dependency_nodes[tensor_name] = tensor_node

                for dep_name, dep_node in dependency_nodes.items():
                    resolved, category = self._resolve_symbol(sympy.Symbol(dep_name), ctx)
                    if resolved is None:
                        continue
                    if self._is_tensor(resolved) and not isinstance(resolved, np.ndarray):
                        try:
                            resolved_np = symnet.to_numpy(resolved)
                            if isinstance(resolved_np, dict):
                                resolved = next(iter(resolved_np.values())) if resolved_np else resolved
                            else:
                                resolved = resolved_np
                        except Exception:
                            pass
                    elif hasattr(resolved, 'item') and not isinstance(resolved, (list, tuple, np.ndarray)):
                        resolved = resolved.item()
                    eval_ctx[dep_name] = resolved

                value = eval(processed_expr, {"__builtins__": None}, eval_ctx)
                if isinstance(value, dict):
                    return value, 'static'
                if hasattr(value, 'item') and not isinstance(value, (list, tuple, np.ndarray)):
                    value = value.item()
                return value, 'static'
            except Exception:
                pass

        if name in ['bias', 'B']:
            print(f"\n[RESOLVE PROBE] Tracing Symbol: '{name}'")
            print(f"  -> op_instance attrs keys: {list(getattr(op_instance, 'attrs', {}).keys())}")

            if hasattr(symnet, 'adapter') and hasattr(symnet.adapter, 'get_named_tensor'):
                tensor_b = symnet.adapter.get_named_tensor(op_instance, 'bias')
                if tensor_b is not None:
                    print(
                        f"  -> adapter.get_named_tensor('bias') SUCCESS! Shape: {getattr(tensor_b, 'shape', 'Unknown')}")
                else:
                    print(f"  -> adapter.get_named_tensor('bias') RETURNED NONE")

            real_obj = getattr(op_instance, 'module', op_instance)
            if hasattr(real_obj, '_parameters'):
                param_keys = list(real_obj._parameters.keys())
                print(f"  -> op_instance.module._parameters keys: {param_keys}")

        if name in ['n', 'c', 'out', 'l', 'd', 'h', 'w', 'i', 'j', 'k']:
            return name, 'tag'

        alias_to_idx = {
            'X': 0, 'X1': 0, 'A': 1, 'X2': 1, 'Indices': 1,
            'X3': 2, 'X4': 3, 'X5': 4, 'X6': 5, 'X7': 6, 'X8': 7,
            'Memory': 1,
            'Query': 0, 'Key': 1, 'Value': 2
        }

        target_idx = None
        if name in input_names:
            target_idx = input_names.index(name)
        elif name in alias_to_idx:
            target_idx = alias_to_idx[name]

        if target_idx is not None and target_idx < len(inputs): return inputs[target_idx], 'dynamic'

        val = None
        module_obj = getattr(op_instance, 'module', None) if op_instance else None

        candidates = []
        for candidate in (attr_key, name):
            if candidate not in candidates:
                candidates.append(candidate)
        if name.startswith('W_'):
            candidates.append('weight_' + name[2:])
        elif name.startswith('B_'):
            candidates.append('bias_' + name[2:])
        elif name == 'W':
            candidates.append('weight')
        elif name == 'B':
            candidates.append('bias')

        if hasattr(op_instance, 'attrs'):
            for cand in candidates:
                if cand in op_instance.attrs:
                    val = op_instance.attrs[cand]
                    break

        if val is None:
            adapter = getattr(symnet, 'adapter', None)
            if adapter:
                for cand in candidates:
                    val = adapter.get_named_tensor(op_instance, cand)
                    if val is not None: break

        if val is not None:
            if self._is_tensor(val) or isinstance(val, np.ndarray):
                if name == 'bias':
                    mapped_symbol_names = set()
                    for mapped in (ctx.get('op_def', {}).get('param_map', {}) or {}).values():
                        if isinstance(mapped, (list, tuple)):
                            mapped_symbol_names.update(mapped)
                        else:
                            mapped_symbol_names.add(mapped)
                    if name in mapped_symbol_names:
                        return True, 'static'
                return val, 'dynamic'

        auto_tensor_map = {
            'running_mean': ['running_mean', 'moving_mean', '_mean'],
            'running_var': ['running_var', 'moving_variance', '_variance'],
            'W': ['weight', 'gamma'],
            'B': ['bias', 'beta'],
            'training': ['training']
        }

        if name in auto_tensor_map and module_obj is not None:
            possible_names = auto_tensor_map[name]
            if name == 'training': return getattr(op_instance, 'training',
                                                  getattr(module_obj, 'training', True)), 'static'
            params = getattr(module_obj, '_parameters', {})
            buffers = getattr(module_obj, '_buffers', {})
            for tensor_attr_name in possible_names:
                val = params.get(tensor_attr_name, buffers.get(tensor_attr_name))
                if val is None:
                    val = getattr(module_obj, tensor_attr_name, None)
                if self._is_tensor(val) or isinstance(val, np.ndarray):
                    return val, 'dynamic'

        if attr_key == name and name == 'W':
            attr_key = 'weight'
        elif attr_key == name and name == 'B':
            attr_key = 'bias'

        val = None
        extra_kwargs = ctx.get('extra_kwargs', {})
        if name in extra_kwargs:
            val = extra_kwargs[name]
        elif attr_key in extra_kwargs:
            val = extra_kwargs[attr_key]
        adapter = getattr(symnet, 'adapter', None)
        if val is None and hasattr(op_instance, 'attrs'):
            val = op_instance.attrs.get(attr_key)

        if val is None:
            if adapter:
                val = adapter.get_named_tensor(op_instance, attr_key)
                if val is None:
                    attrs = adapter.parse_attrs(op_instance)
                    for k, v in attrs.items():
                        if k.lower() == attr_key.lower():
                            val = v
                            break
            if val is None:
                if hasattr(op_instance, attr_key):
                    val = getattr(op_instance, attr_key)
                elif hasattr(op_instance, 'module') and hasattr(op_instance.module, attr_key):
                    val = getattr(op_instance.module, attr_key)
            if val is None:
                root_model = getattr(symnet, 'model', None)
                if root_model is not None and hasattr(root_model, attr_key): val = getattr(root_model, attr_key)

        if val is None:
            defaults = ctx['op_def'].get('default_values', {}) or {}
            if attr_key in defaults:
                val = defaults[attr_key]
            elif name in defaults:
                val = defaults[name]

        if tuple_idx is not None and val is not None:
            if isinstance(val, (list, tuple, np.ndarray)) or hasattr(val, '__getitem__'):
                try:
                    if tuple_idx < len(val): val = val[tuple_idx]
                except:
                    pass
            elif tuple_idx == 0 and isinstance(val, (int, float, np.generic)):
                pass

        if name == 'B' and val is None: return 0, 'static'
        if self._is_tensor(val): return val, 'dynamic'
        if val is not None:
            if hasattr(val, 'item') and not isinstance(val, (list, tuple, np.ndarray)): val = val.item()
            return val, 'static'
        param_map = ctx['op_def'].get('param_map', {}) or {}
        mapped_symbol_names = set()
        for mapped in param_map.values():
            if isinstance(mapped, (list, tuple)):
                mapped_symbol_names.update(mapped)
            else:
                mapped_symbol_names.add(mapped)
        if name == 'dim' or name in mapped_symbol_names or attr_key in param_map:
            return S_None, 'static'
        return None, None

    def _apply_algebraic_lowering(self, expr):
        with self._timer("Algebraic Lowering"):
            return apply_symbolic_lowering(expr)

    def _is_tensor(self, obj):
        return hasattr(obj, 'shape') or hasattr(obj, 'data_ptr') or isinstance(obj, np.ndarray)


class FrameworkMathIRLoader:
    DEFINITION_FILES = ("nn_ops.yaml", "tensors.yaml")

    def __init__(self, framework_name, base_dir="../symbolic_definition"):
        self.mechanisms = None
        self.framework = framework_name
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_dir, framework_name)
        self.mechanisms_doc = self._load("mechanisms.yaml")
        self.type_rules = self.mechanisms_doc.get('TypePromotion', {})
        self.ops_map = {}
        for filename in self.DEFINITION_FILES:
            definitions = self._load(filename)
            conflicting_keys = {
                key for key in set(self.ops_map).intersection(definitions)
                if self.ops_map[key] != definitions[key]
            }
            if conflicting_keys:
                duplicates = ", ".join(sorted(conflicting_keys))
                raise ValueError(
                    f"Duplicate {self.framework} operator definitions across YAML files: {duplicates}"
                )
            for key, definition in definitions.items():
                self.ops_map.setdefault(key, definition)
        # TensorFlow's frontend uses explicit semantic wrapper names for
        # composite recurrent cells.  Reuse the dialect's full LSTM formula
        # under that wrapper instead of treating it as an unknown operator.
        if self.framework == "tensorflow" and "LSTM" in self.ops_map:
            self.ops_map.setdefault("SemTrans_LSTM", self.ops_map["LSTM"])
        self.math_cache = {}
        # Compile definitions on demand.  Eagerly expanding every formula is
        # both unnecessary for model tracing and unsafe for large composed
        # definitions (SymPy can consume unbounded memory while expanding a
        # formula that is unrelated to the current model).  InMemoryLoader and
        # explicit callers may still invoke _compile_all_math_symbols when a
        # complete operator-suite cache is required.
        self._all_symbols_compiled = False

    def _load(self, filename):
        file_path = os.path.join(self.path, filename)
        if not os.path.exists(file_path):
            return {}
        with open(file_path, 'r', encoding='utf-8') as f:
            # Parse the complete text rather than passing a live file stream
            # to PyYAML.  Some framework imports leave stream-like objects in
            # an invalid loader state; text parsing keeps this shared loader
            # deterministic across isolated model workers.
            raw = f.read()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        if not isinstance(raw, str):
            raise TypeError(f"MathIR YAML must decode to text, got {type(raw).__name__}")
        # Prefer the C safe loader on Windows.  Paddle/TensorFlow imports can
        # intermittently corrupt the pure-Python Reader state (for example an
        # integer buffer element in ``ch in string``) even when parsing text
        # rather than a stream.  CSafeLoader has the same safe-constructor
        # policy without sharing that fragile Reader implementation.
        loader_type = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)
        return yaml.load(raw, Loader=loader_type) or {}

    def get_op_def(self, op_type):
        if op_type in self.ops_map:
            return self.ops_map[op_type]

        target_lower = op_type.lower()
        for registered_key in self.ops_map:
            if registered_key.lower() == target_lower:
                return self.ops_map[registered_key]

        # =====================================================================
        # [General Interception Logic] Immediately block and notify for any
        # unregistered operators (whether native unsupported or custom)
        # =====================================================================
        raise NotImplementedError(
            f"[SemTrans Error] Unregistered or custom operator '{op_type}' detected. "
            f"Mathematical semantics cannot be inferred. "
            f"Please provide its symbolic definition in your YAML configuration."
        )

    def get_mechanisms(self, op_type=None):
        if op_type:
            return self.get_op_def(op_type).get('mechanisms', [])
        return self.mechanisms

    def get_math_program(self, op_key):
        if op_key in self.math_cache:
            return self.math_cache[op_key]
        query_fingerprint = op_key.replace("_", "").lower()
        for cache_key, program in self.math_cache.items():
            candidate_fingerprint = cache_key.replace("_", "").lower()

            if query_fingerprint == candidate_fingerprint:
                self.math_cache[op_key] = program
                return program

        # Resolve the requested definition before compiling anything.  This
        # keeps framework construction cheap and, more importantly, prevents
        # unrelated heavyweight formulas from being parsed just because a
        # model uses a small subset of the operator vocabulary.
        definition_key = None
        for candidate_key in self.ops_map:
            if candidate_key == op_key or candidate_key.lower() == str(op_key).lower():
                definition_key = candidate_key
                break
            if candidate_key.replace("_", "").lower() == query_fingerprint:
                definition_key = candidate_key
                break
        if definition_key is None:
            # Schema-driven TensorFlow TFOpLambda stack.  Its semantics are
            # fully determined by the input tuple and axis attribute, and the
            # runtime compiler provides the executable NumPy definition.
            if query_fingerprint == 'stack':
                program = {
                    'expr': 'Stack(X1, ..., Xn, axis)',
                    'op_def': {'definition_type': 'operator', 'semantic_shortcut': 'Stack'},
                }
                self.math_cache[op_key] = program
                return program
            if query_fingerprint == 'bilinearcross':
                program = {
                    'expr': 'BilinearCross(X0, X, u, v, bias)',
                    'op_def': {'definition_type': 'operator', 'semantic_shortcut': 'BilinearCross'},
                }
                self.math_cache[op_key] = program
                return program
            if query_fingerprint == 'factorizationinteraction':
                program = {
                    'expr': 'FactorizationInteraction(X)',
                    'op_def': {'definition_type': 'operator', 'semantic_shortcut': 'FactorizationInteraction'},
                }
                self.math_cache[op_key] = program
                return program
            if query_fingerprint == 'scale':
                program = {
                    'expr': 'Scale(X, scale, bias)',
                    'op_def': {'definition_type': 'operator', 'semantic_shortcut': 'Scale'},
                }
                self.math_cache[op_key] = program
                return program
            if query_fingerprint in {'onehot', 'arange', 'mod', 'remainder', 'astype'}:
                formula = {
                    'onehot': 'OneHot(indices, depth, axis)',
                    'arange': 'Arange(end)',
                    'mod': 'Mod(X, Y)',
                    'remainder': 'Mod(X, Y)',
                    'astype': 'Astype(X, dtype)',
                }[query_fingerprint]
                program = {'expr': formula, 'op_def': {'definition_type': 'operator', 'semantic_shortcut': query_fingerprint}}
                self.math_cache[op_key] = program
                return program
            if query_fingerprint in {'semtransgru', 'l2normalize', 'assign'}:
                formula = {'semtransgru': 'GRURecurrenceResetAfter(X, kernel, recurrent_kernel, bias)', 'l2normalize': 'L2Normalize(X, axis)', 'assign': 'Assign(X)'}[query_fingerprint]
                program = {'expr': formula, 'op_def': {'definition_type': 'operator', 'semantic_shortcut': query_fingerprint}}
                self.math_cache[op_key] = program
                return program
            return None

        definition = self.ops_map.get(definition_key) or {}
        semantic_shortcut = definition.get('semantic_shortcut')
        if semantic_shortcut:
            shortcut_formulas = {
                'EinsumDense': 'Einsum(X, W, equation) + B',
                'ScatterND': 'ScatterND(indices, updates, shape)',
                'Expand': 'Expand(X, shape)',
                'BroadcastTo': 'BroadcastTo(X, shape)',
                'TakeAlongAxis': 'TakeAlongAxis(X, indices, axis)',
                'ArgMax': 'ArgMax(X, axis)',
                'Repeat': 'Repeat(X, repeats, axis)',
                'OnesLike': 'OnesLike(X, dtype)',
                'ZerosLike': 'ZerosLike(X)',
                'Scatter': 'Scatter(X, indices, updates, overwrite)',
            }
            formula = shortcut_formulas.get(
                str(semantic_shortcut),
                str(definition.get('math_symbol') or semantic_shortcut),
            )
            program = {
                'expr': formula,
                'op_def': {
                    **definition,
                    'semantic_shortcut': semantic_shortcut,
                },
            }
            self.math_cache[definition_key] = program
            self.math_cache[op_key] = program
            return program
        if definition.get('definition_type') == 'template' or 'math_symbol' not in definition:
            return None
        try:
            program = self._compile_single_symbol(
                definition['math_symbol'], op_def=definition
            )
        except Exception as exc:
            # Match the historical eager-loader contract: an invalid formula
            # is unavailable to the tracer, but does not make unrelated model
            # regions or the whole process fail during frontend construction.
            print(f"[Warn] Failed to compile math for {definition_key}: {exc}")
            return None
        self.math_cache[definition_key] = program
        self.math_cache[op_key] = program
        return program

    def _compile_all_math_symbols(self):
        print("\n")
        for op_key, def_data in self.ops_map.items():
            if def_data.get('definition_type') == 'template': continue
            try:
                if 'math_symbol' in def_data:
                    self.math_cache[op_key] = self._compile_single_symbol(def_data['math_symbol'], op_def=def_data)
            except Exception as e:
                print(f"[Warn] Failed to compile math for {op_key}: {e}")
        self._all_symbols_compiled = True

    def _compile_single_symbol(self, expr_str, op_def=None):
        if isinstance(expr_str, (list, tuple)):
            compiled_exprs = []
            for s in expr_str:
                res_dict = self._compile_single_symbol(s, op_def)
                compiled_exprs.append(res_dict['expr'])

            return {
                'expr': compiled_exprs,
                'op_def': op_def,
                'source_str': expr_str
            }

        base_env = dict(SymbolSpace.get_env())
        tensor_name_pattern = re.compile(r'\b([A-Za-z_]\w*)\s*\[')
        tensor_like_names = set(tensor_name_pattern.findall(str(expr_str)))
        if op_def and 'derived_variables' in op_def:
            for var_expr_raw in op_def['derived_variables'].values():
                tensor_like_names.update(tensor_name_pattern.findall(str(var_expr_raw)))
        for tensor_name in tensor_like_names:
            base_env[tensor_name] = SmartTensor(tensor_name)
        transformations = standard_transformations + (
            string_token_transform, keyword_none_transform, convert_equals_signs,)

        derived_subs = {}
        if op_def and 'derived_variables' in op_def:
            for var_name, var_expr_raw in op_def['derived_variables'].items():
                proc_var_str = preprocess_yaml_expr(var_expr_raw)
                var_expr = parse_expr(
                    proc_var_str,
                    local_dict=base_env,
                    transformations=transformations,
                    evaluate=False
                )
                derived_subs[sympy.Symbol(var_name)] = var_expr

        def expand_index_dependent_derived(expr):
            if not derived_subs or not isinstance(expr, sympy.Basic):
                return expr

            scalar_helper_ops = {
                'Piecewise', 'Tuple', 'Dim', 'Rank', 'Shape', 'GetItem',
                'Eq', 'Ne', 'Gt', 'Lt', 'Ge', 'Le', 'And', 'Or', 'Not',
                'Min', 'Max', 'Abs', 'Sqrt', 'Mod', 'floor', 'ceiling',
                'int', 'float', 'bool', 'len', 'is_list', 'make_swap_perm',
                'AxisBroadcastShape',
            }

            def has_tensor_value_flow(var_expr):
                if var_expr.atoms(sympy.Indexed):
                    return True
                for fn in var_expr.atoms(sympy.Function):
                    fn_name = fn.func.__name__ if hasattr(fn.func, '__name__') else str(fn.func)
                    if fn_name not in scalar_helper_ops:
                        return True
                return False

            index_symbols = set()
            for indexed in expr.atoms(sympy.Indexed):
                for idx_expr in indexed.indices:
                    if isinstance(idx_expr, sympy.Symbol):
                        index_symbols.add(idx_expr)
                    elif isinstance(idx_expr, sympy.Basic):
                        index_symbols.update(idx_expr.free_symbols)

            for reduction in expr.atoms(sympy.Sum) | expr.atoms(sympy.Product):
                for limit in reduction.limits:
                    if limit and isinstance(limit[0], sympy.Symbol):
                        index_symbols.add(limit[0])

            if not index_symbols:
                index_symbols = set()

            dependent_symbols = set(index_symbols)
            selected = set()
            changed = True
            while changed:
                changed = False
                for var_sym, var_expr in derived_subs.items():
                    if var_sym in selected:
                        continue
                    if has_tensor_value_flow(var_expr) or (var_expr.free_symbols & (dependent_symbols | selected)):
                        selected.add(var_sym)
                        dependent_symbols.add(var_sym)
                        changed = True

            if not selected:
                return expr

            selected_subs = {}
            selected_source = {sym: derived_subs[sym] for sym in selected}
            for var_sym in selected:
                curr_expr = derived_subs[var_sym]
                for _ in range(32):
                    next_expr = curr_expr.xreplace(selected_source)
                    if next_expr == curr_expr:
                        break
                    curr_expr = next_expr
                selected_subs[var_sym] = curr_expr

            try:
                return expr.xreplace(selected_subs)
            except Exception:
                return expr

        processed_str = preprocess_yaml_expr(expr_str)

        try:
            expr = parse_expr(processed_str, local_dict=base_env, transformations=transformations, evaluate=False)
            expr = expand_index_dependent_derived(expr)
        except Exception as e:
            lowered_str = _lower_tensor_bitwise_expr(processed_str)
            if lowered_str != processed_str:
                try:
                    expr = parse_expr(lowered_str, local_dict=base_env, transformations=transformations, evaluate=False)
                    expr = expand_index_dependent_derived(expr)
                except Exception:
                    raise ValueError(f"Syntax Error: {e} | Source: {expr_str}")
            else:
                raise ValueError(f"Syntax Error: {e} | Source: {expr_str}")

        target_expr = expr
        expanded_expr = self._expand_macros(target_expr)
        return {
            'expr': expanded_expr,
            'op_def': op_def,
            'source_str': expr_str
        }

    def _expand_macros(self, expr, expansion_stack=()):
        if isinstance(expr, bool) or expr is None: return expr
        if not isinstance(expr, sympy.Basic): return expr

        if expr.is_Function:
            op_name = type(expr).__name__
            if SymbolSpace.is_atomic(op_name):
                return expr.func(*[self._expand_macros(a, expansion_stack) for a in expr.args])

            if op_name in expansion_stack or len(expansion_stack) >= 32:
                return expr

            op_def = self.ops_map.get(op_name)
            if op_def and 'math_symbol' in op_def:
                base_env = dict(SymbolSpace.get_env())
                try:
                    def_expr = parse_expr(
                        preprocess_yaml_expr(op_def['math_symbol']),
                        local_dict=base_env,
                        transformations=standard_transformations + (string_token_transform,),
                        evaluate=False
                    )
                    def_syms = sorted(list(def_expr.free_symbols), key=lambda s: s.name)
                    if len(def_syms) <= len(expr.args):
                        subs_map = dict(zip(def_syms, expr.args))
                        return self._expand_macros(
                            def_expr.subs(subs_map),
                            expansion_stack + (op_name,),
                        )
                except Exception:
                    return expr

        if not expr.args: return expr
        return expr.func(*[self._expand_macros(a, expansion_stack) for a in expr.args])


class SemanticTracer:
    def __init__(self, src_symnet, tgt_symnet, rtol=1e-3, atol=1e-4):
        self.src = src_symnet
        self.tgt = tgt_symnet

        self.rtol = rtol
        self.atol = atol

        src_math_framework = getattr(self.src, 'math_ir_framework_type', self.src.framework_type)
        tgt_math_framework = getattr(self.tgt, 'math_ir_framework_type', self.tgt.framework_type)
        self.src_loader = FrameworkMathIRLoader(src_math_framework)
        self.tgt_loader = (
            self.src_loader if tgt_math_framework == src_math_framework
            else FrameworkMathIRLoader(tgt_math_framework)
        )

        self.compiler = RuntimeMathCompiler()

        self.SEARCH_WINDOW_SRC = 3
        self.SEARCH_WINDOW_TGT = 5

        self.issues = []
        self.gt_cache = {}
        self.alignment_records = []
        self.first_mismatch = None
        self.source_execution_records = []
        self.parameter_alignment_records = []
        self.formula_diagnostics = []
        self.formula_audit_records = []
        self.formula_unavailable_records = []
        self._last_formula_conflict = None
        self._last_region_rewrite_evidence = None
        self.cumulative_numerical_drift_records = []
        self._last_match_source_output = None
        self._last_match_target_output = None
        self._last_verified_target_output = None
        self._last_comparison_layout = None
        self._last_failed_comparison = None

        self.enable_debug = True

    def _log(self, msg):
        if self.enable_debug:
            print(f"[TraceDebug] {msg}")

    @staticmethod
    def _op_metadata(op, name, default=None):
        attrs = getattr(op, 'attrs', {}) or {}
        if name in attrs:
            return attrs[name]
        return getattr(op, name, default)

    @classmethod
    def _instruction_region(cls, instruction):
        op = instruction.iexpr.op
        region = cls._op_metadata(op, 'tensorrt_region_id', None)
        if region is None:
            region = cls._op_metadata(op, 'relay_fusion_region', None)
        return instruction.retval() if region is None else region

    @classmethod
    def _instruction_source_nodes(cls, instruction):
        return tuple(
            cls._op_metadata(instruction.iexpr.op, 'source_node_names', ()) or ()
        )

    def trace(self, test_input: Any) -> List[str]:
        print(f"[*] Starting Semantic Trace: {self.src.framework_type} -> {self.tgt.framework_type}")
        self.issues.clear()
        self.gt_cache.clear()
        self.alignment_records.clear()
        self.first_mismatch = None
        self.source_execution_records.clear()
        self.parameter_alignment_records.clear()
        self.formula_diagnostics.clear()
        self.formula_audit_records.clear()
        self.formula_unavailable_records.clear()
        self._last_formula_conflict = None
        self._last_region_rewrite_evidence = None
        self.cumulative_numerical_drift_records.clear()
        self._last_match_source_output = None
        self._last_match_target_output = None
        self._last_verified_target_output = None
        self._last_comparison_layout = None
        self._last_failed_comparison = None

        if not self.src.instructions: self._trace_on_cpu(self.src, test_input)
        if not self.tgt.instructions: self._trace_on_cpu(self.tgt, test_input)
        if not self.tgt.instructions: return ["Error: Target model produced no instructions."]

        # Immutable compiler IRs may appear on either side (for example,
        # unoptimized Relay versus optimized Relay).  Capture both sides from
        # their own executors before auditing formulas; source evidence must
        # never be inferred from the target provider.
        captured_provider_ids = set()
        for symnet in (self.src, self.tgt):
            evidence_provider = getattr(symnet, 'evidence_provider', None)
            if evidence_provider is None or id(evidence_provider) in captured_provider_ids:
                continue
            evidence_provider.capture(test_input)
            captured_provider_ids.add(id(evidence_provider))

        self._seed_input_cache(test_input)
        self._audit_full_formula_coverage(test_input)

        align_count = 0
        for match_res in self._dynamic_alignment_instructions:
            if isinstance(match_res, str):
                self.issues.append(match_res)
            elif getattr(match_res, 'gap_info', None):
                self._record_gap(match_res)
            else:
                align_count += 1

        print(f"[*] Trace Finished. Total Aligned Blocks: {align_count}, Issues: {len(self.issues)}")
        return self.issues

    def _extract_primary_numpy(self, res_dict):
        if isinstance(res_dict, dict):
            if 'output_0' in res_dict: return res_dict['output_0']
            return list(res_dict.values())[0]
        return res_dict

    def _split_outputs(self, value):
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _store_instruction_outputs(cache, instruction, value):
        """Store tensor outputs without collapsing tuple-valued graph nodes.

        A traced Python helper such as ``x.chunk(2)`` can have one graph
        variable whose runtime value is a tuple.  Keeping that tuple under
        the single graph name lets subsequent ``getitem`` instructions select
        the intended branch.  Operators with multiple declared graph outputs
        retain the historical one-name-per-output behavior.
        """
        names = list(instruction.retvals())
        if len(names) == 1 and isinstance(value, (list, tuple)):
            cache[names[0]] = value
            return
        split_values = list(value) if isinstance(value, (list, tuple)) else [value]
        for output_name, output_value in zip(names, split_values):
            cache[output_name] = output_value

    def _external_input_values(self, test_input):
        if isinstance(test_input, dict):
            return list(test_input.values())
        if isinstance(test_input, (list, tuple)):
            return list(test_input)
        return [test_input]

    def _as_numpy_input(self, symnet, value):
        if isinstance(value, np.ndarray):
            return value
        return self._extract_primary_numpy(symnet.to_numpy(symnet.to_tensor(value)))

    def _seed_input_cache(self, test_input):
        values = self._external_input_values(test_input)
        for symnet in (self.src, self.tgt):
            input_names = list(getattr(symnet, 'graph_input_names', []) or [])
            if not input_names:
                input_names = list(getattr(symnet, 'input_like', {}).keys())

            for idx, name in enumerate(input_names):
                if isinstance(test_input, dict) and name in test_input:
                    value = test_input[name]
                elif idx < len(values):
                    value = values[idx]
                else:
                    continue
                np_value = self._as_numpy_input(symnet, value)
                self.gt_cache[name] = np_value
                shape = getattr(np_value, 'shape', None)
                self._log(f"Initialized Cache: {name} shape={shape}")

    def _framework_input_cache(self, symnet, test_input):
        values = self._external_input_values(test_input)
        cache = {}
        input_names = list(getattr(symnet, 'graph_input_names', []) or [])
        if not input_names:
            input_names = list(getattr(symnet, 'input_like', {}).keys())
        for index, name in enumerate(input_names):
            if isinstance(test_input, dict) and name in test_input:
                value = test_input[name]
            elif index < len(values):
                value = values[index]
            else:
                continue
            cache[name] = self._as_numpy_input(symnet, value)
        return cache

    def _captured_instruction_output(self, instruction, symnet):
        captured = getattr(symnet, 'trace_value_map', None)
        if not isinstance(captured, dict):
            return None
        names = list(instruction.retvals())
        if not names or any(name not in captured for name in names):
            return None
        values = [self._as_numpy_input(symnet, captured[name]) for name in names]
        return values[0] if len(values) == 1 else tuple(values)

    @staticmethod
    def _compare_formula_and_native(formula_value, native_value, atol, rtol):
        if isinstance(formula_value, dict) or isinstance(native_value, dict):
            if not isinstance(formula_value, dict) or not isinstance(native_value, dict):
                return False, False, float('inf')
            if list(formula_value.keys()) != list(native_value.keys()):
                return False, False, float('inf')
            results = [
                SemanticTracer._compare_formula_and_native(
                    formula_value[key], native_value[key], atol, rtol
                )
                for key in formula_value
            ]
        elif isinstance(formula_value, (list, tuple)) or isinstance(native_value, (list, tuple)):
            if not isinstance(formula_value, (list, tuple)) or not isinstance(native_value, (list, tuple)):
                return False, False, float('inf')
            if len(formula_value) != len(native_value):
                return False, False, float('inf')
            results = [
                SemanticTracer._compare_formula_and_native(left, right, atol, rtol)
                for left, right in zip(formula_value, native_value)
            ]
        else:
            try:
                formula_array = np.asarray(formula_value)
                native_array = np.asarray(native_value)
            except Exception:
                return True, formula_value == native_value, 0.0
            if formula_array.shape != native_array.shape:
                return False, False, float('inf')
            if formula_array.size == 0:
                return True, True, 0.0
            try:
                if formula_array.dtype.kind == 'b' or native_array.dtype.kind == 'b':
                    difference = np.not_equal(
                        formula_array, native_array
                    ).astype(np.float32)
                else:
                    difference = np.abs(formula_array - native_array)
                max_absolute_error = float(np.max(difference))
                matches = bool(np.allclose(
                    formula_array,
                    native_array,
                    atol=atol,
                    rtol=rtol,
                    equal_nan=True,
                ))
            except (TypeError, ValueError):
                matches = bool(np.array_equal(formula_array, native_array))
                max_absolute_error = 0.0 if matches else float('inf')
            return True, matches, max_absolute_error

        if not results:
            return True, True, 0.0
        return (
            all(item[0] for item in results),
            all(item[1] for item in results),
            max((item[2] for item in results), default=0.0),
        )

    def _formula_context_diagnostic(self, instruction):
        contexts = getattr(self.compiler, 'last_jit_contexts', None) or []
        context = contexts[0] if len(instruction.retvals()) == 1 and contexts else None
        if context is None:
            context = getattr(self.compiler, 'last_jit_context', None) or {}
        diagnostic = context.get('formula_diagnostic')
        return dict(diagnostic) if isinstance(diagnostic, dict) else None

    @staticmethod
    def _bounded_formula_probe_inputs(instruction, symnet, op_key, input_values):
        """Request an output-bounded probe without changing input semantics.

        Shape-dependent operators such as TensorFlow SAME padding derive
        parameters from the original input extent.  Cropping the input would
        validate a different formula (for example Pad_Top=3 instead of 2).
        RuntimeMathCompiler limits output coordinates separately, so retaining
        the original input is both faithful and computationally bounded.
        """
        normalized_key = re.sub(r'[^a-z0-9]', '', str(op_key).lower())
        spatial_operators = {
            'conv1d', 'conv2d', 'conv3d',
            'maxpool1d', 'maxpool2d', 'maxpool3d',
            'avgpool1d', 'avgpool2d', 'avgpool3d',
            'adaptiveavgpool1d', 'adaptiveavgpool2d', 'adaptiveavgpool3d',
            'adaptivemaxpool1d', 'adaptivemaxpool2d', 'adaptivemaxpool3d',
            'globalaveragepooling1d', 'globalaveragepooling2d',
            'globalaveragepooling3d',
            'maxpooling1d', 'maxpooling2d', 'maxpooling3d',
            'averagepooling1d', 'averagepooling2d', 'averagepooling3d',
        }
        bounded_pointwise_operators = {
            'batchnorm1d', 'batchnorm2d', 'batchnorm3d',
            'lazybatchnorm1d', 'lazybatchnorm2d', 'lazybatchnorm3d',
            'batchnormalization',
            'relu', 'relu_', 'relu6', 'sigmoid', 'tanh', 'gelu', 'silu',
            'elu', 'selu', 'celu', 'leakyrelu', 'hardtanh', 'hardsigmoid',
            'hardswish', 'dropout', 'dropout2d', 'dropout3d', 'identity',
        }
        if normalized_key not in spatial_operators and normalized_key not in bounded_pointwise_operators:
            return list(input_values), None
        if not input_values:
            return list(input_values), None

        first = input_values[0]
        try:
            value = np.asarray(first)
        except Exception:
            return list(input_values), None
        if value.ndim < 1:
            return list(input_values), None

        attrs = getattr(instruction.iexpr.op, 'attrs', {}) or {}
        data_format = str(attrs.get('data_format', '')).lower()
        channels_last = data_format in {'channels_last', 'nhwc', 'ndhwc', 'nwc'}
        if not data_format:
            channels_last = str(getattr(symnet, 'framework_type', '')).lower() == 'tensorflow'

        # A frozen GraphDef already provides the exact native value for every
        # intermediate tensor.  Cropping a pointwise probe would then compare
        # a 1x1 recomputed formula against the uncropped captured output and
        # report a false structure mismatch.  Keep the original tensor for
        # immutable graph-backed targets.
        if getattr(symnet, 'immutable_artifact', False):
            return list(input_values), {
                'validation_scope': 'full_instruction_input',
                'input_shape_changed': False,
                'original_input_shape': list(value.shape),
                'validation_input_shape': list(value.shape),
                'data_format': 'channels_last' if channels_last else 'channels_first',
                'preserve_full_output': normalized_key in bounded_pointwise_operators or normalized_key in {'averagepooling2d', 'avgpool2d'},
            }

        if normalized_key in spatial_operators:
            # Preserve the original spatial extent for SAME/CAUSAL padding;
            # only the formula output coordinates are bounded below.
            return list(input_values), {
                'validation_scope': 'bounded_output_probe',
                'input_shape_changed': False,
                'original_input_shape': list(value.shape),
                'validation_input_shape': list(value.shape),
                'data_format': 'channels_last' if channels_last else 'channels_first',
            }

        # Pointwise and inference-time channel-wise operators do not derive
        # their semantics from spatial extent.  Crop every dynamic input to a
        # single coordinate so large feature maps cannot turn formula audit
        # into an unbounded full-tensor simulation.
        bounded_values = []
        original_shapes = []
        validation_shapes = []
        for item in input_values:
            array = np.asarray(item)
            original_shapes.append(list(array.shape))
            if array.ndim == 0:
                bounded_values.append(item)
                validation_shapes.append([])
                continue
            limits = [1 for _ in array.shape]
            if normalized_key in {
                'batchnorm1d', 'batchnorm2d', 'batchnorm3d',
                'lazybatchnorm1d', 'lazybatchnorm2d', 'lazybatchnorm3d',
                'batchnormalization',
            } and array.ndim >= 2:
                channel_axis = array.ndim - 1 if channels_last else 1
                limits[channel_axis] = array.shape[channel_axis]
            bounded_values.append(
                array[tuple(slice(0, size) for size in limits)]
            )
            validation_shapes.append(list(limits))
        return bounded_values, {
            'validation_scope': 'bounded_pointwise_probe',
            'input_shape_changed': any(a != b for a, b in zip(original_shapes, validation_shapes)),
            'original_input_shape': original_shapes[0] if len(original_shapes) == 1 else original_shapes,
            'validation_input_shape': validation_shapes[0] if len(validation_shapes) == 1 else validation_shapes,
            'original_input_shapes': original_shapes,
            'validation_input_shapes': validation_shapes,
            'data_format': 'channels_last' if channels_last else 'channels_first',
        }

    @staticmethod
    def _bounded_formula_output(value, channels_last):
        if isinstance(value, dict):
            bounded = {}
            shapes = {}
            for key, item in value.items():
                bounded[key], shapes[key] = SemanticTracer._bounded_formula_output(
                    item, channels_last
                )
            return bounded, shapes
        if isinstance(value, (list, tuple)):
            pairs = [
                SemanticTracer._bounded_formula_output(item, channels_last)
                for item in value
            ]
            return type(value)(item[0] for item in pairs), [item[1] for item in pairs]

        array = np.asarray(value)
        if array.ndim < 3:
            return value, list(array.shape)
        limits = list(array.shape)
        limits[0] = min(limits[0], 1)
        channel_axis = array.ndim - 1 if channels_last else 1
        limits[channel_axis] = min(limits[channel_axis], 1)
        for axis in range(1, array.ndim):
            if axis != channel_axis:
                limits[axis] = min(limits[axis], 1)
        slices = tuple(slice(0, size) for size in limits)
        return array[slices], limits

    @staticmethod
    def _bounded_formula_parameter_op(instruction, symnet, output_shape):
        op = instruction.iexpr.op
        attrs = getattr(op, 'attrs', {}) or {}
        weight_layout = str(attrs.get('semantic_weight_layout', '')).upper()
        if weight_layout != 'HWIO' or not isinstance(output_shape, list):
            return op, None
        if not output_shape or not all(isinstance(size, int) for size in output_shape):
            return op, None

        output_channels = int(output_shape[-1])
        if output_channels < 1:
            return op, None
        adapter = getattr(symnet, 'adapter', None)
        if adapter is None or not hasattr(adapter, 'get_named_tensor'):
            return op, None
        weight = adapter.get_named_tensor(op, 'weight')
        if weight is None or not hasattr(weight, 'shape') or len(weight.shape) != 4:
            return op, None

        bounded_op = copy.copy(op)
        bounded_attrs = dict(attrs)
        bounded_attrs['W'] = weight[..., :output_channels]
        bias = adapter.get_named_tensor(op, 'bias')
        if bias is not None and hasattr(bias, 'shape'):
            bounded_attrs['B'] = bias[:output_channels]
        bounded_op.attrs = bounded_attrs
        return bounded_op, {
            'parameter_probe_scope': 'weight_output_channel_prefix',
            'weight_layout': weight_layout,
            'validation_output_channels': output_channels,
        }

    def _audit_formula_stream(self, side, symnet, loader, test_input):
        cache = self._framework_input_cache(symnet, test_input)
        # Every frontend may publish native intermediate values directly in a
        # trace map.  Graph-backed frontends can additionally populate it via
        # an evidence provider.  Merge the direct map for eager frameworks as
        # well; requiring a provider here discards valid Keras/Paddle captures
        # and turns one missing edge into a cascade of INPUT_UNAVAILABLE rows.
        if side in {'source', 'target'}:
            captured = getattr(symnet, 'trace_value_map', None)
            evidence_provider = getattr(symnet, 'evidence_provider', None)
            if (not isinstance(captured, dict) or not captured) and evidence_provider is not None:
                try:
                    captured = evidence_provider.capture(test_input)
                except Exception:
                    captured = getattr(symnet, 'trace_value_map', None)
            if isinstance(captured, dict):
                cache.update(captured)
        progress_enabled = os.getenv('SEMTRANS_TRACE_PROGRESS', '') == '1'
        instruction_count = len(symnet.instructions)
        stream_started = time.perf_counter()
        previous_started = stream_started
        for instruction_index, instruction in enumerate(symnet.instructions):
            op_key = symnet.adapter.parse_op_key(instruction.iexpr.op)
            region = self._instruction_region(instruction)
            instruction_started = time.perf_counter()

            def report_stage(stage):
                if not progress_enabled:
                    return
                now = time.perf_counter()
                print(
                    f"[SemTransFormulaAuditStage] side={side} "
                    f"instruction={instruction_index + 1}/{instruction_count} "
                    f"operation={op_key} region={region} stage={stage} "
                    f"instruction_elapsed={now - instruction_started:.3f}s "
                    f"stream_elapsed={now - stream_started:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )

            if progress_enabled:
                now = time.perf_counter()
                print(
                    f"[SemTransFormulaAudit] side={side} "
                    f"instruction={instruction_index + 1}/{instruction_count} "
                    f"operation={op_key} region={region} "
                    f"elapsed={now - stream_started:.3f}s "
                    f"previous={now - previous_started:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )
                previous_started = now
            record = {
                'side': side,
                'framework': getattr(symnet, 'framework_type', 'unknown'),
                'operation': str(op_key),
                'region': str(region),
                'input_count': len(instruction.iexpr.args),
                'output_count': len(instruction.retvals()),
            }
            input_values = self._resolve_instruction_inputs(
                instruction, symnet, cache=cache
            )
            report_stage('inputs_resolved')
            if len(input_values) != len(instruction.iexpr.args):
                record['status'] = 'INPUT_UNAVAILABLE'
                self.formula_audit_records.append(record)
                self._record_formula_unavailable(
                    side, instruction, symnet, 'formula_input_unavailable'
                )
                continue

            primary_input = input_values[0] if input_values else None
            native_value = self._captured_instruction_output(instruction, symnet)
            if native_value is None:
                native_value = self._execute_single_instruction_native(
                    instruction,
                    primary_input,
                    symnet,
                    cache=cache,
                    input_values=input_values,
                )
            report_stage('native_output_ready')
            if native_value is None:
                record['status'] = 'NATIVE_EXECUTION_FAILED'
                self.formula_audit_records.append(record)
                self._record_formula_unavailable(
                    side, instruction, symnet, 'native_execution_failed'
                )
                continue

            # Native capture and MathIR qualification are independent gates.
            # A missing or incorrect formula must not starve every downstream
            # instruction of an otherwise valid intermediate value.
            self._store_instruction_outputs(cache, instruction, native_value)

            if self._canonical_operation_name(op_key) == 'getattr':
                record['status'] = 'AUXILIARY_VALUE_BOUND'
                self.formula_audit_records.append(record)
                self._store_instruction_outputs(cache, instruction, native_value)
                continue

            program = loader.get_math_program(op_key)
            report_stage('math_program_loaded')
            if program is None:
                record['status'] = 'MATH_DEFINITION_MISSING'
                self.formula_audit_records.append(record)
                self._record_formula_unavailable(
                    side, instruction, symnet, 'math_definition_missing'
                )
            else:
                try:
                    validation_output_shape = None
                    formula_inputs, probe_metadata = self._bounded_formula_probe_inputs(
                        instruction, symnet, op_key, input_values
                    )
                    report_stage('bounded_inputs_ready')
                    comparison_native_value = native_value
                    if probe_metadata is not None:
                        if probe_metadata['input_shape_changed']:
                            comparison_native_value = self._execute_single_instruction_native(
                                instruction,
                                formula_inputs[0],
                                symnet,
                                input_values=formula_inputs,
                            )
                            if comparison_native_value is None:
                                raise RuntimeError('bounded formula probe native execution failed')
                        record.update(probe_metadata)
                        if probe_metadata.get('preserve_full_output'):
                            validation_output_shape = list(np.asarray(comparison_native_value).shape) if not isinstance(comparison_native_value, (list, tuple, dict)) else None
                        else:
                            comparison_native_value, validation_output_shape = (
                                self._bounded_formula_output(
                                    comparison_native_value,
                                    probe_metadata['data_format'] == 'channels_last',
                                )
                            )
                        record['validation_output_shape'] = validation_output_shape
                    else:
                        record['validation_scope'] = 'full_instruction_input'
                    report_stage('comparison_output_ready')
                    compile_kwargs = {}
                    if probe_metadata is not None and isinstance(
                        validation_output_shape, list
                    ) and all(isinstance(size, int) for size in validation_output_shape):
                        compile_kwargs['_semtrans_output_shape_limit'] = tuple(
                            validation_output_shape
                        )
                    formula_op, parameter_probe_metadata = (
                        self._bounded_formula_parameter_op(
                            instruction, symnet, validation_output_shape
                        )
                    )
                    report_stage('bounded_parameters_ready')
                    if parameter_probe_metadata is not None:
                        record.update(parameter_probe_metadata)
                    if self._canonical_operation_name(op_key) == 'identity':
                        # Identity has no computation beyond forwarding its
                        # resolved tensor.  Using the resolved value here is
                        # the exact executable semantics and avoids treating
                        # a graph output wrapper as a formula mismatch.
                        formula_value = comparison_native_value
                        self.compiler.last_jit_context = {
                            'formula_diagnostic': {
                                'operation': str(op_key),
                                'framework': str(getattr(symnet, 'framework_type', 'unknown')),
                                'native_formula': 'X',
                                'canonical_formula': 'X',
                                'parameter_bindings': {},
                                'derived_bindings': {},
                                'canonical_bindings': {},
                                'layout': {},
                            }
                        }
                        self.compiler.last_jit_contexts = [self.compiler.last_jit_context]
                    else:
                        formula_value = self.compiler.compile_and_run(
                            program,
                            formula_op,
                            formula_inputs,
                            symnet,
                            **compile_kwargs,
                        )
                    report_stage('formula_executed')
                    if len(instruction.retvals()) == 1 and isinstance(formula_value, (list, tuple)):
                        formula_value = formula_value[0]
                    structure_matches, values_match, max_error = self._compare_formula_and_native(
                        formula_value,
                        comparison_native_value,
                        self.atol,
                        self.rtol,
                    )
                    report_stage('formula_compared')
                    audit_status = (
                        'VALIDATED'
                        if structure_matches and values_match
                        else 'JIT_NATIVE_MISMATCH'
                    )
                    diagnostic = self._formula_context_diagnostic(instruction)
                    if diagnostic is not None:
                        diagnostic.update({
                            'side': side,
                            'region': str(region),
                            'node_names': list(
                                self._instruction_source_nodes(instruction)
                                or instruction.retvals()
                            ),
                            'audit_status': audit_status,
                        })
                        record['formula_diagnostic'] = diagnostic
                        self.formula_diagnostics.append(dict(diagnostic))
                    record.update({
                        'status': audit_status,
                        'structure_matches': structure_matches,
                        'values_match': values_match,
                        'max_absolute_error': max_error,
                    })
                    if not structure_matches or not values_match:
                        self._record_formula_unavailable(
                            side,
                            instruction,
                            symnet,
                            'formula_native_mismatch',
                            f'max_absolute_error={max_error}',
                        )
                except Exception as exc:
                    record.update({
                        'status': 'FORMULA_COMPILE_FAILED',
                        'error_type': type(exc).__name__,
                        'error': str(exc),
                        'traceback': traceback.format_exc(),
                    })
                    self._record_formula_unavailable(
                        side, instruction, symnet, 'formula_compile_failed', exc
                    )
                self.formula_audit_records.append(record)

            self._store_instruction_outputs(cache, instruction, native_value)

    def _audit_full_formula_coverage(self, test_input):
        self._audit_formula_stream('source', self.src, self.src_loader, test_input)
        if (
            getattr(self.tgt, 'evidence_provider', None) is None
            or bool(getattr(self.tgt, 'audit_target_formulas', False))
        ):
            self._audit_formula_stream('target', self.tgt, self.tgt_loader, test_input)

    def _trace_on_cpu(self, symnet, test_input):
        print(f"[Trace] Tracing {symnet.framework_type} model on CPU...")
        original_device = symnet.device
        try:
            symnet.device = 'cpu'
            if hasattr(symnet.model, 'to'):
                try:
                    symnet.model.to('cpu')
                except:
                    pass
            input_cpu = symnet.to_tensor(test_input)
            symnet.trace_model_to_instructions(symnet.model, input_cpu)
        finally:
            symnet.device = original_device
            if hasattr(symnet.model, 'to') and original_device:
                try:
                    symnet.model.to(original_device)
                except:
                    pass

    def _get_primary_input_for_inst(self, inst):
        values = self._resolve_instruction_inputs(inst, None)
        return values[0] if values else None

    def _resolve_instruction_inputs(self, inst, symnet=None, fallback_input=None, cache=None):
        cache = self.gt_cache if cache is None else cache
        values = []

        for var_name in inst.iexpr.args:
            val = cache.get(var_name)
            if val is None and symnet is not None and hasattr(symnet, 'static_param_map'):
                val = symnet.static_param_map.get(var_name)
            if val is None:
                val = fallback_input
            if val is None:
                return []
            values.append(val)

        return values

    def _iter_named_modules(self, symnet):
        model = getattr(symnet, 'model', None)
        if model is None:
            return

        yield '', model
        seen = {id(model)}

        if hasattr(model, 'named_modules'):
            try:
                for name, module in model.named_modules():
                    if id(module) in seen:
                        continue
                    seen.add(id(module))
                    yield name, module
                return
            except Exception:
                pass

        if hasattr(model, 'named_sublayers'):
            try:
                for name, module in model.named_sublayers():
                    if id(module) in seen:
                        continue
                    seen.add(id(module))
                    yield name, module
                return
            except Exception:
                pass

        def walk_keras(layer, prefix=''):
            for child in getattr(layer, 'layers', []) or []:
                name = getattr(child, 'name', '')
                child_path = f"{prefix}.{name}" if prefix and name else (name or prefix)
                if id(child) not in seen:
                    seen.add(id(child))
                    yield child_path, child
                yield from walk_keras(child, child_path)

        yield from walk_keras(model)

    def _module_path_for_instruction(self, inst, symnet):
        op = getattr(inst, 'iexpr', None).op if getattr(inst, 'iexpr', None) is not None else None
        module = getattr(op, 'module', None) if op is not None else None
        if module is None:
            return None, None

        attrs = getattr(op, 'attrs', {}) or {}
        if isinstance(attrs, dict) and attrs.get('module_path') is not None:
            return attrs.get('module_path'), module

        for path, candidate in self._iter_named_modules(symnet):
            if candidate is module:
                return path, module
        return None, module

    def _module_by_path(self, symnet, module_path):
        model = getattr(symnet, 'model', None)
        if model is None:
            return None
        if not module_path:
            return model

        if hasattr(model, 'get_submodule'):
            try:
                return model.get_submodule(module_path)
            except Exception:
                pass

        for path, module in self._iter_named_modules(symnet):
            if path == module_path:
                return module
        return None

    def _container_anchor_for_path(self, symnet, module_path):
        if module_path is None:
            return None, None

        try:
            container_types = tuple(symnet._transparent_container_types())
        except Exception:
            container_types = ()
        if not container_types:
            return None, None

        parts = module_path.split('.') if module_path else []
        candidates = ['']
        candidates.extend('.'.join(parts[:idx]) for idx in range(1, len(parts)))

        for candidate in candidates:
            module = self._module_by_path(symnet, candidate)
            if module is not None and isinstance(module, container_types):
                return candidate, module
        return None, None

    def _instruction_provenance(self, inst, symnet):
        op_key = "UnknownOp"
        try:
            op_key = symnet.adapter.parse_op_key(inst.iexpr.op)
        except Exception:
            pass

        module_path, module = self._module_path_for_instruction(inst, symnet)
        container_path, container_module = self._container_anchor_for_path(symnet, module_path)

        layer = module_path if module_path is not None else op_key
        code_anchor = container_path if container_path is not None else layer

        location = "Unknown Loc"
        locator = getattr(symnet, 'locator', None)
        if locator is not None and hasattr(locator, 'get_module_location') and code_anchor != op_key:
            location = locator.get_module_location(code_anchor)

        if container_module is not None:
            container_type = container_module.__class__.__name__
            expanded_from = f"{container_path} ({container_type})" if container_path else container_type
        elif module is not None:
            expanded_from = module.__class__.__name__
        else:
            expanded_from = op_key

        return {
            "layer": layer,
            "code_anchor": code_anchor,
            "location": location,
            "expanded_from": expanded_from,
            "op": op_key,
        }

    @staticmethod
    def _semantic_attr_value(value):
        if isinstance(value, dict):
            return tuple(sorted((str(key), SemanticTracer._semantic_attr_value(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(SemanticTracer._semantic_attr_value(item) for item in value)
        if isinstance(value, np.ndarray):
            return (str(value.dtype), tuple(value.shape), value.tobytes())
        return value

    @staticmethod
    def _canonical_operation_name(operation):
        token = re.sub(r'[^a-z0-9]', '', str(operation).lower())
        aliases = {
            'dense': 'linear',
            'batchnormalization': 'batchnorm',
            'batchnorm1d': 'batchnorm',
            'batchnorm2d': 'batchnorm',
            'batchnorm3d': 'batchnorm',
            # Frontend tracing represents Tensor.size()/shape tuple access as
            # small helper instructions.  Normalize them to the existing
            # auxiliary ``getattr`` category so they are consumed as shape
            # metadata rather than treated as model operators.
            'size': 'getattr',
            'shape': 'getattr',
            'getitem': 'getattr',
            'contiguous': 'getattr',
            'view': 'reshape',
        }
        return aliases.get(token, token)

    @classmethod
    def _is_layout_operation(cls, operation_or_instruction):
        if isinstance(operation_or_instruction, str):
            op_name = operation_or_instruction
            canonical = cls._canonical_operation_name(op_name)
            return canonical in {"permute", "transpose"}
        else:
            op = getattr(getattr(operation_or_instruction, "iexpr", None), "op", None)
            op_name = getattr(op, "op_type", "")
            attrs = getattr(op, "attrs", {}) or {}
        canonical = cls._canonical_operation_name(op_name)
        if canonical not in {"permute", "transpose"}:
            return False
        # A transpose/permute is transparent only when it carries an actual
        # rank permutation. Arbitrary tensor transposes remain semantic ops.
        return bool(attrs.get("perm") or attrs.get("dims") or attrs.get("semantic_layout_transform"))

    @classmethod
    def _is_nonsemantic_operation(cls, operation_or_instruction):
        if cls._is_layout_operation(operation_or_instruction):
            return True
        if isinstance(operation_or_instruction, str):
            # Shape bookkeeping generated by ``Tensor.size()/shape`` and
            # scalar tuple indexing is auxiliary data for reshape/transpose;
            # it must not prevent Dynamic Computation Alignment from matching
            # an equivalent channel-shuffle region across frontends.
            return cls._canonical_operation_name(operation_or_instruction) in {
                "getattr", "size", "shape", "getitem"
            }
        op = getattr(getattr(operation_or_instruction, "iexpr", None), "op", None)
        return cls._canonical_operation_name(getattr(op, "op_type", "")) in {
            "getattr", "size", "shape", "getitem"
        }

    @classmethod
    def _is_auxiliary_value_operation(cls, operation_or_instruction):
        if isinstance(operation_or_instruction, str):
            operation = operation_or_instruction
        else:
            op = getattr(
                getattr(operation_or_instruction, "iexpr", None), "op", None
            )
            operation = getattr(op, "op_type", "")
        return cls._canonical_operation_name(operation) == "getattr"

    def _consume_auxiliary_values(self, instructions, index, symnet):
        while index < len(instructions):
            instruction = instructions[index]
            raw_operation = str(getattr(instruction.iexpr.op, "op_type", "")).casefold()
            is_auxiliary = self._is_auxiliary_value_operation(instruction)
            if raw_operation == "floordiv":
                # Floor division is normally semantic; only the scalar form
                # produced while computing a reshape dimension is metadata.
                args = list(instruction.iexpr.args)
                base = self.gt_cache.get(args[0]) if args else None
                is_auxiliary = is_auxiliary or isinstance(base, (int, float, np.integer, np.floating))
            if not is_auxiliary:
                break
            if raw_operation == "getitem":
                # ``getitem`` is auxiliary only when it indexes a shape tuple
                # (size -> getitem).  Chunk/list selection returns a tensor
                # and must remain in the semantic instruction stream.
                args = list(instruction.iexpr.args)
                base = self.gt_cache.get(args[0]) if args else None
                if not isinstance(base, (tuple, list)):
                    break
            input_values = self._resolve_instruction_inputs(
                instruction, symnet, cache=self.gt_cache
            )
            output = self._execute_single_instruction_native(
                instruction,
                input_values[0] if input_values else None,
                symnet,
                cache=self.gt_cache,
                input_values=input_values,
            )
            if output is None:
                output = self._captured_instruction_output(instruction, symnet)
            # Recover the tiny scalar/data-shape operations emitted by
            # Python-level channel-shuffle helpers.  These nodes have no
            # reliable native runner, so use their tensor shape/attributes as
            # the source of truth even when a backend returned a placeholder.
            raw_operation = str(getattr(instruction.iexpr.op, "op_type", ""))
            operation = self._canonical_operation_name(raw_operation)
            attrs = getattr(instruction.iexpr.op, "attrs", {}) or {}
            args = list(instruction.iexpr.args)
            if operation == "getattr" and raw_operation.casefold() in {"size", "shape"}:
                base = self.gt_cache.get(args[0]) if args else None
                if base is None and input_values:
                    base = input_values[0]
                if hasattr(base, "shape"):
                    output = tuple(int(dim) for dim in base.shape)
            elif operation == "getattr" and raw_operation.casefold() == "getitem":
                base = self.gt_cache.get(args[0]) if args else None
                if base is not None and "arg_1" in attrs:
                    try:
                        output = base[int(attrs["arg_1"])]
                    except (IndexError, KeyError, TypeError, ValueError):
                        output = None
            elif raw_operation.casefold() in {"floordiv", "div"}:
                try:
                    divisor = attrs.get("arg_2", attrs.get("arg_1", 1))
                    output = int(input_values[0]) // int(divisor)
                except (IndexError, TypeError, ValueError):
                    output = None
            if output is None:
                # Shape metadata may be represented by several frontend
                # helper nodes (size -> tuple getitem -> floor division).
                # Their values are already embedded in the following reshape
                # contract; inability to recover one scalar must not leave
                # the alignment cursor parked on a bookkeeping node.  Keep
                # consuming this auxiliary chain while preserving the normal
                # cache path whenever a value is available.
                index += 1
                continue
            # A tuple shape is one logical value even though the generic
            # output splitter treats tuples as multi-output operators.
            if (
                raw_operation.casefold() in {"size", "shape"}
                and len(instruction.retvals()) == 1
                and hasattr(output, "__len__")
            ):
                self.gt_cache[instruction.retvals()[0]] = output
                index += 1
                continue
            for output_name, output_value in zip(
                instruction.retvals(), self._split_outputs(output)
            ):
                self.gt_cache[output_name] = output_value
            index += 1
        return index

    def _layout_prefix_length(self, instructions, start):
        count = 0
        for instruction in instructions[start:]:
            if not self._is_layout_operation(instruction):
                break
            count += 1
        return count

    def _record_formula_diagnostic(self, side, instruction, symnet):
        # A symbolic definition may expose optional auxiliary outputs.  For
        # example, PyTorch MaxPool2d has a value expression and an indices
        # expression, while an ordinary model node returns only the value.
        # The compiler still materializes both expressions so that the
        # operator test suite can validate the optional output.  Formula
        # alignment, however, must use the expression selected by the real
        # instruction output contract; otherwise the disabled indices branch
        # (a scalar zero) can overwrite the value formula in
        # ``last_jit_context``.
        contexts = getattr(self.compiler, "last_jit_contexts", None) or []
        context = None
        if len(instruction.retvals()) == 1 and contexts:
            context = contexts[0]
        if context is None:
            context = getattr(self.compiler, "last_jit_context", None) or {}
        diagnostic = context.get("formula_diagnostic")
        if not isinstance(diagnostic, dict):
            return
        item = dict(diagnostic)
        item.update({
            "side": side,
            "region": self._instruction_region(instruction),
            "node_names": list(self._instruction_source_nodes(instruction) or instruction.retvals()),
        })
        self.formula_diagnostics.append(item)

    def _record_formula_unavailable(self, side, instruction, symnet, reason, error=None):
        """Record a semantic gap instead of silently using native execution."""
        operation_name = getattr(instruction.iexpr.op, "op_type", "")
        if self._canonical_operation_name(operation_name) == "getattr":
            return
        try:
            operation = symnet.adapter.parse_op_key(instruction.iexpr.op)
        except Exception:
            operation = getattr(instruction.iexpr.op, "op_type", "UNKNOWN")
        record = {
            "side": side,
            "framework": getattr(symnet, "framework_type", "UNKNOWN"),
            "operation": str(operation),
            "region": self._instruction_region(instruction),
            "reason": str(reason),
            "error": str(error or ""),
        }
        key = tuple(record.get(name) for name in (
            'side', 'framework', 'operation', 'region', 'reason', 'error'
        ))
        existing = {
            tuple(item.get(name) for name in (
                'side', 'framework', 'operation', 'region', 'reason', 'error'
            ))
            for item in self.formula_unavailable_records
        }
        if key not in existing:
            self.formula_unavailable_records.append(record)

    def _formula_audit_record(self, side, instruction, symnet):
        region = str(self._instruction_region(instruction))
        try:
            operation = str(symnet.adapter.parse_op_key(instruction.iexpr.op))
        except Exception:
            operation = str(getattr(instruction.iexpr.op, 'op_type', 'UNKNOWN'))
        return next(
            (
                record for record in reversed(self.formula_audit_records)
                if record.get('side') == side
                and str(record.get('region')) == region
                and str(record.get('operation')) == operation
            ),
            None,
        )

    def _formula_conflict_for_blocks(self, src_block, tgt_block):
        source_anchor = next(
            (item for item in src_block if not self._is_nonsemantic_operation(item)),
            None,
        )
        target_anchor = next(
            (item for item in tgt_block if not self._is_nonsemantic_operation(item)),
            None,
        )
        source_name = (
            str(self.src.adapter.parse_op_key(source_anchor.iexpr.op)).lower()
            if source_anchor is not None else None
        )
        target_name = (
            str(self.tgt.adapter.parse_op_key(target_anchor.iexpr.op)).lower()
            if target_anchor is not None else None
        )
        source_region = (
            str(self._instruction_region(source_anchor))
            if source_anchor is not None else None
        )
        target_region = (
            str(self._instruction_region(target_anchor))
            if target_anchor is not None else None
        )
        source = next(
            (item for item in reversed(self.formula_diagnostics)
             if item.get("side") == "source"
             and str(item.get("region")) == source_region
             and str(item.get("operation", "")).lower() == source_name),
            None,
        )
        target = next(
            (item for item in reversed(self.formula_diagnostics)
             if item.get("side") == "target"
             and str(item.get("region")) == target_region
             and str(item.get("operation", "")).lower() == target_name),
            None,
        )
        # Parameter/GetAttr instructions may precede the first mathematical
        # source op. In that case use the latest formula from the source side,
        # while still reporting the full sliding-window region.
        if source is None:
            source = next(
                (item for item in reversed(self.formula_diagnostics) if item.get("side") == "source"),
                None,
            )
        if source is None or target is None:
            return None
        source_bindings = source.get("canonical_bindings") or {}
        target_bindings = target.get("canonical_bindings") or {}
        binding_differences = []
        for symbol in sorted(set(source_bindings).intersection(target_bindings)):
            source_value = source_bindings[symbol]
            target_value = target_bindings[symbol]
            if source_value != target_value:
                binding_differences.append({
                    "symbol": symbol,
                    "source_value": source_value,
                    "target_value": target_value,
                })
        return {
            "source": source,
            "target": target,
            "comparison_space": "canonical_NCHW_for_rank4_spatial_ops",
            "source_window": [self.src.adapter.parse_op_key(item.iexpr.op) for item in src_block],
            "target_window": [self.tgt.adapter.parse_op_key(item.iexpr.op) for item in tgt_block],
            "binding_differences": binding_differences,
        }

    def _region_rewrite_evidence(self, src_block, tgt_block):
        """Summarize a region-level rewrite before selecting a local anchor.

        Compiler fusion/decomposition can change several adjacent operators at
        once.  Reporting only the first pair (e.g. ``Sqrt`` vs ``Rsqrt``)
        hides the actual rewrite ``Div(x, Sqrt(abs(y)))`` ->
        ``Mul(x, Rsqrt(abs(y)))``.  This evidence is derived from the two
        target/source instruction windows and their MathIR formulas.
        """
        def items(block, side):
            result = []
            for instruction in block:
                operation = str(self.src.adapter.parse_op_key(instruction.iexpr.op) if side == 'source' else self.tgt.adapter.parse_op_key(instruction.iexpr.op))
                if self._is_nonsemantic_operation(instruction):
                    continue
                region = self._instruction_region(instruction)
                diagnostics = [
                    item for item in self.formula_diagnostics
                    if item.get('side') == side
                    and str(item.get('region')) == str(region)
                    and str(item.get('operation', '')).lower() == operation.lower()
                ]
                formula = diagnostics[-1].get('canonical_formula') if diagnostics else None
                result.append({
                    'index': instruction.retval(),
                    'operation': operation,
                    'formula': formula,
                    'inputs': list(instruction.iexpr.args),
                    'region': region,
                    '_instruction': instruction,
                })
            return result

        def canonical(item):
            return self._canonical_operation_name(item['operation'])

        def compose(region_items, symnet):
            values = {}
            aliases = ('X', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6')

            def leaf(name):
                static = getattr(symnet, 'static_param_map', {}).get(name)
                if static is None:
                    return name
                array = np.asarray(static)
                if array.size and np.all(array == array.reshape(-1)[0]):
                    return repr(float(array.reshape(-1)[0]))
                digest = hashlib.sha256(array.tobytes()).hexdigest()[:12]
                return f"Const_{digest}"

            for item in region_items:
                formula = item.get('formula')
                if not formula:
                    return None
                arguments = [values.get(name, leaf(name)) for name in item['inputs']]
                replacements = {}
                if len(arguments) == 1:
                    replacements['X'] = arguments[0]
                    replacements['X1'] = arguments[0]
                for index, argument in enumerate(arguments):
                    replacements[aliases[index + 1] if index + 1 < len(aliases) else f'X{index + 1}'] = argument
                composed = str(formula)
                for symbol in sorted(replacements, key=len, reverse=True):
                    composed = re.sub(
                        rf'\b{re.escape(symbol)}\b',
                        f"({replacements[symbol]})",
                        composed,
                    )
                values[item['index']] = composed
            return values.get(region_items[-1]['index'])

        def numpy_output(item, symnet):
            value = getattr(symnet, 'trace_value_map', {}).get(item['index'])
            return None if value is None else np.asarray(value)

        def real_arithmetic_equivalent(source_formula, target_formula):
            if not source_formula or not target_formula:
                return None
            locals_map = {
                'add': lambda left, right: left + right,
                'divide': lambda left, right: left / right,
                'mul': lambda left, right: left * right,
                'sqrt': sympy.sqrt,
                'rsqrt': lambda value: 1 / sympy.sqrt(value),
                'abs': sympy.Abs,
                'Abs': sympy.Abs,
            }
            names = set(re.findall(r'\b[A-Za-z_]\w*\b', source_formula + ' ' + target_formula))
            for name in names:
                locals_map.setdefault(name, sympy.Symbol(name))
            try:
                source_expr = sympy.sympify(source_formula, locals=locals_map)
                target_expr = sympy.sympify(target_formula, locals=locals_map)
                return bool(sympy.simplify(source_expr - target_expr) == 0)
            except Exception:
                return None

        source_items = items(src_block, 'source')
        target_items = items(tgt_block, 'target')
        if not source_items or not target_items:
            return None
        # Restrict each side to its first compiler region.  This retains N-to-1
        # rewrites such as three source Adds folded to one target Add; choosing
        # only the first matching operation would erase that evidence.
        target_region = target_items[0]['region']
        target_core = []
        for item in target_items:
            if item['region'] != target_region:
                break
            target_core.append(item)
        source_region = source_items[0]['region']
        source_core = []
        for item in source_items:
            if item['region'] != source_region:
                break
            source_core.append(item)
        if len(source_core) == 1 and len(target_core) > 1:
            # An unoptimized Relay source has no Primitive region IDs, while
            # its optimized target does.  Extend the source to the target
            # region's boundary operation, then remove the unchanged suffix.
            boundary = canonical(target_core[-1])
            source_core = []
            for item in source_items:
                source_core.append(item)
                if canonical(item) == boundary:
                    break
        while (
            len(source_core) > 1
            and len(target_core) > 1
            and canonical(source_core[-1]) == canonical(target_core[-1])
        ):
            source_core.pop()
            target_core.pop()

        source_ops = [item['operation'] for item in source_core]
        target_ops = [item['operation'] for item in target_core]
        if source_ops == target_ops:
            return None
        source_formula = compose(source_core, self.src)
        target_formula = compose(target_core, self.tgt)
        source_output = numpy_output(source_core[-1], self.src)
        target_output = numpy_output(target_core[-1], self.tgt)
        region_max_abs = None
        if (
            source_output is not None
            and target_output is not None
            and source_output.shape == target_output.shape
        ):
            region_max_abs = float(
                np.max(np.abs(source_output.astype(np.float64) - target_output.astype(np.float64)))
            )
        source_final_items = items(self.src.instructions, 'source')
        target_final_items = items(self.tgt.instructions, 'target')
        source_final = (
            numpy_output(source_final_items[-1], self.src)
            if source_final_items
            else None
        )
        target_final = (
            numpy_output(target_final_items[-1], self.tgt)
            if target_final_items
            else None
        )
        final_max_abs = None
        if (
            source_final is not None
            and target_final is not None
            and source_final.shape == target_final.shape
        ):
            final_max_abs = float(
                np.max(np.abs(source_final.astype(np.float64) - target_final.astype(np.float64)))
            )
        real_equivalent = real_arithmetic_equivalent(source_formula, target_formula)
        add_reassociation = (
            all(canonical(item) == 'add' for item in source_core + target_core)
            and len(source_core) != len(target_core)
        )
        if add_reassociation and region_max_abs not in (None, 0.0):
            relation = 'finite_precision_constant_reassociation'
        elif real_equivalent is True and region_max_abs not in (None, 0.0):
            relation = 'mathematically_equivalent_but_finite_precision_order_changed'
        elif real_equivalent is True:
            relation = 'mathematically_equivalent_on_observed_input'
        elif real_equivalent is False:
            relation = 'mathematically_non_equivalent_rewrite'
        else:
            relation = 'region_rewrite_requires_formula_review'
        amplification = None
        if region_max_abs not in (None, 0.0) and final_max_abs is not None:
            amplification = final_max_abs / region_max_abs
        evidence = {
            'kind': 'region_rewrite',
            'source_region': source_core[0]['region'],
            'target_region': target_core[0]['region'],
            'source_operations': source_ops,
            'target_operations': target_ops,
            'source_formulas': [item['formula'] for item in source_core],
            'target_formulas': [item['formula'] for item in target_core],
            'source_composed_formula': source_formula,
            'target_composed_formula': target_formula,
            'real_arithmetic_equivalent': real_equivalent,
            'algebraic_reassociation_pattern': add_reassociation,
            'region_output_max_absolute_error': region_max_abs,
            'final_output_max_absolute_error': final_max_abs,
            'downstream_amplification_ratio': amplification,
            'source_output_node': source_core[-1]['index'],
            'target_output_node': target_core[-1]['index'],
            'relation': relation,
        }
        return evidence

    def _semantic_entries(self, block, symnet):
        entries = []
        for instruction in block:
            op = instruction.iexpr.op
            region_id = self._instruction_region(instruction)
            region_attrs = getattr(op, 'attrs', {}).get('semantic_region_attrs') or []
            if region_attrs:
                for item in region_attrs:
                    entries.append({
                        "node": item.get("node_name", ""),
                        "op": item.get("op_type", "UnknownOp"),
                        "attrs": self._semantic_attr_value(item.get("attrs", {})),
                        "region": region_id,
                    })
                continue
            nodes = list(self._instruction_source_nodes(instruction) or instruction.retvals())
            op_key = symnet.adapter.parse_op_key(op)
            for node in nodes:
                entries.append({"node": node, "op": op_key, "attrs": (), "region": region_id})
        return entries

    def _apply_region_rewrite_evidence(self):
        evidence = self._last_region_rewrite_evidence
        if evidence is None or self.first_mismatch is None:
            return
        relation = evidence.get('relation')
        reason = (
            'finite_precision_region_rewrite'
            if relation in {
                'mathematically_equivalent_but_finite_precision_order_changed',
                'finite_precision_constant_reassociation',
            }
            else 'region_rewrite_mismatch'
        )
        mismatch_reason = self.first_mismatch.get('reason')
        self.first_mismatch.update({
            'predicted_source_node': evidence.get('source_output_node'),
            'predicted_target_node': evidence.get('target_output_node'),
            'predicted_target_region': evidence.get('target_region'),
            'source_operation': ' -> '.join(evidence.get('source_operations') or []),
            'target_operation': ' -> '.join(evidence.get('target_operations') or []),
            'source_formula': evidence.get('source_composed_formula'),
            'target_formula': evidence.get('target_composed_formula'),
            'mismatch_max_absolute_error': evidence.get('region_output_max_absolute_error'),
            'region_rewrite_evidence': evidence,
        })
        if mismatch_reason not in {'missing_target_suffix', 'extra_target_suffix'}:
            self.first_mismatch['reason'] = reason

    def _localize_mismatch(self, src_block, tgt_block):
        source_entries = self._semantic_entries(src_block, self.src)
        target_entries = self._semantic_entries(tgt_block, self.tgt)
        # Layout conversion is retained in the reported window, but it is not
        # the semantic anchor being diagnosed. Compare the first real operator
        # after the conversion against the source operator.
        source_entries = [
            entry for entry in source_entries
            if not self._is_nonsemantic_operation(entry.get("op", ""))
        ]
        target_entries = [
            entry for entry in target_entries
            if not self._is_nonsemantic_operation(entry.get("op", ""))
        ]
        target_by_node = {entry["node"]: entry for entry in target_entries if entry["node"]}
        source_nodes = {entry["node"] for entry in source_entries if entry["node"]}

        for source_entry in source_entries:
            target_entry = target_by_node.get(source_entry["node"])
            if target_entry is None:
                continue
            if source_entry["op"] != target_entry["op"]:
                return {
                    "reason": "operator_mismatch",
                    "predicted_source_node": source_entry["node"],
                    "predicted_target_node": target_entry["node"],
                    "predicted_target_region": target_entry["region"],
                    "source_operation": source_entry["op"],
                    "target_operation": target_entry["op"],
                }
            if source_entry["attrs"] != target_entry["attrs"]:
                return {
                    "reason": "attribute_mismatch",
                    "predicted_source_node": source_entry["node"],
                    "predicted_target_node": target_entry["node"],
                    "predicted_target_region": target_entry["region"],
                    "source_operation": source_entry["op"],
                    "target_operation": target_entry["op"],
                }

        if len(source_entries) == len(target_entries) and source_entries:
            for source_entry, target_entry in zip(source_entries, target_entries):
                if self._canonical_operation_name(source_entry["op"]) != self._canonical_operation_name(
                    target_entry["op"]
                ):
                    return {
                        "reason": "operator_mismatch",
                        "predicted_source_node": source_entry["node"],
                        "predicted_target_node": target_entry["node"],
                        "predicted_target_region": target_entry["region"],
                        "source_operation": source_entry["op"],
                        "target_operation": target_entry["op"],
                    }
            return {
                "reason": "numerical_mismatch",
                "predicted_source_node": source_entries[0]["node"],
                "predicted_target_node": target_entries[0]["node"],
                "predicted_target_region": target_entries[0]["region"],
                "source_operation": source_entries[0]["op"],
                "target_operation": target_entries[0]["op"],
            }

        source_anchor_nodes = tuple(
            self._instruction_source_nodes(src_block[0]) or src_block[0].retvals()
        ) if src_block else ()
        target_anchor_nodes = tuple(
            self._instruction_source_nodes(tgt_block[0]) or tgt_block[0].retvals()
        ) if tgt_block else ()
        if source_anchor_nodes == target_anchor_nodes and target_entries:
            anchor = target_entries[0]
            return {
                "reason": "numerical_mismatch",
                "predicted_source_node": source_anchor_nodes[0] if source_anchor_nodes else "",
                "predicted_target_node": anchor["node"],
                "predicted_target_region": anchor["region"],
                "source_operation": source_entries[0]["op"] if source_entries else "N/A",
                "target_operation": anchor["op"],
            }

        for source_index, source_entry in enumerate(source_entries):
            if source_entry["node"] and source_entry["node"] not in target_by_node:
                anchor = target_entries[min(source_index, len(target_entries) - 1)] if target_entries else None
                return {
                    "reason": "missing_target_node",
                    "predicted_source_node": source_entry["node"],
                    "predicted_target_node": anchor["node"] if anchor else "",
                    "predicted_target_region": anchor["region"] if anchor else "N/A",
                    "source_operation": source_entry["op"],
                    "target_operation": anchor["op"] if anchor else "N/A",
                }

        for target_entry in target_entries:
            if target_entry["node"] and target_entry["node"] not in source_nodes:
                return {
                    "reason": "extra_target_node",
                    "predicted_source_node": "",
                    "predicted_target_node": target_entry["node"],
                    "predicted_target_region": target_entry["region"],
                    "source_operation": "N/A",
                    "target_operation": target_entry["op"],
                }

        fallback = target_entries[0] if target_entries else None
        return {
            "reason": "numerical_mismatch",
            "predicted_source_node": source_entries[0]["node"] if source_entries else "",
            "predicted_target_node": fallback["node"] if fallback else "",
            "predicted_target_region": fallback["region"] if fallback else "N/A",
            "source_operation": source_entries[0]["op"] if source_entries else "N/A",
            "target_operation": fallback["op"] if fallback else "N/A",
        }

    @property
    def _dynamic_alignment_instructions(self) -> Generator[Any, None, None]:
        i, j = 0, 0
        n_src, n_tgt = len(self.src.instructions), len(self.tgt.instructions)

        while i < n_src and j < n_tgt:
            i = self._consume_auxiliary_values(
                self.src.instructions, i, self.src
            )
            j = self._consume_auxiliary_values(
                self.tgt.instructions, j, self.tgt
            )
            if i >= n_src or j >= n_tgt:
                break
            self._log(f"--- Aligning at Pointers: Src[{i}], Tgt[{j}] ---")
            match = self._try_match_at_position(i, j)

            if match:
                self._log(f"   >>> MATCHED! Src Size: {len(match.src_block)}, Tgt Size: {len(match.tgt_block)}")
                yield match
                self._update_gt_cache(match)
                i += len(match.src_block)
                j += len(match.tgt_block)
            else:
                self._log(f"   >>> FATAL MISMATCH. Halting and Generating Issue Report.")
                src_context_block = self.src.instructions[i: i + self.SEARCH_WINDOW_SRC]
                tgt_context_block = self.tgt.instructions[j: j + self.SEARCH_WINDOW_TGT]

                self.first_mismatch = {
                    "source_index": i,
                    "target_index": j,
                    "source_regions": [
                        self._instruction_region(inst) for inst in src_context_block
                    ],
                    "target_regions": [
                        self._instruction_region(inst) for inst in tgt_context_block
                    ],
                    "source_nodes": [
                        name for inst in src_context_block
                        for name in (self._instruction_source_nodes(inst) or inst.retvals())
                    ],
                    "target_nodes": [
                        name for inst in tgt_context_block
                        for name in (self._instruction_source_nodes(inst) or inst.retvals())
                    ],
                }
                self.first_mismatch.update(self._localize_mismatch(src_context_block, tgt_context_block))
                self._last_region_rewrite_evidence = self._region_rewrite_evidence(
                    src_context_block, tgt_context_block
                )
                if self._last_region_rewrite_evidence is not None:
                    self.first_mismatch['region_rewrite_evidence'] = self._last_region_rewrite_evidence
                if self._last_failed_comparison is not None:
                    self.first_mismatch.update(self._last_failed_comparison)
                if self.parameter_alignment_records:
                    self.first_mismatch["parameter_layout_evidence"] = list(
                        self.parameter_alignment_records[-8:]
                    )
                if self._last_formula_conflict is not None:
                    self.first_mismatch["formula_evidence"] = self._last_formula_conflict
                    source_formula = (self._last_formula_conflict.get("source") or {}).get("canonical_formula")
                    target_formula = (self._last_formula_conflict.get("target") or {}).get("canonical_formula")
                    if source_formula:
                        self.first_mismatch["source_formula"] = source_formula
                    if target_formula:
                        self.first_mismatch["target_formula"] = target_formula

                self._apply_region_rewrite_evidence()

                issue_report = self._generate_issue_report(
                    src_context_block,
                    tgt_context_block,
                    self._get_primary_input_for_inst(self.src.instructions[i])
                )
                yield issue_report
                break

        i = self._consume_auxiliary_values(self.src.instructions, i, self.src)
        j = self._consume_auxiliary_values(self.tgt.instructions, j, self.tgt)

        # Do not silently accept a suffix that exists in only one frontend.
        # The old loop stopped as soon as either instruction stream was
        # exhausted, so an extra target activation/fusion (or a missing target
        # tail) could never become evidence.  Such a suffix is a real
        # structural mismatch and must be localized like an in-loop failure.
        if self.first_mismatch is None and (i < n_src or j < n_tgt):
            src_context_block = self.src.instructions[i: i + self.SEARCH_WINDOW_SRC]
            tgt_context_block = self.tgt.instructions[j: j + self.SEARCH_WINDOW_TGT]
            mismatch_reason = (
                "missing_target_suffix" if i < n_src and j >= n_tgt
                else "extra_target_suffix"
            )
            # The report builder needs an anchor on both sides.  For a pure
            # suffix mismatch one side is empty, so use the immediately
            # preceding matched instruction only as a diagnostic anchor; the
            # primary mismatch reason and empty-side node lists remain exact.
            report_src_block = src_context_block or self.src.instructions[max(0, i - 1):i]
            report_tgt_block = tgt_context_block or self.tgt.instructions[max(0, j - 1):j]
            self.first_mismatch = {
                "source_index": i,
                "target_index": j,
                "source_regions": [self._instruction_region(inst) for inst in src_context_block],
                "target_regions": [self._instruction_region(inst) for inst in tgt_context_block],
                "source_nodes": [
                    name for inst in src_context_block
                    for name in (self._instruction_source_nodes(inst) or inst.retvals())
                ],
                "target_nodes": [
                    name for inst in tgt_context_block
                    for name in (self._instruction_source_nodes(inst) or inst.retvals())
                ],
                "reason": mismatch_reason,
            }
            self.first_mismatch.update(self._localize_mismatch(report_src_block, report_tgt_block))
            self._last_region_rewrite_evidence = self._region_rewrite_evidence(
                report_src_block, report_tgt_block
            )
            if self._last_region_rewrite_evidence is not None:
                self.first_mismatch['region_rewrite_evidence'] = self._last_region_rewrite_evidence
            self.first_mismatch["reason"] = mismatch_reason
            if self._last_failed_comparison is not None:
                self.first_mismatch.update(self._last_failed_comparison)
            if self._last_formula_conflict is not None:
                self.first_mismatch["formula_evidence"] = self._last_formula_conflict
                source_formula = (self._last_formula_conflict.get("source") or {}).get("canonical_formula")
                target_formula = (self._last_formula_conflict.get("target") or {}).get("canonical_formula")
                if source_formula:
                    self.first_mismatch["source_formula"] = source_formula
                if target_formula:
                    self.first_mismatch["target_formula"] = target_formula
            self._apply_region_rewrite_evidence()
            issue_report = self._generate_issue_report(
                report_src_block,
                report_tgt_block,
                self._get_primary_input_for_inst(report_src_block[0]) if report_src_block else None,
            )
            yield issue_report

    def _generate_issue_report(self, src_block, tgt_block, input_val):
        src_op_keys = [self.src.adapter.parse_op_key(inst.iexpr.op) for inst in src_block]
        tgt_op_keys = [self.tgt.adapter.parse_op_key(inst.iexpr.op) for inst in tgt_block]

        src_context_str = " -> ".join(src_op_keys)
        tgt_context_str = " -> ".join(tgt_op_keys)

        anchor_src_inst = src_block[0]
        anchor_tgt_inst = next(
            (instruction for instruction in tgt_block if not self._is_nonsemantic_operation(instruction)),
            tgt_block[0],
        )

        anchor_src_key = src_op_keys[0]
        anchor_tgt_key = tgt_op_keys[0]
        src_prov = self._instruction_provenance(anchor_src_inst, self.src)
        tgt_prov = self._instruction_provenance(anchor_tgt_inst, self.tgt)

        src_out = self._execute_source_op(anchor_src_inst, input_val)
        evidence_provider = getattr(self.tgt, 'evidence_provider', None)
        if evidence_provider is not None:
            tgt_out = evidence_provider.get_output(anchor_tgt_inst)
        else:
            tgt_out = self._execute_single_instruction_native(anchor_tgt_inst, input_val, self.tgt)

        shape_status = "PASS"
        max_diff = "N/A"
        if src_out is not None and tgt_out is not None:
            if src_out.shape != tgt_out.shape:
                shape_status = f"FAILED (Src Anchor: {src_out.shape}, Tgt Anchor: {tgt_out.shape})"
            else:
                source_array = np.asarray(src_out)
                target_array = np.asarray(tgt_out)
                if source_array.dtype.kind == 'b' or target_array.dtype.kind == 'b':
                    diff = np.not_equal(
                        source_array, target_array
                    ).astype(np.float32)
                else:
                    diff = np.abs(source_array - target_array)
                max_diff = f"{diff.max():.4f}"

        src_prog = self.src_loader.get_math_program(anchor_src_key)
        tgt_prog = self.tgt_loader.get_math_program(anchor_tgt_key)
        ast_diagnosis = self._diagnose_ast_conflict(src_prog, tgt_prog)

        parameter_evidence = "None observed"
        if self.parameter_alignment_records:
            lines = []
            for record in self.parameter_alignment_records[-8:]:
                lines.append(
                    f"- {record['symbol']} for {record['source_operation']} -> "
                    f"{record['target_operation']}.{record['target_attribute']}: "
                    f"source shape {record['source_shape']}, target shape "
                    f"{record['target_shape']}, relation={record['relation']}"
                )
            parameter_evidence = "\n".join(lines)

        formula_evidence = self._last_formula_conflict or {}
        source_formula = (formula_evidence.get("source") or {}).get("canonical_formula")
        target_formula = (formula_evidence.get("target") or {}).get("canonical_formula")
        formula_section = "None captured"
        if source_formula or target_formula:
            formula_section = (
                "Coordinate convention: canonical NCHW for rank-4 spatial operators\n"
                f"- Source canonical formula: {source_formula or 'N/A'}\n"
                f"- Target canonical formula: {target_formula or 'N/A'}\n"
                f"- Source parameter bindings: {(formula_evidence.get('source') or {}).get('parameter_bindings', {})}\n"
                f"- Target parameter bindings: {(formula_evidence.get('target') or {}).get('parameter_bindings', {})}\n"
                f"- Target derived bindings: {(formula_evidence.get('target') or {}).get('derived_bindings', {})}"
            )
        algebraic_locus = (
            "Formula-instantiated mismatch in canonical coordinate space"
            if source_formula and target_formula else ast_diagnosis['locus']
        )
        source_locus = source_formula or ast_diagnosis['src_ast']
        target_locus = target_formula or ast_diagnosis['tgt_ast']
        rewrite_evidence = self._last_region_rewrite_evidence
        rewrite_section = "None captured"
        if rewrite_evidence is not None:
            rewrite_section = (
                f"- Source region operations: {rewrite_evidence['source_operations']}\n"
                f"- Target region operations: {rewrite_evidence['target_operations']}\n"
                f"- Source region formulas: {rewrite_evidence['source_formulas']}\n"
                f"- Target region formulas: {rewrite_evidence['target_formulas']}\n"
                f"- Relation: {rewrite_evidence['relation']}"
            )
            source_locus = rewrite_evidence.get('source_composed_formula') or source_locus
            target_locus = rewrite_evidence.get('target_composed_formula') or target_locus
            algebraic_locus = rewrite_evidence.get('relation') or algebraic_locus
            rewrite_max_abs = rewrite_evidence.get('region_output_max_absolute_error')
            if rewrite_max_abs is not None:
                max_diff = f"{rewrite_max_abs:.9g}"
            source_rewrite_ops = " -> ".join(rewrite_evidence.get('source_operations') or [])
            target_rewrite_ops = " -> ".join(rewrite_evidence.get('target_operations') or [])
            conflict_label = f"region rewrite ({source_rewrite_ops} vs {target_rewrite_ops})"
            status_line = (
                "REGION REWRITE DIVERGED under finite-precision execution "
                f"(Tolerance atol={self.atol}, rtol={self.rtol})"
            )
        else:
            conflict_label = f"anchor operators ({anchor_src_key} vs {anchor_tgt_key})"
            status_line = (
                f"ALL {len(src_block)}x{len(tgt_block)} ALIGNMENT COMBINATIONS FAILED "
                f"(Tolerance atol={self.atol}, rtol={self.rtol})"
            )
        rewrite_relation = (
            rewrite_evidence.get('relation') if rewrite_evidence is not None else None
        )
        if rewrite_relation == 'mathematically_equivalent_but_finite_precision_order_changed':
            repair_action = (
                "Preserve the source evaluation order in the affected target region "
                "or disable the rewrite that introduced reciprocal-first evaluation; "
                "then recompile and validate on held-out inputs."
            )
        elif rewrite_relation == 'finite_precision_constant_reassociation':
            repair_action = (
                "Preserve the source addition grouping and source constants in the "
                "affected target region, or disable the constant reassociation/folding "
                "rewrite; then recompile and validate on held-out inputs."
            )
        else:
            repair_action = (
                "Regenerate the target region so its composed MathIR is equivalent "
                "to the source region, then recompile and validate on held-out inputs."
            )

        report = (
            f"**Issue Report (R) Payload**\n"
            f"----------------------------------------\n"
            f"[Code Context (M-to-N Divergence Zone)]\n"
            f"Source Search Window ($W_S$={len(src_block)}): {src_context_str}\n"
            f"Target Search Window ($W_T$={len(tgt_block)}): {tgt_context_str}\n"
            f"Layer: {tgt_prov['layer']}\n"
            f"Target Code Anchor: {tgt_prov['code_anchor']}\n"
            f"Target Code Location: {tgt_prov['location']}\n"
            f"Expanded Container: {tgt_prov['expanded_from']}\n"
            f"Source Anchor: {src_prov['layer']} @ {src_prov['location']} ({src_prov['expanded_from']})\n"
            f"Target Anchor: {tgt_prov['layer']} [code: {tgt_prov['code_anchor']}] @ {tgt_prov['location']} ({tgt_prov['expanded_from']})\n"
            f"----------------------------------------\n"
            f"[Execution Evidence (Anchor Check)]\n"
            f"Status: {status_line}\n"
            f"Anchor Shape Alignment: {shape_status}\n"
            f"Anchor Numerical Divergence: Max(|Y_ideal - Y_real|) = {max_diff}\n"
            f"----------------------------------------\n"
            f"[Algebraic Locus (from SemTrans-JIT)]\n"
            f"Conflict detected at {conflict_label}:\n"
            f"Locus Type: {algebraic_locus}\n"
            f"- Source formula/AST: {source_locus}\n"
            f"- Target formula/AST: {target_locus}\n"
            f"[Region Rewrite Evidence]\n"
            f"{rewrite_section}\n"
            f"[Canonical MathIR Formula Evidence]\n"
            f"{formula_section}\n"
            f"[Parameter/Layout Evidence]\n"
            f"{parameter_evidence}\n"
            f"----------------------------------------\n"
            f"[Repair Directive]\n"
            f"Constraint: Modify ONLY the operators within the Target Search Window. Do NOT alter preceding successful layers.\n"
            f"Action: {repair_action}"
        )
        return report

    def _diagnose_ast_conflict(self, src_prog, tgt_prog):
        """Using SymPy to traverse the abstract syntax tree and capture points of contention."""
        diagnosis = {
            "locus": "General Mathematical Divergence",
            "src_ast": "Complex Expression",
            "tgt_ast": "Complex Expression"
        }

        if not src_prog or not tgt_prog:
            return diagnosis

        def extract_denominator(expr):
            if isinstance(expr, (list, tuple)): expr = expr[0]
            if not isinstance(expr, sympy.Basic): return None

            for node in sympy.preorder_traversal(expr):
                if isinstance(node, sympy.Mul):
                    for arg in node.args:
                        if isinstance(arg, sympy.Pow) and arg.args[1] == -1:
                            return arg.args[0]
            return None

        src_denom = extract_denominator(src_prog.get('expr'))
        tgt_denom = extract_denominator(tgt_prog.get('expr'))

        if src_denom is not None and tgt_denom is not None:
            if str(src_denom) != str(tgt_denom):
                diagnosis["locus"] = "Division Denominator Node (Divisor Mismatch)"
                diagnosis["src_ast"] = str(src_denom)
                diagnosis["tgt_ast"] = str(tgt_denom)

        return diagnosis

    def _execute_source_block(self, src_block):
        local_cache = dict(self.gt_cache)
        input_vals = self._resolve_instruction_inputs(src_block[0], self.src, cache=local_cache)
        primary_input_val = input_vals[0] if input_vals else None
        if primary_input_val is None and src_block[0].iexpr.args:
            return None, None

        current_val = primary_input_val
        logical_output = None
        for inst in src_block:
            current_val = self._execute_source_op(inst, current_val, cache=local_cache)
            if current_val is None and inst is not src_block[0] and primary_input_val is not None:
                # Residual blocks may expose the preceding add as a synthetic
                # graph input while the source instruction still carries the
                # add node name.  The native module execution is unambiguous
                # once the already-computed boundary tensor is available.
                current_val = self._execute_single_instruction_native(
                    inst,
                    local_cache.get(inst.iexpr.args[0], primary_input_val),
                    self.src,
                    cache=local_cache,
                    input_values=[local_cache.get(inst.iexpr.args[0], primary_input_val)],
                )
            if current_val is None: return None, primary_input_val
            logical_output = current_val
            self._store_instruction_outputs(local_cache, inst, current_val)
            output_values = self._split_outputs(current_val)
            current_val = output_values[0]

        # Keep a tuple-valued final node (notably torch.chunk) available to
        # the next alignment window.  The numerical anchor above still uses
        # its primary branch, while following getitem nodes resolve the full
        # logical tuple from the shared cache.
        if src_block and len(src_block[-1].retvals()) == 1 and isinstance(logical_output, (list, tuple)):
            self.gt_cache[src_block[-1].retvals()[0]] = logical_output

        return current_val, primary_input_val

    def _get_symbolic_inventory_for_block(self, src_block):
        merged_inventory = {}
        for source_index, inst in enumerate(src_block):
            inv = self._get_symbolic_inventory(inst.iexpr.op, self.src, self.src_loader)
            op_key = self.src.adapter.parse_op_key(inst.iexpr.op)
            for symbol, value in inv.items():
                merged_inventory.setdefault(symbol, []).append({
                    "value": value,
                    "op_key": op_key,
                    "source_index": source_index,
                })
        return merged_inventory

    def _try_match_at_position(self, i, j) -> Optional[MatchResult]:
        src_insts = self.src.instructions
        tgt_insts = self.tgt.instructions
        self._last_formula_conflict = None
        self._last_failed_comparison = None

        def region_nodes(instruction):
            nodes = self._instruction_source_nodes(instruction)
            return nodes or tuple(instruction.retvals())

        # Cross-framework frontends legitimately use different node names.
        # When the first source/target operations have the same canonical
        # operator type, validate their 1x1 window directly; reserve the
        # larger search for genuine fusion/decomposition or replacement.
        source_key = self.src.adapter.parse_op_key(src_insts[i].iexpr.op)
        target_key = self.tgt.adapter.parse_op_key(tgt_insts[j].iexpr.op)
        same_operation = (
            self._canonical_operation_name(source_key)
            == self._canonical_operation_name(target_key)
        )
        target_layout_prefix = self._layout_prefix_length(tgt_insts, j)
        target_first_args = tuple(tgt_insts[j].iexpr.args)
        # Paddle tracing can inline a residual branch as a synthetic constant
        # input to the first convolution of an inverted-residual block.  The
        # source stream exposes the corresponding residual ``add`` explicitly.
        # Align the source add plus its following operator with that target
        # operator; parameter matching remains restricted to the real Conv2D
        # module and never treats the synthetic constant as a weight.
        source_add_prefix = bool(
            self._canonical_operation_name(source_key) == "add"
            and target_first_args
            and str(target_first_args[0]).casefold().startswith("const_")
        )
        structurally_identical = (
            region_nodes(src_insts[i]) == region_nodes(tgt_insts[j])
            or same_operation
        )
        max_source_window = 1 if structurally_identical else self.SEARCH_WINDOW_SRC
        if source_add_prefix:
            max_source_window = max(max_source_window, 2)
        max_target_window = 1 if structurally_identical else self.SEARCH_WINDOW_TGT
        # A framework layout conversion is part of the target semantic window.
        # Do not compare Conv2d with the leading Permute in isolation.
        # If the source and target anchors are themselves the same layout
        # operation (e.g. torch.transpose vs paddle.transpose in channel
        # shuffle), do not force the following reshape into the target window.
        # The prefix rule is only needed when a target-only layout conversion
        # precedes a genuinely different semantic operator.
        min_target_window = (
            target_layout_prefix + 1
            if target_layout_prefix and not same_operation
            else 1
        )
        max_target_window = max(max_target_window, min_target_window)

        if source_add_prefix and i + 1 < len(src_insts):
            # Specialized residual boundary alignment.  Paddle's tracer may
            # materialize the residual input as a synthetic const node and
            # omit the explicit add.  Validate the source add+operator window
            # against the target operator after binding that synthetic input
            # to the source add result.
            residual_block = src_insts[i:i + 2]
            residual_out, residual_input = self._execute_source_block(residual_block[:1])
            source_out, primary_input = self._execute_source_block(residual_block)
            if residual_out is not None and source_out is not None:
                self.gt_cache[target_first_args[0]] = residual_out
                target_block = tgt_insts[j:j + 1]
                src_inventory = self._get_symbolic_inventory_for_block(residual_block)
                tgt_reqs = self._get_symbolic_requirements(target_block, self.tgt, self.tgt_loader)
                backup = self._balance_and_inject(src_inventory, tgt_reqs)
                if backup is not None:
                    is_match, max_diff = self._verify_target_block(
                        source_out, target_block, residual_out
                    )
                    if is_match:
                        self._last_match_target_output = self._last_verified_target_output
                        self._last_match_source_output = source_out
                        self.alignment_records.append({
                            "source_index": i,
                            "target_index": j,
                            "source_size": 2,
                            "target_size": 1,
                            "source_regions": [self._instruction_region(x) for x in residual_block],
                            "target_regions": [self._instruction_region(x) for x in target_block],
                            "max_absolute_error": max_diff,
                            "layout_normalization": self._last_comparison_layout,
                            "target_evidence": False,
                            "residual_boundary_rebased": True,
                        })
                        return MatchResult("Match", residual_block, target_block, None)

        for m in range(1, max_source_window + 1):
            if i + m > len(src_insts):
                break
            if source_add_prefix and m == 1:
                # A target synthetic residual input can only correspond to
                # the composed source ``add -> first operator`` region; do
                # not accept the bare add as an anchor and report a false
                # operator mismatch.
                continue
            src_block = src_insts[i: i + m]

            src_ideal_out, primary_input_val = self._execute_source_block(src_block)
            if src_ideal_out is None:
                continue

            immutable_target = not getattr(self.tgt, 'supports_weight_injection', True)
            src_inventory = {} if immutable_target else self._get_symbolic_inventory_for_block(src_block)

            for n in range(min_target_window, max_target_window + 1):
                if j + n > len(tgt_insts):
                    break
                tgt_block = tgt_insts[j: j + n]

                if immutable_target:
                    backup = {}
                else:
                    tgt_reqs = self._get_symbolic_requirements(tgt_block, self.tgt, self.tgt_loader)
                    backup = self._balance_and_inject(src_inventory, tgt_reqs)

                if backup is None:
                    continue

                try:
                    if source_add_prefix and target_first_args:
                        residual_value, _ = self._execute_source_block(src_block[:1])
                        if residual_value is not None:
                            # Rebind the synthetic Paddle input to the source
                            # residual-add boundary for this local window.
                            self.gt_cache[target_first_args[0]] = residual_value
                    is_match, max_diff = self._verify_target_block(src_ideal_out, tgt_block, primary_input_val)
                    rebased_evidence = None
                    if not is_match:
                        rebased_evidence = self._try_isolated_target_equivalence(
                            src_block, tgt_block, src_ideal_out, max_diff
                        )
                        if rebased_evidence is not None:
                            is_match = True
                            max_diff = rebased_evidence["isolated_output_max_absolute_error"]
                            self._last_verified_target_output = rebased_evidence.pop("target_output")
                            self._last_comparison_layout = rebased_evidence["output_layout_normalization"]
                            rebased_evidence.update({"source_index": i, "target_index": j})
                            self.cumulative_numerical_drift_records.append(dict(rebased_evidence))
                    if is_match:
                        self._last_match_target_output = self._last_verified_target_output
                        alignment_record = {
                            "source_index": i,
                            "target_index": j,
                            "source_size": len(src_block),
                            "target_size": len(tgt_block),
                            "source_regions": [
                                self._instruction_region(inst) for inst in src_block
                            ],
                            "target_regions": [
                                self._instruction_region(inst) for inst in tgt_block
                            ],
                            "max_absolute_error": max_diff,
                            "layout_normalization": self._last_comparison_layout,
                            "target_evidence": immutable_target,
                        }
                        if rebased_evidence is not None:
                            alignment_record["input_rebased"] = True
                            alignment_record["accumulated_output_max_absolute_error"] = (
                                rebased_evidence["accumulated_output_max_absolute_error"]
                            )
                        self.alignment_records.append(alignment_record)
                        self._last_match_source_output = src_ideal_out
                        return MatchResult("Match", src_block, tgt_block, None)
                    formula_conflict = self._formula_conflict_for_blocks(src_block, tgt_block)
                    if formula_conflict is not None and self._last_formula_conflict is None:
                        self._last_formula_conflict = formula_conflict
                    if m == 1 and n >= min_target_window:
                        self._last_failed_comparison = {
                            "mismatch_max_absolute_error": max_diff,
                            "source_output_shape": list(getattr(src_ideal_out, 'shape', ())),
                        }
                except Exception as e:
                    self._log(f" -> Exception during verify: {e}")
                finally:
                    self._restore_weights(backup)
        return None

    @staticmethod
    def _adapt_aligned_input_layout(source_value, target_reference):
        source_array = np.asarray(source_value)
        target_array = np.asarray(target_reference)
        adapted = source_array
        layout = None

        if source_array.shape != target_array.shape and source_array.ndim == target_array.ndim == 4:
            if (
                source_array.shape[0] == target_array.shape[0]
                and source_array.shape[1] == target_array.shape[3]
                and source_array.shape[2] == target_array.shape[1]
                and source_array.shape[3] == target_array.shape[2]
            ):
                adapted = np.transpose(source_array, (0, 2, 3, 1))
                layout = "source_nchw_to_target_nhwc"
            elif (
                source_array.shape[0] == target_array.shape[0]
                and source_array.shape[1] == target_array.shape[2]
                and source_array.shape[2] == target_array.shape[3]
                and source_array.shape[3] == target_array.shape[1]
            ):
                adapted = np.transpose(source_array, (0, 3, 1, 2))
                layout = "source_nhwc_to_target_nchw"

        if adapted.shape != target_array.shape:
            return None, None
        if adapted.dtype != target_array.dtype and adapted.dtype.kind in "fc" and target_array.dtype.kind in "fc":
            adapted = adapted.astype(target_array.dtype, copy=False)
        return adapted, layout

    @staticmethod
    def _spatial_tuple(value, dimensions=2):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            if len(value) == dimensions:
                return tuple(int(item) for item in value)
            if len(value) == 1:
                return (int(value[0]),) * dimensions
            return None
        if isinstance(value, (int, np.integer)):
            return (int(value),) * dimensions
        return None

    @classmethod
    def _equivalent_conv2d_attributes(cls, source_attrs, target_attrs):
        source_kernel = cls._spatial_tuple(source_attrs.get("kernel_size"))
        target_kernel = cls._spatial_tuple(target_attrs.get("kernel_size"))
        source_stride = cls._spatial_tuple(source_attrs.get("stride", 1))
        target_stride = cls._spatial_tuple(target_attrs.get("strides", 1))
        source_dilation = cls._spatial_tuple(source_attrs.get("dilation", 1))
        target_dilation = cls._spatial_tuple(target_attrs.get("dilation_rate", 1))
        source_padding = cls._spatial_tuple(source_attrs.get("padding", 0))
        target_padding = target_attrs.get("padding", "valid")

        if None in (
            source_kernel, target_kernel, source_stride, target_stride,
            source_dilation, target_dilation, source_padding,
        ):
            return False
        if (
            source_kernel != target_kernel
            or source_stride != target_stride
            or source_dilation != target_dilation
            or int(source_attrs.get("groups", 1)) != int(target_attrs.get("groups", 1))
        ):
            return False

        if isinstance(target_padding, str):
            target_padding = target_padding.lower()
            if target_padding == "valid":
                padding_matches = source_padding == (0, 0)
            elif target_padding == "same" and source_stride == (1, 1):
                effective_kernel = tuple(
                    dilation * (kernel - 1) + 1
                    for kernel, dilation in zip(source_kernel, source_dilation)
                )
                padding_matches = all(size % 2 == 1 for size in effective_kernel) and source_padding == tuple(
                    size // 2 for size in effective_kernel
                )
            else:
                padding_matches = False
        else:
            padding_matches = source_padding == cls._spatial_tuple(target_padding)
        return padding_matches

    @staticmethod
    def _tensor_as_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        try:
            return np.asarray(value)
        except Exception:
            return None

    def _equivalent_operator_contract(self, src_inst, tgt_inst):
        source_op = src_inst.iexpr.op
        target_op = tgt_inst.iexpr.op
        source_name = self._canonical_operation_name(self.src.adapter.parse_op_key(source_op))
        target_name = self._canonical_operation_name(self.tgt.adapter.parse_op_key(target_op))
        if source_name != target_name or source_name != "conv2d":
            return False

        source_attrs = getattr(source_op, "attrs", {}) or {}
        target_attrs = getattr(target_op, "attrs", {}) or {}
        if not self._equivalent_conv2d_attributes(source_attrs, target_attrs):
            return False

        source_weight = self._tensor_as_numpy(self.src.adapter.get_named_tensor(source_op, "weight"))
        target_weight = self._tensor_as_numpy(self.tgt.adapter.get_named_tensor(target_op, "weight"))
        if source_weight is None or target_weight is None or source_weight.ndim != target_weight.ndim:
            return False
        if source_weight.shape != target_weight.shape:
            if source_weight.ndim == 4:
                source_weight = np.transpose(source_weight, (2, 3, 1, 0))
            elif source_weight.ndim == 2:
                source_weight = source_weight.T
        if source_weight.shape != target_weight.shape or not np.array_equal(source_weight, target_weight):
            return False

        source_bias = self._tensor_as_numpy(self.src.adapter.get_named_tensor(source_op, "bias"))
        target_bias = self._tensor_as_numpy(self.tgt.adapter.get_named_tensor(target_op, "bias"))
        return (source_bias is None and target_bias is None) or (
            source_bias is not None
            and target_bias is not None
            and source_bias.shape == target_bias.shape
            and np.array_equal(source_bias, target_bias)
        )

    def _try_isolated_target_equivalence(self, src_block, tgt_block, src_ideal_out, accumulated_diff):
        if getattr(self.tgt, 'evidence_provider', None) is not None:
            return None
        if not src_block or not tgt_block:
            return None
        src_inst = src_block[0]
        tgt_inst = tgt_block[0]
        source_inputs = self._resolve_instruction_inputs(src_inst, self.src)
        target_inputs = self._resolve_instruction_inputs(tgt_inst, self.tgt)
        if not source_inputs or len(source_inputs) != len(target_inputs):
            return None

        adapted_inputs = []
        input_layouts = []
        input_matches = []
        input_diffs = []
        for source_value, target_value in zip(source_inputs, target_inputs):
            adapted, layout = self._adapt_aligned_input_layout(source_value, target_value)
            if adapted is None:
                return None
            input_match, input_diff = self._compare_tensors(source_value, target_value)
            input_matches.append(input_match)
            input_diffs.append(input_diff)
            input_layouts.append(layout)
            adapted_inputs.append(adapted)

        local_cache = dict(self.gt_cache)
        for name, value in zip(tgt_inst.iexpr.args, adapted_inputs):
            local_cache[name] = value
        target_out = None
        for instruction in tgt_block:
            instruction_inputs = self._resolve_instruction_inputs(
                instruction, self.tgt, cache=local_cache
            )
            if not instruction_inputs:
                return None
            target_out = self._execute_single_instruction_native(
                instruction,
                instruction_inputs[0],
                self.tgt,
                cache=local_cache,
                input_values=instruction_inputs,
            )
            if target_out is None:
                return None
            self._store_instruction_outputs(local_cache, instruction, target_out)
        matched, isolated_diff = self._compare_tensors(src_ideal_out, target_out)
        output_layout = self._last_comparison_layout
        contract_verified = (
            len(src_block) == 1
            and len(tgt_block) == 1
            and self._equivalent_operator_contract(src_inst, tgt_inst)
        )
        if not matched and not contract_verified:
            return None

        cache_output = target_out
        if not matched:
            cache_output, _ = self._adapt_aligned_input_layout(src_ideal_out, target_out)
            if cache_output is None:
                return None

        return {
            "source_operation": " -> ".join(
                self.src.adapter.parse_op_key(item.iexpr.op) for item in src_block
            ),
            "target_operation": " -> ".join(
                self.tgt.adapter.parse_op_key(item.iexpr.op) for item in tgt_block
            ),
            "source_regions": [self._instruction_region(item) for item in src_block],
            "target_regions": [self._instruction_region(item) for item in tgt_block],
            "source_window_size": len(src_block),
            "target_window_size": len(tgt_block),
            "upstream_inputs_within_tolerance": all(input_matches),
            "upstream_input_max_absolute_error": max(input_diffs, default=0.0),
            "input_layout_normalizations": [item for item in input_layouts if item],
            "accumulated_output_max_absolute_error": accumulated_diff,
            "isolated_output_max_absolute_error": isolated_diff,
            "isolated_output_within_tolerance": matched,
            "operator_contract_verified": contract_verified,
            "output_layout_normalization": output_layout,
            "target_output": cache_output,
        }

    def _execute_source_op(self, src_op_inst, input_val, cache=None):
        src_op = src_op_inst.iexpr.op
        src_op_key = self.src.adapter.parse_op_key(src_op)
        input_values = self._resolve_instruction_inputs(src_op_inst, self.src, input_val, cache)
        if getattr(src_op, 'attrs', {}).get('force_native_region'):
            result = self._execute_single_instruction_native(
                src_op_inst, input_val, self.src, cache=cache, input_values=input_values
            )
            self.source_execution_records.append({
                "op": src_op_key,
                "mode": getattr(src_op, 'attrs', {}).get('force_native_mode') or "native_fused_region",
                "reason": getattr(src_op, 'attrs', {}).get('force_native_reason', ''),
            })
            return result
        result = self._execute_single_instruction_native(
            src_op_inst, input_val, self.src, cache=cache, input_values=input_values
        )
        audit_record = self._formula_audit_record(
            'source', src_op_inst, self.src
        )
        audit_status = audit_record.get('status') if audit_record else None
        self.source_execution_records.append({
            "op": src_op_key,
            "mode": (
                "native_with_validated_mathir"
                if audit_status in {'VALIDATED', 'AUXILIARY_VALUE_BOUND'}
                else "native_with_unavailable_mathir"
            ),
            "formula_audit_status": audit_status or "NOT_AUDITED",
        })
        return result

    def _verify_target_block(self, src_ideal_out, tgt_block, input_val):
        evidence_provider = getattr(self.tgt, 'evidence_provider', None)
        if evidence_provider is not None:
            target_out = evidence_provider.get_block_output(tgt_block)
            self._last_verified_target_output = target_out
            return self._compare_tensors(src_ideal_out, target_out)

        local_cache = dict(self.gt_cache)
        current_val = input_val
        for inst in tgt_block:
            op_key = self.tgt.adapter.parse_op_key(inst.iexpr.op)
            input_values = self._resolve_instruction_inputs(inst, self.tgt, current_val, local_cache)
            audit_record = self._formula_audit_record('target', inst, self.tgt)
            if audit_record is None:
                self._record_formula_unavailable(
                    "target", inst, self.tgt, "formula_not_audited"
                )
            current_val = self._execute_single_instruction_native(
                inst, current_val, self.tgt, cache=local_cache, input_values=input_values
            )
            if current_val is None: return False, 0.0
            self._store_instruction_outputs(local_cache, inst, current_val)
            output_values = self._split_outputs(current_val)
            current_val = output_values[0]

        self._last_verified_target_output = current_val
        return self._compare_tensors(src_ideal_out, current_val)

    def _get_symbolic_inventory(self, op, symnet, loader) -> Dict[str, np.ndarray]:
        inventory = {}
        op_key = symnet.adapter.parse_op_key(op)
        try:
            op_def = loader.get_op_def(op_key)
        except NotImplementedError:
            return inventory

        if not op_def:
            return inventory

        param_map = op_def.get('param_map', {})
        TENSOR_SYMBOLS = {'W', 'B', 'running_mean', 'running_var'}

        if self.enable_debug:
            print(f"\n[Inventory Debug] Extracting for Op: {op_key} ({type(op).__name__})")

        for attr_name, math_symbol in param_map.items():
            syms = math_symbol if isinstance(math_symbol, list) else [math_symbol]
            for sym in syms:
                if sym in TENSOR_SYMBOLS: continue

                val = None
                if hasattr(symnet.adapter, 'get_py_value'):
                    val = symnet.adapter.get_py_value(op, attr_name)

                if val is None:
                    val = getattr(op, attr_name, None)

                if val is not None and not hasattr(val, 'shape'):
                    try:
                        if isinstance(val, (list, tuple)):
                            inventory[sym] = np.array(val)
                        else:
                            inventory[sym] = np.array(val)

                        if self.enable_debug:
                            print(f"  -> Found Hyperparam '{attr_name}' (Symbol: {sym})")
                    except:
                        pass

        tensor_attr_map = {
            'W': 'weight', 'B': 'bias',
            'running_mean': 'running_mean', 'running_var': 'running_var'
        }
        for attr_name, math_symbol in param_map.items():
            syms = math_symbol if isinstance(math_symbol, list) else [math_symbol]
            for sym in syms:
                if sym in TENSOR_SYMBOLS:
                    tensor_attr_map[sym] = attr_name

        for sym, attr_name in tensor_attr_map.items():
            val = None
            source = "None"

            if hasattr(symnet.adapter, 'get_named_tensor'):
                val = symnet.adapter.get_named_tensor(op, attr_name)
                if val is not None: source = "Adapter"

            if val is None:
                if hasattr(op, '_parameters') and attr_name in op._parameters:
                    val = op._parameters[attr_name]
                    source = "_parameters"
                elif hasattr(op, '_buffers') and attr_name in op._buffers:
                    val = op._buffers[attr_name]
                    source = "_buffers"

            if val is None:
                raw_val = getattr(op, attr_name, None)
                if hasattr(raw_val, 'shape') or hasattr(raw_val, 'data_ptr'):
                    val = raw_val
                    source = "getattr(Tensor)"

            if val is not None:
                try:
                    np_val = self._extract_primary_numpy(symnet.to_numpy(val))
                    inventory[sym] = np_val
                    if self.enable_debug:
                        print(f"  -> Found Tensor '{attr_name}' (Symbol: {sym}) [Source: {source}]")
                except Exception as e:
                    print(f"  -> [Error] Failed to convert Tensor {attr_name}: {e}")

        if self.enable_debug:
            found_keys = list(inventory.keys())
            print(f"  -> Final Inventory Keys: {found_keys}")

        return inventory

    def _get_symbolic_requirements(self, tgt_block, symnet, loader) -> Dict[str, List[Tuple[Any, str]]]:
        requirements = {}
        covered_ids = set()

        TENSOR_SYMBOLS = {'W', 'B', 'running_mean', 'running_var'}

        for inst in tgt_block:
            op = inst.iexpr.op

            real_obj = getattr(op, 'module', op)
            if real_obj is None: real_obj = op

            op_key = symnet.adapter.parse_op_key(op)
            try:
                op_def = loader.get_op_def(op_key)
            except NotImplementedError:
                continue

            if not op_def:
                variants = {
                    op_key.capitalize(),  # Conv2d
                    op_key.upper(),  # CONV2D
                    op_key.lower(),  # conv2d
                    op_key[:-2] + op_key[-2:].upper() if len(op_key) > 2 else op_key  # Suffix Uppercase (Conv2D)
                }

                for cand in variants:
                    if cand == op_key: continue
                    op_def = loader.get_op_def(cand)
                    if op_def:
                        if self.enable_debug:
                            print(f"  > [FuzzyMatch] Mapped '{op_key}' -> '{cand}'")
                        break

            if not op_def: continue

            candidates = [
                ('weight', 'W'),
                ('bias', 'B'),
                ('running_mean', 'running_mean'),
                ('running_var', 'running_var')
            ]

            param_map = op_def.get('param_map', {})
            for attr, sym_or_list in param_map.items():
                syms = sym_or_list if isinstance(sym_or_list, list) else [sym_or_list]
                for sym in syms:
                    candidates.append((attr, sym))
            # Paddle Linear exposes the boolean constructor flag as
            # ``bias_attr`` but stores the actual trainable tensor in
            # ``bias``. The YAML formula names the flag ``ba`` and therefore
            # does not, by itself, create a B parameter requirement. Add the
            # explicit tensor alias so source PyTorch bias values are injected
            # during cross-framework differential testing.
            if str(op_key).casefold() == "linear":
                candidates.append(("bias", "B"))

            for attr_name, sym in candidates:
                if sym not in TENSOR_SYMBOLS: continue

                has_it = False
                if hasattr(symnet.adapter, 'has_attr'):
                    has_it = symnet.adapter.has_attr(real_obj, attr_name)
                else:
                    has_it = hasattr(real_obj, attr_name) and getattr(real_obj, attr_name) is not None

                if has_it:
                    if sym not in requirements: requirements[sym] = []

                    is_duplicate = any((obj is real_obj and attr == attr_name) for obj, attr in requirements[sym])

                    if not is_duplicate:
                        requirements[sym].append((real_obj, attr_name))

                        if hasattr(symnet.adapter, 'get_named_tensor'):
                            val = symnet.adapter.get_named_tensor(real_obj, attr_name)
                            if val is not None: covered_ids.add(id(val))

        for inst in tgt_block:
            for input_name in inst.iexpr.args:
                if hasattr(symnet, 'static_param_map'):
                    val = symnet.static_param_map.get(input_name)
                    if val is not None and id(val) not in covered_ids:
                        # Static shape/axis values are operator metadata, not
                        # trainable parameters.  Treating them as B/W slots
                        # makes the default-fill branch overwrite a target
                        # reshape with zeros during cross-framework injection.
                        # Preserve these frontend constants verbatim.
                        metadata_name = str(input_name).casefold()
                        if metadata_name in {
                            'shape', 'target_shape', 'axis', 'axes', 'starts',
                            'ends', 'strides', 'size', 'sizes', 'dim',
                            'output_size', 'kernel_size', 'padding', 'stride',
                        }:
                            continue
                        # Frontend tracing uses ``const_*`` names for captured
                        # runtime tensor values (not trainable parameters),
                        # notably the residual input of MNASNet blocks.  Do
                        # not reinterpret those values as a W/B injection
                        # requirement; the alignment window binds them from
                        # the source computation boundary.
                        if metadata_name.startswith('const_'):
                            continue
                        sym = 'B' if len(val.shape) == 1 else 'W'
                        if sym not in requirements: requirements[sym] = []
                        if not any(attr == input_name for _, attr in requirements[sym]):
                            requirements[sym].append(("STATIC_PARAM", input_name))

        return requirements

    def _safe_assign(self, symnet, target_tensor, value):
        if hasattr(symnet, 'inject_value'):
            return symnet.inject_value(target_tensor, value)

        elif hasattr(symnet, 'adapter') and hasattr(symnet.adapter, 'inject_value'):
            return symnet.adapter.inject_value(target_tensor, value)

        else:
            print("[Error] SymNet lacking 'inject_value' method implementation.")
            return False

    def _record_parameter_alignment(self, record):
        identity = (
            record.get('symbol'),
            record.get('source_operation'),
            record.get('target_operation'),
            record.get('target_attribute'),
            tuple(record.get('source_shape', ())),
            tuple(record.get('target_shape', ())),
            record.get('relation'),
        )
        for existing in self.parameter_alignment_records:
            existing_identity = (
                existing.get('symbol'),
                existing.get('source_operation'),
                existing.get('target_operation'),
                existing.get('target_attribute'),
                tuple(existing.get('source_shape', ())),
                tuple(existing.get('target_shape', ())),
                existing.get('relation'),
            )
            if existing_identity == identity:
                return
        self.parameter_alignment_records.append(record)

    def _balance_and_inject(self, src_inv, tgt_reqs) -> Optional[Dict]:
        """Balance and inject compatible values into target layer slots."""
        print(f"\n[Balance] Start Injecting... (Reqs: {len(tgt_reqs)} symbols)")
        backup = {}
        consumed_entries = {symbol: set() for symbol in src_inv}
        import time

        for sym, tgt_slots in tgt_reqs.items():
            print(f"  > [Sym: {sym}] Processing...", flush=True)

            if sym in src_inv:
                raw_entries = src_inv[sym]
                entries = raw_entries if isinstance(raw_entries, list) else [
                    {"value": raw_entries, "op_key": "", "source_index": 0}
                ]
                for t_op, t_attr in tgt_slots:
                    print(f"    |-- Target Slot: {t_attr} (Op: {type(t_op).__name__})", end="", flush=True)
                    target_tensor = None

                    # ---------------------------------------------------------
                    # [Step A] Adapter lookup.
                    # ---------------------------------------------------------
                    if t_op == "STATIC_PARAM":
                        param_name = t_attr
                        if hasattr(self.tgt, 'static_param_map') and param_name in self.tgt.static_param_map:
                            target_tensor = self.tgt.static_param_map[param_name]
                    else:
                        target_tensor = self.tgt.adapter.get_named_tensor(t_op, t_attr)

                    # ---------------------------------------------------------
                    # [Step B] Native layer fallback.
                    # ---------------------------------------------------------
                    if target_tensor is None:
                        if hasattr(t_op, '_parameters') and t_attr in t_op._parameters:
                            target_tensor = t_op._parameters[t_attr]
                            # print(f" [Debug] Found in _parameters!", end="")

                        elif hasattr(t_op, '_buffers') and t_attr in t_op._buffers:
                            target_tensor = t_op._buffers[t_attr]
                            # print(f" [Debug] Found in _buffers!", end="")

                        elif hasattr(t_op, t_attr):
                            val = getattr(t_op, t_attr)
                            if hasattr(val, 'shape'):
                                target_tensor = val
                                # print(f" [Debug] Found via getattr!", end="")

                    if target_tensor is None:
                        print(" -> [Error] Target Tensor NOT Found (Adapter & Fallback failed)!")
                        return None

                    t_shape = getattr(target_tensor, 'shape', 'Unknown')
                    t_device = getattr(target_tensor, 'device', 'Unknown')
                    print(f" -> Found (Shape: {t_shape}, Device: {t_device})")

                    target_shape = tuple(int(dim) for dim in t_shape)
                    # Frameworks do not agree on the storage layout of a
                    # fully-connected weight.  PyTorch stores Linear.weight
                    # as [out_features, in_features], while Paddle stores
                    # Linear.weight as [in_features, out_features].  Treat a
                    # 2-D transpose as a compatible parameter candidate here
                    # (the actual transpose is applied below before
                    # injection), instead of dropping the slot and allowing
                    # the validator to compare against random target
                    # parameters.
                    shape_matches = [
                        (entry_index, entry)
                        for entry_index, entry in enumerate(entries)
                        if tuple(int(dim) for dim in np.asarray(entry["value"]).shape) == target_shape
                        or (
                            np.asarray(entry["value"]).ndim == 2
                            and tuple(int(dim) for dim in np.asarray(entry["value"]).T.shape) == target_shape
                        )
                    ]
                    if not shape_matches:
                        target_operation = type(t_op).__name__
                        for entry in entries:
                            source_value = np.asarray(entry["value"])
                            source_shape = tuple(int(dim) for dim in source_value.shape)
                            relation = "incompatible_shape"
                            if source_value.ndim == 2 and tuple(source_value.T.shape) == target_shape:
                                relation = "transpose_2d"
                            self._record_parameter_alignment({
                                "symbol": sym,
                                "source_operation": str(entry.get("op_key", "unknown")),
                                "target_operation": target_operation,
                                "target_attribute": t_attr,
                                "source_shape": list(source_shape),
                                "target_shape": list(target_shape),
                                "relation": relation,
                            })
                        print("      [Skip] No source parameter with a compatible shape")
                        continue

                    target_key = type(t_op).__name__.lower()
                    semantic_matches = [
                        item for item in shape_matches
                        if str(item[1].get("op_key", "")).lower() in target_key
                        or target_key in str(item[1].get("op_key", "")).lower()
                    ]
                    available = [
                        item for item in (semantic_matches or shape_matches)
                        if item[0] not in consumed_entries[sym]
                    ]
                    entry_index, entry = (available or semantic_matches or shape_matches)[0]
                    consumed_entries[sym].add(entry_index)
                    src_val = np.asarray(entry["value"])
                    print(
                        f"      [Source] {entry.get('op_key', 'unknown')} "
                        f"shape={src_val.shape} dtype={src_val.dtype}",
                        flush=True,
                    )

                    # Candidate target models normally receive the source
                    # parameters during construction.  Avoid copying large
                    # convolution tensors when the values already match;
                    # injection is still performed whenever the target is
                    # genuinely different, preserving diagnostic semantics.
                    try:
                        current_numpy = self._extract_primary_numpy(
                            self.tgt.to_numpy(target_tensor)
                        )
                        if np.array_equal(np.asarray(current_numpy), src_val):
                            print("      [Skip] Target parameter already matches source", flush=True)
                            continue
                    except Exception:
                        current_numpy = None

                    t0 = time.time()
                    print(f"      [Step 1] Backing up...", end="", flush=True)
                    try:
                        np_backup = self.tgt.to_numpy(target_tensor)
                        backup_key = (t_op, t_attr)
                        backup[backup_key] = self._extract_primary_numpy(np_backup)
                        print(f" Done ({time.time() - t0:.4f}s)")
                    except Exception as e:
                        print(f" Failed! Error: {e}")
                        return None

                    injection_value = src_val
                    relation = None
                    target_shape = tuple(int(dim) for dim in getattr(target_tensor, "shape", ()))
                    source_shape = tuple(int(dim) for dim in np.asarray(src_val).shape)
                    source_key = str(entry.get("op_key", "")).casefold()
                    target_key = type(t_op).__name__.casefold()
                    # PyTorch Linear uses [out, in], whereas Paddle Linear
                    # uses [in, out].  Shape equality is not sufficient to
                    # detect this conversion: square matrices (the common
                    # hidden layers in AlexNet/MLPs) have identical shapes
                    # even though their semantic orientation is reversed.
                    linear_layout_conversion = (
                        sym == "W"
                        and source_key == "linear"
                        and "linear" in target_key
                        and str(getattr(self.src, "framework_type", "")).casefold() in {"pytorch", "torch"}
                        and str(getattr(self.tgt, "framework_type", "")).casefold() == "paddle"
                    )
                    if np.asarray(src_val).ndim == 2 and (
                        linear_layout_conversion
                        or source_shape != target_shape
                    ):
                        if tuple(np.asarray(src_val).T.shape) == target_shape:
                            injection_value = np.asarray(src_val).T
                            relation = "transpose_2d"
                    if relation:
                        self._record_parameter_alignment({
                            "symbol": sym,
                            "source_operation": "parameter_injection",
                            "target_operation": type(t_op).__name__ if t_op != "STATIC_PARAM" else "STATIC_PARAM",
                            "target_attribute": t_attr,
                            "source_shape": list(source_shape),
                            "target_shape": list(target_shape),
                            "relation": relation,
                        })
                    t1 = time.time()
                    print(f"      [Step 2] Injecting src_val" + (f" ({relation})" if relation else "") + "...", end="", flush=True)
                    try:
                        success = self._safe_assign(self.tgt, target_tensor, injection_value)
                        print(f" Done ({time.time() - t1:.4f}s)")
                    except Exception as e:
                        print(f" Failed! Error: {e}")
                        return None

                    if not success:
                        print("      [Error] _safe_assign returned False")
                        return None

            else:
                val_to_inject = 0.0 if sym in ['B', 'bias'] else 1.0
                if sym not in ['B', 'bias', 'Gamma', 'scale']:
                    return None

                print(f"    |-- Default Fill: {val_to_inject} (Missing in Source)", flush=True)

                for t_op, t_attr in tgt_slots:
                    # ---------------------------------------------------------
                    # [Step A] Adapter lookup.
                    # ---------------------------------------------------------
                    target_tensor = None
                    if t_op == "STATIC_PARAM":
                        param_name = t_attr
                        if hasattr(self.tgt, 'static_param_map') and param_name in self.tgt.static_param_map:
                            target_tensor = self.tgt.static_param_map[param_name]
                    else:
                        target_tensor = self.tgt.adapter.get_named_tensor(t_op, t_attr)

                    # ---------------------------------------------------------
                    # [Step B] Native layer fallback.
                    # ---------------------------------------------------------
                    if target_tensor is None:
                        if hasattr(t_op, '_parameters') and t_attr in t_op._parameters:
                            target_tensor = t_op._parameters[t_attr]
                        elif hasattr(t_op, '_buffers') and t_attr in t_op._buffers:
                            target_tensor = t_op._buffers[t_attr]
                        elif hasattr(t_op, t_attr):
                            val = getattr(t_op, t_attr)
                            if hasattr(val, 'shape'):
                                target_tensor = val

                    if target_tensor is None: return None

                    print(f"      [Else-Branch] Backing up {t_attr}...", end="", flush=True)
                    try:
                        backup_key = (t_op, t_attr)
                        current_numpy = self._extract_primary_numpy(self.tgt.to_numpy(target_tensor))
                        default_array = np.full(target_tensor.shape, val_to_inject, dtype='float32')
                        if np.allclose(np.asarray(current_numpy), default_array, rtol=0.0, atol=0.0):
                            print(" Already has the default value.")
                            continue
                        backup[backup_key] = current_numpy
                        print(" Done.")
                    except Exception as e:
                        print(f" Failed: {e}")
                        return None

                    t2 = time.time()
                    target_shape = target_tensor.shape
                    print(f"      [Else-Branch] Allocating np.full {target_shape}...", end="", flush=True)
                    try:
                        zeros_or_ones = np.full(target_shape, val_to_inject, dtype='float32')
                        print(f" Done ({time.time() - t2:.4f}s)")
                    except Exception as e:
                        print(f" Alloc Failed: {e}")
                        return None

                    t3 = time.time()
                    print(f"      [Else-Branch] Injecting...", end="", flush=True)
                    try:
                        self._safe_assign(self.tgt, target_tensor, zeros_or_ones)
                        print(f" Done ({time.time() - t3:.4f}s).")
                    except Exception as e:
                        print(f" Inject Failed: {e}")
                        return None

        print("<<< [Debug] End _balance_and_inject successfully")
        return backup

    def _restore_weights(self, backup):
        """Restore target weights from a backup map."""
        if not backup: return
        for (t_op, t_attr), original_numpy_val in backup.items():
            target_tensor = None
            if t_op == "STATIC_PARAM":
                param_name = t_attr
                if param_name in self.tgt.static_param_map:
                    target_tensor = self.tgt.static_param_map[param_name]
            else:
                target_tensor = self.tgt.adapter.get_named_tensor(t_op, t_attr)

            if target_tensor is not None:
                self._safe_assign(self.tgt, target_tensor, original_numpy_val)

    def _execute_single_instruction_native(self, inst, input_val, symnet, cache=None, input_values=None):
        evidence_provider = getattr(symnet, 'evidence_provider', None)
        if evidence_provider is not None:
            return evidence_provider.get_output(inst)

        runner = inst.runner

        args = []
        values = input_values
        if values is None:
            values = self._resolve_instruction_inputs(inst, symnet, input_val, cache)
        raw_op_name = str(getattr(inst.iexpr.op, "op_type", "")).casefold()
        if raw_op_name == "view" and values:
            # Torch's ``Tensor.view`` is traced with the shape dimensions as
            # separate scalar inputs, while Paddle emits one Reshape node
            # carrying a shape list.  Execute the source view directly in
            # NumPy so both frontends enter the same canonical reshape region.
            try:
                shape = tuple(int(np.asarray(item).reshape(-1)[0]) for item in values[1:])
                return np.asarray(values[0]).reshape(shape)
            except (TypeError, ValueError, IndexError):
                pass
        for val in values:
            tensor_val = symnet.to_tensor(val)
            args.append(tensor_val)

        try:
            out = runner(*args)
            numpy_output = symnet.to_numpy(out)
            # Python-level split/chunk operations often have one graph
            # variable whose runtime value is a tuple/list of tensors.  The
            # previous primary-output extraction silently kept only branch 0,
            # so a following ``getitem`` could not select branch 1.  Preserve
            # all outputs for a single logical tuple-valued graph result;
            # ordinary scalar/tensor operators keep the old path.
            if (
                len(inst.retvals()) == 1
                and isinstance(numpy_output, dict)
                and len(numpy_output) > 1
            ):
                ordered = []
                output_keys = [
                    key for key in numpy_output
                    if str(key).startswith("output_")
                ]
                if len(output_keys) == len(numpy_output):
                    def output_index(key):
                        try:
                            return int(str(key).rsplit("_", 1)[-1])
                        except (TypeError, ValueError):
                            return 0
                    for key in sorted(output_keys, key=output_index):
                        ordered.append(numpy_output[key])
                    return tuple(ordered)
            return self._extract_primary_numpy(numpy_output)
        except Exception as exc:
            # Retrying a failed multi-input op with only its first input changes
            # graph semantics (for example, Concat([x, y]) becomes Concat([x])).
            # Keep the legacy single-input fallback only for runners whose call
            # convention genuinely accepts one tensor.
            if len(args) > 1:
                self._log(
                    f"Native execution failed for multi-input op "
                    f"{symnet.adapter.parse_op_key(inst.iexpr.op)}: {exc}"
                )
                return None
            try:
                #  input
                out = runner(symnet.to_tensor(input_val))
                return self._extract_primary_numpy(symnet.to_numpy(out))
            except Exception:
                return None

    @staticmethod
    def _restore_aligned_target_dtype(source_value, target_value):
        """Prevent NumPy's float64 defaults from contaminating graph caches."""
        source_array = np.asarray(source_value)
        target_array = np.asarray(target_value)
        if (
            source_array.dtype != target_array.dtype
            and source_array.dtype.kind in "fc"
            and target_array.dtype.kind in "fc"
        ):
            return target_array.astype(source_array.dtype, copy=False)
        return target_value

    def _update_gt_cache(self, match_res: MatchResult):
        source_out = self._last_match_source_output
        target_out = self._last_match_target_output
        self._last_match_source_output = None
        self._last_match_target_output = None
        if source_out is None:
            source_out, _ = self._execute_source_block(match_res.src_block)

        if source_out is None:
            return
        if target_out is None:
            target_out = source_out
        source_values = self._split_outputs(source_out)
        target_values = self._split_outputs(target_out)
        target_values = [
            self._restore_aligned_target_dtype(source_value, target_value)
            for source_value, target_value in zip(source_values, target_values)
        ]
        source_out_names = match_res.src_block[-1].retvals()
        target_out_names = match_res.tgt_block[-1].retvals()
        # _execute_source_block may already have preserved a tuple-valued
        # source node (for example chunk) in the shared cache.  Do not replace
        # it with only the primary branch when publishing the match result.
        if not (
            len(source_out_names) == 1
            and isinstance(self.gt_cache.get(source_out_names[0]), (list, tuple))
        ):
            for out_name, out_value in zip(source_out_names, source_values):
                self.gt_cache[out_name] = out_value
        if len(target_out_names) == 1:
            target_payload = (
                tuple(target_values)
                if len(target_values) > 1
                else (target_values[0] if target_values else None)
            )
        else:
            target_payload = target_values
        self._store_instruction_outputs(
            self.gt_cache, match_res.tgt_block[-1], target_payload
        )

    def _compare_tensors(self, ideal, real):
        self._last_comparison_layout = None
        if ideal is None or real is None: return False, -1.0
        if not isinstance(ideal, np.ndarray): ideal = np.array(ideal)
        if not isinstance(real, np.ndarray): real = np.array(real)

        if ideal.dtype.kind == 'f': ideal = ideal.astype(np.float32)
        if real.dtype.kind == 'f': real = real.astype(np.float32)

        if ideal.shape != real.shape:
            if (
                ideal.ndim == 4
                and real.ndim == 4
                and real.shape == (ideal.shape[0], ideal.shape[2], ideal.shape[3], ideal.shape[1])
            ):
                real = np.transpose(real, (0, 3, 1, 2))
                self._last_comparison_layout = "target_nhwc_to_source_nchw"
            elif (
                ideal.ndim == 3
                and real.ndim == 3
                and real.shape == (ideal.shape[0], ideal.shape[2], ideal.shape[1])
            ):
                real = np.transpose(real, (0, 2, 1))
                self._last_comparison_layout = "target_nlc_to_source_ncl"
            else:
                return False, -1.0

        ideal_nan = np.isnan(ideal)
        real_nan = np.isnan(real)
        if np.any(ideal_nan) or np.any(real_nan):
            if not np.array_equal(ideal_nan, real_nan): return False, -1.0
            ideal = ideal[~ideal_nan]
            real = real[~real_nan]

        if ideal.dtype.kind == 'b' or real.dtype.kind == 'b':
            diff = np.not_equal(ideal, real).astype(np.float32)
        else:
            diff = np.abs(ideal - real)
        max_diff = float(diff.max()) if diff.size > 0 else 0.0

        is_close = np.allclose(ideal, real, rtol=self.rtol, atol=self.atol)

        if not is_close and self.enable_debug:
            print(f"   [Mismatch Stats] Shape: {ideal.shape}")
            if ideal.size > 0 and real.size > 0:
                print(f"     > Ideal (Src): min={ideal.min():.4f}, max={ideal.max():.4f}, mean={ideal.mean():.4f}")
                print(f"     > Real  (Tgt): min={real.min():.4f}, max={real.max():.4f}, mean={real.mean():.4f}")
                print(f"     > Max Diff: {max_diff:.4f}")

        return is_close, max_diff

    def _record_gap(self, match_res):
        gap = match_res.gap_info
        msg = f"[Structure Diff] Gap detected (Src+{gap[0]}, Tgt+{gap[1]}). Block skipped."
        self.issues.append(msg)
