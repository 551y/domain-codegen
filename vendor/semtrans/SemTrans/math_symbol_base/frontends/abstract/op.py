# Adapted from NNSmith and modified for SemTrans; see THIRD_PARTY_NOTICES.md.

import random
from abc import ABC, abstractmethod
from functools import reduce
from inspect import signature
from typing import Any, Dict, List, Tuple, Type, Callable, Optional

from SemTrans.math_symbol_base.frontends.abstract.arith import *
from SemTrans.math_symbol_base.frontends.abstract.dtype import (
    DTYPE_GEN_ALL,
    DType,
)
from SemTrans.math_symbol_base.frontends.abstract.extension import ACTIVATED_PATCH
from SemTrans.math_symbol_base.frontends.abstract.tensor import AbsTensor
from SemTrans.math_symbol_base.frontends.abstract.error import ConstraintCheck, SanityCheck

__MIN_RANK__ = 0
__MAX_RANK__ = 5

FULL_OPERATOR_SETS: Dict[str, List[Type["AbsOpBase"]]] = dict()

_PRAGMA_ONCE_CORE_OP = True  # have to use this dirty hack to avoid namespace violation.


class mark_abstract:
    def __init__(self, dialect):
        assert (
            dialect != "core" or _PRAGMA_ONCE_CORE_OP
        ), "`core` is exclusive to nnsmith.abstract.op. Please use other dilect names"
        self.dialect = dialect

    def __call__(self, op_type: Type["AbsOpBase"]) -> Type["AbsOpBase"]:
        op_type.dialect = self.dialect
        return op_type


class mark_materialize:
    def __init__(self, dialect: str, limit_domain=False):
        self.limit_domain = limit_domain
        self.dialect = dialect

    def __call__(self, op_type: Type["AbsOpBase"]) -> Type["AbsOpBase"]:
        op_list = FULL_OPERATOR_SETS.setdefault(self.dialect, [])

        if op_type not in op_list:
            op_type = mark_abstract(self.dialect)(op_type)
            op_type.limit_domain = self.limit_domain
            op_list.append(op_type)
            op_list.sort(key=lambda x: x.__name__)

        return op_type


class Placeholder:
    def __init__(self, ttype: AbsTensor):
        self.ttype = ttype
        self.inp_ranks = []
        self.out_ranks = [(ttype.ndims,)]
        self.input_like = []
        self.output_like = [ttype]

    @staticmethod
    def n_input():
        return 0

    @staticmethod
    def n_output():
        return 1

    def __repr__(self):
        return f"Placeholder({self.ttype})"

    def const(self):
        const_node = Constant(self.ttype.ndims)
        const_node.abs_tensor = self.ttype
        return const_node

    def input(self):
        input_node = Input(self.ttype.ndims)
        input_node.abs_tensor = self.ttype
        return input_node

    def __str__(self):
        return "Placeholder"


def rank_from(start):
    return tuple(range(start, __MAX_RANK__ + 1))


def rank_range(start, end):
    return tuple(range(start, end + 1))


def rank_until(end):
    return tuple(range(__MIN_RANK__, end + 1))


def rank_all():
    return tuple(range(__MIN_RANK__, __MAX_RANK__ + 1))


def check_shape_fn(func):
    def wrapper_check_shape_fn(self, input_shapes):
        SanityCheck.true(
            self.out_ranks,
            "Empty output dimensions in {}".format(self.__class__.__name__),
        )
        SanityCheck.eq(
            len(input_shapes),
            self.n_input(),
            "{} requires {} inputs, but got {}".format(
                self.__class__.__name__, self.n_input(), len(input_shapes)
            ),
        )
        res = func(self, [s.deepcopy() for s in input_shapes])
        SanityCheck.eq(
            len(res),
            self.n_output(),
            "{} requires {} outputs, but got {}".format(
                self.__class__.__name__, self.n_output(), len(res)
            ),
        )
        return res

    return wrapper_check_shape_fn


def check_require_fn(func):
    def wrapper_check_require_fn(self, input_shapes: List[AbsTensor]):
        SanityCheck.eq(
            len(input_shapes),
            self.n_input(),
            "{} requires {} inputs, but got {}".format(
                self.__class__.__name__, self.n_input(), len(input_shapes)
            ),
        )
        return func(self, [s.deepcopy() for s in input_shapes])

    return wrapper_check_require_fn


def _prepend_to(x, max_dim):
    return [1 for _ in range(max_dim - len(x))] + x


def z3_bcast(
    x: Union[int, z3.ExprRef], y: Union[int, z3.ExprRef], *args: Union[int, z3.ExprRef]
):
    x, y = align_bvs(x, y)
    return (
        z3.simplify(z3.If(tensor_eq(y, 1), x, y))
        if len(args) == 0
        else z3_bcast(z3_bcast(x, y), *args)
    )


def int_bcast(x: int, *args: int):
    return x if (x != 1 or len(args) == 0) else int_bcast(*args)


def broadcast_shapes(
    *shapes: List[Union[z3.ExprRef, int]]
) -> List[Union[z3.ExprRef, int]]:
    """this function does not check the validity of broadcast. Please always pair it with broadcast_cons"""
    SanityCheck.gt(len(shapes), 0)
    if len(shapes) == 1:
        return shapes[0]
    max_dim = max(map(lambda x: len(x), shapes))
    out_shape = [None] * max_dim
    for j in range(max_dim):
        i = -j - 1
        args_dim_sz = [_prepend_to(x, max_dim)[i] for x in shapes]
        if any(isinstance(s, z3.ExprRef) for s in args_dim_sz):
            out_shape[i] = z3.simplify(z3_bcast(*args_dim_sz))
        else:
            out_shape[i] = int_bcast(*args_dim_sz)
    return out_shape


def broadcast_cons(*shapes: List[Union[z3.ExprRef, int]]) -> List[z3.BoolRef]:
    tgt_shape = broadcast_shapes(*shapes)
    cons = []
    max_dim = len(tgt_shape)
    for j in range(max_dim):
        i = -j - 1
        if isinstance(tgt_shape[i], z3.ExprRef):
            axis_cons = []
            for x in shapes:
                if len(x) > j:
                    axis_cons.append(
                        z3.Or(tensor_eq(x[i], tgt_shape[i]), tensor_eq(x[i], 1))
                    )
            axis_cons = z3.simplify(z3.And(*axis_cons))
            cons.append(axis_cons)
        else:
            args_dim_sz = [_prepend_to(x, max_dim)[i] for x in shapes]
            valid = all(
                tensor_eq(s, tgt_shape[i]) or tensor_eq(s, 1) for s in args_dim_sz
            )
            # TODO: enable this after fixing issue #2
            # assert valid, "Invalid broadcast shapes {}. Specific dim sizes: {}".format(shapes, args_dim_sz)
            cons.append(z3.BoolVal(valid))
    return cons


def broadcast_cons_binary(*shapes: List[Union[z3.ExprRef, int]]) -> List[z3.BoolRef]:
    SanityCheck.eq(len(shapes), 2)
    tgt_shape = broadcast_shapes(*shapes)
    cons = []
    max_dim = len(tgt_shape)
    lhs, rhs = shapes
    lhs = _prepend_to(lhs, max_dim)
    rhs = _prepend_to(rhs, max_dim)
    for j in range(max_dim):
        i = -j - 1
        if isinstance(tgt_shape[i], z3.ExprRef):
            cons.append(
                z3.simplify(
                    z3.Or(
                        tensor_eq(lhs[i], 1),
                        tensor_eq(rhs[i], 1),
                        tensor_eq(lhs[i], rhs[i]),
                    )
                )
            )
        else:
            valid = (
                    tensor_eq(lhs[i], 1)
                    or tensor_eq(rhs[i], 1)
                    or tensor_eq(lhs[i], rhs[i])
            )
            # TODO: enable this after fixing issue #2
            # assert valid, "Invalid broadcast shapes lhs={}, rhs={}".format(lhs, rhs)
            cons.append(z3.BoolVal(valid))
    return cons


def broadcast_to_cons(*shapes: List[Union[z3.ExprRef, int]]) -> List[z3.BoolRef]:
    """Unidirectional broadcast. Last input is the target shape.

    Examples of valid unidirectional broadcast:
    [1, 2, 3] -> [0, 1, 2, 3]
    [1] -> [3]

    Examples of invalid unidirectional broadcast:
    [0, 1, 2, 3] -> [1, 2, 3]
    [3] -> [1]

    Logic: for each dim: src_dim == tgt_dim or src_dim == 1
    """
    srcs, tgt = shapes[:-1], shapes[-1]
    cons = []
    max_dim = len(tgt)
    for src in srcs:
        ConstraintCheck.true(len(src) <= max_dim)
        src = _prepend_to(src, max_dim)
        for i in range(max_dim):
            if isinstance(tgt[i], z3.ExprRef) or isinstance(src[i], z3.ExprRef):
                cons.append(
                    z3.simplify(
                        z3.Or(tensor_eq(src[i], 1), tensor_eq(src[i], tgt[i]))
                    )
                )
            else:
                valid = tensor_eq(src[i], 1) or tensor_eq(src[i], tgt[i])
                # TODO: enable this after fixing issue #2
                # assert valid, "Invalid broadcast shapes lhs={}, rhs={}".format(lhs, rhs)
                cons.append(z3.BoolVal(valid))
    return cons


@mark_abstract("core")
class AbsOpBase(ABC):
    # number of parameters; None means it's fixed that can be inferred through `signature`.
    num_var_param = None
    # Require the input dimension sizes to be equivalent.
    same_inp_dims = False
    # input dtypes: enumerates all possible input dtype combinations. Size of the list is the number of combinations.
    # Each element is a tuple of allowed input dtypes. NOTE: len(list) can >= the # of inputs, for handling ops with arbitrary arity.
    # For example, [(DType.float32, DType.float32), (DType.float64, DType.float64), (DType.int32, DType.int32)] means that
    # this op can accept one of float32xfloat32, float64xfloat64, and int32xint32 as input dtypes.
    in_dtypes: List[Tuple[DType, ...]] = None  # Overwrite me!
    out_dtypes: List[Tuple[DType, ...]] = None
    # lambdas based on the first element.
    irank_relation = None
    orank_relation = None

    limit_domain = False

    dialect = None

    def __init__(self):
        assert self.dialect, "Set dialect with `mark_dialect` or `mark_realize`"

        # `[3, 3]` this means this op requires 2 inputs. Where the 1st one has 2 dimensions, and the 2nd one has 3 dimensions.
        # `-1` means arbitrary dimantions; NOTE: but should be concretized during execution.
        # All symbols of correponding operator must be the constructor's parameters.
        # [ <inp0>(support_dim0, support_dim1, ...), <inp1>(...), ... ]
        self.inp_ranks = []
        # NOTE: the concrete values of out_ranks are not useful. Just make sure the length is correct.
        # NOTE: the output shape of input dimensions should be concretized during the execution.
        self.out_ranks = []
        # NOTE: the input of operator constructors are all Union[int, z3.ExprRef].
        self.extra_attrs = {}
        self._input_like = [None] * self.n_input()
        self._output_like = [None] * self.n_output()

    @classmethod
    def n_input(cls):
        assert cls.in_dtypes is not None
        return len(cls.in_dtypes[0]) if len(cls.in_dtypes) > 0 else 0

    @classmethod
    def n_output(cls):
        assert cls.out_dtypes is not None
        return len(cls.out_dtypes[0]) if len(cls.out_dtypes) > 0 else 0

    @classmethod
    def get_num_var_param(cls):
        if cls.num_var_param is None:
            return len(signature(cls.__init__).parameters) - 1
        return random.choice(cls.num_var_param)

    def bind_input_like(self, input_like: List[AbsTensor]):
        self._input_like = input_like

    def bind_output_like(self, output_like: List[AbsTensor]):
        self._output_like = output_like

    @property
    def input_like(self):
        return self._input_like

    @property
    def output_like(self):
        return self._output_like

    @abstractmethod  # Overload me!
    # Exception means rejection.
    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        raise NotImplementedError

    @check_shape_fn  # Public API.
    def checked_type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        return self.type_transfer(input_shapes)

    # Overload me!
    # Extra constraints for the input tensors.
    # Exception means rejection.
    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.ExprRef, bool]]:
        return []

    @abstractmethod
    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        raise NotImplementedError

    @check_require_fn  # Public API.
    def checked_requires(self, input_shapes):
        extra = []
        for f in ACTIVATED_PATCH.get(self.name(), []):
            extra.extend(f(self, input_shapes))
        return self.requires(input_shapes) + extra

    def n_floats(self, input_shapes: List[AbsTensor]) -> z3.ExprRef:
        return reduce(tensor_add, [i.nelement() for i in self.output_like])

    def __repr__(self) -> str:
        return self.__class__.__name__

    @classmethod
    def __str__(cls) -> str:
        return cls.name()

    @classmethod
    def name(cls) -> str:
        if hasattr(cls, "dialect"):
            return cls.dialect + "." + cls.__name__.split(".")[-1]
        return cls.__name__.split(".")[-1]


class Constant(AbsOpBase):
    in_dtypes = [()]
    out_dtypes = [(i,) for i in DTYPE_GEN_ALL]

    def __str__(self) -> str:
        return self.name() + " " + str(self.extra_attrs).replace(":", "=")

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.inp_ranks = []
        self.out_ranks = [(dim,)]
        self.abs_tensor: AbsTensor = None

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 0)
        return [self.abs_tensor]

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        SanityCheck.eq(len(input_shapes), 0)
        return []

    def __str__(self):
        return "Constant"

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        pass

    @property
    def input_like(self):
        return []

    @property
    def output_like(self):
        return [self.abs_tensor]
class Input(AbsOpBase):
    in_dtypes = [()]
    out_dtypes = [(i,) for i in DTYPE_GEN_ALL]

    def __init__(self, dim: int):
        super().__init__()
        self.inp_ranks = []
        self.out_ranks = [(dim,)]
        self.abs_tensor: AbsTensor = None

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        SanityCheck.eq(len(input_shapes), 0)
        return [self.abs_tensor]

    def requires(self, input_shapes: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
        SanityCheck.eq(len(input_shapes), 0)
        return []

    def __str__(self):
        return "Input"

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        pass

    @property
    def input_like(self):
        return []

    @property
    def output_like(self):
        return [self.abs_tensor]

class ParsedOp(AbsOpBase):
    dialect = 'paddle'

    def __init__(self, op_type: str, attrs: Dict[str, Any], input_names: List[str], output_names: List[str]):
        self.op_type = op_type
        self.attrs = attrs
        self.input_names = input_names
        self.output_names = output_names
        super().__init__()
        self.inp_ranks = [-1] * len(input_names)
        self.out_ranks = [-1] * len(output_names)
        self._input_like = [None] * len(input_names)
        self._output_like = [None] * len(output_names)

    def n_input(self):
        return len(self.input_names)

    def n_output(self):
        return len(self.output_names)

    def __repr__(self):
        return f"{self.__class__.__name__}(inputs={self.n_input()}, attrs={len(self.attrs)})"

    def type_transfer(self, input_shapes): return self._output_like

    def bind_concrete_types(self, tensors, is_input=True):
        abs_tensors = []

        for t in tensors:
            shape = list(t.shape)
            dtype = self._convert_dtype(t.dtype)
            abs_tensors.append(AbsTensor(shape, dtype))
        if is_input:
            self.bind_input_like(abs_tensors)
        else:
            self.bind_output_like(abs_tensors)

    def _convert_dtype(self, framework_dtype):
        raise NotImplementedError("Subclasses must implement _convert_dtype")

    def deduct_inp_ranks_and_dtype(self, out_abs_tensor): raise NotImplementedError


class OpFactory:
    @staticmethod
    @abstractmethod
    def get_op_instance(op_type: str, attrs: Dict, input_names: List, output_names: List) -> ParsedOp:
        pass


class OpLossManager:
    
    _registry: Dict[str, Dict[str, Callable]] = {
        'common': {}
    }

    @classmethod
    def register(cls, op_name_str: str, framework: str = 'common'):

        def wrapper(func):
            fw_key = framework.lower()
            if fw_key not in cls._registry:
                cls._registry[fw_key] = {}

            
            cls._registry[fw_key][op_name_str.lower()] = func
            return func

        return wrapper

    @classmethod
    def dispatch(cls, op_instance: Any, framework: str = None) -> Optional[Callable]:
        
        
        op_type = getattr(op_instance, 'op_type',
                          getattr(op_instance, 'paddle_op_type',
                                  op_instance.__class__.__name__))

        if not op_type:
            return None

        op_key = str(op_type).lower()

        
        if framework:
            fw_key = framework.lower()
            if fw_key in cls._registry:
                if op_key in cls._registry[fw_key]:
                    return cls._registry[fw_key][op_key]

        
        
        return cls._registry['common'].get(op_key, None)


class BaseAdapter(ABC):
    @staticmethod
    @abstractmethod
    def to_abs(obj):
        pass

    @staticmethod
    @abstractmethod
    def is_tensor(obj):
        pass

    @staticmethod
    @abstractmethod
    def parse_op_key(op_obj) -> str:
        pass

    @staticmethod
    @abstractmethod
    def parse_attrs(op_obj) -> dict:
        pass

    @staticmethod
    def has_attr(op, name):
        
        target = getattr(op, 'module', op)

        
        if hasattr(target, name):
            return getattr(target, name) is not None

        
        if hasattr(op, 'attrs') and isinstance(op.attrs, dict):
            return name in op.attrs

        return False

    @classmethod
    def run_reference_inference(cls, op, inputs, symnet=None):
        if hasattr(op, 'module') and op.module is not None:
            return op.module(*inputs)

        return cls.recover_and_run(op, inputs, symnet)

    @classmethod
    def recover_and_run(cls, op, inputs, symnet):
        raise NotImplementedError(f"Adapter {cls.__name__} must implement recover_and_run")
