import numpy as np
import sympy
from sympy import Function, S, Add, Symbol, exp, Integer, AtomicExpr, IndexedBase, Basic, Expr, Indexed, Float
from sympy.logic.boolalg import Boolean


# =========================================================================
# 0. Helper Functions
# =========================================================================
def _is_numpy_array(obj):
    return type(obj).__module__ == 'numpy' and type(obj).__name__ == 'ndarray'


def _is_numpy_scalar(obj):
    return type(obj).__module__ == 'numpy' and hasattr(obj, 'dtype')


def _ensure_symbol(obj):
    if type(obj).__name__ == 'SymStr':
        return Symbol(obj.name, integer=True)
    return obj


def _clean_limits(limits):
    new_limits = []
    for lim in limits:
        l_args = list(lim)
        if len(l_args) >= 1:
            l_args[0] = _ensure_symbol(l_args[0])
        new_limits.append(tuple(l_args))
    return tuple(new_limits)


# =========================================================================
# 1. Custom Math Symbol Class
# =========================================================================

class SymStr(Symbol):
    def __new__(cls, name):
        if isinstance(name, str):
            name = name.strip("'").strip('"')
        return super().__new__(cls, name)

    def _sympystr(self, printer):
        return f"'{self.name}'"

    def _latex(self, printer):
        return f"\\text{{{self.name}}}"

    def _eval_Eq(self, other):
        if isinstance(other, SymStr):
            return S.true if self.name == other.name else S.false
        if isinstance(other, str):
            return S.true if self.name == other.strip("'").strip('"') else S.false
        return None

class GetItem(Function):
    nargs = 2

    @classmethod
    def eval(cls, obj, idx):
        try:
            if hasattr(idx, 'is_Integer') and idx.is_Integer:
                idx = int(idx)
            elif isinstance(idx, int):
                pass
            else:
                return None
        except:
            return None

        if isinstance(obj, (list, tuple, np.ndarray)):
            try:
                if idx < 0: idx += len(obj)
                if 0 <= idx < len(obj):
                    return _ensure_sympy(obj[idx])
            except:
                pass

        if isinstance(obj, sympy.Tuple):
            try:
                if idx < 0: idx += len(obj)
                if 0 <= idx < len(obj):
                    return obj[idx]
            except:
                pass

        if isinstance(obj, Function) and isinstance(obj, join):
            pass

        return None


def _ensure_sympy(val):
    if isinstance(val, (int, float)):
        return sympy.sympify(val)
    return val


class TensorOp(Function):
    is_Tensor = True
    @classmethod
    def eval(cls, *args): return None


class NoneSymbol(AtomicExpr):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NoneSymbol, cls).__new__(cls)
        return cls._instance

    def _sympystr(self, printer):
        return "None"

    def __repr__(self):
        return "None"

    def _hashable_content(self):
        return (self.__class__.__name__,)


S_None = NoneSymbol()


# =========================================================================
# 2. Reduction Ops
# =========================================================================
class TensorSum(TensorOp):
    nargs = tuple(range(1, 100))

    @classmethod
    def eval(cls, *args):
        if not args: return None
        expr = args[0]
        if expr == 0 or expr is S.Zero:
            return S.Zero
        return None

    def _eval_expand_add(self, **hints):
        if len(self.args) == 0: return self
        expr = self.args[0]
        limits = self.args[1:]
        if expr.is_Add:
            return Add(*[self.func(arg, *limits) for arg in expr.args])
        return self

    def as_native_loop(self):
        if len(self.args) <= 1: return None
        try:
            return sympy.Sum(self.args[0], *_clean_limits(self.args[1:]))
        except Exception:
            return None


class TensorProd(TensorOp):
    nargs = tuple(range(1, 100))

    @classmethod
    def eval(cls, *args):
        if not args: return None
        expr = args[0]
        if expr == 0 or expr is S.Zero: return S.Zero
        if expr == 1 or expr is S.One: return S.One

        if len(args) <= 2 and hasattr(expr, 'func') and expr.func == exp:
            return exp(TensorSum(expr.args[0], *args[1:]))
        return None

    def as_native_loop(self):
        if len(self.args) <= 1: return None
        try:
            return sympy.Product(self.args[0], *_clean_limits(self.args[1:]))
        except Exception:
            return None


class BitwiseAnd(Function):
    nargs = 2

    @classmethod
    def eval(cls, x, y):
        if x == y: return x

        if x == S.Zero or y == S.Zero: return S.Zero

        if x == S.NegativeOne: return y
        if y == S.NegativeOne: return x

        if isinstance(x, Integer) and isinstance(y, Integer):
            return Integer(int(x) & int(y))
        return None


class BitwiseOr(Function):

    nargs = 2

    @classmethod
    def eval(cls, x, y):
        if x == y: return x
        if x == S.Zero: return y
        if y == S.Zero: return x
        if x == S.NegativeOne or y == S.NegativeOne: return S.NegativeOne

        if isinstance(x, Integer) and isinstance(y, Integer):
            return Integer(int(x) | int(y))
        return None


class BitwiseXor(Function):
    nargs = 2

    @classmethod
    def eval(cls, x, y):
        if x == y: return S.Zero
        if x == S.Zero: return y
        if y == S.Zero: return x

        if isinstance(x, Integer) and isinstance(y, Integer):
            return Integer(int(x) ^ int(y))
        return None


class BitwiseNot(Function):
    nargs = 1

    @classmethod
    def eval(cls, x):
        if isinstance(x, cls):
            return x.args[0]

        if isinstance(x, Integer):
            return Integer(~int(x))
        return None


# =========================================================================

# =========================================================================
class Mean(TensorOp):
    def _eval_expand_add(self, **hints):
        expr = self.args[0]
        dims = self.args[1:]

        if isinstance(expr, Expr) and expr.is_Add:
            return Add(*[self.func(arg, *dims) for arg in expr.args])
        return self

    def _eval_expand_mul(self, **hints):
        expr = self.args[0]
        dims = self.args[1:]

        if isinstance(expr, Expr) and expr.is_Mul:
            coeff, remaining = expr.as_coeff_Mul()
            if coeff != 1:
                return coeff * self.func(remaining, *dims)
        return self

    def _eval_simplify(self, **kwargs):
        arg = self.args[0]
        dims = self.args[1:]

        if isinstance(arg, Mean) and arg.args[1:] == dims:
            return arg

        if hasattr(arg, 'is_Number') and arg.is_Number:
            return arg

        return self


class Var(TensorOp):
    def _eval_expand_mul(self, **hints):
        expr = self.args[0]
        dims = self.args[1:]

        if isinstance(expr, Expr) and expr.is_Mul:
            coeff, remaining = expr.as_coeff_Mul()

            if coeff.is_Number and coeff != 1:
                return (coeff ** 2) * self.func(remaining, *dims)
        return self

    def _eval_simplify(self, **kwargs):
        expr = self.args[0]
        dims = self.args[1:]

        if isinstance(expr, Expr) and expr.is_Add:
            new_add_args = []
            for arg in expr.args:
                if hasattr(arg, 'is_Number') and arg.is_Number:
                    continue
                new_add_args.append(arg)

            if len(new_add_args) < len(expr.args):
                if not new_add_args: return S.Zero
                return self.func(Add(*new_add_args), *dims)

        return self


class Reshape(TensorOp):
    nargs = (2, 99)
    def _eval_simplify(self, **kwargs):
        tensor = self.args[0]
        new_shape = self.args[1:]
        if isinstance(tensor, Reshape):
            return self.func(tensor.args[0], *new_shape)
        return self


class join(Function):
    nargs = None

    @classmethod
    def eval(cls, *args):
        new_args = []
        all_resolved = True

        for arg in args:
            if isinstance(arg, sympy.Tuple):
                new_args.extend(arg.args)
            elif isinstance(arg, (list, tuple)):
                new_args.extend(arg)
            elif _is_numpy_array(arg):
                try:
                    new_args.extend(arg.tolist())
                except:
                    try:
                        new_args.append(arg.item())
                    except:
                        all_resolved = False
            elif isinstance(arg, (Symbol, Function, IndexedBase)) and not arg.is_Number:
                all_resolved = False
                break
            else:
                if _is_numpy_scalar(arg):
                    try:
                        arg = arg.item()
                    except:
                        pass
                new_args.append(arg)

        if all_resolved:
            return sympy.Tuple(*new_args)
        return None


class Rank(Function):
    nargs = 1

    @classmethod
    def eval(cls, obj):
        if isinstance(obj, (list, tuple, np.ndarray)):
            return Integer(len(obj))

        if isinstance(obj, sympy.Tuple):
            return Integer(len(obj))

        if isinstance(obj, join):
            total_len = 0
            for arg in obj.args:
                sub_len = cls.eval(arg)
                if isinstance(sub_len, (int, Integer)):
                    total_len += sub_len
                else:
                    return None
            return Integer(total_len)

        if isinstance(obj, (IndexedBase, Indexed)):
            pass

        return None


class is_list(Function, Boolean):
    nargs = 1
    is_Boolean = True

    @classmethod
    def eval(cls, arg):
        if isinstance(arg, (list, tuple, np.ndarray, sympy.Tuple)):
            return S.true

        if isinstance(arg, join):
            return S.true

        if isinstance(arg, (int, float, str, bool, Integer, Float, SymStr)):
            return S.false

        if isinstance(arg, (IndexedBase, Indexed)):
            return S.false

        return None


class Transpose(TensorOp):
    nargs = (1, 99)


class Concat(TensorOp):
    nargs = (2, 99)

