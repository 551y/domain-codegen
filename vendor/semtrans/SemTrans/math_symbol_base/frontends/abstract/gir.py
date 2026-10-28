# Adapted from NNSmith and modified for SemTrans; see THIRD_PARTY_NOTICES.md.

import inspect
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple, Union

from SemTrans.math_symbol_base.frontends.abstract.op import (
    AbsOpBase,
    Placeholder
)


@dataclass
class InstExpr:
    op: Union[AbsOpBase, Placeholder]
    args: Tuple[str, ...]

    def __str__(self):
        return f"{self.op}({', '.join(self.args)})"

    def n_input(self):
        assert len(self.args) == self.op.n_input()
        return len(self.args)

    def n_output(self):
        return self.op.n_output()


def _make_new_id_from_used(used_ids: Set[int]) -> int:
    for i in range(2 ** 16):
        if i not in used_ids:
            return i

    raise RuntimeError("Cannot find a new id within [0, 2**16)")


def id_checker(id):
    assert id >= 0, "Identifier must be non-negative"


VARNAME_SPLITTER = "_"


class Instruction:

    def __init__(self, iexpr: InstExpr, retvals: Tuple[str, ...], runner=None):
        self.iexpr = iexpr
        self.retvals_list = retvals  
        self.runner = runner  

    def retvals(self) -> List[str]:
        return list(self.retvals_list)

    def retval(self, index=0) -> str:
        return self.retvals_list[index]

    def n_input(self):
        return self.iexpr.n_input()

    def n_output(self):
        return len(self.retvals_list)

    def __str__(self):
        return f"{', '.join(self.retvals_list)} = {self.iexpr}"


def binary_associative_chain(
        input_names: Sequence[str], output_name: str, intermediate_prefix: str
) -> List[Tuple[Tuple[str, ...], str]]:
    """Return a deterministic binary chain for an associative tensor op.

    Frontends use this for variadic concatenation.  Keeping the graph rewrite
    here gives PyTorch, TensorFlow, and Paddle the same structural MathIR while
    leaving the framework-specific runner in each SymbolNet.
    """
    names = tuple(str(name) for name in input_names)
    if not names:
        raise ValueError("An associative tensor operation requires at least one input")
    if len(names) == 1:
        return [((names[0],), str(output_name))]

    steps = []
    current = names[0]
    for index, next_name in enumerate(names[1:], start=1):
        is_last = index == len(names) - 1
        step_output = str(output_name) if is_last else f"{intermediate_prefix}_{index}"
        steps.append(((current, next_name), step_output))
        current = step_output
    return steps


InitEvent = namedtuple('InitEvent', [
    'framework',  # 'pytorch' or 'paddle'
    'api_name',  # e.g., 'kaiming_uniform_', 'XavierUniform'
    'args',  
    'kwargs',  
    'filepath',  
    'lineno',  
    'code_line',  
    'normalized_args',  
    'is_explicit'  
])


class BaseInitTracer:
    def __init__(self, user_dirs=None):
        self.events = []
        self._processing = False

        if user_dirs:
            self.user_dirs = [os.path.abspath(d).lower() for d in user_dirs]
        else:
            self.user_dirs = []
            print("[InitTracer] WARNING: No user_dirs provided. Detection might fail.")

    def _get_call_origin(self):
        stack = inspect.stack()

        
        for i in range(2, len(stack)):
            frame = stack[i]

            
            raw_path = frame.filename
            norm_path = os.path.abspath(raw_path).lower()

            
            if "init_trace" in norm_path or "tracers" in norm_path:
                continue

            
            
            
            is_target_model_code = False
            for d in self.user_dirs:
                if norm_path.startswith(d):
                    is_target_model_code = True
                    break

            if is_target_model_code:
                
                code = ""
                try:
                    code = frame.code_context[0].strip() if frame.code_context else ""
                except:
                    pass

                
                
                
                
                

                
                
                

                
                
                
                is_explicit = (i == 2)

                
                
                direct_caller_path = os.path.abspath(stack[2].filename).lower()
                is_direct_caller_system = any(
                    x in direct_caller_path for x in ['torch', 'paddle', 'site-packages', 'dist-packages'])

                if is_direct_caller_system:
                    is_explicit = False

                return raw_path, frame.lineno, code, is_explicit

        
        return "system_internal", 0, "ignored", False

    def _normalize_args(self, api_name, args, kwargs):
        return {}


